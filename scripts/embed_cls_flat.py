"""
DINOv3 per-image CLS + patch embedder, saev-shard ready.

Reads a CSV of images, resizes each to 752x752 (47*16, so patch_size=16
divides evenly -> 47x47 = 2209 patch grid), embeds with DINOv3, and writes
output in exactly the layout saev/scripts/h5_to_saev_shards.py consumes:

    <out_dir>/
        embeddings_index.csv          # split, dataset_index, label, grid_h, grid_w, embedding_filepath
        <split>/<split>_000000.h5     # datasets: cls_embedding (d,), patch_embeddings (n_patches, d)
        <split>/<split>_000001.h5
        ...

One .h5 per image. Because gzip-compressing + writing thousands of little h5s
is the bottleneck (not the GPU), saving is offloaded to a pool of background
writer threads fed by a bounded queue: the main loop keeps DINOv3 batches
flowing on the GPU while the writers drain the buffer to disk. The bounded
queue also backpressures the GPU so we never pile up unbounded numpy arrays in
RAM. h5py/zlib release the GIL during compression + IO, so threads overlap.

The index CSV lists only files confirmed on disk and is flushed every
--save-every batches for redundancy.

Usage:
    python embed_cls_flat.py --input imgs.csv --out-dir /path/to/out
    python embed_cls_flat.py --input imgs.csv --image-col crop_path \\
        --split-col split --label-col label --batch-size 16 \\
        --writer-threads 6 --buffer 128
"""

import argparse
import os
import queue
import sys
import threading

import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dino_patch.patch_embedding import (
    load_dinov3_model,
    get_dinov3_class_patch_embeddings_batch,
)

IMAGE_SIZE = 752  # 47 * 16, so patches divide evenly for patch_size=16

INDEX_COLUMNS = [
    "split",
    "dataset_index",
    "label",
    "grid_h",
    "grid_w",
    "embedding_filepath",
    "image_path",
]


def save_patch_embedding_h5(out_path, patch_emb, cls_emb, grid_hw, meta):
    """Save one image's patch + CLS embeddings in the saev-shard input layout."""
    gh, gw = grid_hw
    with h5py.File(out_path, "w") as f:
        f.create_dataset("patch_embeddings", data=patch_emb, compression="gzip")
        f.create_dataset("cls_embedding", data=cls_emb)
        f.attrs["grid_h"] = gh
        f.attrs["grid_w"] = gw
        for k, v in meta.items():
            f.attrs[k] = v


class H5WriterPool:
    """Background thread pool that writes per-image h5s from a bounded queue.

    Producers call `submit(job)`; workers write the file and record the job's
    index row (only after a successful write) so the CSV reflects on-disk state.
    """

    def __init__(self, out_dir, n_threads, buffer):
        self.out_dir = out_dir
        self.q = queue.Queue(maxsize=buffer)
        self._done_rows = []
        self._lock = threading.Lock()
        self._error = None
        self.threads = [
            threading.Thread(target=self._worker, daemon=True) for _ in range(n_threads)
        ]
        for t in self.threads:
            t.start()

    def _worker(self):
        while True:
            job = self.q.get()
            if job is None:
                self.q.task_done()
                return
            try:
                if self._error is None:  # skip remaining work once something failed
                    rel, patch_emb, cls_emb, grid_hw, meta, row = job
                    save_patch_embedding_h5(
                        os.path.join(self.out_dir, rel), patch_emb, cls_emb, grid_hw, meta
                    )
                    with self._lock:
                        self._done_rows.append(row)
            except Exception as e:  # surface writer failures to the main thread
                with self._lock:
                    if self._error is None:
                        self._error = e
            finally:
                self.q.task_done()

    def submit(self, job):
        # Re-raise a writer error on the producer side (backpressure point).
        if self._error is not None:
            raise self._error
        self.q.put(job)  # blocks when the buffer is full -> throttles the GPU

    def completed_rows(self):
        with self._lock:
            return list(self._done_rows)

    def raise_if_error(self):
        if self._error is not None:
            raise self._error

    def close(self):
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()
        self.raise_if_error()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV with an image-path column.")
    ap.add_argument("--out-dir", required=True, help="Output dir for h5s + index csv.")
    ap.add_argument("--image-col", default="crop_path", help="Image-path column.")
    ap.add_argument(
        "--split-col",
        default=None,
        help="Column with train/test split; if absent, uses --split for all rows.",
    )
    ap.add_argument("--split", default="train", help="Split label when --split-col unset.")
    ap.add_argument(
        "--label-col",
        default=None,
        help="Column with a class label; if absent, label is left empty.",
    )
    ap.add_argument(
        "--model-id",
        default="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        help="DINOv3 checkpoint id.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Flush the index csv every N batches for redundancy.",
    )
    ap.add_argument(
        "--writer-threads",
        type=int,
        default=6,
        help="Background threads writing h5s in parallel with GPU batches.",
    )
    ap.add_argument(
        "--buffer",
        type=int,
        default=128,
        help="Max pending images in the writer queue (backpressures the GPU).",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "embeddings_index.csv")

    df = pd.read_csv(args.input)
    if args.image_col not in df.columns:
        raise SystemExit(
            f"Column '{args.image_col}' not in {args.input}. Columns: {list(df.columns)}"
        )
    n = len(df)
    if n == 0:
        raise SystemExit(f"No rows in {args.input}")

    paths = df[args.image_col].astype(str).tolist()
    splits = (
        df[args.split_col].astype(str).tolist()
        if args.split_col and args.split_col in df.columns
        else [args.split] * n
    )
    labels = (
        df[args.label_col].tolist()
        if args.label_col and args.label_col in df.columns
        else [""] * n
    )

    print(f"{n} images. Loading model {args.model_id} ...")
    processor, model = load_dinov3_model(args.model_id, device=args.device)
    patch_size = getattr(model.config, "patch_size", 16)
    assert IMAGE_SIZE % patch_size == 0, (
        f"IMAGE_SIZE {IMAGE_SIZE} not divisible by patch_size {patch_size}"
    )

    # Pre-create split subdirs so writer threads never race on os.makedirs.
    for sp in set(splits):
        os.makedirs(os.path.join(args.out_dir, sp), exist_ok=True)

    pool = H5WriterPool(args.out_dir, args.writer_threads, args.buffer)
    split_counts = {}  # per-split running counter, assigned in the main thread

    def flush():
        rows = pool.completed_rows()
        pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(index_path, index=False)

    try:
        n_batches = (n + args.batch_size - 1) // args.batch_size
        for bi in tqdm.tqdm(range(n_batches), desc="Embedding"):
            pool.raise_if_error()
            sl = slice(bi * args.batch_size, (bi + 1) * args.batch_size)
            b_paths, b_splits, b_labels = paths[sl], splits[sl], labels[sl]

            images, meta = [], []
            for p, sp, lab in zip(b_paths, b_splits, b_labels):
                try:
                    img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                except Exception as e:
                    print(f"Skipping {p}: {e}")
                    continue
                images.append(img)
                meta.append((p, sp, lab))
            if not images:
                continue

            cls_tokens, patches = get_dinov3_class_patch_embeddings_batch(
                images, processor, model, device=args.device
            )
            cls_np = cls_tokens.cpu().float().numpy()
            patch_np = patches.cpu().float().numpy()

            num_patches = patch_np.shape[1]
            grid = int(round(num_patches**0.5))
            assert grid * grid == num_patches, f"non-square patch grid: {num_patches}"

            # Copy each row out of the batch arrays so the writer owns its own
            # contiguous buffer (the batch arrays are freed next iteration).
            for i, (p, sp, lab) in enumerate(meta):
                idx = split_counts.get(sp, 0)
                split_counts[sp] = idx + 1
                rel = os.path.join(sp, f"{sp}_{idx:06d}.h5")
                row = {
                    "split": sp,
                    "dataset_index": idx,
                    "label": lab,
                    "grid_h": grid,
                    "grid_w": grid,
                    "embedding_filepath": rel,
                    "image_path": p,
                }
                job = (
                    rel,
                    np.ascontiguousarray(patch_np[i]),
                    np.ascontiguousarray(cls_np[i]),
                    (grid, grid),
                    {"split": sp, "dataset_index": idx, "label": lab, "model_id": args.model_id},
                    row,
                )
                pool.submit(job)  # blocks if the disk falls behind

            if (bi + 1) % args.save_every == 0:
                flush()
    finally:
        pool.close()  # drain the queue, join writers, re-raise any writer error
        flush()

    print(
        f"Done. Wrote {len(pool.completed_rows())} embeddings across splits {dict(split_counts)}.\n"
        f"  index: {index_path}\n"
        f"Feed --embeddings-dir {args.out_dir} to h5_to_saev_shards.py"
    )


if __name__ == "__main__":
    main()
