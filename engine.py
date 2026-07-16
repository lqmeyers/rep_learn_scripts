#########################################
# Engine for processing images using detection, segmentation, and embedding pipelines.
#
# This script reads an input CSV and configuration YAML, processes images (optionally in batch),
# runs detection/segmentation, embeds crops, and saves results to an output CSV.
#
# Requirements:
# - .env file with HF_TOKEN for HuggingFace authentication
# Configuration YAML specifying model checkpoints, options, and output paths
#
# Author: L Meyers
#########################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import pandas as pd
import yaml
from PIL import Image
import numpy as np
import torch
import tqdm

from scripts.detect_segment_crop_batch import *
from scripts.detect_segment_crop import *
from scripts.embed import (
    save_patch_embedding_h5,
    dinov3_class_patch_embed_batch,
)

from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables and authenticate with HuggingFace
load_dotenv(".env")
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def _to_numpy(x):
    """CLS/patch embeddings come out of DINOv3 as torch tensors; the CSV path may hand us plain arrays. Normalize to float32 numpy either way."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def save_saev_h5_from_embeddings(
    cls_batch, patch_batch, rows, config, output_dir, crop_prefix="crop", verbose=True
):
    """
    Write already-computed DINOv3 CLS + patch embeddings into the per-image .h5 +
    embeddings_index.csv layout consumed by saev/scripts/h5_to_saev_shards.py.

    Runs no model: `cls_batch` and `patch_batch` are the outputs of the main embedding
    pass, aligned one-to-one with `rows`. Every crop becomes its own .h5 (token 0 = CLS,
    tokens 1: = patches, matching saev's convention). All crops must already share one
    patch grid (resize to a fixed square before embedding); the converter needs a fixed
    content_tokens_per_example.

    Args:
        cls_batch (list): Per row, (n_crops, D) CLS embeddings (np.ndarray or torch.Tensor).
        patch_batch (list): Per row, (n_crops, N, D) patch embeddings (np.ndarray or torch.Tensor).
        rows (list of dict): Per-image metadata rows, aligned with cls_batch / patch_batch.
        config (dict): Parsed YAML config (patch_embedding_dir / split_col / label_col / ...).
        output_dir (str): Pipeline output dir, used to derive the default embedding dir.
        crop_prefix (str): Filename prefix for saved .h5 files.
    """
    assert len(cls_batch) == len(patch_batch) == len(rows), (
        f"cls/patch/rows length mismatch: {len(cls_batch)}, {len(patch_batch)}, {len(rows)}."
    )

    patch_dir = config.get("patch_embedding_dir", os.path.join(output_dir, "saev_h5"))
    model_id = config.get(
        "dino_patch_model_id", "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    )
    split_col = config.get("split_col", None)
    label_col = config.get("label_col", None)

    os.makedirs(patch_dir, exist_ok=True)
    if verbose:
        print(f"Writing saev-format patch embeddings to {patch_dir}")

    index_rows = []
    split_counters = {}
    grid_ref = None

    for row, cls_crops, patch_crops in tqdm.tqdm(
        zip(rows, cls_batch, patch_batch), total=len(rows), desc="Saving saev h5"
    ):
        cls_crops = _to_numpy(cls_crops)  # (n_crops, D)
        patch_crops = _to_numpy(patch_crops)  # (n_crops, N, D)
        if cls_crops.size == 0:
            continue
        assert cls_crops.shape[0] == patch_crops.shape[0], (
            f"cls/patch crop count mismatch: {cls_crops.shape[0]} != {patch_crops.shape[0]}."
        )

        split = str(row.get(split_col, "train")) if split_col else "train"
        label = (
            int(row.get(label_col, -1))
            if (label_col and row.get(label_col) is not None)
            else -1
        )

        n_patches = patch_crops.shape[1]
        gh = gw = int(round(n_patches**0.5))
        assert gh * gw == n_patches, f"Non-square patch grid: {n_patches} patches."
        if grid_ref is None:
            grid_ref = (gh, gw)
        assert (gh, gw) == grid_ref, (
            f"Non-uniform patch grid across crops: {(gh, gw)} != {grid_ref}. Resize crops to a fixed square before embedding."
        )

        os.makedirs(os.path.join(patch_dir, split), exist_ok=True)
        for j in range(cls_crops.shape[0]):
            ds_idx = split_counters.get(split, 0)
            split_counters[split] = ds_idx + 1
            rel = os.path.join(split, f"{crop_prefix}_{ds_idx:06d}.h5")
            meta = {
                "split": split,
                "dataset_index": ds_idx,
                "label": label,
                "model_name": model_id,
            }
            save_patch_embedding_h5(
                os.path.join(patch_dir, rel),
                patch_crops[j],
                cls_crops[j],
                (gh, gw),
                meta,
            )
            index_rows.append(
                {
                    "split": split,
                    "dataset_index": ds_idx,
                    "label": label,
                    "grid_h": gh,
                    "grid_w": gw,
                    "embedding_filepath": rel,
                }
            )

    index_path = os.path.join(patch_dir, "embeddings_index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    if verbose:
        print(f"Wrote {len(index_rows)} embeddings and index to {index_path}")


def process_csv_with_pipeline(
    input_csv: str, config_yaml: str, crop_prefix: str = "crop", verbose: bool = True
):
    """
    Processes images listed in a CSV using detection/segmentation and embedding pipelines.

    Args:
        input_csv (str): Path to input CSV file containing image paths and metadata.
        config_yaml (str): Path to YAML config file with pipeline options.
        crop_prefix (str): Prefix for saved crop image filenames.
        verbose (bool): If True, prints progress information.
    """
    # Load CSV and YAML config
    df = pd.read_csv(input_csv)
    if verbose:
        print(f"Loaded {len(df)} rows from {input_csv}")
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "./output")
    output_csv = config.get("output_csv", "./output/results.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output records
    output_records = []

    # make image paths work from local paths
    filepath_col = config.get("filepath_col", "image_path")
    filepath_prefix = config.get("filepath_prefix", "")
    df[filepath_col] = df[filepath_col].apply(
        lambda x: os.path.join(filepath_prefix, x)
    )

    # Filter out rows with missing image files
    df = df[df[filepath_col].apply(lambda x: os.path.isfile(x))]

    if verbose:
        print(f"Processing {len(df)} images...")

    if config.get("batch_size", 1) <= 1:
        # Single-image processing
        single_crops_batch, single_rows = [], []
        for idx, row in df.iterrows():
            image_path = row[filepath_col]
            text_prompt = config.get("text_prompt", "")
            sam2_checkpoint = config["sam2_checkpoint"]
            model_cfg = config["sam2_model_cfg"]
            coco_json_path = config.get(
                "coco_json_path", f"{output_dir}/nms_boxes_{idx}.json"
            )
            device = config.get("device", None)
            visualize = config.get("visualize", False)

            if verbose:
                print(
                    f"Processing row {idx + 1}/{len(df)}: {image_path}, with prompt '{text_prompt}'"
                )

            # Detection/segmentation pipeline
            crops, *_ = detect_segment_crop_pipeline(
                image_path,
                text_prompt,
                sam2_checkpoint,
                model_cfg,
                coco_json_path=coco_json_path,
                device=device,
                visualize=visualize,
            )

            # Embedding
            embeddings = bioclip_embed_batch(crops)

            single_crops_batch.append(crops)
            single_rows.append(row.to_dict())

            # Save crops and record results
            for crop_idx, (crop, emb) in enumerate(zip(crops, embeddings)):
                crop_filename = f"{crop_prefix}_{idx}_{crop_idx}.png"
                crop_path = os.path.join(output_dir, crop_filename)
                if config.get("save_images", True):
                    crop.save(crop_path)
                output_row = row.to_dict()
                output_row["crop_path"] = crop_path
                output_row["embedding"] = emb.tolist()
                output_records.append(output_row)

        # Optional: export DINOv3 patch embeddings in the saev .h5 + index format. The
        # single-image path embeds with BioCLIP (CLS only), so run DINOv3 once here to get
        # the patch tensors the saev converter needs.
        if config.get("save_saev_h5", False):
            input_size = int(config.get("shard_input_size", 750))
            cls_batch, patch_batch = dinov3_class_patch_embed_batch(
                single_crops_batch,
                model_id=config.get(
                    "dino_patch_model_id", "facebook/dinov3-vit7b16-pretrain-lvd1689m"
                ),
                input_size=input_size,
                batch_size=int(config.get("shard_embed_batch_size", 16)),
            )
            save_saev_h5_from_embeddings(
                cls_batch,
                patch_batch,
                single_rows,
                config,
                output_dir,
                crop_prefix=crop_prefix,
                verbose=verbose,
            )
    else:
        # Batch processing
        image_paths = df[filepath_col].tolist()
        text_prompts = [config.get("text_prompt", "")] * len(image_paths)
        # Only required when detection/segmentation actually runs; use .get so a pure
        # save_saev_h5 export on full images works without SAM2 config keys.
        sam2_checkpoint = config.get("sam2_checkpoint")
        model_cfg = config.get("sam2_model_cfg")
        coco_json_paths = [
            f"{output_dir}/nms_boxes_{i}.json" for i in range(len(image_paths))
        ]
        device = config.get("device", None)
        visualize = config.get("visualize", False)

        if config.get("segmentation", True) or config.get("detection", True):
            if verbose:
                print(
                    f"Running batched detection and segmentation on {len(image_paths)} images..."
                )
            crops_batch, *_ = detect_segment_crop_pipeline_batch(
                image_paths,
                text_prompts,
                sam2_checkpoint,
                model_cfg,
                coco_json_paths=coco_json_paths,
                device=device,
                visualize=visualize,
                kwargs=config,
            )
            if verbose:
                print(
                    f"Found {sum(len(crops) for crops in crops_batch)} total crops across {len(image_paths)} images."
                )
        else:
            if verbose:
                print(
                    f"Skipping detection and segmentation, using full images for {len(image_paths)} images..."
                )
            # Keep the list-of-lists shape every downstream consumer expects:
            # one inner list of crops per row (here a single full image).
            crops_batch = []
            for img_path in tqdm.tqdm(image_paths, "Opening Images"):
                crops_batch.append([Image.open(img_path).convert("RGB")])

        # Embedding selection. DINOv3 returns CLS + patches in one pass so the saev h5
        # export below can reuse them instead of recomputing.
        embedding_model = config.get("embedding_model", "Bioclip")
        cls_batch = patch_batch = None
        if embedding_model == "Bioclip":
            embeddings_batch = bioclip_embed_batch(crops_batch)
        elif embedding_model == "DINOv3":
            # A fixed square input_size gives every crop the same patch grid, which the saev converter requires.
            input_size = (
                int(config.get("shard_input_size", 750))
                if config.get("save_saev_h5", False)
                else None
            )
            cls_batch, patch_batch = dinov3_class_patch_embed_batch(
                crops_batch,
                model_id=config.get(
                    "dino_patch_model_id", "facebook/dinov3-vitl16-pretrain-lvd1689m"
                ),
                input_size=input_size,
                batch_size=int(config.get("shard_embed_batch_size", 8)),
            )
            embeddings_batch = cls_batch
        else:
            print(f"Unknown embedding model {embedding_model}, skipping embedding.")
            embeddings_batch = [np.array([]) for _ in crops_batch]

        # Save crops and record results
        for idx, (row, crops, embeddings) in tqdm.tqdm(
            enumerate(zip(df.to_dict(orient="records"), crops_batch, embeddings_batch)),
            total=len(df),
            desc="Saving Crops",
        ):
            if embeddings is None or len(embeddings) == 0:
                embeddings = [None] * len(crops)
            for crop_idx, (crop, emb) in enumerate(zip(crops, embeddings)):
                crop_filename = f"{crop_prefix}_{idx}_{crop_idx}.png"
                crop_path = os.path.join(output_dir, crop_filename)
                if config.get("save_images", True):
                    crop.save(crop_path)
                output_row = row.copy()
                output_row["crop_path"] = crop_path
                output_row["embedding"] = emb.tolist() if emb is not None else None
                output_records.append(output_row)

        # Optional: export DINOv3 patch embeddings in the saev .h5 + index format, reusing
        # the CLS + patch tensors already computed above (no recompute).
        if config.get("save_saev_h5", False):
            assert cls_batch is not None and patch_batch is not None, (
                "save_saev_h5 needs embedding_model: DINOv3 (it produces the patch embeddings)."
            )
            save_saev_h5_from_embeddings(
                cls_batch,
                patch_batch,
                df.to_dict(orient="records"),
                config,
                output_dir,
                crop_prefix=crop_prefix,
                verbose=verbose,
            )

    # Write output CSV
    out_df = pd.DataFrame(output_records)
    out_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_csv> <config_yaml>")
        sys.exit(1)
    input_csv = sys.argv[1]
    config_yaml = sys.argv[2]
    process_csv_with_pipeline(input_csv, config_yaml)
