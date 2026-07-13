#########################################
#Engine for processing images using detection, segmentation, and embedding pipelines.
#
#This script reads an input CSV and configuration YAML, processes images (optionally in batch),
#runs detection/segmentation, embeds crops, and saves results to an output CSV.
#
#Requirements:
#- .env file with HF_TOKEN for HuggingFace authentication
# Configuration YAML specifying model checkpoints, options, and output paths
#
#Author: L Meyers
#########################################

import os
import pandas as pd
import yaml
from PIL import Image
import numpy as np
import tqdm

from scripts.detect_segment_crop_batch import *
from scripts.detect_segment_crop import *
from scripts.embed import (
    save_patch_embedding_h5,
    load_dinov3_model,
    get_dinov3_class_patch_embeddings_batch,
)

from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables and authenticate with HuggingFace
load_dotenv(".env")
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

def save_saev_h5_from_crops(crops_batch, rows, config, output_dir, crop_prefix="crop", verbose=True):
    """
    Export DINOv3 patch embeddings for every crop into the per-image .h5 +
    embeddings_index.csv layout consumed by saev/scripts/h5_to_saev_shards.py.

    Every crop is resized to a fixed square `shard_input_size` so all examples share one
    patch grid (a hard precondition of the saev converter). Reuses the existing DINOv3
    extraction functions; writes nothing that depends on the saev package.

    Args:
        crops_batch (list of list of PIL.Image): One inner list of crops per input row.
        rows (list of dict): Per-image metadata rows, aligned with crops_batch.
        config (dict): Parsed YAML config (reads save_saev_h5 / patch_embedding_dir / ...).
        output_dir (str): Pipeline output dir, used to derive the default embedding dir.
        crop_prefix (str): Filename prefix for saved .h5 files.
    """
    patch_dir = config.get("patch_embedding_dir", os.path.join(output_dir, "saev_h5"))
    input_size = int(config.get("shard_input_size", 750))
    model_id = config.get("dino_patch_model_id", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
    split_col = config.get("split_col", None)
    label_col = config.get("label_col", None)
    embed_batch_size = int(config.get("shard_embed_batch_size", 8))

    os.makedirs(patch_dir, exist_ok=True)
    if verbose:
        print(f"Exporting saev-format patch embeddings to {patch_dir} (input_size={input_size}, model={model_id})")

    processor, model = load_dinov3_model(model_id, device="cuda")

    # get_dinov3_class_patch_embeddings_batch runs with do_resize=False, so the input side
    # must be an exact multiple of the model's patch size. Snap it down to be safe.
    patch_size = getattr(model.config, "patch_size", 16)
    snapped = (input_size // patch_size) * patch_size
    if snapped != input_size:
        if verbose:
            print(f"Adjusting shard_input_size {input_size} -> {snapped} (multiple of patch_size {patch_size})")
        input_size = snapped
    assert input_size >= patch_size, f"shard_input_size too small for patch_size {patch_size}."

    index_rows = []
    split_counters = {}
    grid_ref = None

    for row, crops in tqdm.tqdm(zip(rows, crops_batch), total=len(rows), desc="Saving saev h5"):
        split = str(row.get(split_col, "train")) if split_col else "train"
        label = int(row.get(label_col, -1)) if (label_col and row.get(label_col) is not None) else -1

        # Resize every crop to a fixed square so the patch grid is uniform across examples.
        resized = []
        for crop in crops:
            if crop.mode != "RGB":
                crop = crop.convert("RGB")
            resized.append(crop.resize((input_size, input_size)))
        if not resized:
            continue

        # Extract CLS + patch embeddings in mini-batches.
        cls_chunks, patch_chunks = [], []
        for i in range(0, len(resized), embed_batch_size):
            batch = resized[i:i + embed_batch_size]
            cls_tokens, patches = get_dinov3_class_patch_embeddings_batch(batch, processor, model, device="cuda")
            cls_chunks.append(cls_tokens.cpu().float().numpy())
            patch_chunks.append(patches.cpu().float().numpy())
        cls_all = np.concatenate(cls_chunks, axis=0)     # (n_crops, D)
        patch_all = np.concatenate(patch_chunks, axis=0)  # (n_crops, N, D)

        n_patches = patch_all.shape[1]
        gh = gw = int(round(n_patches ** 0.5))
        assert gh * gw == n_patches, f"Non-square patch grid: {n_patches} patches."

        os.makedirs(os.path.join(patch_dir, split), exist_ok=True)
        for j in range(cls_all.shape[0]):
            ds_idx = split_counters.get(split, 0)
            split_counters[split] = ds_idx + 1
            rel = os.path.join(split, f"{crop_prefix}_{ds_idx:06d}.h5")
            meta = {
                "split": split,
                "dataset_index": ds_idx,
                "label": label,
                "model_name": model_id,
                "input_size": input_size,
            }
            save_patch_embedding_h5(os.path.join(patch_dir, rel), patch_all[j], cls_all[j], (gh, gw), meta)

            if grid_ref is None:
                grid_ref = (gh, gw)
            assert (gh, gw) == grid_ref, f"Non-uniform patch grid across crops: {(gh, gw)} != {grid_ref}."

            index_rows.append({
                "split": split,
                "dataset_index": ds_idx,
                "label": label,
                "grid_h": gh,
                "grid_w": gw,
                "embedding_filepath": rel,
            })

    index_path = os.path.join(patch_dir, "embeddings_index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    if verbose:
        print(f"Wrote {len(index_rows)} embeddings and index to {index_path}")


def process_csv_with_pipeline(
    input_csv: str,
    config_yaml: str,
    crop_prefix: str = "crop",
    verbose: bool = True
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
    df[filepath_col] = df[filepath_col].apply(lambda x: os.path.join(filepath_prefix, x))

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
            coco_json_path = config.get("coco_json_path", f"{output_dir}/nms_boxes_{idx}.json")
            device = config.get("device", None)
            visualize = config.get("visualize", False)

            if verbose:
                print(f"Processing row {idx+1}/{len(df)}: {image_path}, with prompt '{text_prompt}'")

            # Detection/segmentation pipeline
            crops, *_ = detect_segment_crop_pipeline(
                image_path,
                text_prompt,
                sam2_checkpoint,
                model_cfg,
                coco_json_path=coco_json_path,
                device=device,
                visualize=visualize
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

        # Optional: export DINOv3 patch embeddings in the saev .h5 + index format.
        if config.get("save_saev_h5", False):
            save_saev_h5_from_crops(
                single_crops_batch, single_rows, config, output_dir,
                crop_prefix=crop_prefix, verbose=verbose,
            )
    else:
        # Batch processing
        image_paths = df[filepath_col].tolist()
        text_prompts = [config.get("text_prompt", "")] * len(image_paths)
        # Only required when detection/segmentation actually runs; use .get so a pure
        # save_saev_h5 export on full images works without SAM2 config keys.
        sam2_checkpoint = config.get("sam2_checkpoint")
        model_cfg = config.get("sam2_model_cfg")
        coco_json_paths = [f"{output_dir}/nms_boxes_{i}.json" for i in range(len(image_paths))]
        device = config.get("device", None)
        visualize = config.get("visualize", False)

        if config.get("segmentation", True) or config.get("detection", True):
            if verbose:
                print(f"Running batched detection and segmentation on {len(image_paths)} images...")
            crops_batch, *_ = detect_segment_crop_pipeline_batch(
                image_paths,
                text_prompts,
                sam2_checkpoint,
                model_cfg,
                coco_json_paths=coco_json_paths,
                device=device,
                visualize=visualize,
                kwargs=config
            )
            if verbose:
                print(f"Found {sum(len(crops) for crops in crops_batch)} total crops across {len(image_paths)} images.")
        else:
            if verbose:
                print(f"Skipping detection and segmentation, using full images for {len(image_paths)} images...")
            crops_batch = [[Image.open(img_path).convert("RGB")] for img_path in image_paths]

        # Embedding selection
        embedding_model = config.get("embedding_model", "Bioclip")
        if embedding_model == "Bioclip":
            embeddings_batch = bioclip_embed_batch(crops_batch)
        elif embedding_model == "DINOv3":
            embeddings_batch = Dinov3_predict_batch(crops_batch)
        else:
            print(f"Unknown embedding model {embedding_model}, skipping embedding.")
            embeddings_batch = [np.array([]) for _ in crops_batch]

        # Save crops and record results
        for idx, (row, crops, embeddings) in tqdm.tqdm(
            enumerate(zip(df.to_dict(orient="records"), crops_batch, embeddings_batch)),
            total=len(df),
            desc="Saving Crops"
        ):
            if not embeddings[0]:
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

        # Optional: export DINOv3 patch embeddings in the saev .h5 + index format.
        if config.get("save_saev_h5", False):
            save_saev_h5_from_crops(
                crops_batch, df.to_dict(orient="records"), config, output_dir,
                crop_prefix=crop_prefix, verbose=verbose,
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
