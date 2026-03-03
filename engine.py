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

from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables and authenticate with HuggingFace
load_dotenv(".env")
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

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
    else:
        # Batch processing
        image_paths = df[filepath_col].tolist()
        text_prompts = [config.get("text_prompt", "")] * len(image_paths)
        sam2_checkpoint = config["sam2_checkpoint"]
        model_cfg = config["sam2_model_cfg"]
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
