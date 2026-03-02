from detect_segment_crop import *
from detect_segment_crop_batch import *
#from embed import *
import tqdm

# TODO impliment option from dir as well
from dotenv import load_dotenv
from huggingface_hub import login, upload_folder #load_dataset
load_dotenv(".env")  # Loads variables from .env into environment
hf_token = os.getenv("HF_TOKEN")
#print(hf_token)
login(token=hf_token)

def process_csv_with_pipeline(
    input_csv,
    config_yaml,
    crop_prefix="crop",
    verbose=True
):
    # Load CSV and YAML config
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "./output")
    output_csv = config.get("output_csv", "./output/results.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output records
    output_records = []

    #Make sure images exist and remove if not
    df = df[df[config.get("filepath_col","image_path")].apply(lambda x: os.path.isfile(x))]

    print(f"Processing {len(df)} images...")
    if not config.get("batch_mode", True):
        for idx, row in df.iterrows():
            image_path = row[config.get("filepath_col", "image_path")]
            # Extract config values
            text_prompt = config.get("text_prompt", "")
            sam2_checkpoint = config["sam2_checkpoint"]
            model_cfg = config["sam2_model_cfg"]
            coco_json_path = config.get("coco_json_path", f"{output_dir}/nms_boxes_{idx}.json")
            device = config.get("device", None)
            visualize = config.get("visualize", False)

            if verbose:
                print(f"Processing row {idx+1}/{len(df)}: {image_path}, with prompt '{text_prompt}'")

            # Run pipeline
            crops, all_label_masks_squeezed, all_labels_scores, nms_boxes, nms_labels, nms_scores = detect_segment_crop_pipeline(
                image_path,
                text_prompt,
                sam2_checkpoint,
                model_cfg,
                coco_json_path=coco_json_path,
                device=device,
                visualize=visualize
            )

            # Embed crops
            embeddings = bioclip_embed_batch(crops)

            # Save crops and prepare output rows
            for crop_idx, (crop, emb) in enumerate(zip(crops, embeddings)):
                crop_filename = f"{crop_prefix}_{idx}_{crop_idx}.png"
                crop_path = os.path.join(output_dir, crop_filename)
                if config.get("save_images", True):
                    crop.save(crop_path)
                # Prepare output row: duplicate metadata, add crop path and embedding
                output_row = row.to_dict()
                output_row["crop_path"] = crop_path
                output_row["embedding"] = emb.tolist()  # Save as list for CSV compatibility
                output_records.append(output_row)
    else:
        # batch processing
        image_paths = df[config.get("filepath_col", "image_path")].tolist()
        text_prompts = [config.get("text_prompt", "")] * len(image_paths)
        sam2_checkpoint = config["sam2_checkpoint"] #TODO make this better just kwargs
        model_cfg = config["sam2_model_cfg"]
        coco_json_paths = [f"{output_dir}/nms_boxes_{i}.json" for i in range(len(image_paths))]
        device = config.get("device", None)
        visualize = config.get("visualize", False)

        #FIXME Detection model from config
        if config.get("segmentation", True) or config.get("detection", True):
            if verbose:
                print(f"Running batched detection and segmentation on {len(image_paths)} images...")
            crops_batch, all_label_masks_squeezed_batch, all_labels_scores_batch, nms_boxes_batch, nms_labels_batch, nms_scores_batch = detect_segment_crop_pipeline_batch(
                image_paths,
                text_prompts,
                sam2_checkpoint,
                model_cfg,
                coco_json_paths=coco_json_paths,
                device=device,
                visualize=visualize,
                kwargs=config
            )
            print(f"Found {sum(len(crops) for crops in crops_batch)} total crops across {len(image_paths)} images.")
        else:
            if verbose:
                print(f"Skipping detection and segmentation, using full images for {len(image_paths)} images...")
            crops_batch = []
            for img_path in image_paths:
                image = Image.open(img_path).convert("RGB")
                crops_batch.append([image])  # single crop which is the whole image

        if config.get("embedding_model", "Bioclip") == "Bioclip":
            embeddings_batch = bioclip_embed_batch(crops_batch)
        elif config.get("embedding_model", "Bioclip") == "DINOv3":
            embeddings_batch = Dinov3_predict_batch(crops_batch)
        else:
            print(f"Unknown embedding model {config.get('embedding_model')}, skipping embedding.")
            embeddings_batch = [np.array([]) * len(crops) for crops in crops_batch]

        for idx, (row, crops, embeddings) in tqdm(enumerate(zip(df.to_dict(orient="records"), crops_batch, embeddings_batch)), total=len(df), desc="Saving Crops"):
            if not embeddings:
                embeddings = [None] * len(crops)

            for crop_idx, (crop, emb) in enumerate(zip(crops, embeddings)): #TODO verbose + tqdm
                crop_filename = f"{crop_prefix}_{idx}_{crop_idx}.png"
                crop_path = os.path.join(output_dir, crop_filename)
                #print(f"Saving crop {crop_idx} for image {idx} to {crop_path}")
                if config.get("save_images", True):
                    crop.save(crop_path)
                output_row = row.copy()
                output_row["crop_path"] = crop_path
                output_row["embedding"] = emb.tolist() if emb is not None else None
                output_records.append(output_row)

    # Write output CSV
    out_df = pd.DataFrame(output_records)
    out_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_csv> <config_yaml>")
        sys.exit(1)

    input_csv = sys.argv[1]
    config_yaml = sys.argv[2]

    process_csv_with_pipeline(
        input_csv,
        config_yaml
    )
