from PIL import Image
import tqdm
from transformers import AutoModel, AutoImageProcessor
from bioclip import TreeOfLifeClassifier, Rank
import numpy as np
import pandas as pd
import h5py
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "../")
from dino_patch.patch_embedding import *


############################################################ Main Embedding Functions ########################################################
def bioclip_embed_batch(crops_batch, classifier=None):
    """
    Batched embedding of crops using BioCLIP.
    Args:
        crops_batch (list of list of PIL.Image)
        classifier (TreeOfLifeClassifier, optional)
    Returns:
        embeddings_batch (list of np.ndarray)
    """
    if classifier is None:
        classifier = TreeOfLifeClassifier()
    embeddings_batch = []
    for crops in crops_batch:
        embeddings = []
        for crop in crops:
            if crop.mode == "RGBA":
                crop = crop.convert("RGB")
            emb = classifier.create_image_features([crop]).cpu().numpy().squeeze()
            embeddings.append(emb)
        embeddings_batch.append(np.array(embeddings))
    return embeddings_batch


####################3
def Dinov3_predict_batch(crops_batch):

    preprocessor, model = load_dinov3_model(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m", device="cuda"
    )

    embeddings_batch = []
    for crops in tqdm.tqdm(
        crops_batch, desc="Embedding crops with DINOv3", total=len(crops_batch)
    ):
        batch_embeddings = []
        for crop in crops:
            if crop.mode == "RGBA":
                crop = crop.convert("RGB")
            inputs = preprocessor(crop, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            batch_embeddings.append(embedding)
        embeddings_batch.append(np.array(batch_embeddings))
    return embeddings_batch


def dinov3_class_patch_embed_batch(
    crops_batch,
    model_id="facebook/dinov3-vit7b16-pretrain-lvd1689m",
    input_size=None,
    batch_size=8,
    device="cuda",
):
    """
    Embed grouped crops with DINOv3, returning CLS and patch embeddings aligned with crops_batch.

    Loads the model once (unlike calling the low-level helper repeatedly). When input_size is given, every crop is resized to (input_size, input_size) so all crops share one patch grid, which the saev shard export requires. input_size is snapped down to a multiple of the model's patch size.

    Args:
        crops_batch (list of list of PIL.Image): One inner list of crops per input row.
        model_id (str): DINOv3 checkpoint id.
        input_size (int | None): Square side to resize every crop to, or None to leave crops as-is.
        batch_size (int): Forward-pass mini-batch size.
        device (str): torch device.

    Returns:
        cls_batch (list of np.ndarray): Per row, (n_crops, D) float32 CLS embeddings.
        patch_batch (list of np.ndarray): Per row, (n_crops, N, D) float32 patch embeddings.
    """
    processor, model = load_dinov3_model(model_id, device=device)

    if input_size is not None:
        patch_size = getattr(model.config, "patch_size", 16)
        input_size = (input_size // patch_size) * patch_size
        assert input_size >= patch_size, (
            f"input_size too small for patch_size {patch_size}."
        )

    cls_batch, patch_batch = [], []
    for crops in tqdm.tqdm(
        crops_batch, total=len(crops_batch), desc="Embedding crops with DINOv3"
    ):
        prepped = []
        for crop in crops:
            if crop.mode != "RGB":
                crop = crop.convert("RGB")
            if input_size is not None:
                crop = crop.resize((input_size, input_size))
            prepped.append(crop)
        if not prepped:
            cls_batch.append(np.empty((0,), dtype=np.float32))
            patch_batch.append(np.empty((0,), dtype=np.float32))
            continue
        cls_chunks, patch_chunks = [], []
        for i in range(0, len(prepped), batch_size):
            cls_tokens, patches = get_dinov3_class_patch_embeddings_batch(
                prepped[i : i + batch_size], processor, model, device=device
            )
            cls_chunks.append(cls_tokens.cpu().float().numpy())
            patch_chunks.append(patches.cpu().float().numpy())
        cls_batch.append(np.concatenate(cls_chunks, axis=0))
        patch_batch.append(np.concatenate(patch_chunks, axis=0))
    return cls_batch, patch_batch


def save_patch_embedding_h5(out_path, patch_emb, cls_emb, grid_hw, meta):
    """
    Save one crop's DINOv3 patch embeddings + CLS token to its own .h5 file, in the
    layout consumed by saev/scripts/h5_to_saev_shards.py.

    Args:
        out_path (str): Destination .h5 path.
        patch_emb (np.ndarray): Patch embeddings of shape (num_patches, D), float32.
        cls_emb (np.ndarray): CLS token of shape (D,), float32.
        grid_hw (tuple): Patch grid (grid_h, grid_w) so num_patches == grid_h * grid_w.
        meta (dict): Extra attributes to store (split, dataset_index, label, model_name...).
    """
    gh, gw = grid_hw
    with h5py.File(out_path, "w") as f:
        f.create_dataset("patch_embeddings", data=patch_emb, compression="gzip")
        f.create_dataset("cls_embedding", data=cls_emb)
        f.attrs["grid_h"] = gh
        f.attrs["grid_w"] = gw
        for k, v in meta.items():
            f.attrs[k] = v


############################################ BioCLIP Embedding Function ############################################
def bioclip_embed(crops, classifier=None):
    """
    Embed crops using BioCLIP.

    Args:
        crops (list of PIL.Image): List of cropped images to embed.
        classifier (TreeOfLifeClassifier, optional): Pre-initialized BioCLIP classifier.

    Returns:
        embeddings (np.ndarray): Array of embeddings for each crop.
    """
    if classifier is None:
        classifier = TreeOfLifeClassifier()

    embeddings = []
    for crop in crops:
        # Convert RGBA crop to RGB before embedding
        if crop.mode == "RGBA":
            crop = crop.convert("RGB")
        emb = classifier.create_image_features([crop]).cpu().numpy().squeeze()
        embeddings.append(emb)

    return np.array(embeddings)


def bioclip_embed_batch_from_df(
    df,
    image_path_col="local_path",
    classifier=None,
    batch_size=32,
    embedding_col="embedding",
):
    """
    Batch embed images using BioCLIP and add embeddings as a list to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing image paths.
        image_path_col (str): Column name for image paths.
        classifier (TreeOfLifeClassifier, optional): Pre-initialized BioCLIP classifier.
        batch_size (int): Number of images per batch.
        embedding_col (str): Name of the column to store embeddings.

    Returns:
        pd.DataFrame: DataFrame with embeddings added.
    """
    if classifier is None:
        device = "cuda"
        classifier = TreeOfLifeClassifier(device=device)

    embeddings = []
    predictions = []
    df.drop_duplicates()
    paths = df[image_path_col].tolist()
    for i in tqdm.tqdm(
        range(0, len(paths), batch_size), total=len(paths) // batch_size
    ):
        batch_paths = paths[i : i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        batch_emb = classifier.create_image_features(batch_images).cpu().numpy()
        batch_preds = classifier.predict(
            batch_images, Rank.SPECIES, k=1, batch_size=batch_size
        )
        # Update 'file_name' in batch_preds to actual image paths
        for idx, pred in enumerate(batch_preds):
            pred["file_name"] = batch_paths[idx]
            pred["embedding"] = batch_emb[idx].tolist()

        predictions.extend(batch_preds)
        embeddings.extend(batch_emb.tolist())

    pred_df = pd.DataFrame(predictions)
    pred_df.rename(columns={"score": "class_score"}, inplace=True)

    # Fix col name
    df.rename(columns={"score": "detection_score"}, inplace=True)
    cols_to_delete = [
        col for col in pred_df.columns if col in df.columns and col != "file_name"
    ]
    df.drop(cols_to_delete, axis=1, inplace=True)

    df = df.merge(pred_df, left_on=image_path_col, right_on="file_name", how="left")

    emb_save_path = os.path.splitext(df_path)[0] + "_embeddings.pth"
    torch.save(torch.tensor(embeddings), emb_save_path)

    df = df[~df[image_path_col].isna()]

    # if len(embeddings) == len(df):
    #     df[embedding_col] = embeddings
    # Add detection metadata
    df["has_y_edge"] = df.apply(
        lambda row: (row["y_min"] <= 5) | (row["y_max"] >= 1075), axis=1
    )
    df["horizontal_detection"] = df.apply(
        lambda row: (row["x_max"] - row["x_min"]) > (row["y_max"] - row["y_min"]),
        axis=1,
    )
    df["has_x_edge"] = df.apply(
        lambda row: (row["x_min"] <= 5) | (row["x_max"] >= 1915), axis=1
    )
    df["near_edge"] = ((~df["has_x_edge"]) & (~df["horizontal_detection"])) | (
        (~df["has_y_edge"]) & (df["horizontal_detection"])
    )

    return df


#############################
# Dino inference
def dinov3_embed_from_df(
    df,
    image_path_col="crop_path",
    model_name="facebook/dinov3-vit7b16-pretrain-lvd1689",
    batch_size=32,
    embedding_col="embedding",
    limited_mem=False,
):

    # Check GPU
    print("CUDA available:", torch.cuda.is_available())
    num_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_devices)
    for idx in range(num_devices):
        print(f"Device {idx}: {torch.cuda.get_device_name(idx)}")

    preprocessor, model = load_dinov3_model(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m", device="cuda"
    )

    # Prepare paths and output arrays
    paths = df[image_path_col].tolist()
    n_samples = len(paths)
    embedding_dim = (
        model.config.hidden_size if hasattr(model.config, "hidden_size") else 768
    )

    # Preallocate arrays for embeddings
    embeddings = []

    # Batch inference
    for i in tqdm.tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        batch_idx = slice(i, min(i + batch_size, n_samples))
        batch_paths = paths[batch_idx]
        images = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                images.append(Image.new("RGB", input_size))
        inputs = processor(
            images=images, return_tensors="pt", size=list(input_size)
        ).to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        batch_embeddings = outputs.pooler_output.cpu().numpy()
        embeddings.extend(batch_embeddings.tolist())
        torch.cuda.empty_cache()

    df[embedding_col] = embeddings
    return df


if __name__ == "__main__":
    # Usage example:
    df_path = "/home/lmeyers/GH010037/detections.csv"
    dfm = pd.read_csv(df_path)
    dfcm = bioclip_embed_batch_from_df(dfm, image_path_col="crop_path", batch_size=64)
    outpath = os.path.join(
        os.path.dirname(df_path),
        os.path.basename(df_path)[:-4] + "_with_all_embeddings.csv",
    )
    dfcm.to_csv(outpath, index=False)
