from PIL import Image
import tqdm
from transformers import AutoModel, AutoImageProcessor
from bioclip import TreeOfLifeClassifier, Rank
import numpy as np
import pandas as pd
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "/home/lmeyers/rep_learn_scripts/dino_patch")
from patch_embedding import *


#df = pd.read_csv("/home/lmeyers/ciclidos/data/combined_ciclid_w_metadata_paths.csv")
#print(len(df))
def bioclip_embed_batch(df, image_path_col="local_path", classifier=None, batch_size=32, embedding_col="embedding"):
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
    for i in tqdm.tqdm(range(0, len(paths), batch_size),total=len(paths)//batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        batch_emb = classifier.create_image_features(batch_images).cpu().numpy()
        batch_preds = classifier.predict(batch_images,Rank.SPECIES,k=1,batch_size=batch_size)
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
    cols_to_delete = [col for col in pred_df.columns if col in df.columns and col != "file_name"]
    df.drop(cols_to_delete, axis=1, inplace=True)

    df = df.merge(pred_df,left_on=image_path_col,right_on="file_name",how="left")
    
    emb_save_path = os.path.splitext(df_path)[0] + "_embeddings.pth"
    torch.save(torch.tensor(embeddings), emb_save_path)

   
    df = df[~df[image_path_col].isna()]

    # if len(embeddings) == len(df):
    #     df[embedding_col] = embeddings
    # Add detection metadata
    df["has_y_edge"] = df.apply(lambda row: (row["y_min"] <= 5) | (row["y_max"] >= 1075),axis=1)
    df["horizontal_detection"] = df.apply(lambda row: (row["x_max"] - row["x_min"]) > (row["y_max"] - row["y_min"]),axis=1)
    df["has_x_edge"] = df.apply(lambda row: (row["x_min"] <= 5) | (row["x_max"] >= 1915),axis=1)
    df["near_edge"] = ((~df["has_x_edge"]) & (~df["horizontal_detection"])) | ((~df["has_y_edge"]) & (df["horizontal_detection"]))

    return df

#############################
# Dino inference
def dinov3_embed(df, image_path_col="crop_path", model_name="facebook/dinov3-vit7b16-pretrain-lvd1689", batch_size=32, embedding_col="embedding",limited_mem=False):
   
    # Check GPU
    print("CUDA available:", torch.cuda.is_available())
    num_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_devices)
    for idx in range(num_devices):
        print(f"Device {idx}: {torch.cuda.get_device_name(idx)}")
    
    preprocessor, model = load_dinov3_model("facebook/dinov3-vit7b16-pretrain-lvd1689m", device='cuda')

    # Prepare paths and output arrays
    paths = df[image_path_col].tolist()
    n_samples = len(paths)
    embedding_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else 768

   
   
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
        inputs = processor(images=images, return_tensors="pt", size=list(input_size)).to(model.device)
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
    dfcm = batch_bioclip_embed(dfm, image_path_col="crop_path",batch_size=64)
    outpath = os.path.join(os.path.dirname(df_path),os.path.basename(df_path)[:-4] + "_with_all_embeddings.csv")
    dfcm.to_csv(outpath, index=False)
