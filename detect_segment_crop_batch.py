##################################### BioCLIP - Detect, Segment, Crop Script #####################################
# 
# Luke Meyers, Funcapalooza-Cicli2, 2025
############################################################################
import sys
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision.ops as ops

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from bioclip import TreeOfLifeClassifier

from sam_utils import *
from visualize import *
from detect import *

import pandas as pd
import yaml
import tqdm
import os
import sys

sys.path.insert(0, "/home/lmeyers/rep_learn_scripts/dino_patch")
from patch_embedding import load_dinov3_model

def extract_masks_from_image_batch(images, masks_batch, scores_batch, box_coords_batch, input_labels_batch=None, borders=True):
    """
    Batched extraction of crops from images.
    Args:
        images (list of PIL.Image or np.ndarray)
        masks_batch (list of np.ndarray)
        scores_batch (list of np.ndarray)
        box_coords_batch (list of torch.Tensor)
        input_labels_batch (list, optional)
        borders (bool)
    Returns:
        crops_batch (list of list of PIL.Image)
    """
    crops_batch = []
    for image, masks, scores, box_coords in zip(images, masks_batch, scores_batch, box_coords_batch):
        crops = []
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        sorted_items = sorted(enumerate(zip(masks, scores, box_coords)), key=lambda x: -x[1][1])
        for idx, (mask, score, box) in sorted_items: #TODO better functional breakdown here
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            x_min, y_min, x_max, y_max = [int(v) for v in box]
            cropped_img = image_pil.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask_image.crop((x_min, y_min, x_max, y_max))
            cropped_mask_np = np.array(cropped_mask)
            _, binary_mask = cv2.threshold(cropped_mask_np, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                hull_mask = np.zeros_like(binary_mask)
                cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
                cropped_mask = Image.fromarray(hull_mask)
            masked_crop = Image.new("RGBA", cropped_img.size)
            masked_crop.paste(cropped_img, mask=cropped_mask)
            crops.append(masked_crop)
        crops_batch.append(crops)
    return crops_batch

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
     
    preprocessor, model = load_dinov3_model("facebook/dinov3-vit7b16-pretrain-lvd1689m", device='cuda')

    embeddings_batch = []
    for crops in tqdm.tqdm(crops_batch, desc="Embedding crops with DINOv3", total=len(crops_batch)):
        batch_embeddings = []
        for crop in crops:
            if crop.mode == "RGBA":
                crop = crop.convert("RGB")
            inputs = preprocessor(crop, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            batch_embeddings.append(embedding)
        embeddings_batch.append(np.array(batch_embeddings))
    return embeddings_batch


#################################################### Main Batched Pipeline ########################################################
def detect_segment_crop_pipeline_batch(
    image_paths,
    text_prompts,
    sam2_checkpoint,
    model_cfg,
    coco_json_paths=None,
    device=None,
    save_coco=True,
    visualize=False,
    **kwargs
):
    """
    Batched pipeline to detect, segment, and crop objects from images.
    Args:
        image_paths (list of str): Paths to input images.
        text_prompts (list of str): Text prompts for object detection.
        sam2_checkpoint (str): Path to SAM2 checkpoint.
        model_cfg (str): Path to SAM2 config file.
        coco_json_paths (list of str, optional): Paths to save/load COCO JSON for NMS boxes.
        device (str, optional): Device to use ("cuda" or "cpu").
        visualize (bool, optional): Whether to visualize results.
    Returns:
        crops_batch, all_label_masks_squeezed_batch, all_labels_scores_batch, nms_boxes_batch, nms_labels_batch, nms_scores_batch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if coco_json_paths is None:
        coco_json_paths = [f"nms_boxes_coco_{i}.json" for i in range(len(image_paths))]

    # Step 1: Detect objects (batched)
    if kwargs.get("detection",True):
        batch_boxes, batch_labels, batch_scores = gdino_predict_batch(image_paths, text_prompts)
        nms_boxes_batch, nms_labels_batch, nms_scores_batch = nms_batch(batch_boxes, batch_labels, batch_scores)
    else:
        #use whole image
        nms_boxes_batch, nms_labels_batch, nms_scores_batch = [], [], []
        for img_path in image_paths:
            image = Image.open(img_path)
            width, height = image.size
            box = torch.tensor([[0, 0, width, height]], dtype=torch.float32)
            nms_boxes_batch.append(box)
            nms_labels_batch.append([0])  # Dummy label
            nms_scores_batch.append(torch.tensor([1.0]))  # Dummy score

    # Step 2: Save/load NMS results as COCO JSON (batched)
    for i, (img_path, nms_boxes, nms_labels, nms_scores, coco_json_path) in enumerate(
        zip(image_paths, nms_boxes_batch, nms_labels_batch, nms_scores_batch, coco_json_paths)
    ):
        if save_coco:
            save_coco_json(img_path, nms_boxes, nms_labels, nms_scores, coco_json_path)
        nms_boxes, nms_labels, nms_scores = load_coco_nms_boxes_labels_scores(coco_json_path)
        nms_boxes_batch[i] = nms_boxes
        nms_labels_batch[i] = nms_labels
        nms_scores_batch[i] = nms_scores

    if kwargs.get("segmentation",True):
        # Step 3: Build SAM2 model and predictor (shared for batch)
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        # Step 4: Prepare input points and labels (batched)
        box_centers_batch = [
            np.array([[(x_min + x_max) / 2, (y_min + y_max) / 2] for x_min, y_min, x_max, y_max in nms_boxes.cpu().numpy()])
            for nms_boxes in nms_boxes_batch
        ]
        input_labels_batch = [np.eye(len(box_centers), dtype=int) for box_centers in box_centers_batch]

        # Step 5: Load images and set for predictor (batched)
        images = [np.array(Image.open(img_path)) for img_path in image_paths]

        # Step 6: Predict masks for each label (batched)
        all_label_masks_batch = []
        all_label_scores_batch = []
        all_label_logits_batch = []
        for image, input_points, input_labels in zip(images, box_centers_batch, input_labels_batch):
            predictor.set_image(image)
            masks_list, scores_list, logits_list = [], [], []
            for label in input_labels:
                masks_, scores_, logits_ = predictor.predict(
                    point_coords=input_points,
                    point_labels=label,
                    multimask_output=False,
                )
                masks_list.append(masks_)
                scores_list.append(scores_)
                logits_list.append(logits_)
            all_label_masks = np.stack(masks_list)
            all_label_scores = np.stack(scores_list)
            all_label_logits = np.stack(logits_list)
            all_label_masks_squeezed = np.squeeze(all_label_masks, axis=1)
            all_labels_scores = np.squeeze(all_label_scores, axis=1)
            all_label_masks_batch.append(all_label_masks_squeezed)
            all_label_scores_batch.append(all_labels_scores)
            all_label_logits_batch.append(all_label_logits)
    else:
        all_label_masks_batch = [np.ones((len(nms_boxes), images[0].shape[0], images[0].shape[1]), dtype=np.uint8) for nms_boxes in nms_boxes_batch]
        all_label_scores_batch = [np.ones((len(nms_boxes),), dtype=np.float32) for nms_boxes in nms_boxes_batch]
        all_label_logits_batch = [np.ones((len(nms_boxes), images[0].shape[0], images[0].shape[1]), dtype=np.float32) for nms_boxes in nms_boxes_batch]
        

    # Step 7: Visualization (optional, batched)
    if visualize:
        for image, masks, scores, box_centers in zip(images, all_label_masks_batch, all_label_scores_batch, box_centers_batch):
            plot_predicted_masks_overlay(image, masks, scores, box_centers)

    # Step 8: Extract crops (batched)
    crops_batch = extract_masks_from_image_batch(
        images,
        all_label_masks_batch,
        all_label_scores_batch,
        nms_boxes_batch,
        input_labels_batch
    )

    if visualize:
        for crops, scores in zip(crops_batch, all_label_scores_batch):
            for i, crop in enumerate(crops):
                plt.figure(figsize=(6, 6))
                plt.imshow(crop)
                plt.title(f"Crop {i+1} - Score: {scores[i]:.2f}")
                plt.axis('off')
                plt.show()

    return crops_batch, all_label_masks_batch, all_label_scores_batch, nms_boxes_batch, nms_labels_batch, nms_scores_batch
