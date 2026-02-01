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

from detect import *
from sam_utils import *
from visualize import *
import pandas as pd
import yaml
import os

################################# Crop Masks from Image Function #####################################

def extract_masks_from_image(image,masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Extract masks from an image based on the provided masks and scores.
    
    Args:
        image (PIL.Image): The input image.
        masks (np.ndarray): Array of masks to extract.
        scores (np.ndarray): Scores for each mask.
        point_coords (np.ndarray, optional): Coordinates of points used for mask prediction.
        box_coords (np.ndarray, optional): Coordinates of bounding boxes.
        input_labels (np.ndarray, optional): Labels for the input points.
        borders (bool, optional): Whether to draw borders around masks.
        
    Returns:
        Crops (list): list of cropped images for each mask.
    """
   
    crops = []
    # Ensure image is a PIL.Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    # If box_coords is not provided, skip cropping
    if box_coords is None:
        raise ValueError("box_coords must be provided to crop to box boundary.")

    # Sort masks and scores by score descending
    sorted_items = sorted(enumerate(zip(masks, scores, box_coords)), key=lambda x: -x[1][1])
    for idx, (mask, score, box) in sorted_items:
        # Convert mask to PIL Image
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        # Crop image to box boundary
        x_min, y_min, x_max, y_max = [int(v) for v in box]
        cropped_img = image_pil.crop((x_min, y_min, x_max, y_max))
        # Crop mask to box boundary and use as alpha
        cropped_mask = mask_image.crop((x_min, y_min, x_max, y_max))
        # Use OpenCV to find the largest contour and create a convex hull
        cropped_mask_np = np.array(cropped_mask)
        # Threshold to binary mask
        _, binary_mask = cv2.threshold(cropped_mask_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if idx == 0:
                plt.imshow(cropped_mask_np, cmap='gray')
                plt.title(f"Mask {idx+1} - Score: {score:.2f}")
                plt.axis('off')
                plt.show()
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if idx == 0:
                print(f"Largest contour area for mask {idx+1}: {cv2.contourArea(largest_contour)}")
            hull = cv2.convexHull(largest_contour)
            # Create blank mask and fill convex hull
            hull_mask = np.zeros_like(binary_mask)
            if idx == 0:
                print(f"Hull mask shape for mask {idx+1}: {hull_mask.shape}")
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            if idx == 0:
                plt.imshow(hull_mask, cmap='gray')
                plt.title(f"Hull Mask for Mask {idx+1} - Score: {score:.2f}")
                plt.axis('off')
                plt.show()
            cropped_mask = Image.fromarray(hull_mask)
            
        # Apply mask to cropped image
        masked_crop = Image.new("RGBA", cropped_img.size)
        masked_crop.paste(cropped_img, mask=cropped_mask)
        crops.append(masked_crop)
    return crops
 
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

#################################################### Main Pipeline ########################################################
def detect_segment_crop_pipeline(
    image_path,
    text_prompt,
    sam2_checkpoint,
    model_cfg,
    coco_json_path="nms_boxes_coco.json",
    device=None,
    save_coco=False,
    visualize=False,
    segmentation=False,
):
    """
    Pipeline to detect, segment, and crop objects from an image.

    Args:
        image_path (str): Path to the input image.
        text_prompt (str): Text prompt for object detection.
        sam2_checkpoint (str): Path to SAM2 checkpoint.
        model_cfg (str): Path to SAM2 config file.
        coco_json_path (str): Path to save/load COCO JSON for NMS boxes.
        device (str, optional): Device to use ("cuda" or "cpu").
        visualize (bool, optional): Whether to visualize results.

    Returns:
        crops (list): List of cropped images (PIL.Image).
        all_label_masks_squeezed (np.ndarray): Segmentation masks.
        all_labels_scores (np.ndarray): Scores for each mask.
        nms_boxes (torch.Tensor): NMS bounding boxes.
        nms_labels (list): NMS labels.
        nms_scores (torch.Tensor): NMS scores.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Detect objects
    boxes, labels, scores = gdino_predict(image_path, text_prompt)
    print(f"Detected {len(boxes)} objects with scores: {scores}")
    
    nms_boxes, nms_labels, nms_scores = nms(boxes, labels, scores)

    print(f"Detected {len(nms_boxes)} objects with NMS.")

    #Keep just the most confident box for testing
    nms_boxes = nms_boxes[:1]
    nms_labels = nms_labels[:1]
    nms_scores = nms_scores[:1]
    
    # Step 2: Save/load NMS results as COCO JSON
    if save_coco:
        save_coco_json(image_path, nms_boxes, nms_labels, nms_scores, coco_json_path)
        nms_boxes, nms_labels, nms_scores = load_coco_nms_boxes_labels_scores(coco_json_path)

    if segmentation:
        # Step 3: Build SAM2 model and predictor
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        # Step 4: Prepare input points and labels
        box_centers = np.array([[(x_min + x_max) / 2, (y_min + y_max) / 2] for x_min, y_min, x_max, y_max in nms_boxes.cpu().numpy()])
        input_points = box_centers
        input_labels = np.eye(len(input_points), dtype=int)

        # Step 5: Load image and set for predictor
        image = np.array(Image.open(image_path))
        predictor.set_image(image)

        # Step 6: Predict masks for each label
        all_label_masks = []
        all_label_scores = []
        all_label_logits = []
        for label in input_labels:
            masks_, scores_, logits_ = predictor.predict(
                point_coords=input_points,
                point_labels=label,
                multimask_output=False,
            )
            all_label_masks.append(masks_)
            all_label_scores.append(scores_)
            all_label_logits.append(logits_)

        all_label_masks = np.stack(all_label_masks)
        all_label_scores = np.stack(all_label_scores)
        all_label_logits = np.stack(all_label_logits)
        all_label_masks_squeezed = np.squeeze(all_label_masks, axis=1)
        all_labels_scores = np.squeeze(all_label_scores, axis=1)

        # Step 7: Visualization (optional)
        if visualize:
            plot_predicted_masks_overlay(image, all_label_masks_squeezed, all_labels_scores, box_centers)


    else: # just use crop from box
        image = np.array(Image.open(image_path))
        all_label_masks_squeezed = np.array([np.ones((image.shape[0],image.shape[1])) for _ in range(len(nms_boxes))])
        all_labels_scores = nms_scores.cpu().numpy()
        input_labels = np.eye(len(nms_boxes), dtype=int)

    # Step 8: Extract crops
    crops = extract_masks_from_image(
        image,
        all_label_masks_squeezed,
        all_labels_scores,
        box_coords=nms_boxes,
        input_labels=input_labels
    )

    if visualize:
        for i, crop in enumerate(crops):
            plt.figure(figsize=(6, 6))
            plt.imshow(crop)
            plt.title(f"Crop {i+1} - Score: {all_labels_scores[i]:.2f}")
            plt.axis('off')
            plt.show()
            #FIXME save them??

    return crops, all_label_masks_squeezed, all_labels_scores, nms_boxes, nms_labels, nms_scores
