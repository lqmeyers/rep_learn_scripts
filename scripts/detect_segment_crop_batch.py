##################################### BioCLIP - Detect, Segment, Crop Script #####################################
# 
# Luke Meyers, Funcapalooza-Cicli2, 2025
############################################################################
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
import yaml
import tqdm
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from PIL import Image

# ML imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision.ops as ops
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from bioclip import TreeOfLifeClassifier

# Utils and pipeline functions
sys.path.insert(0, "../")
from utils.sam_utils import *
from utils.visualize import *
from scripts.detect import *
from scripts.segment import *
from scripts.crop import * 
from scripts.embed import *
from dino_patch.patch_embedding import load_dinov3_model

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
        save_coco (bool, optional): Whether to save COCO JSON files.
        visualize (bool, optional): Whether to visualize results.
        **kwargs: Additional keyword arguments for detection/segmentation toggles.

    Returns:
        tuple: crops_batch, all_label_masks_batch, all_label_scores_batch, nms_boxes_batch, nms_labels_batch, nms_scores_batch
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if coco_json_paths is None:
        coco_json_paths = [f"nms_boxes_coco_{i}.json" for i in range(len(image_paths))]

    # Load images
    images = [np.array(Image.open(img_path).convert("RGB")) for img_path in image_paths]

    # Step 1: Detect objects (batched)
    if kwargs.get("detection", True):
        batch_boxes, batch_labels, batch_scores = owl_predict_batch(image_paths, text_prompts)
        nms_boxes_batch, nms_labels_batch, nms_scores_batch = nms_batch(batch_boxes, batch_labels, batch_scores)
    else:
        # Use whole image as a single box
        nms_boxes_batch, nms_labels_batch, nms_scores_batch = [], [], []
        for img in images:
            height, width = img.shape[:2]
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

    # Step 3: Batch segmentation with SAM2
    all_label_masks_batch = []
    all_label_scores_batch = []
    all_label_logits_batch = []
    if kwargs.get("segmentation", True):
        all_labels_masks_squeezed, all_labels_scores, all_label_logits = batch_segmentation(
            image_paths,nms_boxes_batch, model_cfg, sam2_checkpoint, device=device
        )
        all_label_masks_batch = all_labels_masks_squeezed
        all_label_scores_batch = all_labels_scores
        all_label_logits_batch = all_label_logits
    else:
        for img, nms_boxes in zip(images, nms_boxes_batch):
            mask_shape = (len(nms_boxes), img.shape[0], img.shape[1])
            all_label_masks_batch.append(np.ones(mask_shape, dtype=np.uint8))
            all_label_scores_batch.append(np.ones((len(nms_boxes),), dtype=np.float32))
            all_label_logits_batch.append(np.ones(mask_shape, dtype=np.float32))

    # Step 4: Visualization (optional, batched)
    if visualize:
        box_centers_batch = [get_box_centers(nms_boxes) for nms_boxes in nms_boxes_batch]
        for image, masks, scores, box_centers in zip(images, all_label_masks_batch, all_label_scores_batch, box_centers_batch):
            plot_predicted_masks_overlay(image, masks, scores, box_centers)

    # Step 5: Extract crops (batched)
    input_labels_batch = nms_labels_batch
    crops_batch = extract_masks_from_image_batch(
        images,
        all_label_masks_batch,
        all_label_scores_batch,
        nms_boxes_batch,
        input_labels_batch
    )

    # Step 6: Visualization of crops (optional)
    if visualize:
        for crops, scores in zip(crops_batch, all_label_scores_batch):
            for i, crop in enumerate(crops):
                plt.figure(figsize=(6, 6))
                plt.imshow(crop)
                plt.title(f"Crop {i+1} - Score: {scores[i]:.2f}")
                plt.axis('off')
                plt.show()

    return crops_batch, all_label_masks_batch, all_label_scores_batch, nms_boxes_batch, nms_labels_batch, nms_scores_batch
