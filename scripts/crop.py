########################################################################################################
# File: crop.py
# Description: Functions for cropping images based on predicted masks and bounding boxes.
# Author: L. Meyers
# Date: 2025
########################################################################################################

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

################################# Crop Masks from Image Functions #####################################

def extract_masks_from_image(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True):
    """
    Extracts cropped regions from an image using predicted masks and bounding boxes.

    Args:
        image (PIL.Image or np.ndarray): The input image.
        masks (np.ndarray): Array of binary masks.
        scores (np.ndarray): Confidence scores for each mask.
        point_coords (np.ndarray, optional): Coordinates of points used for mask prediction.
        box_coords (np.ndarray): Coordinates of bounding boxes (required).
        input_labels (np.ndarray, optional): Labels for the input points.
        borders (bool, optional): Whether to draw borders around masks.

    Returns:
        list of PIL.Image: Cropped images for each mask.
    """
    if box_coords is None:
        raise ValueError("box_coords must be provided to crop to box boundary.")

    # Ensure image is a PIL.Image
    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    # Sort masks and scores by descending score
    sorted_items = sorted(
        enumerate(zip(masks, scores, box_coords)),
        key=lambda x: -x[1][1]
    )

    crops = []
    for idx, (mask, score, box) in sorted_items:
        # Convert mask to PIL Image and crop to bounding box
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_img = image_pil.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask_image.crop((x_min, y_min, x_max, y_max))

        # Convert mask to numpy and threshold to binary
        cropped_mask_np = np.array(cropped_mask)
        _, binary_mask = cv2.threshold(cropped_mask_np, 127, 255, cv2.THRESH_BINARY)

        # Find contours and create convex hull mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            hull_mask = np.zeros_like(binary_mask)
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            cropped_mask = Image.fromarray(hull_mask)

        # Apply mask to cropped image
        masked_crop = Image.new("RGBA", cropped_img.size)
        masked_crop.paste(cropped_img, mask=cropped_mask)
        crops.append(masked_crop)

    return crops


def extract_masks_from_image_batch(
    images,
    masks_batch,
    scores_batch,
    box_coords_batch,
    input_labels_batch=None,
    borders=True):
    """
    Batch extraction of cropped regions from multiple images using predicted masks and bounding boxes.

    Args:
        images (list of PIL.Image or np.ndarray): List of input images.
        masks_batch (list of np.ndarray): List of mask arrays for each image.
        scores_batch (list of np.ndarray): List of score arrays for each image.
        box_coords_batch (list of np.ndarray): List of bounding box arrays for each image.
        input_labels_batch (list, optional): List of input labels for each image.
        borders (bool, optional): Whether to draw borders around masks.

    Returns:
        list of list of PIL.Image: Cropped images for each mask in each image.
    """
    crops_batch = []
    for image, masks, scores, box_coords in zip(images, masks_batch, scores_batch, box_coords_batch):
        image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        sorted_items = sorted(
            enumerate(zip(masks, scores, box_coords)),
            key=lambda x: -x[1][1]
        )

        crops = []
        for idx, (mask, score, box) in sorted_items:
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            x_min, y_min, x_max, y_max = map(int, box)
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