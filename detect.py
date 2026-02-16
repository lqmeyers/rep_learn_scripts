import sys
import json
import cv2
import tqdm
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
from transformers import Owlv2Processor, Owlv2ForObjectDetection
#     import torch
#     from PIL import Image

from bioclip import TreeOfLifeClassifier

from sam_utils import *
from visualize import *
import pandas as pd
import yaml
import os

#########################################################################################3
# Helper functions 
def owl_predict(image, text):
    """
    Predict bounding boxes and labels using OWL-ViT (Open-Vocabulary Object Detection).
    Args:
        image (str or PIL.Image): Path to the image or PIL Image object.
        text (str or list): Text prompt(s) for object detection. Can be a string or list of strings.
    Returns:
        boxes (torch.Tensor): Detected bounding boxes (xyxy format).
        labels (list): Detected labels.
        scores (torch.Tensor): Detection scores.
    """
  
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    model_id = "google/owlvit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Owlv2ImageProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(torch.device)

    # OWL-ViT expects a list of queries
    if isinstance(text, str):
        text = [text]

    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get boxes, scores, and labels
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.25)
    print(results)
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    # Convert labels from tensor to string
    labels = [l for l in labels]

    return boxes, labels, scores

####################################################################
def gdino_predict(image,text):
    """ Predict bounding boxes and labels using Grounding DINO.
    Args:
        image (str or PIL.Image): Path to the image or PIL Image object.
        text (str): Text prompt for object detection.
    Returns:
        boxes (torch.Tensor): Detected bounding boxes.
        labels (list): Detected labels.
        scores (torch.Tensor): Detection scores.
    """
    image = Image.open(image)
    # Check for cats and remote controls

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
   
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"]  # Example boxes
    labels = results[0]["labels"] # Example labels
    scores = results[0]["scores"] # Example scores
   
    return boxes, labels, scores


################################# Non-Maximal Suppression (NMS) Function #####################################
def nms(boxes,labels, scores):
    # Convert boxes and scores to the required format
    boxes_tensor = boxes
    scores_tensor = scores

    # Perform Non-Maximal Suppression
    nms_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

    # Filter boxes, labels, and scores based on NMS indices
    nms_boxes = boxes_tensor[nms_indices]
    nms_labels = [labels[i] for i in nms_indices]
    nms_scores = scores_tensor[nms_indices]

    return nms_boxes, nms_labels, nms_scores


# #########################################################################################
# Helper functions for batched processing

def gdino_predict_batch(images, texts):
    """
    Batched prediction of bounding boxes and labels using Grounding DINO.
    Args:
        images (list of str or PIL.Image): List of image paths or PIL Image objects.
        texts (list of str): List of text prompts for object detection.
    Returns:
        batch_boxes (list of torch.Tensor): Detected bounding boxes per image.
        batch_labels (list of list): Detected labels per image.
        batch_scores (list of torch.Tensor): Detection scores per image.
    """
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    batch_boxes, batch_labels, batch_scores = [], [], []
    for image, text in zip(images, texts):
        if isinstance(image, str):
            image = Image.open(image)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        batch_boxes.append(results[0]["boxes"])
        batch_labels.append(results[0]["labels"])
        batch_scores.append(results[0]["scores"])
    return batch_boxes, batch_labels, batch_scores

def nms_batch(batch_boxes, batch_labels, batch_scores):
    """
    Batched Non-Maximal Suppression.
    Args:
        batch_boxes (list of torch.Tensor)
        batch_labels (list of list)
        batch_scores (list of torch.Tensor)
    Returns:
        nms_boxes_batch, nms_labels_batch, nms_scores_batch
    """
    nms_boxes_batch, nms_labels_batch, nms_scores_batch = [], [], []
    for boxes, labels, scores in zip(batch_boxes, batch_labels, batch_scores):
        nms_indices = ops.nms(boxes, scores, iou_threshold=0.5)
        nms_boxes = boxes[nms_indices]
        nms_labels = [labels[i] for i in nms_indices]
        nms_scores = scores[nms_indices]
        nms_boxes_batch.append(nms_boxes)
        nms_labels_batch.append(nms_labels)
        nms_scores_batch.append(nms_scores)
    return nms_boxes_batch, nms_labels_batch, nms_scores_batch

def owl_predict_batch(images, texts, batch_size=16):
    """
    Batched prediction of bounding boxes and labels using OWL-ViT.
    Args:
        images (list of str or PIL.Image): List of image paths or PIL Image objects.
        texts (list of str or list): List of text prompts for object detection.
        batch_size (int): Number of images to process per batch.
    Returns:
        batch_boxes (list of torch.Tensor): Detected bounding boxes per image.
        batch_labels (list of list): Detected labels per image.
        batch_scores (list of torch.Tensor): Detection scores per image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

    batch_boxes, batch_labels, batch_scores = [], [], []

    for i in tqdm.tqdm(range(0, len(images), batch_size), desc="Processing detection batches"):
        batch_images = images[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
        processed_images = []
        for image in batch_images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            processed_images.append(image)
        
        processed_texts = []
        for text in batch_texts:
            if isinstance(text, str):
                text = [text]
            processed_texts.append(text)
        
        inputs = processor(text=processed_texts, images=processed_images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([img.size[::-1] for img in processed_images])
        results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=0.25)
        
        for result in results:
            batch_boxes.append(result["boxes"])
            batch_labels.append(result["labels"])
            batch_scores.append(result["scores"])
    
    return batch_boxes, batch_labels, batch_scores

#TODO Load models function
# Pred function that only loads models once and reuses them