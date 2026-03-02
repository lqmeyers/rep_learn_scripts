import numpy as np
from PIL import Image
import sys

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sys.path.insert(0, "../")
from utils.sam_utils import *
from utils.visualize import *

def batch_segmentation(image_paths, nms_boxes_batch, model_cfg, sam2_checkpoint, device="cuda"):
    """
    Performs segmentation on a batch of images using the SAM2 model.

    Args:
        image_paths (list of str): Paths to input images.
        nms_boxes_batch (list of torch.Tensor): List of bounding boxes for each image (shape: [N, 4]).
        model_cfg (dict): Configuration for the SAM2 model.
        sam2_checkpoint (str): Path to the SAM2 model checkpoint.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        tuple: (masks, scores, logits) for all labels in the batch.
            - masks: np.ndarray of shape [batch_size, num_labels, H, W]
            - scores: np.ndarray of shape [batch_size, num_labels]
            - logits: np.ndarray of shape [batch_size, num_labels, H, W]
    """
    # Build SAM2 model and predictor
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Prepare input points and labels for each image
    box_centers_batch = [
        np.array([[(x_min + x_max) / 2, (y_min + y_max) / 2] for x_min, y_min, x_max, y_max in nms_boxes.cpu().numpy()])
        for nms_boxes in nms_boxes_batch
    ]
    input_labels_batch = [np.eye(len(box_centers), dtype=int) for box_centers in box_centers_batch]

    # Load images
    images = [np.array(Image.open(img_path)) for img_path in image_paths]

    # Predict masks, scores, and logits for each image and label
    all_masks, all_scores, all_logits = [], [], []
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
        masks = np.squeeze(np.stack(masks_list), axis=1)
        scores = np.squeeze(np.stack(scores_list), axis=1)
        logits = np.stack(logits_list)
        all_masks.append(masks)
        all_scores.append(scores)
        all_logits.append(logits)

    # Stack results for the batch
    all_masks = np.stack(all_masks)
    all_scores = np.stack(all_scores)
    all_logits = np.stack(all_logits)
    return all_masks, all_scores, all_logits