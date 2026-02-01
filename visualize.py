#######################################################################
# Luke Meyers Funcapalooza Cicli2
# Utils for Visualization
# ################################# 
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import matplotlib.patches as patches
def visualize_bounding_boxes(image, boxes, labels=None, scores=None):
    """
    Visualize bounding boxes on an image.

    Args:
        image_path (str): Path to the image file.
        boxes (list of list of float): List of bounding boxes, each box is [x_min, y_min, x_max, y_max].
        labels (list of str, optional): List of labels for each bounding box.
        scores (list of float, optional): List of scores for each bounding box.
    """
    # Load the image
    # image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Add bounding boxes
    for i, box in enumerate(boxes.cpu().numpy()):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add label and score if available
        if labels and scores != None:
            label = f"{labels[i]}: {scores[i]:.2f}"
        elif labels:
            label = labels[i]
        elif scores:
            label = f"{scores[i]:.2f}"
        else:
            label = None

        if label:
            plt.text(x_min, y_min, label, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

#######################################################
# Show masks on OG Image
###########################################################
def plot_predicted_masks_overlay(image, all_label_masks_squeezed, all_labels_scores, box_centers, title="Predicted Masks Overlay"):
    """
    Plot predicted masks overlay on the original image.
    Args:
        image (np.ndarray): Original image.
        all_label_masks_squeezed (np.ndarray): Squeezed masks array of shape (num_labels, H, W).
        all_labels_scores (np.ndarray): Scores for each mask.
        box_centers (np.ndarray): Centers of the bounding boxes.
        title (str): Title for the plot.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Get a colormap with enough unique colors
    cmap = cm.get_cmap('tab10', all_label_masks_squeezed.shape[0])
    for i in range(all_label_masks_squeezed.shape[0]):
        mask = all_label_masks_squeezed[i]
        color = cmap(i)
        ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.75, cmap=cm.colors.ListedColormap([color]))
        score = all_labels_scores[i]
        center = box_centers[i]
        ax.plot(center[0], center[1], 'wo')
        ax.text(center[0], center[1], f"{score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.title(title)
    plt.show()
