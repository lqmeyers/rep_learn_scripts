####################################
# Scripts for using DinoV3 patch embeddings
####################################

import os
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def load_dinov3_model(model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",device="cuda"):
    """
    Load the DINOv3 model and image processor.

    Args:
        model_id (str): The model identifier from Hugging Face.

    Returns:
        processor: The image processor for DINOv3.
        model: The DINOv3 model.
    """
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained(model_id, device_map="auto")
    model.eval()
    return processor, model

###############################################################
def get_dinov3_class_patch_embeddings(image, processor, model, device="cuda"):
    """
    Get patch embeddings from a DINOv3 model for a given image.

    Args:
        image (PIL.Image.Image): The input image.
        processor: The image processor for DINOv3.
        model: The DINOv3 model.

    Returns:
        class_token: The class token embedding from the model.
        patch_embeddings: The patch embeddings from the model.
    """
    inputs = processor(images=image, return_tensors="pt",do_resize=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    patch_embeddings = outputs.last_hidden_state[:, 5:, :]  # Exclude CLS and registers token
    class_token = outputs.last_hidden_state[:, 0, :]  # CLS token
    return class_token, patch_embeddings

def get_dinov3_class_patch_embeddings_batch(images, processor, model, device="cuda"):
    """
    Get patch embeddings from a DINOv3 model for a batch of images.

    Args:
        images (list of PIL.Image.Image): The input images.
        processor: The image processor for DINOv3.
        model: The DINOv3 model.

    Returns:
        class_tokens: The class token embeddings from the model (B, D).
        patch_embeddings: The patch embeddings from the model (B, N, D).
    """
    inputs = processor(images=images, return_tensors="pt", do_resize=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    patch_embeddings = outputs.last_hidden_state[:, 5:, :]  # Exclude CLS and registers token
    class_tokens = outputs.last_hidden_state[:, 0, :]  # CLS token
    return class_tokens, patch_embeddings


################################################################

def get_grid_size_from_patch_embeddings(patch_embeddings):
    """
    Calculate the grid size (H, W) from patch embeddings.

    Args:
        patch_embeddings (torch.Tensor): The patch embeddings of shape (B, N, D).   
    Returns:
        H (int): Height of the grid.
        W (int): Width of the grid.
    """
    B, N, D = patch_embeddings.shape  # N = number of patches
    H = W = int(N**0.5)  # Assuming square grid
    return H, W

###############################################################

def calc_similarities_to_patches(patch_embeddings, reference_vector):
    """
    Calculate cosine similarities between patch embeddings and a reference vector.

    Args:
        patch_embeddings (torch.Tensor): The patch embeddings of shape (N, D).
        reference_vector (torch.Tensor): The reference vector of shape (D,). 
    Returns:
        similarities (torch.Tensor): Cosine similarities of shape (N,).
    """
    patch_embeddings_norm = torch.nn.functional.normalize(patch_embeddings, dim=-1)
    reference_vector_norm = torch.nn.functional.normalize(reference_vector, dim=-1)
    similarities = torch.nn.functional.cosine_similarity(
        patch_embeddings_norm, reference_vector_norm.unsqueeze(0), dim=-1
    )
    return similarities

###############################################################

def save_similarity_map(image, similarities, grid, save_path):
    """
    Save the similarity map as a NPY image.

    Args:
        image (PIL.Image.Image): The input image (unused).
        similarities (torch.Tensor): Cosine similarities of shape (N,).
        grid (tuple): The grid size (H, W).
        save_path (str): Path to save the similarity map NPY.
    """
    sim_map = similarities.reshape(grid[0], grid[1]).cpu().numpy()
    np.save(save_path, sim_map)
    # Normalize to 0-255 for visualization
    # sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    # sim_map_uint8 = (sim_map_norm * 255).astype('uint8')
    # imageio.imwrite(save_path, sim_map_uint8)
    return

def display_similarity_map(image, similarities, grid, save_path=None):
    """
    Display the similarity map overlaid on the image.

    Args:
        image (PIL.Image.Image): The input image.
        similarities (torch.Tensor): Cosine similarities of shape (N,).
        grid (tuple): The grid size (H, W).
        save_path (str, optional): Path to save the similarity map. If None, the map is displayed.
    """
    similarities = similarities.reshape(grid[0], grid[1]).cpu().numpy()  # (grid, grid)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image.resize((grid[1], grid[0])), alpha=0.5)
    im = ax.imshow(similarities, cmap='bwr', alpha=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title('Cosine Similarity between Class Token and Patch Tokens')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        return
    plt.show()
    return

def get_patch_embedding_at_clicks(image, patches, clicks, device="cuda"):
    """
    Get patch embeddings at specified click locations.

    Args:
        patches (torch.Tensor): Patch embeddings of shape (N, D).
        clicks (list of tuples): List of (x, y) click coordinates.

    Returns:
        embeddings_at_clicks (list of torch.Tensor): List of patch embeddings at the click locations.
    """

    grid_size = get_grid_size_from_patch_embeddings(patches)[0]
    print("Patches shape:", patches.shape, "Grid size:", grid_size)
    image_width, image_height = image.size
    print("Image size:", image.size)
    embeddings_at_clicks = []
    for (x, y) in clicks:
        patch_x = min(int(x / image_width * grid_size), grid_size - 1)
        patch_y = min(int(y / image_height * grid_size), grid_size - 1)
        print(f"Click at ({x:.2f}, {y:.2f}) maps to patch ({patch_x}, {patch_y})")
        patch_index = patch_y * grid_size + patch_x
        print(f"Patch index: {patch_index}")
        embeddings_at_clicks.append(patches[0, patch_index].to(device))
    return embeddings_at_clicks