############
# Script for saving similarity map between DINO features and reference image patch prespecified by coordinates.
############
from patch_embedding import *
import argparse
import sys
import os

sys.path.insert(0, "/home/lmeyers/rep_learn_scripts/")
from image_utils import *



## Load ref image 
def load_ref_image(path, patch_idx, preprocessor, model):
    # IMAGE_PATH ="/home/lmeyers/random_images_and_figures/Screenshot 2025-04-17 154938.png"#"/home/lmeyers/random_images_and_figures/Screenshot 2026-01-28 121814.png"#"/home/lmeyers/random_images_and_figures/5786744133_cf32f127cb_b.jpg"  # change it
    ref_image = crop_square(path, crop_size=400, translation=(0,0))
    ref_cls_token, ref_patches = get_dinov3_class_patch_embeddings(ref_image, preprocessor, model, device='cuda')  # (N, D)
    print(ref_patches.dtype)
    #select reference patch 
    ref_patch = ref_patches[:, patch_idx, :]  # (1, D)
    return ref_image, ref_patch

def calc_sim_and_save_batch(query_image_paths, ref_patch, preprocessor, model, save_paths):
    # load query image
    cropped_images = []
    for image in query_image_paths:
        query_image = crop_square(image, crop_size=2000, translation=(0,0))
        cropped_images.append(query_image)
    # get query patch embeddings
    query_cls_tokens, query_patcheses = get_dinov3_class_patch_embeddings(cropped_images, preprocessor, model, device='cuda')  # (N, D)
    print("Embeddings extracted with shape:", query_patcheses.shape)
    # for each image in batch
    for i in range(len(cropped_images)):
        query_image = cropped_images[i]
        query_patches = query_patcheses[i].unsqueeze(0)  # (1, N, D)
        # calculate similarities
        similarities = calc_similarities_to_patches(query_patches[0], ref_patch[0])  # (N,)
        # get grid size
        H, W = get_grid_size_from_patch_embeddings(query_patches)
        # display similarity map
        save_similarity_map(query_image, similarities, (H, W), save_path=save_paths[i])
    return  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_images', type=str, nargs='+', help='Paths to query images')
    parser.add_argument('--save_dir', type=str, help='Directory to save similarity maps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing images')
    args = parser.parse_args()

    print("Loading DINOv3 model...")
    preprocessor, model = load_dinov3_model("facebook/dinov3-vit7b16-pretrain-lvd1689m", device='cuda')
    print("DINOv3 model loaded.")
    # Load reference image and patch
    print("Loading reference image and patch...")
    ref_image, ref_patch = load_ref_image("/home/lmeyers/random_images_and_figures/Screenshot 2025-04-17 154938.png", patch_idx=539, preprocessor=preprocessor, model=model)

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory at {args.save_dir}")
        os.makedirs(args.save_dir)

    for i in range(0, len(args.query_images), args.batch_size):
        print(f"Processing batch {i // args.batch_size + 1}...","images:", args.query_images[i:i+args.batch_size])
        batch_images = args.query_images[i:i+args.batch_size]
        save_paths = [os.path.join(args.save_dir, os.path.basename(img) + "_sim_map.png") for img in batch_images]
        print("Save paths:", save_paths)
        calc_sim_and_save_batch(batch_images, ref_patch, preprocessor, model, save_paths)