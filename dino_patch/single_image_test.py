import os
import PIL
import torch
import sys
from patch_embedding import *
import glob

preprocessor, model = load_dinov3_model("facebook/dinov3-vit7b16-pretrain-lvd1689m", device='cuda')

from PIL import Image
from matplotlib import pyplot as plt
sys.path.insert(0, "../")
from image_utils import *



def main(image2_path,save_dir):
    IMAGE_PATH ="../data/ohelo_flower_reference.png"#"/home/lmeyers/random_images_and_figures/Screenshot 2026-01-28 121814.png"#"/home/lmeyers/random_images_and_figures/5786744133_cf32f127cb_b.jpg"  # change it
    image_cropped = crop_square(IMAGE_PATH, crop_size=400, translation=(0,0))

    class_token, patches = get_dinov3_class_patch_embeddings(image_cropped, preprocessor, model, device='cuda')  # (N, D)

    #image_2 ="/users/PAS2136/lqmeyers/cam_traps/ID26-T/R32_26_B_25_02_10_25_02_25/IMAG0021.jpg" # "/home/lmeyers/hawaii/IMAG0019.jpg" #

    img2_cropped = crop_square(image2_path, crop_size=4000, translation=(0,0))
    class2, patches2 = get_dinov3_class_patch_embeddings(img2_cropped, preprocessor, model, device='cuda')  # (N, D)

    target = patches[:,539,:]
    print("Target patch stats:")
    print(patches.mean())
    print(patches.min(),patches.max())

    similarity_to_target = calc_similarities_to_patches(patches2,target[0])
    print("Similarity to target patch stats:")
    print(similarity_to_target.mean())
    print(similarity_to_target.min(),similarity_to_target.max())
    grid_h, grid_w = get_grid_size_from_patch_embeddings(patches2)
    save_path = os.path.join(save_dir, os.path.basename(image2_path).split('.')[0] )
    display_similarity_map(img2_cropped, similarity_to_target, (grid_w,grid_h),save_path=save_path+ "_similarity_map.png")
    save_similarity_map(img2_cropped, similarity_to_target, (grid_w,grid_h),save_path=save_path+ "_similarity_map.npy")

if __name__ == "__main__":
    image_glob = "/users/PAS2136/lqmeyers/cam_traps/ID26-T/R32_26_A_25_02_25_25_03_12/IMAG*.jpg"
    images = sorted(glob.glob(image_glob))
    save_dir = "/users/PAS2136/lqmeyers/rep_learn_scripts/output/sim_maps/R32_26_A_25_02_25_25_03_12"
    for img_path in images:
        print("Processing image:", img_path)
        main(img_path, save_dir)
