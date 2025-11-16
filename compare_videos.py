import os
import sys
import numpy as np
import json # Used for saving results to a JSON file
import glob # Used for finding files easily
from enum import Enum

# --- Setup for CGVQM Import ---
# 1. Get the project root path (e.g., /home/bia/PTAA)
project_root = os.path.dirname(os.path.abspath(__file__))

# 2. Add the 'src' directory to the Python path to find 'cgvqm' package
sys.path.append(os.path.join(project_root, 'src'))

# 3. Add the 'src/cgvqm' directory to the path for internal imports like 'utils'
sys.path.append(os.path.join(project_root, 'src', 'cgvqm'))

# 4. Import the main functions/classes
from cgvqm.cgvqm import run_cgvqm, visualize_emap, CGVQM_TYPE

# Define your main function to replicate the demo logic
def run_custom_cgvqm_demo():

    # --- 1. Custom Paths ---
    # NOTE: These paths are relative to your project root (/home/bia/PTAA)
    # The 'dist_vid_path' is your distorted video, and 'ref_vid_path' is your reference.
    dist_vid_path = os.path.join(project_root, 'data/vary_alpha_weight/vary_alpha_weight_0.1.mp4')
    ref_vid_path = os.path.join(project_root, 'data/16SSAA.mp4')
    
    # Path to save the predicted error map
    errmap_path = os.path.join(project_root, 'output/comparison_emap.mp4') 
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(errmap_path), exist_ok=True)


    # --- 2. Configuration (Copied from your desired demo) ---
    cgvqm_type = CGVQM_TYPE.CGVQM_2 # Select the model version
    device = 'cuda'                   # Change to 'cpu' if no CUDA GPU is available
    patch_scale = 4                   # Adjust if memory is an issue (Higher value = less memory)
    patch_pool='mean'                 # Pooling method


    # --- 3. Run CGVQM and visualize results ---
    print(f"\n--- Running CGVQM Comparison ---")
    print(f"Distorted Video: {dist_vid_path}")
    print(f"Reference Video: {ref_vid_path}")
    print(f"Using CGVQM Type: {cgvqm_type.name}")

    # The function call to run the comparison
    q, emap = run_cgvqm(
        dist_vid_path, 
        ref_vid_path, 
        cgvqm_type = cgvqm_type, 
        device=device, 
        patch_pool=patch_pool, 
        patch_scale=patch_scale
    )
    
    # --- 4. Output Results ---
    qlabels = ['very annoying', 'annoying', 'slightly annoying', 'perceptible but not annoying', 'imperceptible']
    
    # Calculate the label index safely
    score = q.item()
    label_index = min(int(np.round(score / 25)), len(qlabels) - 1) 
    
    print(f'\nQuality Score: {score:.2f}/100 ({qlabels[label_index]})')
    
    # Visualize and save the error map
    visualize_emap(emap, dist_vid_path, 100, errmap_path)
    print(f'Quality map written to {errmap_path}')

if __name__ == '__main__':
    # You might need to add a try-except block here to catch errors related 
    # to missing files, weights, or device setup (e.g., CUDA errors).
    run_custom_cgvqm_demo()