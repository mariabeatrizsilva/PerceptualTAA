import os
import sys
import numpy as np
import json # Used for saving results to a JSON file
import glob # Used for finding files easily
from enum import Enum

REF_VID_PATH_REL = 'data/16SSAA.mp4'
DIST_VID_DIR_REL = 'data/vary_alpha_weight'
ERR_MAP_DIR_REL = 'outputs/err_maps'
ERR_SCORES_PATH_REL = 'outputs/vary_alpha_weight_scores.json'

# --- Setup for CGVQM Import ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'cgvqm'))

CGVQM_CONFIG = {
    'cgvqm_type': 'CGVQM_2', # Will be converted to CGVQM_TYPE.CGVQM_2
    'device': 'cuda',        # Change to 'cpu' if no CUDA GPU is available
    'patch_scale': 4,        # Increase this value if low on available GPU memory
    'patch_pool': 'mean'     # Choose from {'max', 'mean'}
}

qlabels = ['very annoying', 'annoying', 'slightly annoying', 'perceptible but not annoying', 'imperceptible']

# 4. Import the main functions/classes
from cgvqm.cgvqm import run_cgvqm, visualize_emap, CGVQM_TYPE

# Define your main function to replicate the demo logic
def run_single_comparison(dist_vid_path, ref_vid_path, errmap_output_path, config):
    """
    Calculates the CGVQM score and error map for a single pair of videos.
    Returns the scalar quality score.
    """
    
    # Map string config to the Enum type required by the library
    cgvqm_type_enum = getattr(CGVQM_TYPE, config['cgvqm_type'])
    
    # Run CGVQM
    q, emap = run_cgvqm(
        dist_vid_path, 
        ref_vid_path, 
        cgvqm_type = cgvqm_type_enum, 
        device=config['device'], 
        patch_pool=config['patch_pool'], 
        patch_scale=config['patch_scale']
    )
    
    score = q.item()
    
    # Save the error map visualization
    visualize_emap(emap, dist_vid_path, 100, errmap_output_path)
    
    return score
    

if __name__ == '__main__':
    # You might need to add a try-except block here to catch errors related 
    # to missing files, weights, or device setup (e.g., CUDA errors).
    ref_video_rel = REF_VID_PATH_REL
    dist_video_rel = os.path.join(DIST_VID_DIR_REL, 'vary_alpha_weight_0.1.mp4') # Assuming this test video exists
    
    # Resolve absolute paths
    ref_vid_path = os.path.join(project_root, ref_video_rel)
    dist_vid_path = os.path.join(project_root, dist_video_rel)
    
    # Define a specific output path for the test error map
    test_errmap_dir = os.path.join(project_root, 'outputs/test_output')
    os.makedirs(test_errmap_dir, exist_ok=True)
    test_errmap_path = os.path.join(test_errmap_dir, 'single_test_err_map.mp4')

    if not os.path.exists(dist_vid_path):
        print(f"FATAL ERROR: Distorted video not found at: {dist_vid_path}")
        sys.exit(1)
    if not os.path.exists(ref_vid_path):
        print(f"FATAL ERROR: Reference video not found at: {ref_vid_path}")
        sys.exit(1)
    try:
        print(f"Reference: {os.path.basename(ref_vid_path)}")
        print(f"Distorted: {os.path.basename(dist_vid_path)}")
        
        # Call the dedicated function with the resolved paths and global config
        final_score = run_single_comparison(
            dist_vid_path=dist_vid_path, 
            ref_vid_path=ref_vid_path, 
            errmap_output_path=test_errmap_path, 
            config=CGVQM_CONFIG
        )
        
        print("\n--- Test Results ---")
        print(f"✅ CGVQM Score: {final_score:.4f}")
        print(f"Error Map saved to: {os.path.relpath(test_errmap_path, project_root)}")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        print("Please ensure your videos and CUDA/PyTorch setup are correct.")
        
    print("------------------------------------------")