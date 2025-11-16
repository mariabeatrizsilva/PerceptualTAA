import os
import sys
import time
import numpy as np
import json # Used for saving results to a JSON file
import glob # Used for finding files easily
from enum import Enum

REF_VID_PATH_REL = 'data/16SSAA.mp4'

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

import torch

if not torch.cuda.is_available():
    print("!!! WARNING: CUDA is NOT available. Running on CPU.")
    CGVQM_CONFIG['device'] = 'cpu' # Force CPU if CUDA isn't detected
else:
    print(f"CUDA is available. Using device: {CGVQM_CONFIG['device']}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}") # Prints the name of the GPU
    
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
    
def batch_process_cgvqm(subfolder_name):
    """
    Processes all videos within a specific subfolder against the reference video.

    Args:
        subfolder_name (str): The name of the folder inside 'data/' 
                              (e.g., 'vary_alpha_weight').
    """
    
    print(f"\n--- Starting Batch Process for Folder: {subfolder_name} ---")
    
    # Path where distorted videos are located (e.g., /PTAA/data/vary_alpha_weight)
    distorted_vid_path = os.path.join(project_root, 'data', subfolder_name)
    
    # Path for saving error maps (e.g., /PTAA/outputs/err_maps)
    err_map_root = os.path.join(project_root, 'outputs/err_maps')
    
    # Path for saving the score JSON file (e.g., /PTAA/outputs/scores/vary_alpha_weight_scores.json)
    err_scores_root = os.path.join(project_root, 'outputs/scores')
    err_scores_path = os.path.join(err_scores_root, f'{subfolder_name}_scores.json')
    
    # Static Reference Video Path
    ref_vid_path = os.path.join(project_root, REF_VID_PATH_REL)

    # Ensure output directories exist
    os.makedirs(err_map_root, exist_ok=True)
    os.makedirs(err_scores_root, exist_ok=True)

    distorted_videos = sorted(glob.glob(os.path.join(distorted_vid_path, '*.mp4')))
    
    if not distorted_videos:
        print(f"!!! Error: No MP4 files found in {distorted_vid_path}. Skipping.")
        return

    all_results = {}
    total_videos = len(distorted_videos)
    print(f"Found {total_videos} videos to process.")
    
    for i, dist_vid_path in enumerate(distorted_videos):
        video_file = os.path.basename(dist_vid_path)        # (e.g., vary_alpha_weight_0.1.mp4)
        name_only = os.path.splitext(video_file)[0]         # remove extension
        errmap_output_path = os.path.join(err_map_root, f'{name_only}_errmap.mp4')
        
        print(f"\nProcessing {i+1}/{total_videos}: {video_file}")
        
        try:
            score = run_single_comparison(
                dist_vid_path=dist_vid_path, 
                ref_vid_path=ref_vid_path, 
                errmap_output_path=errmap_output_path, 
                config=CGVQM_CONFIG
            )
            
            # Save score and path to results dictionary
            all_results[name_only] = {
                'score': score,
                'errmap_path_rel': os.path.relpath(errmap_output_path, project_root)
            }
            print(f'   ✅ Score: {score:.4f}')

        except Exception as e:
            print(f"   !!! Failed to process {video_file}: {e}")
            all_results[name_only] = {'error': str(e)}


    # Put all scores in one file for the whole folder
    with open(err_scores_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nCompleted processing for {subfolder_name}.")
    print(f"Scores saved to: {os.path.relpath(err_scores_path, project_root)}")
    
if __name__ == '__main__':    
    folders_to_process = [
        'vary_filter_size',
        'vary_num_samples',
        'vary_hist_percent'  # Your current folder
        # 'another_folder_name', # Uncomment and add more folders when ready
        # 'final_runs',
    ]

    for folder_name in folders_to_process:
        batch_process_cgvqm(folder_name)
    
    print("\n\n--- ALL BATCH PROCESSING COMPLETE ---")