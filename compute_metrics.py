""" Python script to compute metrics (either CVVDP or CGVQM) """
import os
import json
from enum import Enum


## Config for CGVQM
import sys
import time
import numpy as np
import glob # Used for finding files easily


## Convig for ColorVideoVDP (CVVDP)S
import subprocess
import shlex
import re 

REF_NAME = '16SSAA'
BASE_MP4 = 'data/'
BASE_FRAMES = 'data/frames/'
FRAMES_SUFFIX = '%04d.png'


class Metric(Enum):
    """Available video quality metrics."""
    CVVDP = "ColorVideoVDP"
    CGVQM = "CGVQM"

# Needed for CGVQM 
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'cgvqm'))
# from cgvqm.cgvqm import run_cgvqm, visualize_emap, CGVQM_TYPE

CGVQM_CONFIG = {
    'cgvqm_type': 'CGVQM_2', # Will be converted to CGVQM_TYPE.CGVQM_2
    'device': 'cuda',        # Change to 'cpu' if no CUDA GPU is available
    'patch_scale': 4,        # Increase this value if low on available GPU memory
    'patch_pool': 'mean'     # Choose from {'max', 'mean'}
}

def get_paths(folder_name: str, metric: Metric):
    """ returns path for folder containing videos (or frames) and error map"""
    score_file_name = f"{folder_name}_scores.json"
    if (metric == Metric.CGVQM):
        video_path = os.path.join(project_root, BASE_MP4, folder_name)
        output_scores_path = os.path.join(project_root, 'outputs/scores_cgvqm', score_file_name)
        err_map_path = os.path.join(project_root, 'outputs/err_map_cgvqm', score_file_name)
    else:
        video_path = os.path.join(project_root, BASE_FRAMES, folder_name)
        output_scores_path = os.path.join(project_root, 'outputs/scores_cvvdp',score_file_name)
        err_map_path = None
    return video_path, output_scores_path, err_map_path

def compute_metric_cgvqm():
    return 6

def compute_metric_cvvdp():
    return 5

def compute_score_single(test_name: str, folder_path: str, ref_path: str, 
                        metric: Metric, err_map_path = None) -> float:
    """
    Compute metric for a single video/frame sequence.
    
    Args:
        test_name: Name of the test (without extension)
        folder_path: Base folder containing videos/frames
        ref_path: Path to reference video or frame pattern
        metric: Which metric to compute
        err_map_path: Optional path for error map output (CGVQM only)
    
    Returns:
        Quality score as float
    """
    if metric == Metric.CGVQM:
        dist_path = os.path.join(folder_path, f"{test_name}.mp4")
        
        if not os.path.exists(dist_path):
            raise FileNotFoundError(f"Video not found: {dist_path}")
        
        score = compute_metric_cgvqm()
        
    else:  # CVVDP
        dist_folder = os.path.join(folder_path, test_name)
        dist_path = os.path.join(dist_folder, FRAMES_SUFFIX)
        
        if not os.path.exists(dist_folder):
            raise FileNotFoundError(f"Frames folder not found: {dist_folder}")
        
        score = compute_metric_cvvdp()
    
    return score

def compute_score_folder(folder_name: str, metric: Metric = Metric.CGVQM):
    """
    Compute metrics for all videos/frames in a folder.
    
    For CGVQM: Processes .mp4 files directly in folder_path
    For CVVDP: Processes PNG sequences in subfolders of folder_path
    """
    folder_path, output_scores_path, err_map_path = get_paths(
        folder_name=folder_name, metric=metric
    )
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_scores_path), exist_ok=True)
    if err_map_path:
        os.makedirs(os.path.dirname(err_map_path), exist_ok=True)
    
    results = {}
    
    # Set up reference path
    if metric == Metric.CGVQM:
        ref_path = os.path.join(folder_path, f"{REF_NAME}.mp4")
        # if not os.path.exists(ref_path):
        #     raise FileNotFoundError(f"Reference video not found: {ref_path}")

        # Get all mp4 files except reference
        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
        test_names = [
            os.path.splitext(os.path.basename(v))[0] 
            for v in video_files 
            if os.path.basename(v) != f"{REF_NAME}.mp4"
        ]
        
    else:  # CVVDP
        ref_path = os.path.join(folder_path, REF_NAME, FRAMES_SUFFIX)
        ref_folder = os.path.join(folder_path, REF_NAME)
        # if not os.path.exists(ref_folder):
        #     raise FileNotFoundError(f"Reference video not found: {ref_path}")

        # Get all subfolders except reference
        test_names = [
            f for f in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, f)) and f != REF_NAME
        ]
    
    print(f"Processing {len(test_names)} items with {metric.value}...")
    print(f"Reference: {REF_NAME}")
    
    # Process each test
    for test_name in sorted(test_names):
        print(f"  Computing metric for: {test_name}")
        
        try:
            score = compute_score_single(
                test_name=test_name,
                folder_path=folder_path,
                ref_path=ref_path,
                metric=metric,
                err_map_path=err_map_path
            )
            
            results[test_name] = score
            print(f"    Score: {score:.4f}")
            
        except FileNotFoundError as e:
            print(f"    Warning: {e}, skipping...")
            continue
        except Exception as e:
            print(f"    Error: {e}, skipping...")
            continue
    
    # Save results
    with open(output_scores_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_scores_path}")
    if results:
        print(f"Processed: {len(results)}/{len(test_names)} items")
        print(f"Average score: {np.mean(list(results.values())):.4f}")
        print(f"Score range: [{min(results.values()):.4f}, {max(results.values()):.4f}]")
    else:
        print("Warning: No results computed!")
    
    return results


if __name__ == '__main__':    
    folder_name = "vary_alpha_weight"
    compute_score_folder(folder_name=folder_name)