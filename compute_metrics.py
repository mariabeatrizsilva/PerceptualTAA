""" Python script to compute metrics (either CVVDP or CGVQM) """
import os
import json
from enum import Enum
import argparse


## Config for CGVQM
import sys
import time
import numpy as np
import glob # Used for finding files easily


## Config for ColorVideoVDP (CVVDP)
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

# CGVQM Setup
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'cgvqm'))
from cgvqm.cgvqm import run_cgvqm, visualize_emap, CGVQM_TYPE

CGVQM_CONFIG = {
    'cgvqm_type': 'CGVQM_2', # Will be converted to CGVQM_TYPE.CGVQM_2
    'device': 'cuda',        # Change to 'cpu' if no CUDA GPU is available
    'patch_scale': 4,        # Increase this value if low on available GPU memory
    'patch_pool': 'mean'     # Choose from {'max', 'mean'}
}

# CVVDP Setup
CVVDP_EXECUTABLE = 'cvvdp'
DISPLAY_MODE = 'standard_4k'
FPS_VALUE = 30

def get_paths(folder_name: str, metric: Metric):
    """Returns path for folder containing videos (or frames) and output scores path"""
    score_file_name = f"{folder_name}_scores.json"
    if metric == Metric.CGVQM:
        video_path = os.path.join(project_root, BASE_MP4, folder_name)
        output_scores_path = os.path.join(project_root, 'outputs/scores_cgvqm', score_file_name)
    else:
        video_path = os.path.join(project_root, BASE_FRAMES, folder_name)
        output_scores_path = os.path.join(project_root, 'outputs/scores_cvvdp', score_file_name)
    return video_path, output_scores_path

def compute_metric_cgvqm(ref_path: str, dist_path: str, config: dict, 
                         err_map_path: str) -> float:
    """
    Compute CGVQM score for a single video pair.
    
    Args:
        ref_path: Path to reference video
        dist_path: Path to distorted video
        config: CGVQM configuration dict
        err_map_path: Full path (including filename) to save error map visualization
    
    Returns:
        Quality score as float
    """
    
    # Map string config to the Enum type required by the library
    cgvqm_type_enum = getattr(CGVQM_TYPE, config['cgvqm_type'])
    
    # Run CGVQM
    q, emap = run_cgvqm(
        dist_path,      # distorted video
        ref_path,       # reference video
        cgvqm_type=cgvqm_type_enum, 
        device=config['device'], 
        patch_pool=config['patch_pool'], 
        patch_scale=config['patch_scale']
    )
    
    score = q.item()
    
    # Save the error map visualization
    os.makedirs(os.path.dirname(err_map_path), exist_ok=True)
    visualize_emap(emap, dist_path, 100, err_map_path)
    print(f"    Error map saved to: {err_map_path}")
    
    return score


def compute_metric_cvvdp(ref_path: str, dist_path: str) -> float:
    """
    Compute ColorVideoVDP score for a frame sequence pair.
    
    Args:
        ref_path: Path pattern to reference frames
        dist_path: Path pattern to distorted frames
    
    Returns:
        Quality score as float
    """
    
    # Construct the command
    command = [
        CVVDP_EXECUTABLE,
        '--test', dist_path,
        '--ref', ref_path,
        '--display', DISPLAY_MODE,
        '--fps', str(FPS_VALUE)
    ]
    
    try:
        # Execute the command
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            shell=False
        )
        
        # Extract the score from output using regex
        score_pattern = re.compile(r"cvvdp=(\d+\.?\d*)")
        match = score_pattern.search(result.stdout)
        
        if match:
            score = float(match.group(1))
            return score
        else:
            raise ValueError("Could not extract score from CVVDP output")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"CVVDP command failed with exit code {e.returncode}: {e.stderr}")
    
    except FileNotFoundError:
        raise RuntimeError(f"CVVDP executable '{CVVDP_EXECUTABLE}' not found. "
                         "Ensure cvvdp is installed and in PATH.")

def compute_score_single(test_name: str, folder_path: str, ref_path: str, 
                        metric: Metric) -> float:
    """
    Compute metric for a single video/frame sequence.
    
    Args:
        test_name: Name of the test (without extension)
        folder_path: Base folder containing videos/frames
        ref_path: Path to reference video or frame pattern
        metric: Which metric to compute
    
    Returns:
        Quality score as float
    """
    if metric == Metric.CGVQM:
        dist_path = os.path.join(folder_path, f"{test_name}.mp4")
        
        if not os.path.exists(dist_path):
            raise FileNotFoundError(f"Video not found: {dist_path}")
        
        # Create error map path for this video
        err_map_name = f"{test_name}_errmap.mp4"
        err_map_path = os.path.join(project_root, 'outputs/err_maps', err_map_name)
        
        score = compute_metric_cgvqm(
            ref_path=ref_path,
            dist_path=dist_path,
            config=CGVQM_CONFIG,
            err_map_path=err_map_path
        )
        
    else:  # CVVDP
        dist_folder = os.path.join(folder_path, test_name)
        dist_path = os.path.join(dist_folder, FRAMES_SUFFIX)
        
        if not os.path.exists(dist_folder):
            raise FileNotFoundError(f"Frames folder not found: {dist_folder}")
                
        score = compute_metric_cvvdp(
            ref_path=ref_path,
            dist_path=dist_path
        )
    
    return score

def compute_score_folder(folder_name: str, metric: Metric = Metric.CGVQM):
    """
    Compute metrics for all videos/frames in a folder.
    
    For CGVQM: Processes .mp4 files directly in folder_path
    For CVVDP: Processes PNG sequences in subfolders of folder_path
    """
    folder_path, output_scores_path = get_paths(
        folder_name=folder_name, metric=metric
    )
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_scores_path), exist_ok=True)
    
    results = {}
    
    # Set up reference path
    if metric == Metric.CGVQM:
        ref_path = os.path.join(folder_path, f"{REF_NAME}.mp4")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference video not found: {ref_path}")

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
        if not os.path.exists(ref_folder):
            raise FileNotFoundError(f"Reference frames folder not found: {ref_folder}")

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
                metric=metric
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


def main():
    """Main function to parse arguments and run metric computation."""
    parser = argparse.ArgumentParser(
        description='Compute video quality metrics (CGVQM or CVVDP) for video folders.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute CGVQM for a single folder
  python compute_metrics.py --metric CGVQM --folders vary_alpha_weight
  
  # Compute CVVDP for multiple folders
  python compute_metrics.py --metric CVVDP --folders vary_filter_size vary_num_samples
  
  # Use short flags
  python compute_metrics.py -m CGVQM -f vary_alpha_weight vary_hist_percent
        """
    )
    
    parser.add_argument(
        '--metric', '-m',
        type=str,
        required=True,
        choices=['CGVQM', 'CVVDP'],
        help='Video quality metric to compute (CGVQM or CVVDP)'
    )
    
    parser.add_argument(
        '--folders', '-f',
        type=str,
        nargs='+',
        required=True,
        help='Folder name(s) to process. Can specify multiple folders separated by spaces.'
    )
    
    args = parser.parse_args()
    
    # Convert string to Metric enum
    metric = Metric.CGVQM if args.metric == 'CGVQM' else Metric.CVVDP
    
    print(f"\n{'='*60}")
    print(f"Video Quality Metric Computation")
    print(f"Metric: {metric.value}")
    print(f"Folders: {', '.join(args.folders)}")
    print(f"{'='*60}\n")
    
    # Process each folder
    for folder_name in args.folders:
        try:
            compute_score_folder(folder_name=folder_name, metric=metric)
        except Exception as e:
            print(f"\n Failed to process folder '{folder_name}': {e}\n")
            continue
    
    print(f"\n{'='*60}")
    print("ALL PROCESSING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()