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
import torch
from PIL import Image
from torch.nn.functional import interpolate


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
from cgvqm.cgvqm import CGVQM, CGVQM_TYPE, visualize_emap, preprocess
from torchvision.models.video import resnet

CGVQM_CONFIG = {
    'cgvqm_type': 'CGVQM_2', # Will be converted to CGVQM_TYPE.CGVQM_2
    'device': 'cuda',        # Change to 'cpu' if no CUDA GPU is available
    'patch_scale': 4,        # Increase this value if low on available GPU memory
    'patch_pool': 'mean',    # Choose from {'max', 'mean'}
    'sleep_between_videos': 2.0  # Seconds to sleep between videos to let GPU cool down
}

# CVVDP Setup
CVVDP_EXECUTABLE = 'cvvdp'
DISPLAY_MODE = 'standard_4k'
FPS_VALUE = 30

def get_reference_paths():
    """Returns absolute paths to reference video and frames."""
    ref_video_path = os.path.join(project_root, BASE_MP4, f'{REF_NAME}.mp4')
    ref_frames_path = os.path.join(project_root, BASE_FRAMES, REF_NAME, FRAMES_SUFFIX)
    return ref_video_path, ref_frames_path

def run_cgvqm_with_tensors(D, R, metadata, cgvqm_type=CGVQM_TYPE.CGVQM_2, device='cpu', 
                           patch_pool='max', patch_scale=4):
    """
    Run CGVQM with pre-loaded tensors instead of video paths.
    This is a modified version of run_cgvqm that accepts tensors directly.
    
    Args:
        D: Distorted video tensor (T, C, H, W)
        R: Reference video tensor (T, C, H, W)
        metadata: Dict with 'shape' and 'fps' keys
        cgvqm_type: CGVQM model type
        device: 'cuda' or 'cpu'
        patch_pool: 'max' or 'mean'
        patch_scale: Spatial patch scale factor
    
    Returns:
        tuple: (quality_score, error_map)
    """
    # Load model
    model = resnet.r3d_18(weights=resnet.R3D_18_Weights.DEFAULT).to(device)
    model.__class__ = CGVQM
    dir_path = os.path.join(project_root, 'src', 'cgvqm')
    
    if cgvqm_type == CGVQM_TYPE.CGVQM_2:
        model.init_weights(os.path.join(dir_path, 'weights', 'cgvqm-2.pickle'), num_layers=3)
    elif cgvqm_type == CGVQM_TYPE.CGVQM_5:
        model.init_weights(os.path.join(dir_path, 'weights', 'cgvqm-5.pickle'), num_layers=6)
    else:
        raise Exception("ERROR: unknown model type")
    
    model.eval()
    
    # Preprocess and add batch dimension
    D = preprocess(D).unsqueeze(0)
    R = preprocess(R).unsqueeze(0)
    
    # Divide video into patches and calculate quality of each patch
    if D.shape[3] % patch_scale != 0 or D.shape[4] % patch_scale != 0:
        print(f'WARNING: Spatial resolution not divisible by {patch_scale}. Error map resolution might not match input videos')
    
    ps = [int(D.shape[3] / patch_scale), int(D.shape[4] / patch_scale)]
    clip_size = int(min(metadata['fps'], 30))  # temporal duration of each patch
    
    # Pad videos in space-time to be divisible by patch size
    D = torch.nn.functional.pad(
        D, 
        (0, (ps[1] - D.shape[4] % ps[1]) % ps[1], 
         0, (ps[0] - D.shape[3] % ps[0]) % ps[0], 
         0, (clip_size - D.shape[2] % clip_size) % clip_size), 
        mode='replicate'
    )
    R = torch.nn.functional.pad(
        R, 
        (0, (ps[1] - R.shape[4] % ps[1]) % ps[1], 
         0, (ps[0] - R.shape[3] % ps[0]) % ps[0], 
         0, (clip_size - R.shape[2] % clip_size) % clip_size), 
        mode='replicate'
    )
    
    emap = torch.zeros([R.shape[2], R.shape[3], R.shape[4]])
    patch_errors = []
    count = 0
    
    for i in range(0, D.shape[2], clip_size):
        for h in range(0, D.shape[3], ps[0]):
            for w in range(0, D.shape[4], ps[1]):
                Cd = D[:, :, i:min(i + clip_size, D.shape[2]), 
                       h:h + min(ps[0], D.shape[3]), 
                       w:w + min(ps[1], D.shape[4])].to(device)
                Cr = R[:, :, i:min(i + clip_size, D.shape[2]), 
                       h:h + min(ps[0], D.shape[3]), 
                       w:w + min(ps[1], D.shape[4])].to(device)
                
                with torch.no_grad():
                    q, em = model.feature_diff(Cd, Cr)
                    emap[i:min(i + clip_size, D.shape[2]), 
                         h:h + min(ps[0], D.shape[3]), 
                         w:w + min(ps[1], D.shape[4])] = em.squeeze()
                    patch_errors.append(q)
                count += 1
    
    emap = emap[:metadata['shape'][0], :metadata['shape'][2], :metadata['shape'][3]]
    
    if patch_pool == 'max':
        q = 100 - max(patch_errors)
    elif patch_pool == 'mean':
        q = 100 - torch.mean(torch.stack(patch_errors))
    else:
        raise Exception("ERROR: unknown patch pooling method")
    
    return q, emap

def load_frames_from_pngs(frames_folder: str) -> torch.Tensor:
    """
    Load video frames from PNG files and convert to tensor.
    
    Args:
        frames_folder: Path to folder containing PNG frames (named 0001.png, 0002.png, etc.)
    
    Returns:
        torch.Tensor of shape (T, C, H, W) with values in [0, 255]
    """
    png_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
    
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {frames_folder}")
    
    print(f"    Loading {len(png_files)} frames from PNGs...")
    
    frames = []
    for png_file in png_files:
        img = Image.open(png_file).convert('RGB')
        img_array = np.array(img)  # (H, W, 3)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
        frames.append(img_tensor)
    
    frames_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
    print(f"    Loaded tensor shape: {frames_tensor.shape}")
    
    return frames_tensor

def load_resize_frames(test_frames_folder: str, ref_frames_folder: str) -> tuple:
    """
    Load and resize frames from PNGs, replicating load_resize_vids behavior.
    
    Args:
        test_frames_folder: Path to folder with test/distorted frames
        ref_frames_folder: Path to folder with reference frames
    
    Returns:
        tuple: (D, R, metadata) matching load_resize_vids output format
    """
    D = load_frames_from_pngs(test_frames_folder)
    R = load_frames_from_pngs(ref_frames_folder)
    
    T, C, H, W = R.shape
    
    # Spatial resize if needed
    if D.shape[2:] != R.shape[2:]:
        print(f"    Resizing spatial dimensions from {D.shape[2:]} to {R.shape[2:]}")
        D = interpolate(D.float(), size=(R.shape[2], R.shape[3]), mode='bilinear')
        D = D.to(torch.uint8)
    
    # Temporal resize if needed
    if D.shape[0] != R.shape[0]:
        print(f"    Resizing temporal dimension from {D.shape[0]} to {R.shape[0]} frames")
        D = D.unsqueeze(0).permute(0, 2, 1, 3, 4)
        R_temp = R.unsqueeze(0).permute(0, 2, 1, 3, 4)
        D = interpolate(D.float(), size=(R_temp.shape[2], R_temp.shape[3], R_temp.shape[4]), mode='nearest')
        D = D.squeeze().permute(1, 0, 2, 3)
        D = D.to(torch.uint8)
    
    metadata = {'shape': (T, C, H, W), 'fps': FPS_VALUE}
    
    return D, R, metadata

def get_paths(folder_name: str, metric: Metric):
    """Returns path for folder containing videos (or frames) and output scores path"""
    score_file_name = f"{folder_name}_scores.json"
    # Both metrics now use frames
    frames_path = os.path.join(project_root, BASE_FRAMES, folder_name)
    
    if metric == Metric.CGVQM:
        output_scores_path = os.path.join(project_root, 'outputs/scores_cgvqm', score_file_name)
    else:
        output_scores_path = os.path.join(project_root, 'outputs/scores_cvvdp', score_file_name)
    
    return frames_path, output_scores_path

def compute_metric_cgvqm(ref_frames_folder: str, dist_frames_folder: str, config: dict, 
                         err_map_path: str, dist_video_path: str = None) -> float:
    """
    Compute CGVQM score for a single video pair using PNG frames.
    
    Args:
        ref_frames_folder: Path to reference frames folder
        dist_frames_folder: Path to distorted frames folder
        config: CGVQM configuration dict
        err_map_path: Full path (including filename) to save error map visualization
        dist_video_path: Optional path to distorted video for error map visualization
    
    Returns:
        Quality score as float
    """
    
    # Load frames from PNGs
    print(f"    Loading frames...")
    D, R, metadata = load_resize_frames(dist_frames_folder, ref_frames_folder)
    
    # Map string config to the Enum type required by the library
    cgvqm_type_enum = getattr(CGVQM_TYPE, config['cgvqm_type'])
    
    # Run CGVQM with frame tensors
    q, emap = run_cgvqm_with_tensors(
        D,              # distorted frames tensor
        R,              # reference frames tensor
        metadata,       # metadata dict
        cgvqm_type=cgvqm_type_enum, 
        device=config['device'], 
        patch_pool=config['patch_pool'], 
        patch_scale=config['patch_scale']
    )
    
    score = q.item()
    
    # Save the error map visualization
    os.makedirs(os.path.dirname(err_map_path), exist_ok=True)
    
    # For visualize_emap, we can either pass the dist_video_path if available,
    # or pass the dist_frames_folder - need to check what visualize_emap expects
    viz_path = dist_video_path if dist_video_path else dist_frames_folder
    visualize_emap(emap, viz_path, 100, err_map_path)
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

def compute_score_single(test_name: str, folder_path: str, ref_frames_folder: str, 
                        metric: Metric) -> float:
    """
    Compute metric for a single video/frame sequence.
    
    Args:
        test_name: Name of the test (without extension)
        folder_path: Base folder containing frames subfolders
        ref_frames_folder: Path to reference frames folder (for CGVQM) or pattern (for CVVDP)
        metric: Which metric to compute
    
    Returns:
        Quality score as float
    """
    if metric == Metric.CGVQM:
        # For CGVQM, we now use frame folders
        dist_frames_folder = os.path.join(folder_path, test_name)
        
        if not os.path.exists(dist_frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {dist_frames_folder}")
        
        # Create error map path for this video
        err_map_name = f"{test_name}_errmap.mp4"
        err_map_path = os.path.join(project_root, 'outputs/err_maps', err_map_name)
        
        # Optional: if you still have MP4s and want to use them for visualization
        dist_video_path = os.path.join(project_root, BASE_MP4, folder_path.split('/')[-1], f"{test_name}.mp4")
        if not os.path.exists(dist_video_path):
            dist_video_path = None
        
        score = compute_metric_cgvqm(
            ref_frames_folder=ref_frames_folder,
            dist_frames_folder=dist_frames_folder,
            config=CGVQM_CONFIG,
            err_map_path=err_map_path,
            dist_video_path=dist_video_path
        )
        
    else:  # CVVDP
        dist_folder = os.path.join(folder_path, test_name)
        dist_path = os.path.join(dist_folder, FRAMES_SUFFIX)
        
        if not os.path.exists(dist_folder):
            raise FileNotFoundError(f"Frames folder not found: {dist_folder}")
                
        score = compute_metric_cvvdp(
            ref_path=ref_frames_folder,  # This is already a pattern for CVVDP
            dist_path=dist_path
        )
    
    return score

def compute_score_folder(folder_name: str, metric: Metric = Metric.CGVQM):
    """
    Compute metrics for all videos/frames in a folder.
    
    For CGVQM: Processes PNG frame sequences in subfolders of folder_path (no longer uses MP4)
    For CVVDP: Processes PNG sequences in subfolders of folder_path
    
    Both metrics now load frames from PNG files to avoid compression artifacts.
    """
    folder_path, output_scores_path = get_paths(
        folder_name=folder_name, metric=metric
    )
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_scores_path), exist_ok=True)
    
    results = {}
    
    # Get reference paths
    ref_video_path, ref_frames_path = get_reference_paths()
    
    # Set up reference path and get test names based on metric
    if metric == Metric.CGVQM:
        # For CGVQM, use reference frames folder (not video)
        ref_frames_folder = os.path.join(project_root, BASE_FRAMES, REF_NAME)
        if not os.path.exists(ref_frames_folder):
            raise FileNotFoundError(f"Reference frames folder not found: {ref_frames_folder}")

        # Get all subfolders except reference (same as CVVDP)
        test_names = [
            f for f in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, f)) and f != REF_NAME
        ]
        ref_path = ref_frames_folder
        
    else:  # CVVDP
        ref_path = ref_frames_path
        ref_folder = os.path.join(project_root, BASE_FRAMES, REF_NAME)
        if not os.path.exists(ref_folder):
            raise FileNotFoundError(f"Reference frames folder not found: {ref_folder}")

        # Get all subfolders except reference
        test_names = [
            f for f in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, f)) and f != REF_NAME
        ]
    
    print(f"Processing {len(test_names)} items with {metric.value}...")
    print(f"Reference: {REF_NAME}")
    print(f"Reference path: {ref_path}")
    
    # Process each test
    for test_name in sorted(test_names):
        print(f"  Computing metric for: {test_name}")
        
        try:
            score = compute_score_single(
                test_name=test_name,
                folder_path=folder_path,
                ref_frames_folder=ref_path,
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
        
        # GPU cooling: sleep between videos if using CGVQM
        if metric == Metric.CGVQM and CGVQM_CONFIG.get('sleep_between_videos', 0) > 0:
            sleep_time = CGVQM_CONFIG['sleep_between_videos']
            print(f"    Sleeping {sleep_time}s to let GPU cool down...")
            time.sleep(sleep_time)
            
            # Optional: Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
  
  # Test on a single video (for debugging/testing)
  python compute_metrics.py --metric CGVQM --folders vary_alpha_weight --single video_name
  
  # Use short flags
  python compute_metrics.py -m CGVQM -f vary_alpha_weight -s video_name
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
    
    parser.add_argument(
        '--single', '-s',
        type=str,
        default=None,
        help='Process only a single video (specify video name without extension). Useful for testing.'
    )
    
    args = parser.parse_args()
    
    # Convert string to Metric enum
    metric = Metric.CGVQM if args.metric == 'CGVQM' else Metric.CVVDP
    
    print(f"\n{'='*60}")
    print(f"Video Quality Metric Computation")
    print(f"Metric: {metric.value}")
    print(f"Folders: {', '.join(args.folders)}")
    if args.single:
        print(f"Single Video Mode: {args.single}")
    print(f"{'='*60}\n")
    
    # Single video mode
    if args.single:
        if len(args.folders) > 1:
            print("Warning: Single video mode only processes the first folder specified.")
        
        folder_name = args.folders[0]
        test_name = args.single
        
        try:
            folder_path, output_scores_path = get_paths(
                folder_name=folder_name, metric=metric
            )
            
            # Get reference paths
            ref_video_path, ref_frames_path = get_reference_paths()
            
            if metric == Metric.CGVQM:
                ref_frames_folder = os.path.join(project_root, BASE_FRAMES, REF_NAME)
                if not os.path.exists(ref_frames_folder):
                    raise FileNotFoundError(f"Reference frames folder not found: {ref_frames_folder}")
                ref_path = ref_frames_folder
            else:  # CVVDP
                ref_path = ref_frames_path
            
            print(f"Processing single video: {test_name}")
            print(f"Folder: {folder_path}")
            print(f"Reference: {ref_path}\n")
            
            score = compute_score_single(
                test_name=test_name,
                folder_path=folder_path,
                ref_frames_folder=ref_path,
                metric=metric
            )
            
            print(f"\n{'='*60}")
            print(f"RESULT: {test_name}")
            print(f"Score: {score:.4f}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\nError processing single video '{test_name}': {e}\n")
            import traceback
            traceback.print_exc()
        
        return
    
    # Normal mode: Process all videos in folders
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