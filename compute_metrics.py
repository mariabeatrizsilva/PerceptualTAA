""" Python script to compute metrics (either CVVDP or CGVQM) """
import os
import json
from enum import Enum
import argparse
import glob



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

# ============================================================================
# CONFIGURATION - SCENE SETTINGS
# ============================================================================

# python compute_metrics.py -m CGVQM --all --scenes fantasticvillage-open quarry-all quarry-rocksonly resto-close resto-pan subway-lookdown
REF_NAME = '16SSAA'
BASE_MP4 = 'data/'
FRAMES_SUFFIX = '%04d.png'

MISC_FOLDER = 'misc_params'
MISC_SCORES = 'misc_scores'

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


def get_base_frames(scene_name):
    return f'data/{scene_name}/'

def get_output_paths(folder_name: str, metric: Metric, scene_name: str):
    """
    Returns output paths based on scene name and metric.
    
    Returns:
        tuple: (scores_dir, err_maps_dir)
    """
    base_output = os.path.join(project_root, 'outputs', scene_name)
    
    if metric == Metric.CGVQM:
        scores_dir = os.path.join(base_output, 'scores_cgvqm')
        err_maps_dir = os.path.join(base_output, 'err_maps_cgvqm')
    else:
        scores_dir = os.path.join(base_output, 'scores_cvvdp')
        err_maps_dir = os.path.join(base_output, 'err_maps_cvvdp')
    
    os.makedirs(scores_dir, exist_ok=True)
    if err_maps_dir:
        os.makedirs(err_maps_dir, exist_ok=True)

    return scores_dir, err_maps_dir


def get_reference_paths(scene_name: str):
    """Returns absolute paths to reference video and frames."""
    ref_video_path = os.path.join(project_root, BASE_MP4, f'{REF_NAME}.mp4')
    ref_frames_path = os.path.join(project_root, get_base_frames(scene_name=scene_name), REF_NAME, FRAMES_SUFFIX)
    return ref_video_path, ref_frames_path

def get_paths(folder_name: str, metric: Metric, scene_name: str, ref_scene: str = None):
    frames_path = os.path.join(project_root, get_base_frames(scene_name=scene_name), folder_name)
    
    scores_dir, err_maps_dir = get_output_paths(folder_name, metric, scene_name)
    
    ref_suffix = f"_ref-{ref_scene}" if ref_scene and ref_scene != scene_name else ""
    score_file_name = f"{folder_name}_scores{ref_suffix}.json"
    output_scores_path = os.path.join(scores_dir, score_file_name)
    
    return frames_path, output_scores_path, err_maps_dir

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

def get_cgvqm_per_frame_errors(emap):
    """
    Extract per-frame error from CGVQM error map.
    
    Args:
        emap: Error map tensor of shape [T, H, W]
    
    Returns:
        per_frame_errors: Tensor of shape [T] with average error per frame
    """
    # Average over spatial dimensions (H, W) for each frame
    per_frame_errors = emap.mean(dim=(1, 2))  # Shape: [T]
    return per_frame_errors

def compute_metric_cgvqm(ref_frames_folder: str, dist_frames_folder: str, config: dict, 
                         err_map_path: str, dist_video_path: str = None):
    """
    Compute CGVQM score for a single video pair using PNG frames.
    
    Args:
        ref_frames_folder: Path to reference frames folder
        dist_frames_folder: Path to distorted frames folder
        config: CGVQM configuration dict
        err_map_path: Full path (including filename) to save error map visualization
        dist_video_path: Optional path to distorted video for error map visualization
    
    Returns:
        tuple: (score, per_frame_errors_array)
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
    per_frame_errors = get_cgvqm_per_frame_errors(emap)
    per_frame_errors_array = per_frame_errors.cpu().numpy().tolist()
    
    # Save error map only if video exists
    if dist_video_path and os.path.exists(dist_video_path):
        os.makedirs(os.path.dirname(err_map_path), exist_ok=True)
        try:
            visualize_emap(emap, dist_video_path, 100, err_map_path)
            print(f"    Error map saved to: {err_map_path}")
        except Exception as e:
            print(f"    Warning: Error map generation failed: {e}")
    else:
        print(f"    Skipping error map (no video available)")
    
    return score, per_frame_errors_array


"""
Minimal changes to fix CVVDP error map generation
Replace your compute_metric_cvvdp function with this version
"""
def compute_metric_cvvdp(ref_path: str, dist_path: str, heatmap_output_dir: str = None):
    """
    Compute ColorVideoVDP score for a frame sequence pair.
    
    Args:
        ref_path: Path pattern to reference frames (e.g., /path/to/ref/%04d.png)
        dist_path: Path pattern to distorted frames (e.g., /path/to/video_name/%04d.png)
        heatmap_output_dir: Optional directory to save heatmap video
    
    Returns:
        score (float)
    """
    
    # Determine if we need to save heatmap
    temp_heatmap_video = None
    if heatmap_output_dir:
        os.makedirs(heatmap_output_dir, exist_ok=True)
        
        # FIXED: Extract video name correctly from the path pattern
        # dist_path is like: /path/to/folder/video_name/%04d.png
        # We want: video_name
        # First remove the frame pattern (%04d.png)
        dist_folder = os.path.dirname(dist_path)  # Gets /path/to/folder/video_name
        video_name = os.path.basename(dist_folder)  # Gets video_name
        
        temp_heatmap_video = os.path.join(heatmap_output_dir, f'{video_name}_heatmap')
    
    # Construct the command
    command = [
        CVVDP_EXECUTABLE,
        '--test', dist_path,
        '--ref', ref_path,
        '--display', DISPLAY_MODE,
        '--fps', str(FPS_VALUE)
    ]
    
    # Add heatmap output if requested
    if temp_heatmap_video:
        command.extend([
            '--heatmap', 'raw',
            '-o', temp_heatmap_video
        ])
    
    try:
        # Execute the command
        print(f"    Running CVVDP...")
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            shell=False
        )
        
        # Try multiple patterns to extract the score
        score_pattern1 = re.compile(r"cvvdp\s*=\s*(\d+\.?\d*)")
        score_pattern2 = re.compile(r"cvvdp\s*:\s*(\d+\.?\d*)")
        score_pattern3 = re.compile(r"(\d+\.\d+)")
        
        match = score_pattern1.search(result.stdout)
        if not match:
            match = score_pattern2.search(result.stdout)
        if not match:
            match = score_pattern3.search(result.stdout)
        
        if match:
            score = float(match.group(1))
        else:
            print(f"    CVVDP stdout: {result.stdout}")
            print(f"    CVVDP stderr: {result.stderr}")
            raise ValueError("Could not extract score from CVVDP output")
        
        # Verify the heatmap was created and is valid
        if temp_heatmap_video:
            if os.path.exists(temp_heatmap_video):
                file_size = os.path.getsize(temp_heatmap_video)
                print(f"    Heatmap saved: {temp_heatmap_video} ({file_size} bytes)")
                
                if file_size < 1000:  # Suspiciously small file
                    print(f"    WARNING: Heatmap file is very small, may be corrupted")
            else:
                print(f"    WARNING: Heatmap file was not created at {temp_heatmap_video}")
        
        return score
        
    except subprocess.CalledProcessError as e:
        print(f"    CVVDP stdout: {e.stdout}")
        print(f"    CVVDP stderr: {e.stderr}")
        raise RuntimeError(f"CVVDP command failed with exit code {e.returncode}: {e.stderr}")
    
    except FileNotFoundError:
        raise RuntimeError(f"CVVDP executable '{CVVDP_EXECUTABLE}' not found. "
                         "Ensure cvvdp is installed and in PATH.")
    

"""
Update your compute_score_single function to handle simplified CVVDP return
"""

def compute_score_single(test_name: str, folder_path: str, ref_frames_folder: str,
                        metric: Metric, scene_name: str, err_maps_dir: str = None):
    """
    Compute metric for a single video/frame sequence.
    """
    if metric == Metric.CGVQM:
        dist_frames_folder = os.path.join(folder_path, test_name)
        
        if not os.path.exists(dist_frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {dist_frames_folder}")
        
        err_map_name = f"{test_name}_errmap.mp4"
        err_map_path = os.path.join(err_maps_dir, err_map_name)
        
        folder_name = os.path.basename(folder_path)
        dist_video_path = os.path.join(
            project_root,
            'outputs',
            scene_name,
            'videos',
            folder_name,
            f"{test_name}.mp4"
        )
        
        if not os.path.exists(dist_video_path):
            print(f"    Warning: Video not found at {dist_video_path}")
            dist_video_path = None
        else:
            print(f"    Using video from: {dist_video_path}")
        
        score, per_frame_errors = compute_metric_cgvqm(
            ref_frames_folder=ref_frames_folder,
            dist_frames_folder=dist_frames_folder,
            config=CGVQM_CONFIG,
            err_map_path=err_map_path,
            dist_video_path=dist_video_path
        )
        
        return score, per_frame_errors
            
def compute_score_folder(folder_name: str, metric: Metric, scene_name: str, ref_scene: str = None):
    ref_scene_name = ref_scene if ref_scene else scene_name  # <-- add this

    folder_path, output_scores_path, err_maps_dir = get_paths(
        folder_name=folder_name, metric=metric, scene_name=scene_name, ref_scene=ref_scene  # <-- add ref_scene
    )
    
    # Load existing results if they exist (for resuming)
    if os.path.exists(output_scores_path):
        print(f"Found existing results file: {output_scores_path}")
        with open(output_scores_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len([k for k in results if not k.startswith('_')])} existing results. Will skip already-processed videos.")
    else:
        results = {
            "_meta": {
                "reference_scene": ref_scene_name,
                "metric": metric.value,
                "scene": scene_name,
            }
        }  # <-- replaces results = {}
    
    ref_video_path, ref_frames_path = get_reference_paths(ref_scene_name)  # <-- was scene_name

    if metric == Metric.CGVQM:
        ref_frames_folder = os.path.join(project_root, f'data/{ref_scene_name}/', REF_NAME)  # <-- was scene_name
        if not os.path.exists(ref_frames_folder):
            raise FileNotFoundError(f"Reference frames folder not found: {ref_frames_folder}")

        test_names = [
            f for f in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, f)) and f != REF_NAME
        ]
        ref_path = ref_frames_folder
        
    else:  # CVVDP
        ref_path = ref_frames_path
        ref_folder = os.path.join(project_root, f'data/{scene_name}/', REF_NAME)
        if not os.path.exists(ref_folder):
            raise FileNotFoundError(f"Reference frames folder not found: {ref_folder}")

        test_names = [
            f for f in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, f)) and f != REF_NAME
        ]
    
    # Filter out already-processed videos
    remaining_test_names = [name for name in test_names if name not in results and not name.startswith('_')]
    
    print(f"Total videos: {len(test_names)}")
    print(f"Already processed: {len(results)}")
    print(f"Remaining to process: {len(remaining_test_names)}")
    print(f"Processing with {metric.value}...")
    print(f"Reference: {REF_NAME}")
    print(f"Reference path: {ref_path}")
    print(f"Scene: {scene_name}")
    print(f"Output scores: {output_scores_path}")
    if err_maps_dir:
        print(f"Error maps: {err_maps_dir}")
    print()
    
    # Process each test
    for idx, test_name in enumerate(sorted(remaining_test_names), 1):
        print(f"[{idx}/{len(remaining_test_names)}] Computing metric for: {test_name}")
        
        try:
            result = compute_score_single(
                test_name=test_name,
                folder_path=folder_path,
                ref_frames_folder=ref_path,
                metric=metric,
                scene_name=scene_name,
                err_maps_dir=err_maps_dir
            )
            
            # Store result based on metric type
            if metric == Metric.CGVQM:
                score, per_frame_errors = result
                results[test_name] = {
                    'score': score,
                    'per_frame_errors': per_frame_errors
                }
                print(f"    Score: {score:.4f}")
                print(f"    Per-frame errors: {len(per_frame_errors)} frames")
            else:  # CVVDP
                score, _ = result
                results[test_name] = score
                print(f"    Score: {score:.4f}")
            
            # Save results immediately after each video
            with open(output_scores_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"    ✓ Results saved to: {output_scores_path}")
            
        except FileNotFoundError as e:
            print(f"    Warning: {e}, skipping...")
            continue
        except Exception as e:
            print(f"    Error: {e}, skipping...")
            import traceback
            traceback.print_exc()
            continue
        
        # GPU cooling for CGVQM only
        if metric == Metric.CGVQM and CGVQM_CONFIG.get('sleep_between_videos', 0) > 0:
            sleep_time = CGVQM_CONFIG['sleep_between_videos']
            print(f"    Sleeping {sleep_time}s to let GPU cool down...")
            time.sleep(sleep_time)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE FOR: {folder_name}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_scores_path}")
    if results:
        print(f"Total processed: {len(results)}/{len(test_names)} items")
        
        if metric == Metric.CGVQM:
            scores = [v['score'] for v in results.values()]
            print(f"Average score: {np.mean(scores):.4f}")
            print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        else:
            scores = []
            for v in results.values():
                if isinstance(v, dict):
                    scores.append(v['score'])
                else:
                    scores.append(v)
            
            if scores:
                print(f"Average score: {np.mean(scores):.4f}")
                print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
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
  # Multiple scenes with specific folders
  python compute_metrics.py -m CGVQM -f vary_alpha_weight --scenes scene1 scene2 scene3
  
  # Multiple scenes with --all flag
  python compute_metrics.py -m CGVQM --all --scenes oldmine-screen-per-25 oldmine-screen-per-50 oldmine-screen-per-75 village-screen-per-25 village-screen-per-50 village-screen-per-75
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
        help='Folder name(s) to process.'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Compute metrics over all subfolders, excluding reference.'
    )

    parser.add_argument(
        '--single', '-s',
        type=str,
        default=None,
        help='Process only a single video (specify video name without extension). Useful for testing.'
    )
    
    parser.add_argument(
        '--scenes',
        type=str,
        nargs='+',
        help='Scene name(s) to process.'
    )

    parser.add_argument(
        '--ref-scene',
        type=str,
        default=None,
        help='Override the scene used for the 16SSAA reference. E.g. --ref-scene oldmine'
    )
    
    args = parser.parse_args()
    
    # Convert string to Metric enum
    metric = Metric.CGVQM if args.metric == 'CGVQM' else Metric.CVVDP
    
    # Loop over scenes
    for scene_idx, scene_name in enumerate(args.scenes, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING SCENE [{scene_idx}/{len(args.scenes)}]: {scene_name}")
        print(f"{'='*80}\n")
        
        # Determine target folders for this scene
        target_folders = []
        if args.all:
            root_dir = os.path.join(project_root, 'data', scene_name)
            if not os.path.exists(root_dir):
                print(f"Warning: Root directory {root_dir} does not exist. Skipping scene.")
                continue

            subfolders = [
                d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d)) and d != REF_NAME
            ]
            target_folders = sorted(subfolders)
            print(f"Found {len(target_folders)} subfolders in {root_dir} (excluding {REF_NAME})")
        elif args.folders:
            target_folders = args.folders
        else:
            print("Error: You must provide either --folders or use the --all flag.")
            continue
        
        print(f"Metric: {metric.value}")
        print(f"Folders to process: {', '.join(target_folders)}")
        if args.single:
            print(f"Single Video Mode: {args.single}")
        print()
        
        # Single video mode
        if args.single:
            if len(target_folders) > 1:
                print("Warning: Single video mode only processes the first folder specified.")
            
            folder_name = target_folders[0]
            test_name = args.single
            
            try:
                folder_path, output_scores_path, err_maps_dir = get_paths(
                    folder_name=folder_name, metric=metric, scene_name=scene_name
                )
                ref_scene_name = args.ref_scene if args.ref_scene else scene_name
                ref_video_path, ref_frames_path = get_reference_paths(ref_scene_name)
                
                if metric == Metric.CGVQM:
                    ref_frames_folder = os.path.join(project_root, f'data/{ref_scene_name}/', REF_NAME)  # <-- was scene_name
                    if not os.path.exists(ref_frames_folder):
                        raise FileNotFoundError(f"Reference frames folder not found: {ref_frames_folder}")
                    ref_path = ref_frames_folder
                else:
                    ref_path = ref_frames_path
                
                print(f"Processing single video: {test_name}")
                print(f"Folder: {folder_path}")
                print(f"Reference: {ref_path}\n")
                
                result = compute_score_single(
                    test_name=test_name,
                    folder_path=folder_path,
                    ref_frames_folder=ref_path,
                    metric=metric,
                    scene_name=scene_name,
                    err_maps_dir=err_maps_dir
                )
                
                print(f"\n{'='*60}")
                print(f"RESULT: {test_name}")
                
                if metric == Metric.CGVQM:
                    score, per_frame_errors = result
                    print(f"Overall Score: {score:.4f}")
                    print(f"Number of frames: {len(per_frame_errors)}")
                    print(f"Mean per-frame error: {np.mean(per_frame_errors):.4f}")
                    print(f"Std per-frame error: {np.std(per_frame_errors):.4f}")
                    print(f"Min per-frame error: {np.min(per_frame_errors):.4f}")
                    print(f"Max per-frame error: {np.max(per_frame_errors):.4f}")
                else:
                    score = result
                    print(f"Score: {score:.4f}")
                
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"\nError processing single video '{test_name}': {e}\n")
                import traceback
                traceback.print_exc()
            
            continue
        
        # Normal mode: Process all videos in folders
        for folder_name in target_folders:
            try:
                compute_score_folder(folder_name=folder_name, metric=metric, scene_name=scene_name, ref_scene=args.ref_scene)  # <-- add ref_scene
            except Exception as e:
                print(f"\nFailed to process folder '{folder_name}': {e}\n")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print("ALL SCENES AND PROCESSING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
