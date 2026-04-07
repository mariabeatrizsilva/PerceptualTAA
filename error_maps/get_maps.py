import os
import sys
import torch
import glob
from torchvision.io import read_video
from torchvision.models.video import resnet

# 1. Setup paths to find your src folder (assumes error_maps is inside project_root)
# Project Root -> error_maps -> gen_error_maps.py
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(os.path.join(project_root, 'src'))

from cgvqm.cgvqm import CGVQM, CGVQM_TYPE, visualize_emap, preprocess

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_SUBFOLDER = os.path.join(script_dir, 'videos')
REF_VIDEO_NAME = '16SSAA.mp4'
REF_PATH = os.path.join(VIDEO_SUBFOLDER, REF_VIDEO_NAME)

CGVQM_CONFIG = {
    'cgvqm_type': CGVQM_TYPE.CGVQM_2,
    'patch_scale': 4,
}

def load_video_to_tensor(path):
    """Loads a video file directly into a torch tensor (T, C, H, W)."""
    # Using pts_unit='sec' to ensure correct indexing
    vframes, _, info = read_video(path, pts_unit='sec', output_format='TCHW')
    return vframes

def process_video_pair(dist_path, ref_path, output_map_path):
    # Load Model
    model = resnet.r3d_18(weights=resnet.R3D_18_Weights.DEFAULT).to(DEVICE)
    model.__class__ = CGVQM
    
    # Locate weights in the src directory
    weights_path = os.path.join(project_root, 'src', 'cgvqm', 'weights', 'cgvqm-2.pickle')
    model.init_weights(weights_path, num_layers=3)
    model.eval()

    # Load Tensors
    print(f"--- Loading Reference: {os.path.basename(ref_path)}")
    R = load_video_to_tensor(ref_path)
    print(f"--- Loading Distorted: {os.path.basename(dist_path)}")
    D = load_video_to_tensor(dist_path)

    # Preprocess (Add batch dim)
    D_proc = preprocess(D).unsqueeze(0).to(DEVICE)
    R_proc = preprocess(R).unsqueeze(0).to(DEVICE)

    print(f"--- Computing CGVQM Error Map...")
    with torch.no_grad():
        score, emap = model.feature_diff(D_proc, R_proc)

    # Save visualization to the error_maps folder
    # 100 is the sensitivity threshold (standard for CGVQM)
    visualize_emap(emap.squeeze().cpu(), dist_path, 100, output_map_path)
    print(f"+++ Success! Saved to: {os.path.basename(output_map_path)}")

def main():
    if not os.path.exists(REF_PATH):
        print(f"Error: Could not find reference video at {REF_PATH}")
        return

    # Find all mp4s in the /videos/ subfolder
    search_pattern = os.path.join(VIDEO_SUBFOLDER, "*.mp4")
    video_files = [f for f in glob.glob(search_pattern) if os.path.basename(f) != REF_VIDEO_NAME]
    
    if not video_files:
        print(f"No distorted videos found in {VIDEO_SUBFOLDER}")
        return

    print(f"Found {len(video_files)} videos to process.")

    for dist_video_path in video_files:
        # Create output filename based on the video name
        video_name = os.path.splitext(os.path.basename(dist_video_path))[0]
        output_map_path = os.path.join(script_dir, f"{video_name}_errmap.mp4")
        
        print(f"\nProcessing: {video_name}")
        try:
            process_video_pair(dist_video_path, REF_PATH, output_map_path)
        except Exception as e:
            print(f"!!! Error processing {video_name}: {e}")
            continue

if __name__ == "__main__":
    main()