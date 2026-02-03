import os
import subprocess
from datetime import datetime
import sys

# --- Configuration ---
# List of scene names (folders inside data/)
SCENE_NAMES = [
    "plantshow",
    "abandoned",
    "abandoned-demo"
]

# The dictionary now just lists the top-level folders we want to process.
# We will detect automatically if they contain subfolders or direct frames.
FOLDERS_TO_PROCESS = [
    "vary_alpha_weight",
    "vary_num_samples",
    "vary_num_samples_TAA",
    "vary_hist_percent",
    "vary_filter_size",
    "16SSAA"
]

# --- Utility Functions ---
def ensure_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: FFmpeg not found. Please ensure it is in your PATH.")
        return False

def run_ffmpeg(input_pattern, output_path, display_name):
    """Executes the ffmpeg command."""
    # Check if video already exists to avoid re-running
    if os.path.exists(output_path):
        print(f"  ⏩ SKIPPING: {display_name}.mp4 already exists.")
        return
    
    ffmpeg_command = [
        'ffmpeg', '-y', '-framerate', '30',
        '-i', input_pattern,
        '-c:v', 'libx264', '-profile:v', 'high', '-crf', '10',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        print(f"  > Creating video for: {display_name}")
        subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print(f"  ✅ SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ ERROR: FFmpeg failed for {display_name}.\n{e.stderr}")

def get_input_pattern(folder_path):
    """Determines if we should use frame.png or %04d.png."""
    if os.path.exists(os.path.join(folder_path, "frame.png")):
        return os.path.join(folder_path, "frame.png")
    return os.path.join(folder_path, "%04d.png")

# --- Main Logic ---
def process_scene(scene_name):
    """Process a single scene directory."""
    base_source_dir = os.path.join("data", scene_name)
    videos_root = os.path.join("outputs", scene_name, "videos")
    
    print(f"\n{'='*60}")
    print(f"Processing Scene: {scene_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(base_source_dir):
        print(f"⚠️  WARNING: Scene directory not found: {base_source_dir}")
        return
    
    os.makedirs(videos_root, exist_ok=True)
    
    for folder_name in FOLDERS_TO_PROCESS:
        path = os.path.join(base_source_dir, folder_name)
        
        if not os.path.exists(path):
            print(f"\nPath not found, skipping: {path}")
            continue
        
        print(f"\n--- Checking: {folder_name} ---")
        
        # Special handling for 16SSAA: always treat as direct frames
        if folder_name == "16SSAA":
            # Output directly into the VIDEOS_ROOT (not in a subfolder)
            output_path = os.path.join(videos_root, "16SSAA.mp4")
            input_pattern = get_input_pattern(path)
            run_ffmpeg(input_pattern, output_path, f"{scene_name}/16SSAA")
        else:
            # Check if this folder contains frames directly
            try:
                direct_frame_exists = any(f.endswith('.png') for f in os.listdir(path))
            except PermissionError:
                print(f"  ⚠️  Permission denied: {path}")
                continue
            
            if direct_frame_exists:
                # Case 1: Direct frames (uncommon for non-16SSAA)
                output_path = os.path.join(videos_root, f"{folder_name}.mp4")
                input_pattern = get_input_pattern(path)
                run_ffmpeg(input_pattern, output_path, f"{scene_name}/{folder_name}")
            else:
                # Case 2: Nested subfolders (vary_alpha_weight style)
                # Create a specific output subfolder for this group
                sub_output_dir = os.path.join(videos_root, folder_name)
                os.makedirs(sub_output_dir, exist_ok=True)
                
                try:
                    subfolders = os.listdir(path)
                except PermissionError:
                    print(f"  ⚠️  Permission denied: {path}")
                    continue
                
                for subfolder in subfolders:
                    subfolder_path = os.path.join(path, subfolder)
                    if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
                        output_path = os.path.join(sub_output_dir, f"{subfolder}.mp4")
                        input_pattern = get_input_pattern(subfolder_path)
                        run_ffmpeg(input_pattern, output_path, f"{scene_name}/{folder_name}/{subfolder}")

def run_video_conversion():
    print(f"--- Starting Video Conversion Script ---")
    print(f"Scenes to process: {', '.join(SCENE_NAMES)}")
    
    if not ensure_ffmpeg_installed():
        sys.exit(1)
    
    for scene_name in SCENE_NAMES:
        process_scene(scene_name)
    
    print("\n" + "="*60)
    print("--- Video Conversion Script Finished ---")
    print("="*60)

if __name__ == "__main__":
    run_video_conversion()
