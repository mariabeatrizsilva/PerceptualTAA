import os
import subprocess
from datetime import datetime
import sys
import glob

# --- Configuration ---
# List of scene names (folders inside data/)
# SCENE_NAMES = [
    # "abandoned",
    # "abandoned-demo",
    # "abandoned-flipped",
    # "cubetest", "fantasticvillage-open", "lightfoliage", "lightfoliage-close", "oldmine", "oldmine-close", 
    # "oldmine-warm", "quarry-all", "quarry-rocksonly", "resto-close", "resto-fwd", "resto-pan", "scifi", "subway-lookdown", 
    # "subway-turn", "wildwest-bar", "wildwest-barzoom", "wildwest-behindcounter", "wildwest-store", "wildwest-town"
    # "oldmine-speed-9", "oldmine-speed-18", "oldmine-speed-35", "oldmine-speed-75",
#     "oldmine-screen-per-25", "oldmine-screen-per-50", "oldmine-screen-per-75", "village-screen-per-25", "village-screen-per-50", "village-screen-per-75"
# ]

BASE_SCENE_NAMES = [
    "abandoned"
    # "village"
]

# The dictionary now just lists the top-level folders we want to process.
# We will detect automatically if they contain subfolders or direct frames.
FOLDERS_TO_PROCESS = [
    "vary_alpha_weight",
    # "vary_num_samples",
    # "vary_num_samples_TAA",
    "vary_hist_percent",
    # "vary_filter_size",
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

def get_all_variant_scenes(base_names):
    """
    Finds all folders in data/ that match the base name or base-screen-per-*.
    """
    all_scenes = set()
    available_folders = os.listdir("data")
    
    for base in base_names:
        # Match exact base name
        if base in available_folders:
            all_scenes.add(base)
        
        # Match patterns like 'village-screen-per-25'
        pattern = f"{base}-screen-per-"
        for folder in available_folders:
            if folder.startswith(pattern):
                all_scenes.add(folder)
                
    return sorted(list(all_scenes))

def run_ffmpeg(input_pattern, output_path, display_name):
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
    if os.path.exists(os.path.join(folder_path, "frame.png")):
        return os.path.join(folder_path, "frame.png")
    return os.path.join(folder_path, "%04d.png")

# --- Main Logic ---
def process_scene(scene_name):
    base_source_dir = os.path.join("data", scene_name)
    videos_root = os.path.join("outputs", scene_name, "videos")
    
    print(f"\n{'='*60}")
    print(f"Processing Scene Folder: {scene_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(base_source_dir):
        return
    
    os.makedirs(videos_root, exist_ok=True)
    
    for folder_name in FOLDERS_TO_PROCESS:
        path = os.path.join(base_source_dir, folder_name)
        if not os.path.exists(path):
            continue
        
        if folder_name == "16SSAA":
            output_path = os.path.join(videos_root, "16SSAA.mp4")
            input_pattern = get_input_pattern(path)
            run_ffmpeg(input_pattern, output_path, f"{scene_name}/16SSAA")
        else:
            try:
                direct_frame_exists = any(f.endswith('.png') for f in os.listdir(path))
            except PermissionError:
                continue
            
            if direct_frame_exists:
                output_path = os.path.join(videos_root, f"{folder_name}.mp4")
                input_pattern = get_input_pattern(path)
                run_ffmpeg(input_pattern, output_path, f"{scene_name}/{folder_name}")
            else:
                sub_output_dir = os.path.join(videos_root, folder_name)
                os.makedirs(sub_output_dir, exist_ok=True)
                
                for subfolder in os.listdir(path):
                    subfolder_path = os.path.join(path, subfolder)
                    if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
                        output_path = os.path.join(sub_output_dir, f"{subfolder}.mp4")
                        input_pattern = get_input_pattern(subfolder_path)
                        run_ffmpeg(input_pattern, output_path, f"{scene_name}/{folder_name}/{subfolder}")

def run_video_conversion():
    print(f"--- Starting Video Conversion Script ---")
    
    if not ensure_ffmpeg_installed():
        sys.exit(1)

    # Automatically expand base names to include all screen-per variations
    scenes_to_process = get_all_variant_scenes(BASE_SCENE_NAMES)
    
    print(f"Found folders: {', '.join(scenes_to_process)}")
                
    for scene_name in scenes_to_process:
        process_scene(scene_name)
    
    print("\n" + "="*60)
    print("--- Video Conversion Script Finished ---")
    print("="*60)

if __name__ == "__main__":
    run_video_conversion()