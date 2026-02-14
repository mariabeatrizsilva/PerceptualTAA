import os
import subprocess
from datetime import datetime
import sys

# --- Configuration ---
# List of scene names (folders inside data/)
SCENE_NAMES = [
    "abandoned",
    "abandoned-demo",
    "abandoned-flipped",
    "cubetest", "fantasticvillage-open", "lightfoliage", "lightfoliage-close", "oldmine", "oldmine-close", 
    "oldmine-warm", "quarry-all", "quarry-rocksonly", "resto-close", "resto-fwd", "resto-pan", "scifi", "subway-lookdown", 
    "subway-turn", "wildwest-bar", "wildwest-barzoom", "wildwest-behindcounter", "wildwest-store", "wildwest-town"
]

<<<<<<< HEAD
# **IMPORTANT**: Update these paths to match your system.
BASE_SOURCE_DIR = "data/lightfoliage-close"

# The root directory where the final MP4 videos will be saved.
VIDEOS_ROOT = "outputs/lightfoliage-close/videos"

# Define the structure of the main folders and their corresponding output directories
# This dictionary makes the logic clean and easy to scale.
FOLDER_CONFIG = {
    # Main Folder Name: (Output Directory Path, Input Frame Extension)
    "vary_alpha_weight": (os.path.join(VIDEOS_ROOT, "vary_alpha_weight"), 'png'),
    "vary_num_samples": (os.path.join(VIDEOS_ROOT, "vary_num_samples"), 'png'),
    "vary_hist_percent": (os.path.join(VIDEOS_ROOT, "vary_hist_percent"), 'png'),
    "vary_filter_size": (os.path.join(VIDEOS_ROOT, "vary_filter_size"), 'png')
}
=======
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
>>>>>>> 3ac9b112b269e851538f714c4daa764a7a2da0bc

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
<<<<<<< HEAD
        sys.exit(1) # Exit if ffmpeg isn't ready
        
    try:
        # 1. Loop through the main folders defined in FOLDER_CONFIG
        for main_folder_name, (output_dir, frame_ext) in FOLDER_CONFIG.items():
            
            main_folder_path = os.path.join(BASE_SOURCE_DIR, main_folder_name)
            
            print(f"\n--- Processing Main Folder: {main_folder_name} ---")
            print(f"Videos will be saved to: {output_dir}")
            
            # Ensure the output directory for this group exists
            os.makedirs(output_dir, exist_ok=True)
            
            if not os.path.exists(main_folder_path):
                 print(f"  WARNING: Main folder not found: {main_folder_path}. Skipping.")
                 continue

            # 2. Loop through the subfolders inside the main folder
            for subfolder_name in os.listdir(main_folder_path):
                subfolder_path = os.path.join(main_folder_path, subfolder_name)
                
                # Check if it's a valid directory
                if os.path.isdir(subfolder_path) and not subfolder_name.startswith('.'):
                    
                    print(f"\n> Processing Subfolder: {subfolder_name}")

                    # A. Define the input pattern inside the current subfolder
                    # Assumes files are zero-padded, e.g., 0001.png, 0002.png
                    input_pattern = os.path.join(subfolder_path, f'%04d.{frame_ext}')
                    
                    # B. Define the output filename: subfolder_name.mp4
                    output_filename = f"{subfolder_name}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # C. Run FFmpeg
                    run_ffmpeg(input_pattern, output_path, subfolder_name)
        main_folder_path = os.path.join(BASE_SOURCE_DIR, "16SSAA")
        output_filename = "16SSAA.mp4"
        input_pattern = os.path.join(main_folder_path, f'%04d.{frame_ext}')
        output_path = os.path.join(VIDEOS_ROOT, output_filename)
        run_ffmpeg(input_pattern, output_path, output_filename)
        # main_folder_path = os.path.join(BASE_SOURCE_DIR, "16SSAA-1")
        # output_filename = "16SSAA-1.mp4"
        # input_pattern = os.path.join(main_folder_path, f'%04d.{frame_ext}')
        # output_path = os.path.join(VIDEOS_ROOT, output_filename)
        # run_ffmpeg(input_pattern, output_path, output_filename)
                
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
    finally:
        print("\n--- Video Conversion Script Finished ---")
=======
        sys.exit(1)
    
    for scene_name in SCENE_NAMES:
        process_scene(scene_name)
    
    print("\n" + "="*60)
    print("--- Video Conversion Script Finished ---")
    print("="*60)
>>>>>>> 3ac9b112b269e851538f714c4daa764a7a2da0bc

if __name__ == "__main__":
    run_video_conversion()
