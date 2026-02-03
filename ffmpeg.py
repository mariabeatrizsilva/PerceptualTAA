import os
import subprocess
from datetime import datetime
import sys

# --- Configuration ---

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

# --- Utility Functions ---

def ensure_ffmpeg_installed():
    """Checks if the ffmpeg command is available in the system's PATH."""
    try:
        # Run a simple, non-intrusive command
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("\nERROR: FFmpeg command failed to run. Check your installation.")
        return False
    except FileNotFoundError:
        print("\nERROR: FFmpeg command not found.")
        print("Please ensure ffmpeg is installed and available in your system's PATH.")
        return False

def run_ffmpeg(input_pattern, output_path, subfolder_name):
    """Executes the ffmpeg command."""
    
    # The base ffmpeg command structure
    ffmpeg_command = [
        'ffmpeg',
        '-y', # Overwrite output files without asking
        '-framerate', '30',
        '-i', input_pattern, # Input file pattern (e.g., /path/to/frames/%04d.png)
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '10', # Lower CRF means higher quality, 10 is very high quality
        '-pix_fmt', 'yuv420p', # Recommended for compatibility (e.g., YouTube, players)
        output_path # Output path
    ]
    
    print(f"  Input Pattern: {input_pattern}")
    print(f"  Output File: {output_path}")

    try:
        # Execute the command
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            check=True # Raise an exception for non-zero return codes
        )
        print(f"  ✅ SUCCESS: Video created for {subfolder_name}.")
        # Optional: Uncomment the line below to see verbose ffmpeg output
        # print(f"FFmpeg Output:\n{result.stderr}") 

    except subprocess.CalledProcessError as e:
        print(f"  ❌ ERROR: FFmpeg failed for folder {subfolder_name}.")
        print(f"  Return Code: {e.returncode}")
        print(f"  STDERR: {e.stderr}")
    
    print("-" * 60)


# --- Main Logic ---

def run_video_conversion():
    """
    Iterates through the four main folders, then their subfolders, and runs 
    the ffmpeg command on the frames in each subfolder.
    """
    
    print(f"--- Starting Video Conversion Script ---")
    print(f"Source Directory Root: {BASE_SOURCE_DIR}")
    print(f"Video Output Root: {VIDEOS_ROOT}")
    print("-" * 60)
    
    if not ensure_ffmpeg_installed():
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

if __name__ == "__main__":
    run_video_conversion()