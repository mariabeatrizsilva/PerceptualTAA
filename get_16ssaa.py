import os
import subprocess
import sys

# --- Configuration ---
BASE_SOURCE_DIR = "./data"
VIDEOS_ROOT = "./output_videos"

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

def get_input_pattern(path):
    """
    Standardizes the input pattern for ffmpeg. 
    Adjust %04d if your frames use a different padding length.
    """
    return os.path.join(path, "%04d.png")

def run_video_conversion():
    print(f"--- Starting Video Conversion Script ---")
    
    if not os.path.exists(BASE_SOURCE_DIR):
        print(f"❌ Error: Source directory {BASE_SOURCE_DIR} not found.")
        return
        
    os.makedirs(VIDEOS_ROOT, exist_ok=True)

    # 1. Iterate through every folder inside /data
    for project_folder in os.listdir(BASE_SOURCE_DIR):
        project_path = os.path.join(BASE_SOURCE_DIR, project_folder)
        
        if not os.path.isdir(project_path) or project_folder.startswith('.'):
            continue

        # 2. Target the specific '16SSAA' folder within the project
        target_path = os.path.join(project_path, "16SSAA")
        
        if not os.path.exists(target_path):
            continue

        print(f"\n--- Processing: {project_folder}/16SSAA ---")

        # 3. Logic: Check if 16SSAA contains frames directly or subfolders
        files_in_target = os.listdir(target_path)
        direct_frame_exists = any(f.endswith('.png') for f in files_in_target)
        
        if direct_frame_exists:
            # Case 1: Frames are directly in 16SSAA
            # Naming the video after the parent project folder
            output_path = os.path.join(VIDEOS_ROOT, f"{project_folder}_16SSAA.mp4")
            input_pattern = get_input_pattern(target_path)
            run_ffmpeg(input_pattern, output_path, f"{project_folder}/16SSAA")
        else:
            # Case 2: 16SSAA contains nested subfolders of frames
            sub_output_dir = os.path.join(VIDEOS_ROOT, project_folder)
            os.makedirs(sub_output_dir, exist_ok=True)

            for subfolder in files_in_target:
                subfolder_path = os.path.join(target_path, subfolder)
                
                if os.path.isdir(subfolder_path) and not subfolder.startswith('.'):
                    output_path = os.path.join(sub_output_dir, f"{subfolder}.mp4")
                    input_pattern = get_input_pattern(subfolder_path)
                    run_ffmpeg(input_pattern, output_path, f"{project_folder}/16SSAA/{subfolder}")

    print("\n--- Video Conversion Script Finished ---")

if __name__ == "__main__":
    run_video_conversion()