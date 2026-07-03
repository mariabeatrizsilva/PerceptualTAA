import os
import subprocess
import argparse

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

REF_FOLDER = "16SSAA"
COMBO_FOLDER = "full_factorial"

def convert_folder_to_video(input_dir, output_video_path, fps=30, crf=15):
    """Compiles a sequential list of PNGs inside a folder into an MP4 video."""
    # Look for the first png to establish format matching
    pngs = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    if not pngs:
        print(f"  Skipping: No PNGs found in {input_dir}")
        return False

    # Assumes sequential formatting like 0001.png, 0002.png etc. 
    # %04d matches a 4-digit zero-padded number sequence
    input_pattern = os.path.join(input_dir, "%04d.png")
    
    # FFmpeg command string
    # -y overwrites existing files
    # -c:v libx264 uses standard H.264 compression
    # -pix_fmt yuv420p ensures maximum video player compatibility
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_video_path
    ]
    
    try:
        # Run silently, checking errors if it collapses
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg Error code {e.returncode} for folder {input_dir}")
        return False
    except FileNotFoundError:
        print("  Error: 'ffmpeg' dependency not found system-wide. Please install FFmpeg.")
        return False

def process_scene_frames(scene_name):
    scene_dir = os.path.join(TESTS_DIR, scene_name)
    
    if not os.path.exists(scene_dir):
        print(f"Error: Scene directory '{scene_name}' does not exist.")
        return

    print(f"\nEncoding frames to video for scene: {scene_name}")
    print("-" * 60)

    # 1. Convert Reference Folder
    ref_src_dir = os.path.join(scene_dir, REF_FOLDER)
    ref_target_vid = os.path.join(scene_dir, f"{REF_FOLDER}.mp4")
    
    if os.path.exists(ref_src_dir) and os.path.isdir(ref_src_dir):
        print(f"Processing reference frames -> {REF_FOLDER}.mp4")
        if convert_folder_to_video(ref_src_dir, ref_target_vid):
            print("  Successfully generated reference video.")
    else:
        print(f"  Notice: No reference folder found at {ref_src_dir}")

    # 2. Convert Full Factorial Combo Folders
    combo_base_dir = os.path.join(scene_dir, COMBO_FOLDER)
    if os.path.exists(combo_base_dir):
        combos = sorted([d for d in os.listdir(combo_base_dir) if os.path.isdir(os.path.join(combo_base_dir, d))])
        print(f"Found {len(combos)} combination directories to compress inside {COMBO_FOLDER}/")
        
        for idx, combo in enumerate(combos, 1):
            input_combo_dir = os.path.join(combo_base_dir, combo)
            output_combo_vid = os.path.join(combo_base_dir, f"{combo}.mp4")
            
            print(f"[{idx}/{len(combos)}] Compressing combo: {combo}")
            convert_folder_to_video(input_combo_dir, output_combo_vid)
    else:
        print(f"  Notice: No factorial folder found at {combo_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sequence folders to mp4 videos via FFmpeg.")
    parser.add_argument("--scene", nargs="+", required=True, help="Scene folder names inside tests/")
    args = parser.parse_args()

    for scene in args.scene:
        process_scene_frames(scene)
        
    print("\nEncoding workflow completed.")