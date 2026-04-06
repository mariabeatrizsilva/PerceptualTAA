import os
import subprocess
import glob

# --- Configuration ---
# Target the "videos" subfolder
VIDEO_DIR = "maps-reg"
# Where to save the extracted frames
OUTPUT_BASE_DIR = "extracted_frames-reg"

def extract_every_10_frames(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a specific folder for this video's frames
    output_folder = os.path.join(OUTPUT_BASE_DIR, video_name)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"--- Processing: {video_name} ---")
    
    # FFmpeg command:
    # -i: input file
    # -vf: video filter
    # select='not(mod(n\,10))': select frame if index % 10 == 0
    # -vsync vfr: variable frame rate output (prevents dropped/duplicated frames)
    # %04d.png: naming convention (0001.png, 0002.png...)
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', "select='not(mod(n\,5))'",
        '-vsync', 'vfr',
        '-q:v', '2',  # High quality
        os.path.join(output_folder, 'frame_%04d.png')
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"✅ Successfully extracted frames to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {video_name}: {e.stderr.decode()}")

def main():
    # Find all mp4s in the video directory
    videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} videos. Starting extraction...")
    
    for v in videos:
        extract_every_10_frames(v)

if __name__ == "__main__":
    main()