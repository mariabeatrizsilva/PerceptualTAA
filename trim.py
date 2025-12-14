import os
import glob

# --- Configuration ---
# ⚠️ Set the correct path to your 'misc_params' folder containing the video subfolders
# The path should be relative to where you run this script, or an absolute path.
BASE_FRAMES_PATH = 'data/frames/'
TARGET_FOLDER_NAME = 'misc_params'
MAX_FRAMES = 420  # The desired number of frames (0001.png through 0419.png)

# The expected file extension
FRAME_EXTENSION = '.png'
# The format for file names (e.g., '0001.png')
FRAME_NAME_FORMAT = f"%04d{FRAME_EXTENSION}"

# --- Script Logic ---

def truncate_frames_in_folder(root_dir: str, max_frames: int):
    """
    Truncates the number of PNG frames in all subfolders of the root_dir.
    It removes any frame whose index is greater than max_frames.
    """
    
    # 1. Check if the target folder exists
    if not os.path.isdir(root_dir):
        print(f"Error: Directory not found at {root_dir}")
        return

    print(f"Scanning for video subfolders in: {root_dir}")
    
    # 2. Iterate through all subdirectories (which are your video tests)
    for subfolder_name in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nProcessing subfolder: **{subfolder_name}**")
        
        # Get a list of all PNG files in the subfolder
        all_frames = sorted(glob.glob(os.path.join(subfolder_path, f"*{FRAME_EXTENSION}")))
        
        if not all_frames:
            print("  No PNG frames found. Skipping.")
            continue
            
        print(f"  Found {len(all_frames)} frames.")
        
        # Frames to delete are those starting at index (MAX_FRAMES + 1)
        frames_to_delete = all_frames[max_frames:]
        
        if not frames_to_delete:
            print("  Frame count is already <= the required 419. No files deleted.")
            continue
            
        # 3. Delete the excess frames
        count_deleted = 0
        for frame_path in frames_to_delete:
            # Optionally check the file name to ensure it follows the expected index
            try:
                os.remove(frame_path)
                count_deleted += 1
            except OSError as e:
                print(f"  Error deleting file {frame_path}: {e}")

        print(f"  Successfully **deleted {count_deleted} frames** to keep the first {max_frames}.")
        
        # Optional: verify the count
        remaining_frames = len(glob.glob(os.path.join(subfolder_path, f"*{FRAME_EXTENSION}")))
        print(f"  Remaining frames: {remaining_frames}")


if __name__ == '__main__':
    target_path = os.path.join(BASE_FRAMES_PATH, TARGET_FOLDER_NAME)
    
    # ⚠️ IMPORTANT: If your script running the metric is in the project root, 
    # and the frames are in 'data/frames/misc_params', the path below is correct.
    # Otherwise, adjust the 'target_path' variable before running!
    
    # Example adjustment if you are running this from a different folder:
    # target_path = '/path/to/your/project/data/frames/misc_params'
    
    print("="*50)
    print(f"Truncating frames to {MAX_FRAMES}...")
    print("="*50)
    
    truncate_frames_in_folder(target_path, MAX_FRAMES)
    
    print("\nProcessing complete.")