import os
import subprocess
import sys

# --- Configuration ---
SCENE_NAMES = [
    # "abandoned",
#    "abandoned-demo",
#    "abandoned-flipped",
#    "cubetest", "fantasticvillage-open", "lightfoliage", "lightfoliage-close", "oldmine", "oldmine-close", 
#    "oldmine-warm", "quarry-all", "quarry-rocksonly", "resto-close", "resto-fwd", "resto-pan", "scifi", "subway-lookdown", 
#    "subway-turn", "wildwest-bar", "wildwest-barzoom", "wildwest-behindcounter", "wildwest-store", "wildwest-town",
    "oldmine-speed-9","oldmine-speed-18", "oldmine-speed-35", "oldmine-speed-75"
    ]

# Source machine (where videos are)
SOURCE_USER = "bia"
SOURCE_HOST = "10.0.0.100"
SOURCE_BASE_PATH = "/home/bia/PTAA/outputs"

# Local destination path (your current machine)
LOCAL_OUTPUTS = "/Users/mariasilva/Documents/PerceptualTAA/outputs"

# rsync options
RSYNC_OPTIONS = [
    "-avhP",  # archive, verbose, human-readable, progress
    "--stats"  # show transfer statistics
]

# --- Functions ---
def check_rsync_installed():
    """Check if rsync is available."""
    try:
        subprocess.run(['rsync', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: rsync not found. Please install rsync first.")
        return False

def rsync_scene(scene_name, dry_run=False):
    """Rsync only the videos folder for a single scene from source machine."""
    # Only sync the videos/ subdirectory
    source_path = f"{SOURCE_USER}@{SOURCE_HOST}:{SOURCE_BASE_PATH}/{scene_name}/videos/"
    dest_path = os.path.join(LOCAL_OUTPUTS, scene_name, "videos") + "/"  # Trailing slash important!
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_path.rstrip("/"), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Pulling videos: {scene_name}")
    print(f"From: {source_path}")
    print(f"To: {dest_path}")
    print(f"{'='*60}")
    
    # Build rsync command
    rsync_command = ['rsync'] + RSYNC_OPTIONS
    
    if dry_run:
        rsync_command.append('--dry-run')
        print("🔍 DRY RUN MODE - No files will be transferred")
    
    rsync_command.extend([source_path, dest_path])
    
    try:
        result = subprocess.run(rsync_command, check=True)
        print(f"✅ SUCCESS: {scene_name}/videos synced")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to sync {scene_name}/videos")
        print(f"Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Transfer interrupted by user")
        return False

def main():
    """Main execution function."""
    print("--- Starting Batch rsync Pull Script (videos only) ---")
    print(f"Pulling from: {SOURCE_USER}@{SOURCE_HOST}:{SOURCE_BASE_PATH}")
    print(f"Pulling to: {LOCAL_OUTPUTS}")
    print(f"Scenes to sync: {', '.join(SCENE_NAMES)}")
    
    # Check for rsync
    if not check_rsync_installed():
        sys.exit(1)
    
    # Ensure local outputs directory exists
    os.makedirs(LOCAL_OUTPUTS, exist_ok=True)
    
    # Ask user for dry run
    print("\nOptions:")
    print("1. Dry run (preview what will be transferred)")
    print("2. Actually transfer files")
    choice = input("Enter choice (1 or 2): ").strip()
    
    dry_run = choice == "1"
    
    if not dry_run:
        confirm = input("\n⚠️  This will transfer files. Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
    
    # Track results
    successful = []
    failed = []
    
    # Process each scene
    for scene_name in SCENE_NAMES:
        success = rsync_scene(scene_name, dry_run=dry_run)
        if success:
            successful.append(scene_name)
        else:
            failed.append(scene_name)
    
    # Summary
    print("\n" + "="*60)
    print("--- Summary ---")
    print(f"✅ Successful: {len(successful)}/{len(SCENE_NAMES)}")
    if successful:
        for scene in successful:
            print(f"   - {scene}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(SCENE_NAMES)}")
        for scene in failed:
            print(f"   - {scene}")
    
    print("="*60)

if __name__ == "__main__":
    main()
