import os
import subprocess
import sys

# --- Configuration ---

# Define the base names of your videos (active scenes for screen-per mode)
BASE_SCENES = [
    # "oldmine",
    # "junkyard-mound2",
    # "fantasticvillage-open",
    "abandoned",
    # "quarry-all",
    # "scifi",
]

# All scene names from scenes.yaml (used for 16xSSAA mode)
ALL_SCENES = [
    # "fantasticvillage-open",
    "oldmine",
    # "oldmine-warm",
    # "junkyard-mound1",
    # "junkyard-mound2",
    # "subway-lookdown",
    # "subway-turn",
    # "cubetest",
    # "abandoned",
    # "abandoned-demo",
    # "abandoned-flipped",
    # "scifi",
    "quarry-all",
    "quarry-rocksonly",
    # "lightfoliage",
    # "lightfoliage-close",
    # "wildwest-bar",
    # "wildwest-barzoom",
    # "wildwest-behindcounter",
    # "wildwest-town",
    # "wildwest-store",
    "resto-close",
    "resto-pan",
    "resto-fwd",
]

# Define the screen-per values you want to sync
SCREEN_PER_VALUES = [
    "50",
    "71",
    "87"
]

# Dynamically generate the scene names: {base name}-screen-per-{value}
SCENE_NAMES_SCREEN_PER = []
for base in BASE_SCENES:
    # 1. Add the original base scene
    SCENE_NAMES_SCREEN_PER.append(base)
    # 2. Add all the screen-per variations
    for val in SCREEN_PER_VALUES:
        SCENE_NAMES_SCREEN_PER.append(f"{base}-screen-per-{val}")

# Source machine (where videos are)
SOURCE_USER = "bia"
SOURCE_HOST = "10.0.0.111"
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

def rsync_16ssaa(scene_name, dry_run=False):
    """Rsync only the 16SSAA.mp4 file for a single base scene from source machine."""
    source_path = f"{SOURCE_USER}@{SOURCE_HOST}:{SOURCE_BASE_PATH}/{scene_name}/videos/16SSAA.mp4"
    dest_dir = os.path.join(LOCAL_OUTPUTS, scene_name, "videos")
    dest_path = os.path.join(dest_dir, "16SSAA.mp4")

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pulling 16xSSAA: {scene_name}")
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
        print(f"✅ SUCCESS: {scene_name}/videos/16SSAA.mp4 synced")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to sync {scene_name}/videos/16SSAA.mp4")
        print(f"Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Transfer interrupted by user")
        return False

def main():
    """Main execution function."""
    print("--- Batch rsync Pull Script ---")
    print(f"Pulling from: {SOURCE_USER}@{SOURCE_HOST}:{SOURCE_BASE_PATH}")
    print(f"Pulling to:   {LOCAL_OUTPUTS}")

    # Ask user which mode to run
    print("\nWhat would you like to sync?")
    print("1. Screen-per variants (original behaviour, uses BASE_SCENES)")
    print("2. 16xSSAA videos only (all scenes from scenes.yaml)")
    mode_choice = input("Enter choice (1 or 2): ").strip()

    if mode_choice == "1":
        scenes = SCENE_NAMES_SCREEN_PER
        sync_fn = rsync_scene
        print(f"\nMode: Screen-per variants")
        print(f"Scenes to sync: {', '.join(scenes)}")
    elif mode_choice == "2":
        scenes = ALL_SCENES
        sync_fn = rsync_16ssaa
        print(f"\nMode: 16xSSAA only")
        print(f"Scenes to sync: {', '.join(scenes)}")
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

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
    for scene_name in scenes:
        success = sync_fn(scene_name, dry_run=dry_run)
        if success:
            successful.append(scene_name)
        else:
            failed.append(scene_name)

    # Summary
    print("\n" + "="*60)
    print("--- Summary ---")
    print(f"✅ Successful: {len(successful)}/{len(scenes)}")
    if successful:
        for scene in successful:
            print(f"   - {scene}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(scenes)}")
        for scene in failed:
            print(f"   - {scene}")

    print("="*60)

if __name__ == "__main__":
    main()