import os
import subprocess
import sys

# --- Configuration ---

# Source machine details
SOURCE_USER = "bia"
SOURCE_HOST = "10.0.0.7"
# The path to the error_maps folder on the remote machine
# Adjust this if 'error_maps' is located elsewhere in your project
REMOTE_ERROR_MAPS_PATH = "/home/bia/PTAA/error_maps"

# Local destination path (your Mac)
LOCAL_PROJECT_ROOT = "/Users/mariasilva/Documents/PerceptualTAA"
LOCAL_DEST = os.path.join(LOCAL_PROJECT_ROOT, "error_maps")

# rsync options: 
# -a (archive), -v (verbose), -h (human-readable), -P (progress)
# --include='*.mp4' --exclude='*' ensures ONLY mp4s are copied
RSYNC_OPTIONS = [
    "-avhP",
    "--include=*.mp4", 
    "--exclude=*",
    "--stats"
]

def sync_maps(dry_run=False):
    """Syncs all .mp4 files from remote error_maps to local error_maps."""
    
    # Ensure local directory exists
    os.makedirs(LOCAL_DEST, exist_ok=True)

    # Note: Trailing slash on source means "contents of the folder"
    source_path = f"{SOURCE_USER}@{SOURCE_HOST}:{REMOTE_ERROR_MAPS_PATH}/"
    dest_path = f"{LOCAL_DEST}/"

    print(f"\n{'='*60}")
    print(f"PULLING ERROR MAPS")
    print(f"From: {source_path}")
    print(f"To:   {dest_path}")
    print(f"{'='*60}")

    rsync_command = ['rsync'] + RSYNC_OPTIONS

    if dry_run:
        rsync_command.append('--dry-run')
        print("🔍 DRY RUN MODE - No files will be transferred")

    rsync_command.extend([source_path, dest_path])

    try:
        subprocess.run(rsync_command, check=True)
        print(f"\n✅ SUCCESS: Error maps synced to {LOCAL_DEST}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Sync failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"\n⚠️  Transfer cancelled by user")

def main():
    print("--- Error Map Sync Script ---")
    
    # Check if rsync is installed
    try:
        subprocess.run(['rsync', '--version'], capture_output=True)
    except FileNotFoundError:
        print("Error: rsync not found on this system.")
        sys.exit(1)

    print("\nOptions:")
    print("1. Dry run (preview filenames)")
    print("2. Actually transfer .mp4 files")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        sync_maps(dry_run=True)
    elif choice == "2":
        confirm = input("Confirm transfer? (y/n): ").lower()
        if confirm == 'y':
            sync_maps(dry_run=False)
        else:
            print("Aborted.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()