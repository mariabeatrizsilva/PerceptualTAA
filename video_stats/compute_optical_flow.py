"""
compute_optical_flow.py
Computes dense optical flow (Farneback) for each video, averages into 8x8 blocks,
and saves the result as a .npy file of shape (T, 2, H, W) — ready for use with
motion_vector() and dynamic_texture_parameter() in video_parameterization.py.

Usage:
    python compute_optical_flow.py

Outputs:
    ./flow/  <video_stem>_flow.npy   e.g. abandoned_flow.npy

Dependencies:
    pip install numpy opencv-python tqdm
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

VIDEO_DIR  = Path("./16SSAA-vids")
OUTPUT_DIR = Path("./flow")
BLOCK_SIZE = 8        # 8x8 blocks to match paper spec
MAX_FRAMES = None     # Set to int to cap frames (must match what you used in compute_video_stats.py)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
# ──────────────────────────────────────────────
# Per-video flow computation
# ──────────────────────────────────────────────
def compute_flow_for_video(video_path, max_frames=None):
    """
    Returns motion_vid of shape (T, 2, H, W), where T matches
    the number of frames in the corresponding vid array.

    Flow at frame 0 is zeros (no previous frame).
    Flow at frame t is computed between frame t-1 and frame t.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    frames_gray = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()

    if not frames_gray:
        raise RuntimeError(f"No frames read from: {video_path}")

    T = len(frames_gray)
    H, W = frames_gray[0].shape
    motion_vid = np.zeros((T, 2, H, W), dtype=np.float32) # one entry per frame, 2 components (dx,dy), pixel spatial grid

    for fid in tqdm(range(1, T), desc=video_path.stem, leave=False, unit="frame"):
        flow = cv2.calcOpticalFlowFarneback(frames_gray[fid-1], frames_gray[fid], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_vid[fid, 0, :, :] = flow[:, :, 0]  # dx
        motion_vid[fid, 1, :, :] = flow[:, :, 1]  # dy

    return motion_vid



# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    videos = sorted([p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])

    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} video(s). Computing optical flow with {BLOCK_SIZE}x{BLOCK_SIZE} block averaging...\n")

    for video_path in tqdm(videos, unit="video"):
        out_path = OUTPUT_DIR / f"{video_path.stem}_flow.npy"
        if out_path.exists():
            tqdm.write(f"  Skipping {video_path.name} (flow file already exists)")
            continue
        try:
            motion_vid = compute_flow_for_video(video_path, max_frames=MAX_FRAMES)
            np.save(out_path, motion_vid)
            tqdm.write(f"  Saved: {out_path}  shape={motion_vid.shape}")
        except Exception as e:
            tqdm.write(f"  [ERROR] {video_path.name}: {e}")

    print(f"\nDone. Flow files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()