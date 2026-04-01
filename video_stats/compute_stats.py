"""
Computes SI, TI, CF, TP for all videos in ./16SSAA-vids/ and saves results to CSV.
MV and DTP are skipped until optical flow / motion vectors are available.

Usage:
    python compute_video_stats.py

Dependencies:
    pip install numpy scipy dtcwt opencv-python tqdm
"""

import csv
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

from video_parameterization import spatial_information, temporal_information, colorfulness, texture_parameter

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

VIDEO_DIR = Path("./16SSAA-vids")
OUTPUT_CSV = "video_stats.csv"
MAX_FRAMES = None  # Set to an int (e.g. 120) to cap frames per video and speed things up

# ──────────────────────────────────────────────
# Video loading
# ──────────────────────────────────────────────

def load_video(path, max_frames=None):
    """Load video into array of shape (T, C, H, W), normalised to [0, 1]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb.transpose(2, 0, 1))  # (H,W,C) -> (C,H,W)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    return np.stack(frames, axis=0)  # (T, C, H, W)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

videos = sorted([p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])
if not videos:
    print(f"No video files found in {VIDEO_DIR}")
    exit()

print(f"Found {len(videos)} video(s). Computing SI, TI, CF, TP ...\n")

fieldnames = ["filename", "SI", "TI", "CF", "TP", "error"]
rows = []

for video_path in tqdm(videos, unit="video"):
    row = {"filename": video_path.name, "SI": "", "TI": "", "CF": "", "TP": "", "error": ""}
    try:
        vid = load_video(video_path, max_frames=MAX_FRAMES)
        row["SI"] = spatial_information(vid)
        row["TI"] = temporal_information(vid)
        row["CF"] = colorfulness(vid)
        row["TP"] = texture_parameter(vid)
    except Exception as e:
        row["error"] = str(e)
        tqdm.write(f"  [ERROR] {video_path.name}: {e}")
    rows.append(row)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

n_ok = sum(1 for r in rows if not r["error"])
print(f"\nDone. {n_ok}/{len(videos)} videos processed successfully.")
print(f"Results saved to: {OUTPUT_CSV}")