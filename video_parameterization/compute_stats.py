"""
Computes SI, TI, CF, TP, MV, DTP for all videos in ./16SSAA-vids/ and saves results to CSV.
Skips computing SI/TI/CF/TP if values already exist in the CSV.

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
from video_parameterization import spatial_information, temporal_information, colorfulness, texture_parameter, motion_vector, dynamic_texture_parameter

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

VIDEO_DIR  = Path("./16SSAA-vids")
FLOW_DIR   = Path("./flow")
OUTPUT_CSV = "video_stats.csv"
MAX_FRAMES = None

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
        frames.append(frame_rgb.transpose(2, 0, 1))  # (H,W,C) -> (C,H,W) because video param videos expect RGB first 
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    return np.stack(frames, axis=0)  # (T, C, H, W)

# ──────────────────────────────────────────────
# Load existing CSV
# ──────────────────────────────────────────────

def load_existing(csv_path):
    """Returns a dict of {filename: row} for any rows already in the CSV."""
    existing = {}
    if Path(csv_path).exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["filename"]] = row
    return existing

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

videos = sorted([p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])
if not videos:
    print(f"No video files found in {VIDEO_DIR}")
    exit()

existing = load_existing(OUTPUT_CSV)
print(f"Found {len(videos)} video(s). {len(existing)} already have existing results.\n")

fieldnames = ["filename", "SI", "TI", "CF", "TP", "MV", "DTP", "error"]
rows = []

for video_path in tqdm(videos, unit="video"):
    name = video_path.name
    # Start from existing row if available, otherwise blank
    row = existing.get(name, {"filename": name, "SI": "", "TI": "", "CF": "", "TP": "", "MV": "", "DTP": "", "error": ""})
    # Ensure new columns exist in case row came from old CSV without them
    row.setdefault("MV", "")
    row.setdefault("DTP", "")

    try:
        # ── SI, TI, CF, TP — skip if already computed ──
        if not all(row.get(k) for k in ["SI", "TI", "CF", "TP"]): 
            vid = load_video(video_path, max_frames=MAX_FRAMES)
            row["SI"] = spatial_information(vid)
            row["TI"] = temporal_information(vid)
            row["CF"] = colorfulness(vid)
            row["TP"] = texture_parameter(vid)
            row["error"] = ""
        else:
            tqdm.write(f"  Skipping SI/TI/CF/TP for {name} (already computed)")
            vid = None  # don't load unless needed below

        # ── MV, DTP — only if flow file exists ──
        flow_path = FLOW_DIR / f"{video_path.stem}_flow.npy"
        if not row.get("MV") and flow_path.exists():
            if vid is None:
                vid = load_video(video_path, max_frames=MAX_FRAMES)
            motion_vid = np.load(flow_path)
            row["MV"]  = motion_vector(motion_vid)
            row["DTP"] = dynamic_texture_parameter(vid, motion_vid)
            row["error"] = ""
        elif not flow_path.exists():
            tqdm.write(f"  No flow file for {name}, skipping MV/DTP")

    except Exception as e:
        row["error"] = str(e)
        tqdm.write(f"  [ERROR] {name}: {e}")

    rows.append(row)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

n_ok = sum(1 for r in rows if not r["error"])
print(f"\nDone. {n_ok}/{len(videos)} videos processed successfully.")
print(f"Results saved to: {OUTPUT_CSV}")