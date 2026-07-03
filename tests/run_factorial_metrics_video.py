import os
import sys
import json
import csv
import argparse
import itertools
import time
import re

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Locate compute_metrics.py one directory above tests/
# ---------------------------------------------------------------------------
TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_DIR)

# CGVQM internals
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.append(SRC_DIR)
sys.path.append(os.path.join(SRC_DIR, "cgvqm"))

# Added load_resize_vids here to let CGVQM handle video decoding
from src.cgvqm.cgvqm import CGVQM, CGVQM_TYPE, preprocess, load_resize_vids
from torchvision.models.video import resnet

# ============================================================================
# CONFIGURATION
# ============================================================================

# Assuming your reference video is named "16SSAA.mp4" and combinations are inside "full_factorial/"
REF_VIDEO_NAME = "16SSAA.mp4"   
COMBO_FOLDER   = "full_factorial"   
FPS            = 30

CGVQM_CONFIG = {
    "cgvqm_type": CGVQM_TYPE.CGVQM_2,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "patch_scale": 4,
    "patch_pool":  "mean",
    "sleep_between": 2.0,   # seconds between combos (GPU cool-down)
}

# The 4 parameters and their 3 levels
FACTORIAL_PARAMS = {
    "alpha_weight": {
        "cvar":   "r.TemporalAACurrentFrameWeight",
        "short":  "aw",
        "values": [0.04, 0.5, 1],
    },
    "num_samples": {
        "cvar":   "r.TemporalAASamples",
        "short":  "ns",
        "values": [4, 16, 64],
    },
    "filter_size": {
        "cvar":   "r.TemporalAAFilterSize",
        "short":  "fs",
        "values": [0.1, 0.5, 1.0],
    },
    "hist_percent": {
        "cvar":   "r.TemporalAA.HistoryScreenPercentage",
        "short":  "hp",
        "values": [100, 150, 200],
    },
}

# ============================================================================
# COMBO NAME HELPERS
# ============================================================================

def combo_name(values_dict):
    """{'alpha_weight': 0.04, 'num_samples': 4, ...} -> 'aw0.04_ns4_fs0.1_hp100'"""
    parts = []
    for pname, cfg in FACTORIAL_PARAMS.items():
        parts.append(f"{cfg['short']}{values_dict[pname]}")
    return "_".join(parts)


def all_combos():
    """Generate all 81 (name, values_dict) pairs."""
    param_names = list(FACTORIAL_PARAMS.keys())
    value_lists = [FACTORIAL_PARAMS[p]["values"] for p in param_names]
    for combo in itertools.product(*value_lists):
        vd = dict(zip(param_names, combo))
        yield combo_name(vd), vd


# ============================================================================
# CGVQM SCORING (VIDEOS)
# ============================================================================

_model_cache = {}   # cache so we don't reload weights for every combo

def get_model(cgvqm_type, device):
    key = (cgvqm_type, device)
    if key not in _model_cache:
        print("  Loading CGVQM model weights...")
        model = resnet.r3d_18(weights=resnet.R3D_18_Weights.DEFAULT).to(device)
        model.__class__ = CGVQM
        weights_dir = os.path.join(PROJECT_DIR, "src", "cgvqm", "weights")
        if cgvqm_type == CGVQM_TYPE.CGVQM_2:
            model.init_weights(os.path.join(weights_dir, "cgvqm-2.pickle"), num_layers=3)
        else:
            model.init_weights(os.path.join(weights_dir, "cgvqm-5.pickle"), num_layers=6)
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def score_video_pair(dist_vid_path, ref_vid_path, cfg):
    """Compute CGVQM score using native video processing."""
    device      = cfg["device"]
    patch_scale = cfg["patch_scale"]
    patch_pool  = cfg["patch_pool"]
    cgvqm_type  = cfg["cgvqm_type"]

    # CGVQM native loaders load, resize, and auto-align videos
    D, R, metadata = load_resize_vids(dist_vid_path, ref_vid_path)
    
    T, _, H, W = metadata['shape']

    model = get_model(cgvqm_type, device)

    D_t = preprocess(D).unsqueeze(0)
    R_t = preprocess(R).unsqueeze(0)

    ps        = [int(D_t.shape[3] / patch_scale), int(D_t.shape[4] / patch_scale)]
    clip_size = int(min(metadata["fps"], 30))

    def pad(x):
        return torch.nn.functional.pad(
            x,
            (0, (ps[1] - x.shape[4] % ps[1]) % ps[1],
             0, (ps[0] - x.shape[3] % ps[0]) % ps[0],
             0, (clip_size - x.shape[2] % clip_size) % clip_size),
            mode="replicate",
        )

    D_t = pad(D_t)
    R_t = pad(R_t)

    emap          = torch.zeros([R_t.shape[2], R_t.shape[3], R_t.shape[4]])
    patch_errors  = []

    for i in range(0, D_t.shape[2], clip_size):
        for h in range(0, D_t.shape[3], ps[0]):
            for w in range(0, D_t.shape[4], ps[1]):
                Cd = D_t[:, :, i:i+clip_size, h:h+ps[0], w:w+ps[1]].to(device)
                Cr = R_t[:, :, i:i+clip_size, h:h+ps[0], w:w+ps[1]].to(device)
                with torch.no_grad():
                    q, em = model.feature_diff(Cd, Cr)
                    emap[i:i+clip_size, h:h+ps[0], w:w+ps[1]] = em.squeeze()
                    patch_errors.append(q)

    emap = emap[:T, :H, :W]

    if patch_pool == "max":
        overall = 100 - max(patch_errors)
    else:
        overall = 100 - torch.mean(torch.stack(patch_errors))

    per_frame = emap.mean(dim=(1, 2)).cpu().numpy().tolist()
    return float(overall.item()), per_frame


# ============================================================================
# PER-SCENE RUNNER
# ============================================================================

def run_scene(scene_name, resume=False):
    scene_dir  = os.path.join(TESTS_DIR, scene_name)
    ref_file   = os.path.join(scene_dir, REF_VIDEO_NAME)
    combo_dir  = os.path.join(scene_dir, COMBO_FOLDER)
    json_path  = os.path.join(scene_dir, "scores.json")
    csv_path   = os.path.join(scene_dir, "scores.csv")

    print(f"\n{'='*70}")
    print(f"SCENE: {scene_name}")
    print(f"{'='*70}")

    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference video file not found: {ref_file}")
    if not os.path.exists(combo_dir):
        raise FileNotFoundError(f"Factorial folder not found: {combo_dir}")

    # Load existing results if resuming
    results = {}
    if resume and os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} combos already scored.")

    combos = list(all_combos())
    print(f"Total combos: {len(combos)}  |  Remaining: {len(combos) - len(results)}\n")

    for idx, (name, values_dict) in enumerate(combos, 1):
        if name in results:
            print(f"[{idx:3d}/{len(combos)}] SKIP (already scored): {name}")
            continue

        # Target video file inside full_factorial folder (e.g. aw0.04_ns4_fs0.1_hp100.mp4)
        dist_vid = os.path.join(combo_dir, f"{name}.mp4")
        if not os.path.exists(dist_vid):
            print(f"[{idx:3d}/{len(combos)}] MISSING video file, skipping: {dist_vid}")
            continue

        print(f"[{idx:3d}/{len(combos)}] Scoring: {name}.mp4")
        try:
            score, per_frame = score_video_pair(dist_vid, ref_file, CGVQM_CONFIG)
            results[name] = {
                "score":            score,
                "per_frame_errors": per_frame,
                "params":           values_dict,
            }
            print(f"          score = {score:.4f}  ({len(per_frame)} frames)")

            # Save JSON after every combo so we can resume if interrupted
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"          ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if CGVQM_CONFIG["sleep_between"] > 0:
            time.sleep(CGVQM_CONFIG["sleep_between"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    param_cols = list(FACTORIAL_PARAMS.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["combo_name"] + param_cols + ["cgvqm_score"])
        writer.writeheader()
        for name, data in sorted(results.items()):
            row = {"combo_name": name, "cgvqm_score": data["score"]}
            row.update(data["params"])
            writer.writerow(row)

    print(f"\nSaved {len(results)} scores -> {json_path}")
    print(f"Saved CSV                -> {csv_path}")

    scores = [d["score"] for d in results.values()]
    if scores:
        print(f"Score stats: mean={np.mean(scores):.4f}  "
              f"min={np.min(scores):.4f}  max={np.max(scores):.4f}")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute CGVQM scores for the full factorial TAA sweep."
    )
    parser.add_argument(
        "--scene", nargs="+", required=True,
        help="Scene folder name(s) inside tests/  e.g. --scene quarry-rocksonly village-day"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip combos that already have a score in scores.json"
    )
    args = parser.parse_args()

    for scene in args.scene:
        run_scene(scene, resume=args.resume)

    print("\nAll scenes complete.")


if __name__ == "__main__":
    main()