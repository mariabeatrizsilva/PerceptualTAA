"""
run_factorial_metrics.py
------------------------
Computes CGVQM scores for all 81 factorial combinations against the 16SSAA reference.
Place this script inside the tests/ folder, one level above scene folders.

EXPECTED STRUCTURE (before running):
  tests/
    run_factorial_metrics.py   <- this script
    {scene_name}/
      16SSAA/
        0001.png ...
      full_factorial/
        aw0.04_ns4_fs0.1_hp100/
          0001.png ...
        ...

  compute_metrics.py           <- your existing script, one level above tests/

OUTPUT (written into tests/{scene_name}/):
  scores.json    — raw scores + per-frame errors keyed by combo name
  scores.csv     — one row per combo with all 4 parameter values + score (for analysis)

USAGE:
  cd tests/
  python run_factorial_metrics.py --scene quarry-rocksonly
  python run_factorial_metrics.py --scene quarry-rocksonly village-day  # multiple scenes
  python run_factorial_metrics.py --scene quarry-rocksonly --resume      # skip already-scored combos
"""

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
from PIL import Image
from torch.nn.functional import interpolate

# ---------------------------------------------------------------------------
# Locate compute_metrics.py one directory above tests/
# ---------------------------------------------------------------------------
TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_DIR)

# CGVQM internals (mirrored from compute_metrics.py so we stay self-contained
# for the parts we actually need, but import the heavy model code from upstream)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.append(SRC_DIR)
sys.path.append(os.path.join(SRC_DIR, "cgvqm"))

from src.cgvqm.cgvqm import CGVQM, CGVQM_TYPE, preprocess
from torchvision.models.video import resnet

# ============================================================================
# CONFIGURATION
# ============================================================================

REF_FOLDER   = "16SSAA"          # subfolder name for the reference render
COMBO_FOLDER = "full_factorial"   # subfolder containing all 81 combos
FPS          = 30

CGVQM_CONFIG = {
    "cgvqm_type": CGVQM_TYPE.CGVQM_2,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "patch_scale": 4,
    "patch_pool":  "mean",
    "sleep_between": 2.0,   # seconds between combos (GPU cool-down)
}

# The 4 parameters and their 3 levels — must match generate_factorial_jobs.py
FACTORIAL_PARAMS = {
    "alpha_weight": {
        "cvar":   "r.TemporalAACurrentFrameWeight",
        "short":  "aw",
        "values": [0.04, 0.1, 0.5],
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


def parse_combo_name(name):
    """'aw0.04_ns4_fs0.1_hp100' -> {'alpha_weight': 0.04, 'num_samples': 4, ...}"""
    pattern = re.compile(
        r"aw(?P<aw>[\d.]+)_ns(?P<ns>[\d.]+)_fs(?P<fs>[\d.]+)_hp(?P<hp>[\d.]+)"
    )
    m = pattern.match(name)
    if not m:
        raise ValueError(f"Cannot parse combo name: {name}")
    return {
        "alpha_weight": float(m.group("aw")),
        "num_samples":  int(float(m.group("ns"))),
        "filter_size":  float(m.group("fs")),
        "hist_percent": int(float(m.group("hp"))),
    }


def all_combos():
    """Generate all 81 (name, values_dict) pairs."""
    param_names = list(FACTORIAL_PARAMS.keys())
    value_lists = [FACTORIAL_PARAMS[p]["values"] for p in param_names]
    for combo in itertools.product(*value_lists):
        vd = dict(zip(param_names, combo))
        yield combo_name(vd), vd


# ============================================================================
# FRAME LOADING
# ============================================================================

def load_frames(folder):
    """Load all PNGs from a folder -> tensor (T, C, H, W) uint8."""
    pngs = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".png")
    )
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in: {folder}")
    frames = []
    for fname in pngs:
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        arr = np.array(img)                          # (H, W, 3)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (C, H, W)
    return torch.stack(frames, dim=0)               # (T, C, H, W)


def align_tensors(D, R):
    """Resize D spatially and temporally to match R if needed."""
    if D.shape[2:] != R.shape[2:]:
        D = interpolate(D.float().unsqueeze(0),
                        size=(R.shape[2], R.shape[3]),
                        mode="bilinear").squeeze(0).to(torch.uint8)
    if D.shape[0] != R.shape[0]:
        D = (D.permute(1, 0, 2, 3).unsqueeze(0).float())
        D = interpolate(D, size=(R.shape[0], R.shape[2], R.shape[3]),
                        mode="nearest").squeeze(0).permute(1, 0, 2, 3).to(torch.uint8)
    return D


# ============================================================================
# CGVQM SCORING
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


def score_pair(dist_folder, ref_folder, cfg):
    """Compute CGVQM score for one distorted/reference folder pair."""
    device      = cfg["device"]
    patch_scale = cfg["patch_scale"]
    patch_pool  = cfg["patch_pool"]
    cgvqm_type  = cfg["cgvqm_type"]

    D = load_frames(dist_folder)
    R = load_frames(ref_folder)
    D = align_tensors(D, R)

    T, C, H, W = R.shape
    metadata = {"shape": (T, C, H, W), "fps": FPS}

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
    ref_dir    = os.path.join(scene_dir, REF_FOLDER)
    combo_dir  = os.path.join(scene_dir, COMBO_FOLDER)
    json_path  = os.path.join(scene_dir, "scores.json")
    csv_path   = os.path.join(scene_dir, "scores.csv")

    print(f"\n{'='*70}")
    print(f"SCENE: {scene_name}")
    print(f"{'='*70}")

    if not os.path.exists(ref_dir):
        raise FileNotFoundError(f"Reference folder not found: {ref_dir}")
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

        dist_dir = os.path.join(combo_dir, name)
        if not os.path.exists(dist_dir):
            print(f"[{idx:3d}/{len(combos)}] MISSING folder, skipping: {name}")
            continue

        print(f"[{idx:3d}/{len(combos)}] Scoring: {name}")
        try:
            score, per_frame = score_pair(dist_dir, ref_dir, CGVQM_CONFIG)
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
