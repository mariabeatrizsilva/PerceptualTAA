"""
score_taa.py  (dataset v2)
--------------------------
Scores every render in a v2 scene (from generate_taa_jobs.py + encode_and_verify.py)
against its reference with CGVQM and, optionally, ColorVideoVDP.

CGVQM settings are IDENTICAL to compute_metrics.py used for the paper
(CGVQM-2, patch_scale 4, mean patch pooling, 30-frame clips), so scores are
comparable with the published numbers.

What's different from compute_metrics.py (on purpose):
  - STRICT: a render whose frame count or resolution doesn't match the
    reference is NOT scored. (The old code silently stretched it in time or
    resized it to fit -- broken renders got plausible-looking scores.)
  - Flags any CGVQM score outside [0, 100] as "suspect" instead of dropping it.
  - Model weights and the reference are loaded once per scene, not per video.
  - Reads lossless videos (falls back to PNG folders if a video doesn't exist).
  - Reads params from manifest.json -- no parsing of folder names.
  - Saves a coarse spatial error map per render (T x 9 x 16 grid) for the
    content-dependence analysis (error vs. local motion/detail).

OUTPUT
  {root}/{scene}/scores/cgvqm.json          per render: score, per-frame errors, params, flags
  {root}/{scene}/scores/cgvqm_emaps/*.npz   per render:
        patch_quality (clips, 4, 4): CGVQM quality of each patch x 30-frame clip
                                     (the units the overall score averages) -> content analysis
            patch_quality[c, r, w] covers frames c*clip_frames .. +clip_frames-1,
            pixel rows r*patch_h .. +patch_h-1, pixel cols w*patch_w .. +patch_w-1
            (patch_h, patch_w, clip_frames are stored in the same file; at 1080p: 270, 480, 30)
        grid (T, 9, 16): coarse per-frame error map, float16
  {root}/{scene}/scores/cvvdp.json          (if --metrics cvvdp)
  {root}/scores_all.csv                     one tidy row per (scene, render, reference, metric)

Every render is scored against the 100 % reference. Legacy-check renders are
ALSO scored against the legacy 87 % reference, to compare with the old
factorial scores.json.

USAGE  (run from the repo root, so src/cgvqm is found)
  python score_taa.py --root ~/Documents/PerceptualTAA/data_v2 --scene oldmine
  python score_taa.py --root ... --all --metrics cgvqm cvvdp
  python score_taa.py --root ... --scene oldmine --old-scores oldmine=tests/oldmine/scores.json
  # re-running resumes: already-scored renders are skipped unless --overwrite
"""

import os
import sys
import csv
import json
import time
import glob
import argparse
import subprocess

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG -- keep CGVQM identical to the paper (compute_metrics.py)
# ----------------------------------------------------------------------------
CGVQM_CONFIG = {
    "cgvqm_type": "CGVQM_2",
    "patch_scale": 4,
    "patch_pool": "mean",
}
EMAP_GRID = (9, 16)            # coarse spatial error map saved per frame (rows, cols)
CVVDP_DISPLAY = "standard_4k"  # what the paper's CVVDP run used; see note in main()
MAIN_REF = "ssaa16_sp100"
LEGACY_REF = "ssaa16_sp087"


# ============================================================================
# LOADING
# ============================================================================

def load_video_rgb(path):
    """Decode a video to (T, H, W, 3) uint8 via ffmpeg."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path],
                           check=True, capture_output=True, text=True).stdout.strip()
    w, h = map(int, probe.split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def load_pngs_rgb(folder):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"no PNGs in {folder}")
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def load_job(scene_dir, job):
    """Returns (T, H, W, 3) uint8, from the video if present, else the PNG folder."""
    if job.get("video"):
        v = os.path.join(scene_dir, job["video"])
        if os.path.exists(v):
            return load_video_rgb(v)
    d = os.path.join(scene_dir, job["png_dir"])
    if os.path.isdir(d):
        return load_pngs_rgb(d)
    raise FileNotFoundError(f"neither video nor PNGs found for {job['kind']}/{job['config']}")


# ============================================================================
# CGVQM  (same math as run_cgvqm_with_tensors in compute_metrics.py)
# ============================================================================

def find_project_dir(explicit=None):
    """Repo root containing src/cgvqm: the given path, else this script's folder or any parent
    (so the script works from the repo root or from tests/), else the current directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [explicit] if explicit else []
    d = here
    while True:
        candidates.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    candidates.append(os.getcwd())
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "src", "cgvqm")):
            return c
    raise SystemExit("Could not find src/cgvqm. Pass --project-dir <path to the PerceptualTAA repo root>.")


class CGVQMScorer:
    def __init__(self, project_dir, device):
        import torch
        sys.path.append(os.path.join(project_dir, "src"))
        sys.path.append(os.path.join(project_dir, "src", "cgvqm"))
        from cgvqm.cgvqm import CGVQM, CGVQM_TYPE, preprocess
        from torchvision.models.video import resnet

        self.torch, self.preprocess = torch, preprocess
        self.device = device
        ctype = getattr(CGVQM_TYPE, CGVQM_CONFIG["cgvqm_type"])
        model = resnet.r3d_18(weights=resnet.R3D_18_Weights.DEFAULT).to(device)
        model.__class__ = CGVQM
        wdir = os.path.join(project_dir, "src", "cgvqm", "weights")
        if ctype == CGVQM_TYPE.CGVQM_2:
            model.init_weights(os.path.join(wdir, "cgvqm-2.pickle"), num_layers=3)
        else:
            model.init_weights(os.path.join(wdir, "cgvqm-5.pickle"), num_layers=6)
        model.eval()
        self.model = model
        self._ref_cache = {}

    def _prep(self, frames_thwc, fps):
        """uint8 (T,H,W,3) -> padded (1,C,T,H,W) tensor, plus patch geometry."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(frames_thwc)).permute(0, 3, 1, 2)  # T,C,H,W
        x = self.preprocess(x).unsqueeze(0)
        ps = [int(x.shape[3] / CGVQM_CONFIG["patch_scale"]), int(x.shape[4] / CGVQM_CONFIG["patch_scale"])]
        clip = int(min(fps, 30))
        x = torch.nn.functional.pad(
            x,
            (0, (ps[1] - x.shape[4] % ps[1]) % ps[1],
             0, (ps[0] - x.shape[3] % ps[0]) % ps[0],
             0, (clip - x.shape[2] % clip) % clip),
            mode="replicate")
        return x, ps, clip

    def score(self, dist, ref, ref_key, fps):
        torch = self.torch
        T, H, W, _ = ref.shape
        if ref_key not in self._ref_cache:
            self._ref_cache = {ref_key: self._prep(ref, fps)}   # keep only the current reference
        R, ps, clip = self._ref_cache[ref_key]
        D, _, _ = self._prep(dist, fps)

        emap = torch.zeros([R.shape[2], R.shape[3], R.shape[4]])
        errs = []
        with torch.no_grad():
            for i in range(0, D.shape[2], clip):
                for h in range(0, D.shape[3], ps[0]):
                    for w in range(0, D.shape[4], ps[1]):
                        Cd = D[:, :, i:i + clip, h:h + ps[0], w:w + ps[1]].to(self.device)
                        Cr = R[:, :, i:i + clip, h:h + ps[0], w:w + ps[1]].to(self.device)
                        q, em = self.model.feature_diff(Cd, Cr)
                        emap[i:i + clip, h:h + ps[0], w:w + ps[1]] = em.squeeze().cpu()
                        errs.append(q)
        emap = emap[:T, :H, :W]
        # Per-patch quality, shape (clips, rows, cols) = e.g. (5, 4, 4) for 150 frames at 1080p.
        # Loop order above is clip -> row -> col, so a plain reshape is correct.
        n_c = D.shape[2] // clip
        n_r = D.shape[3] // ps[0]
        n_w = D.shape[4] // ps[1]
        patch_q = (100 - torch.stack([e.reshape(()) for e in errs]).float().cpu()).reshape(n_c, n_r, n_w).numpy()
        if CGVQM_CONFIG["patch_pool"] == "max":
            q = 100 - max(errs)
        else:
            q = 100 - torch.mean(torch.stack(errs))
        per_frame = emap.mean(dim=(1, 2)).numpy()
        grid = torch.nn.functional.adaptive_avg_pool2d(emap.unsqueeze(1), EMAP_GRID).squeeze(1).numpy()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        # patch geometry so patch_quality[c, r, w] can be located afterwards:
        # frames c*clip .. c*clip+clip-1, rows r*ps[0] .., cols w*ps[1] ..
        self.last_geometry = {"patch_h": int(ps[0]), "patch_w": int(ps[1]), "clip_frames": int(clip),
                              "frame_h": int(H), "frame_w": int(W), "n_frames": int(T)}
        return float(q), per_frame, grid, patch_q


class CVVDPScorer:
    def __init__(self, device):
        import pycvvdp
        self.metric = pycvvdp.cvvdp(display_name=CVVDP_DISPLAY, device=device)

    def score(self, dist, ref, ref_key, fps):
        jod, _ = self.metric.predict(dist, ref, dim_order="FHWC", frames_per_second=fps)
        return float(jod), None, None, None


# ============================================================================
# PER SCENE
# ============================================================================

def pairs_for_scene(manifest):
    """[(job, ref_config)] -- every render vs the 100 % ref; legacy renders also vs the 87 % ref."""
    refs = {j["config"] for j in manifest["jobs"] if j["config"].startswith("ssaa16_")}
    out = []
    for j in manifest["jobs"]:
        if j["kind"] == "aux" or j["config"].startswith("ssaa16_"):
            continue
        out.append((j, MAIN_REF))
        if j["kind"] == "legacy_check" and LEGACY_REF in refs:
            out.append((j, LEGACY_REF))
    return out


def run_scene(scene_dir, scorers, overwrite, only=None):
    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    name = manifest["scene"]["name"]
    fps = manifest["render_settings"]["fps"]
    ref_jobs = {j["config"]: j for j in manifest["jobs"] if j["config"].startswith("ssaa16_")}
    pairs = pairs_for_scene(manifest)
    if only:                                   # re-score just these renders (always overwritten)
        pairs = [(j, r) for j, r in pairs if j["config"] in only]
        missing = set(only) - {j["config"] for j, _ in pairs}
        if missing:
            print(f"  (not in this scene's manifest: {sorted(missing)})")
        overwrite = True
    out_dir = os.path.join(scene_dir, "scores")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}\n{name}: {len(pairs)} render/reference pairs, metrics: {list(scorers)}\n{'=' * 70}")
    problems = []

    for metric, scorer in scorers.items():
        jpath = os.path.join(out_dir, f"{metric}.json")
        results = json.load(open(jpath)) if os.path.exists(jpath) else {}
        emap_dir = os.path.join(out_dir, f"{metric}_emaps")

        # group by reference so each reference is loaded once
        for ref_cfg in sorted({r for _, r in pairs}):
            todo = [j for j, r in pairs if r == ref_cfg
                    and (overwrite or f"{j['kind']}/{j['config']}@{ref_cfg}" not in results)]
            if not todo:
                continue
            if ref_cfg not in ref_jobs:
                problems.append(f"no reference job {ref_cfg} in manifest")
                continue
            try:
                ref = load_job(scene_dir, ref_jobs[ref_cfg])
            except Exception as e:
                problems.append(f"reference {ref_cfg}: {e}")
                continue
            print(f"[{metric}] reference {ref_cfg}: {ref.shape[0]} frames {ref.shape[2]}x{ref.shape[1]}")

            for idx, job in enumerate(todo, 1):
                key = f"{job['kind']}/{job['config']}@{ref_cfg}"
                t0 = time.time()
                try:
                    dist = load_job(scene_dir, job)
                except Exception as e:
                    problems.append(f"{key}: {e}")
                    continue
                if dist.shape != ref.shape:
                    problems.append(f"{key}: shape {dist.shape} != reference {ref.shape} -- NOT scored, re-render")
                    continue
                try:
                    score, per_frame, grid, patch_q = scorer.score(dist, ref, ref_cfg, fps)
                except Exception as e:
                    problems.append(f"{key}: scoring failed: {e}")
                    continue

                flag = ""
                if metric == "cgvqm" and not (0.0 <= score <= 100.0):
                    flag = "suspect_out_of_range"
                    problems.append(f"{key}: CGVQM score {score:.2f} outside [0,100] -- check render/reference")
                entry = {"scene": name, "kind": job["kind"], "config": job["config"], "ref": ref_cfg,
                         "params": job["params"], "score": score, "n_frames": int(ref.shape[0]),
                         "flag": flag}
                if per_frame is not None:
                    entry["per_frame_errors"] = [round(float(x), 5) for x in per_frame]
                if grid is not None:
                    os.makedirs(emap_dir, exist_ok=True)
                    np.savez_compressed(os.path.join(emap_dir, f"{job['kind']}__{job['config']}__{ref_cfg}.npz"),
                                        grid=grid.astype(np.float16),
                                        patch_quality=patch_q.astype(np.float32),
                                        **{k: np.int32(v) for k, v in scorer.last_geometry.items()})
                results[key] = entry
                with open(jpath, "w") as f:          # save after every render -> resumable
                    json.dump(results, f, indent=1)
                print(f"  [{idx:3d}/{len(todo)}] {key}: {score:.4f} {flag}  ({time.time() - t0:.0f}s)")

    if problems:
        print(f"\n!! {len(problems)} problem(s) in {name}:")
        for p in problems:
            print("   " + p)
    return problems


# ============================================================================
# SUMMARY OUTPUTS
# ============================================================================

def write_tidy_csv(root):
    rows = []
    for jpath in sorted(glob.glob(os.path.join(root, "*", "scores", "*.json"))):
        metric = os.path.splitext(os.path.basename(jpath))[0]
        for e in json.load(open(jpath)).values():
            p = e["params"]
            rows.append({"scene": e["scene"], "kind": e["kind"], "config": e["config"], "ref": e["ref"],
                         "metric": metric, "score": e["score"],
                         "sp": p.get("sp"), "aw": p.get("aw"), "hp": p.get("hp"),
                         "ns": p.get("ns"), "fs": p.get("fs"),
                         "warmup_frames": p.get("warmup_frames", ""),
                         "n_frames": e["n_frames"], "flag": e["flag"]})
    if rows:
        path = os.path.join(root, "scores_all.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {path}")


def legacy_comparison(root, scene, old_scores_path):
    """New legacy-check renders (vs 87 % ref) next to the old factorial scores."""
    jpath = os.path.join(root, scene, "scores", "cgvqm.json")
    if not os.path.exists(jpath):
        return
    new = json.load(open(jpath))
    old = json.load(open(old_scores_path))
    print(f"\nLegacy check for {scene} (old factorial vs. re-render; should match closely):")
    print(f"  {'old combo':28s} {'old':>8s} {'new@87ref':>10s} {'new@100ref':>11s}")
    for e in new.values():
        if e["kind"] != "legacy_check" or e["ref"] != LEGACY_REF:
            continue
        oname = e["params"].get("old_factorial_name")
        o = old.get(oname, {}).get("score")
        e100 = new.get(f"legacy_check/{e['config']}@{MAIN_REF}", {}).get("score")
        fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
        print(f"  {oname:28s} {fmt(o):>8s} {fmt(e['score']):>10s} {fmt(e100):>11s}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root, e.g. ~/Documents/PerceptualTAA/data_v2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--metrics", nargs="+", default=["cgvqm"], choices=["cgvqm", "cvvdp"])
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--project-dir", default=None,
                    help="repo root containing src/cgvqm (default: this script's folder)")
    ap.add_argument("--overwrite", action="store_true", help="re-score renders that already have scores")
    ap.add_argument("--only", nargs="+", metavar="CONFIG",
                    help="re-score only these renders, e.g. --only sp087_aw0.300_hp150_ns08_fs1.00 (implies overwrite)")
    ap.add_argument("--old-scores", nargs="*", default=[], metavar="SCENE=PATH",
                    help="old factorial scores.json per scene, for the legacy comparison")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.expanduser(args.root)
    scenes = (sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "manifest.json")))
              if args.all else args.scene)

    scorers = {}
    if "cgvqm" in args.metrics:
        scorers["cgvqm"] = CGVQMScorer(find_project_dir(args.project_dir), device)
    if "cvvdp" in args.metrics:
        # Note: the paper's CVVDP run assumed a 'standard_4k' display. The renders
        # are 1080p, so 'standard_fhd' may be the more defensible choice --
        # decide once and use it for every scene.
        scorers["cvvdp"] = CVVDPScorer(device)

    all_problems = {}
    for s in scenes:
        p = run_scene(os.path.join(root, s), scorers, args.overwrite, set(args.only) if args.only else None)
        if p:
            all_problems[s] = len(p)

    write_tidy_csv(root)
    for item in args.old_scores:
        scene, path = item.split("=", 1)
        legacy_comparison(root, scene, path)

    print("\nDone." + (f" Problems in: {all_problems}" if all_problems else " No problems."))


if __name__ == "__main__":
    main()