"""
scene_content_view.py  (dataset v2)
-----------------------------------
What the CONTENT features look like, per scene, from the 16x SSAA reference only
(no renders needed -- works on any computer that has the reference videos).

For each scene it writes, into {root}/{scene}/visuals/:
  content_sheet.png   one row per 30-frame clip (middle frame of each):
                      frame | motion | newly visible pixels | fine detail,
                      with CGVQM's 4 x 4 patch grid and each patch's value
                      for that clip -- the exact numbers the analysis uses
  content_view.mp4    the same four views for every frame (--no-video to skip)

Patch values come from features/content.csv when compute_features.py has run
(so they match the analysis exactly); otherwise they are computed here.

USAGE  (keep next to compute_features.py and visualize_artifacts.py)
  python scene_content_view.py --root ~/Documents/PerceptualTAA/data_v2 --all
  python scene_content_view.py --root ... --scene oldmine lightfoliage --no-video
"""

import os
import sys
import csv
import json
import argparse
import subprocess

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_features as cf
import visualize_artifacts as va

ROWS = [("frame", None), ("motion (px / frame)", "motion_mean"),
        ("newly visible pixels (red)", "disocc_frac"), ("fine detail", "detail_hf")]
FMT = {"motion_mean": "{:.1f}", "disocc_frac": "{:.0%}", "detail_hf": "{:.0f}"}


def load_content_csv(scene_dir):
    path = os.path.join(scene_dir, "features", "content.csv")
    if not os.path.exists(path):
        return None
    vals = {}
    for r in csv.DictReader(open(path)):
        for k in FMT:
            if r.get(k) not in (None, ""):
                vals.setdefault(k, {})[(int(r["clip"]), int(r["row"]), int(r["col"]))] = float(r[k])
    return vals


def clip_grid(vals, key, c, grid):
    out = np.full((grid.nr, grid.nw), np.nan)
    if vals and key in vals:
        for r in range(grid.nr):
            for w in range(grid.nw):
                out[r, w] = vals[key].get((c, r, w), np.nan)
    return out


def maps_for(Yr, t, dis):
    fb, occ, gx, gy = va.flow_maps(Yr[t - 1], Yr[t], dis)
    motion = np.sqrt((fb ** 2).sum(-1))
    hp = cf.highpass(Yr[t]) ** 2
    return motion, occ, hp


def draw(axrow, rgb, Ylum, motion, occ, hp, grid, values, scales, show_labels):
    H, W = Ylum.shape
    occ_rgba = np.zeros((H, W, 4), np.float32)
    occ_rgba[occ] = [0.89, 0.29, 0.28, 0.85]
    gray = np.repeat(Ylum[..., None] / 255, 3, -1) * 0.6 + 0.4
    va.draw_tile(axrow[0], rgb, grid)
    va.draw_tile(axrow[1], motion, grid, values["motion_mean"], FMT["motion_mean"], va.SEQ, 0, scales["motion"])
    va.draw_tile(axrow[2], gray, grid, values["disocc_frac"], FMT["disocc_frac"], overlay=occ_rgba)
    va.draw_tile(axrow[3], np.log10(cv2.GaussianBlur(hp, (0, 0), 6) + 1), grid, values["detail_hf"],
                 FMT["detail_hf"], va.SEQ, 0, scales["detail"])
    if show_labels:
        for ax, (lab, _) in zip(axrow, ROWS):
            ax.set_title(lab, fontsize=10, color=va.INK)


def run_scene(scene_dir, video, fps_out, preset):
    manifest = json.load(open(os.path.join(scene_dir, "manifest.json")))
    name = manifest["scene"]["name"]
    fps = manifest["render_settings"]["fps"]
    ref_job = next(j for j in manifest["jobs"] if j["config"] == cf.MAIN_REF)
    T = va.total_frames(scene_dir, ref_job, fps)
    print(f"\n=== {name}: {T} reference frames ===")
    rgb = va.job_frames(scene_dir, ref_job, 0, T)
    Yr = va.luma(rgb)
    H, W = Yr.shape[1:]
    grid = cf.Grid(len(Yr), H, W, fps)
    dis = cv2.DISOpticalFlow_create(cf.DIS_PRESETS[preset])
    vals = load_content_csv(scene_dir)
    source = "features/content.csv (clip averages)" if vals else "computed here (this frame)"

    # colour scales: one per scene, from a quick pass over every 10th frame
    sample = list(range(1, len(Yr), 10))
    mot, det = [], []
    for t in sample:
        m, _, hp = maps_for(Yr, t, dis)
        mot.append(np.percentile(m, 99)); det.append(np.log10(np.percentile(cv2.GaussianBlur(hp, (0, 0), 6), 99.5) + 1))
    scales = {"motion": max(1e-3, float(np.max(mot))), "detail": float(np.max(det))}

    out_dir = os.path.join(scene_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)

    def values_for(t, motion, occ, hp):
        c = t // grid.clip
        if vals:
            return {k: clip_grid(vals, k, c, grid) for k in FMT}
        return {"motion_mean": va.patch_means(motion, grid), "disocc_frac": va.patch_means(occ.astype(np.float32), grid),
                "detail_hf": va.patch_means(hp, grid)}

    # ---- sheet: middle frame of each clip
    mids = [min(len(Yr) - 1, max(1, c * grid.clip + grid.clip // 2)) for c in range(grid.nc)]
    tile = 480
    th = tile * H / W / 100
    fig, axes = plt.subplots(len(mids), 4, figsize=(4 * tile / 100 + 0.8, len(mids) * th + 1.2), dpi=100, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, t in enumerate(mids):
        motion, occ, hp = maps_for(Yr, t, dis)
        draw(axes[i], rgb[t], Yr[t], motion, occ, hp, grid, values_for(t, motion, occ, hp), scales, i == 0)
        c = t // grid.clip
        axes[i, 0].set_ylabel(f"clip {c}\nframes {c * grid.clip}–{min((c + 1) * grid.clip, len(Yr)) - 1}",
                              fontsize=9, color=va.MUTED)
    fig.suptitle(f"{name} · content (reference only) · patch values: {source}", fontsize=11, color=va.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    sheet = os.path.join(out_dir, "content_sheet.png")
    fig.savefig(sheet, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"  wrote {sheet}")

    if not video:
        return
    # ---- video: every frame, 2 x 2 tiles
    path = os.path.join(out_dir, "content_view.mp4")
    proc = None
    tile = 800
    th = tile * H / W / 100
    for t in range(1, len(Yr)):
        motion, occ, hp = maps_for(Yr, t, dis)
        fig, axes = plt.subplots(2, 2, figsize=(2 * tile / 100 + 0.4, 2 * th + 0.9), dpi=100)
        fig.patch.set_facecolor("#fcfcfb")
        ax4 = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        draw(ax4, rgb[t], Yr[t], motion, occ, hp, grid, values_for(t, motion, occ, hp), scales, True)
        fig.suptitle(f"{name} · frame {t} · clip {t // grid.clip}", fontsize=11, color=va.INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        if proc is None:
            proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{w}x{h}", "-r", str(fps_out), "-i", "-", "-c:v", "libx264", "-crf", "20",
                                     "-preset", "veryfast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(img[:h, :w]).tobytes())
        if t % 25 == 0:
            print(f"  frame {t}/{len(Yr) - 1}")
    proc.stdin.close(); proc.wait()
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="only the summary sheet (fast)")
    ap.add_argument("--fps", type=int, default=15, help="playback speed of the video (default 15 = half speed)")
    ap.add_argument("--flow-preset", default="medium", choices=list(cf.DIS_PRESETS))
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    scenes = (sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "manifest.json")))
              if args.all else args.scene)
    for s in scenes:
        try:
            run_scene(os.path.join(root, s), not args.no_video, args.fps, args.flow_preset)
        except Exception as e:
            print(f"  !! {s}: {e}")


if __name__ == "__main__":
    main()