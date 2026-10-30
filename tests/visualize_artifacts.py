"""
visualize_artifacts.py  (dataset v2)
------------------------------------
Shows WHERE the artifact numbers come from: per-pixel maps for one frame (PNG)
or one 30-frame clip (MP4), the reference next to 1-3 renders, with CGVQM's
4 x 4 patch grid drawn on top and each patch's value written in it.

Columns: reference | render A | render B | ...
Rows (renders):                      Row (reference column):
  1 frame        + CGVQM patch score    frame
  2 |error|      + patch mean           motion speed (optical flow, px/frame)
  3 flicker      + patch mean           newly visible pixels (red)
  4 detail       + patch detail ratio   fine detail of the reference
     blue = detail lost (blur), red = extra detail (aliasing / noise)
Each row uses ONE colour scale across all renders, so columns are comparable.

USAGE  (needs compute_features.py in the same folder; pip install opencv-python matplotlib)
  # pick renders by settings (same resolution + history, two alphas):
  python visualize_artifacts.py --root ~/Documents/PerceptualTAA/data_v2 --scene oldmine \
         --sp 50 --hp 100 --aw 0.04 1.0 --frame 75
  # or by config name:
  python visualize_artifacts.py --root ... --scene oldmine --configs sp050_aw0.040_hp100_ns08_fs1.00
  # a short video of one 30-frame clip instead of a still (clip 2 = frames 60-89):
  python visualize_artifacts.py --root ... --scene oldmine --sp 50 --hp 100 --aw 0.04 1.0 --clip 2
Output goes to {root}/{scene}/visuals/.
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_features as cf   # same luma, high-pass, flow and grid as the measured numbers

TILE_W = 640                       # width of each tile in the output
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b42d2d"])
INK, MUTED = "#0b0b0b", "#52514e"


# ----------------------------------------------------------------------------- loading
def decode_rgb(path, start, count):
    """Frames start .. start+count-1 as (count, H, W, 3) uint8."""
    w, h = cf._probe_wh(path)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf",
                          f"select='between(n\\,{start}\\,{start + count - 1})'", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def job_frames(scene_dir, job, start, count):
    v = os.path.join(scene_dir, job["video"]) if job.get("video") else None
    if v and os.path.exists(v):
        return decode_rgb(v, start, count)
    d = os.path.join(scene_dir, job["png_dir"])
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))[start:start + count]
    return np.stack([cv2.imread(os.path.join(d, f))[..., ::-1] for f in files])


def total_frames(scene_dir, job, fps):
    v = os.path.join(scene_dir, job["video"]) if job.get("video") else None
    if v and os.path.exists(v):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v],
                             check=True, capture_output=True, text=True).stdout
        return int(round(float(out) * fps))
    return len([f for f in os.listdir(os.path.join(scene_dir, job["png_dir"])) if f.endswith(".png")])


def luma(rgb):
    return rgb.astype(np.float32) @ cf.LUMA_W


# ----------------------------------------------------------------------------- per-pixel maps
def flow_maps(Yprev, Ycur, dis):
    """Same computation as compute_features.Reference, for one frame pair."""
    H, W = Ycur.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    u8p, u8c = np.clip(Yprev, 0, 255).astype(np.uint8), np.clip(Ycur, 0, 255).astype(np.uint8)
    fb = dis.calc(u8c, u8p, None)
    ff = dis.calc(u8p, u8c, None)
    ff_at = cv2.remap(ff, gx + fb[..., 0], gy + fb[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    s = fb + ff_at
    occ = (s ** 2).sum(-1) > 0.01 * ((fb ** 2).sum(-1) + (ff_at ** 2).sum(-1)) + 0.5
    occ |= ((gx + fb[..., 0] < 0) | (gx + fb[..., 0] > W - 1) | (gy + fb[..., 1] < 0) | (gy + fb[..., 1] > H - 1))
    return fb, occ, gx, gy


def local_energy(Y):
    return cv2.GaussianBlur(cf.highpass(Y) ** 2, (0, 0), 6)


def render_maps(Dprev, Dcur, Rprev, Rcur, fb, occ, gx, gy, eref):
    e_prev, e_cur = Dprev - Rprev, Dcur - Rcur
    warped = cv2.remap(e_prev, gx + fb[..., 0], gy + fb[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    flick = np.abs(e_cur - warped)
    flick[occ] = np.nan
    detail = np.log2((local_energy(Dcur) + 1e-3) / (eref + 1e-3))
    return np.abs(e_cur), flick, np.clip(detail, -2, 2)


def patch_means(m, grid, fn=np.nanmean):
    out = np.full((grid.nr, grid.nw), np.nan)
    for r in range(grid.nr):
        for w in range(grid.nw):
            blk = m[r * grid.ph:(r + 1) * grid.ph, w * grid.pw:(w + 1) * grid.pw]
            if np.isfinite(blk).any():
                out[r, w] = fn(blk)
    return out


# ----------------------------------------------------------------------------- drawing
def draw_tile(ax, img, grid, values=None, fmt="{:.1f}", cmap=None, vmin=None, vmax=None, overlay=None):
    if cmap is None:
        ax.imshow(img, interpolation="bilinear")
    else:
        ax.set_facecolor("#c3c2b7")       # NaN (newly visible pixels in the flicker row) shows as grey
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if overlay is not None:
        ax.imshow(overlay, interpolation="nearest")
    for r in range(1, grid.nr):
        ax.axhline(r * grid.ph - 0.5, color="white", lw=0.8, alpha=0.9)
    for w in range(1, grid.nw):
        ax.axvline(w * grid.pw - 0.5, color="white", lw=0.8, alpha=0.9)
    if values is not None:
        for r in range(grid.nr):
            for w in range(grid.nw):
                v = values[r, w]
                if np.isfinite(v):
                    ax.text(w * grid.pw + 8, r * grid.ph + 8, fmt.format(v), color=INK, fontsize=7, va="top",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_figure(t, ref_rgb, rends, ref_maps, rmaps, grid, labels, patchq, scales, scene):
    ncol = 1 + len(rends)
    H, W = ref_rgb.shape[:2]
    tile_h = TILE_W * H / W / 100
    fig, axes = plt.subplots(4, ncol, figsize=(ncol * TILE_W / 100 + 1.2, 4 * tile_h + 1.4), dpi=100, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    c = t // grid.clip
    motion, occ, eref = ref_maps
    occ_rgba = np.zeros((H, W, 4), np.float32); occ_rgba[occ] = [0.89, 0.29, 0.28, 0.85]
    gray = np.repeat(luma(ref_rgb)[..., None] / 255, 3, -1) * 0.6 + 0.4

    draw_tile(axes[0, 0], ref_rgb, grid)
    draw_tile(axes[1, 0], motion, grid, patch_means(motion, grid), "{:.1f}", SEQ, 0, scales["motion"])
    draw_tile(axes[2, 0], gray, grid, patch_means(occ.astype(np.float32), grid), "{:.0%}", overlay=occ_rgba)
    draw_tile(axes[3, 0], np.log10(eref + 1), grid, None, cmap=SEQ, vmin=0, vmax=scales["eref"])
    axes[0, 0].set_title("Reference (16× SSAA)", fontsize=10, color=INK)
    row_ref = ["frame", "motion (px/frame)", "newly visible pixels (red)", "fine detail (log)"]
    for i, lab in enumerate(row_ref):
        axes[i, 0].set_ylabel(lab, fontsize=9, color=MUTED)

    for j, (rgb, (aerr, flick, detail), lab, q) in enumerate(zip(rends, rmaps, labels, patchq), start=1):
        draw_tile(axes[0, j], rgb, grid, q[c] if q is not None else None, "Q {:.1f}")
        draw_tile(axes[1, j], aerr, grid, patch_means(aerr, grid), "{:.1f}", SEQ, 0, scales["err"])
        draw_tile(axes[2, j], flick, grid, patch_means(flick, grid), "{:.2f}", SEQ, 0, scales["flick"])
        ratio = patch_means(detail, grid, lambda b: 2 ** np.nanmean(b))
        draw_tile(axes[3, j], detail, grid, ratio, "×{:.2f}", DIV, -2, 2)
        axes[0, j].set_title(lab, fontsize=10, color=INK)
        if j == 1:
            for i, l in enumerate(["frame  (Q = CGVQM patch score, this clip)", "|error| vs reference (luma levels)",
                                   "flicker: error change vs previous frame", "detail: blue lost · red extra (×ratio)"]):
                axes[i, j].set_ylabel(l, fontsize=9, color=MUTED)
    fig.suptitle(f"{scene} · frame {t} (clip {c}: frames {c * grid.clip}–{min((c + 1) * grid.clip, grid.T) - 1})",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--configs", nargs="*", default=[], help="config names (as in manifest.json)")
    ap.add_argument("--sp", type=int); ap.add_argument("--hp", type=int)
    ap.add_argument("--aw", type=float, nargs="*", default=[], help="alpha values to compare")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--frame", type=int, default=75, help="frame to show as a PNG (default 75)")
    g.add_argument("--clip", type=int, help="render this 30-frame clip as an MP4 instead")
    ap.add_argument("--preset", default="medium", choices=list(cf.DIS_PRESETS))
    args = ap.parse_args()

    scene_dir = os.path.join(os.path.expanduser(args.root), args.scene)
    manifest = json.load(open(os.path.join(scene_dir, "manifest.json")))
    fps = manifest["render_settings"]["fps"]
    jobs = {j["config"]: j for j in manifest["jobs"]}
    ref_job = jobs[cf.MAIN_REF]
    pick = list(args.configs)
    for a in args.aw:
        hits = [j["config"] for j in manifest["jobs"] if j["kind"] == "taa"
                and abs(j["params"].get("aw", -1) - a) < 1e-9
                and (args.sp is None or j["params"].get("sp") == args.sp)
                and (args.hp is None or j["params"].get("hp") == args.hp)]
        if not hits:
            sys.exit(f"No render with aw={a} sp={args.sp} hp={args.hp}")
        pick.append(hits[0])
    if not pick or len(pick) > 3:
        sys.exit("Choose 1-3 renders (--configs or --aw with --sp/--hp).")

    clip_len = int(min(fps, cf.CLIP_MAX))
    if args.clip is not None:
        frames = list(range(max(1, args.clip * clip_len), (args.clip + 1) * clip_len))
    else:
        frames = [max(1, args.frame)]
    start, count = frames[0] - 1, frames[-1] - frames[0] + 2         # include the previous frame

    ref = job_frames(scene_dir, ref_job, start, count)
    rends = [job_frames(scene_dir, jobs[c], start, count) for c in pick]
    frames = frames[:len(ref) - 1]
    if not frames:
        sys.exit("Frame/clip is beyond the end of the video.")
    H, W = ref.shape[1:3]
    grid = cf.Grid(total_frames(scene_dir, ref_job, fps), H, W, fps)
    labels = [", ".join(f"{k}={jobs[c]['params'][k]}" for k in ("sp", "aw", "hp") if k in jobs[c]["params"]) or c
              for c in pick]
    if len(set(labels)) < len(labels):          # settings don't tell them apart -> use config names
        labels = pick
    patchq = []
    for c in pick:
        f = os.path.join(scene_dir, "scores", "cgvqm_emaps", f"{jobs[c]['kind']}__{c}__{cf.MAIN_REF}.npz")
        patchq.append(np.load(f)["patch_quality"] if os.path.exists(f) else None)

    dis = cv2.DISOpticalFlow_create(cf.DIS_PRESETS[args.preset])
    Yr = luma(ref); Yd = [luma(r) for r in rends]
    per_frame = []
    for t in frames:
        i = t - start
        fb, occ, gx, gy = flow_maps(Yr[i - 1], Yr[i], dis)
        motion = np.sqrt((fb ** 2).sum(-1))
        eref = local_energy(Yr[i])
        rm = [render_maps(D[i - 1], D[i], Yr[i - 1], Yr[i], fb, occ, gx, gy, eref) for D in Yd]
        per_frame.append((t, i, (motion, occ, eref), rm))

    # one colour scale per row, shared by all renders and frames
    pct = lambda arrs: max(1e-6, float(np.nanpercentile(np.concatenate([a[np.isfinite(a)].ravel()[::7] for a in arrs]), 99)))
    scales = {"motion": pct([p[2][0] for p in per_frame]),
              "eref": float(np.log10(np.nanpercentile(per_frame[0][2][2], 99.5) + 1)),
              "err": pct([m[0] for p in per_frame for m in p[3]]),
              "flick": pct([m[1] for p in per_frame for m in p[3]])}

    out_dir = os.path.join(scene_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    tag = "_vs_".join(pick)[:150]
    if args.clip is None:
        t, i, refm, rm = per_frame[0]
        fig = make_figure(t, ref[i], [r[i] for r in rends], refm, rm, grid, labels, patchq, scales, args.scene)
        path = os.path.join(out_dir, f"frame{t:04d}__{tag}.png")
        fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)
    else:
        path = os.path.join(out_dir, f"clip{args.clip}__{tag}.mp4")
        proc = None
        for t, i, refm, rm in per_frame:
            fig = make_figure(t, ref[i], [r[i] for r in rends], refm, rm, grid, labels, patchq, scales, args.scene)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)
            h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
            if proc is None:
                proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                         "-s", f"{w}x{h}", "-r", "10", "-i", "-", "-c:v", "libx264", "-crf", "18",
                                         "-preset", "veryfast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
            proc.stdin.write(np.ascontiguousarray(img[:h, :w]).tobytes())
            print(f"  frame {t}")
        proc.stdin.close(); proc.wait()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()