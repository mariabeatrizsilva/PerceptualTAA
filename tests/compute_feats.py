"""
compute_features.py  (dataset v2)
---------------------------------
Per-patch CONTENT features (from the 16x SSAA reference) and ARTIFACT measures
(each render vs. the reference), on exactly the same grid CGVQM uses:
4 x 4 spatial patches x 30-frame clips  ->  80 cells per 150-frame 1080p video.
So every cell lines up with patch_quality[c, r, w] saved by score_taa.py.

CONTENT (reference only -- describes the scene, independent of TAA settings)
  motion_mean      mean optical-flow magnitude, pixels/frame (DIS optical flow)
  disocc_frac      fraction of pixels newly revealed this frame
                   (forward-backward flow consistency check)
  detail_hf        high-frequency energy: mean (Y - GaussianBlur(Y))^2
  detail_grad      mean gradient magnitude (Sobel)
  contrast         mean per-frame standard deviation of luma within the patch
  luma_mean        mean luma (0-255)
  temporal_change  mean |Y_t - Y_{t-1}|  (raw frame difference, not motion-compensated)

ARTIFACTS (render D vs. reference R, luma 0-255, error e_t = D_t - R_t)
  err_mae          mean |e|                                   overall difference
  flicker          mean |e_t - warp(e_{t-1})|, outside disocclusions
                   = how much the ERROR changes from frame to frame after
                   following the motion. A stable blur scores ~0; shimmer,
                   jitter and flicker score high.
  hf_ratio         high-frequency energy of D / of R.  < 1: detail lost (blur)
                                                       > 1: extra detail (aliasing/noise)
  err_disocc       mean |e| in newly revealed pixels  (ghosting/trailing shows up here)
  err_nondisocc    mean |e| everywhere else
  (frame 0 has no previous frame: motion/flicker/disocclusion use frames 1..T-1)

OUTPUT  ({root}/{scene}/features/)
  content.csv        one row per cell
  artifacts.csv      one row per (render, cell)
  patch_table.csv    analysis-ready: artifacts + content + CGVQM patch_quality
                     (if score_taa.py has run) -- one row per (render, cell)
  cache/*.npz        per-render results, so re-runs only process new renders

USAGE
  pip install opencv-python numpy
  python compute_features.py --root ~/Documents/PerceptualTAA/data_v2 --scene oldmine
  python compute_features.py --root ... --all
  # resumable; --overwrite recomputes everything

Runtime: optical flow on the reference takes ~1-2 min per scene (done once per
run); each render then takes ~10-30 s. Peak RAM ~5 GB at 1080p.
"""

import os
import csv
import json
import math
import time
import glob
import argparse
import subprocess

import numpy as np
import cv2

# --- grid: identical to CGVQM in score_taa.py ---------------------------------
PATCH_SCALE = 4
CLIP_MAX = 30
MAIN_REF = "ssaa16_sp100"

# --- feature settings -----------------------------------------------------------
HP_SIGMA = 1.5        # Gaussian sigma (px) separating "high frequency" detail
DIS_PRESETS = {"ultrafast": 0, "fast": 1, "medium": 2}
LUMA_W = np.array([0.2126, 0.7152, 0.0722], np.float32)   # BT.709

CONTENT_KEYS = ["motion_mean", "disocc_frac", "detail_hf", "detail_grad",
                "contrast", "luma_mean", "temporal_change"]
ARTIFACT_KEYS = ["err_mae", "flicker", "hf_ratio", "err_disocc", "err_nondisocc"]


# ============================================================================
# LOADING
# ============================================================================

def _probe_wh(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout.strip()
    return map(int, out.split(",")[:2])


def video_luma(path):
    """Decode a video straight to luma (T, H, W) float32 in 0-255, one frame at a time."""
    w, h = _probe_wh(path)
    n = w * h * 3
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    frames = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) @ LUMA_W)
    proc.wait()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def pngs_luma(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"no PNGs in {folder}")
    return np.stack([cv2.imread(f, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) @ LUMA_W
                     for f in files])


def load_luma(scene_dir, job):
    if job.get("video"):
        v = os.path.join(scene_dir, job["video"])
        if os.path.exists(v):
            return video_luma(v)
    d = os.path.join(scene_dir, job["png_dir"])
    if os.path.isdir(d):
        return pngs_luma(d)
    raise FileNotFoundError(f"no video or PNGs for {job['kind']}/{job['config']}")


# ============================================================================
# GRID + POOLING
# ============================================================================

class Grid:
    """CGVQM's patch grid: patch = int(H/4) x int(W/4) px, clip = min(fps, 30) frames."""

    def __init__(self, T, H, W, fps):
        self.T, self.H, self.W = T, H, W
        self.ph, self.pw = int(H / PATCH_SCALE), int(W / PATCH_SCALE)
        self.clip = int(min(fps, CLIP_MAX))
        self.nc = math.ceil(T / self.clip)
        self.nr = math.ceil(H / self.ph)
        self.nw = math.ceil(W / self.pw)
        self.Hp, self.Wp = self.nr * self.ph, self.nw * self.pw

    def new_acc(self):
        return np.zeros((self.nc, self.nr, self.nw)), np.zeros((self.nc, self.nr, self.nw))

    def add(self, acc, t, pixmap, mask=None):
        """Add one frame's per-pixel values (NaN / mask=False ignored) to the clip/patch sums."""
        s, n = acc
        m = pixmap.astype(np.float64)
        valid = np.isfinite(m) if mask is None else (np.isfinite(m) & mask)
        m = np.where(valid, m, 0.0)
        if (self.Hp, self.Wp) != m.shape:      # pad to a whole number of patches
            m = np.pad(m, ((0, self.Hp - self.H), (0, self.Wp - self.W)))
            valid = np.pad(valid, ((0, self.Hp - self.H), (0, self.Wp - self.W)))
        c = t // self.clip
        s[c] += m.reshape(self.nr, self.ph, self.nw, self.pw).sum(axis=(1, 3))
        n[c] += valid.reshape(self.nr, self.ph, self.nw, self.pw).sum(axis=(1, 3))

    def add_patchwise(self, acc, t, values):
        """Add one value per patch for frame t (for per-frame patch statistics like std)."""
        s, n = acc
        c = t // self.clip
        s[c] += values
        n[c] += 1

    def patch_std(self, img):
        x = img.astype(np.float64)
        if (self.Hp, self.Wp) != x.shape:
            x = np.pad(x, ((0, self.Hp - self.H), (0, self.Wp - self.W)), mode="edge")
        b = x.reshape(self.nr, self.ph, self.nw, self.pw)
        return b.std(axis=(1, 3))

    @staticmethod
    def mean(acc):
        s, n = acc
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    def cells(self):
        for c in range(self.nc):
            for r in range(self.nr):
                for w in range(self.nw):
                    yield c, r, w, {
                        "clip": c, "row": r, "col": w,
                        "frame_start": c * self.clip, "frame_end": min((c + 1) * self.clip, self.T) - 1,
                        "y0": r * self.ph, "y1": min((r + 1) * self.ph, self.H) - 1,
                        "x0": w * self.pw, "x1": min((w + 1) * self.pw, self.W) - 1}


# ============================================================================
# REFERENCE: optical flow, disocclusion, content features
# ============================================================================

def highpass(Y):
    return Y - cv2.GaussianBlur(Y, (0, 0), HP_SIGMA)


class Reference:
    def __init__(self, Y, fps, preset):
        self.Y = Y
        T, H, W = Y.shape
        self.grid = g = Grid(T, H, W, fps)
        dis = cv2.DISOpticalFlow_create(DIS_PRESETS[preset])
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.gx, self.gy = gx, gy

        # flow_b[t]: on frame t's pixel grid, where that pixel was in frame t-1
        self.flow_b = np.zeros((T, H, W, 2), np.float16)
        self.disocc = np.zeros((T, H, W), bool)

        acc = {k: g.new_acc() for k in CONTENT_KEYS}
        acc_hp_energy = g.new_acc()           # sum of highpass^2 -> denominator of hf_ratio
        u8 = [np.clip(f, 0, 255).astype(np.uint8) for f in Y]

        for t in range(T):
            hp = highpass(Y[t])
            g.add(acc["detail_hf"], t, hp ** 2)
            g.add(acc_hp_energy, t, hp ** 2)
            gxs = cv2.Sobel(Y[t], cv2.CV_32F, 1, 0, ksize=3)
            gys = cv2.Sobel(Y[t], cv2.CV_32F, 0, 1, ksize=3)
            g.add(acc["detail_grad"], t, np.sqrt(gxs ** 2 + gys ** 2))
            g.add(acc["luma_mean"], t, Y[t])
            g.add_patchwise(acc["contrast"], t, g.patch_std(Y[t]))
            if t == 0:
                continue
            g.add(acc["temporal_change"], t, np.abs(Y[t] - Y[t - 1]))

            fb = dis.calc(u8[t], u8[t - 1], None)        # t-grid -> position in t-1
            ff = dis.calc(u8[t - 1], u8[t], None)        # (t-1)-grid -> position in t
            # forward-backward consistency (Sundaram et al. 2010)
            ff_at = cv2.remap(ff, gx + fb[..., 0], gy + fb[..., 1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
            s = fb + ff_at
            occ = (s ** 2).sum(-1) > 0.01 * ((fb ** 2).sum(-1) + (ff_at ** 2).sum(-1)) + 0.5
            out_of_frame = ((gx + fb[..., 0] < 0) | (gx + fb[..., 0] > W - 1) |
                            (gy + fb[..., 1] < 0) | (gy + fb[..., 1] > H - 1))
            occ |= out_of_frame                            # content entering at the border
            self.flow_b[t] = fb
            self.disocc[t] = occ
            g.add(acc["motion_mean"], t, np.sqrt((fb ** 2).sum(-1)))
            g.add(acc["disocc_frac"], t, occ.astype(np.float32))

        self.content = {k: Grid.mean(a) for k, a in acc.items()}
        self.hp_energy = Grid.mean(acc_hp_energy)

    def warp_prev(self, img_prev, t):
        """Bring frame t-1 values onto frame t's pixel grid using the reference flow."""
        fb = self.flow_b[t].astype(np.float32)
        return cv2.remap(img_prev, self.gx + fb[..., 0], self.gy + fb[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def artifacts(self, D):
        g = self.grid
        acc = {k: g.new_acc() for k in ARTIFACT_KEYS if k != "hf_ratio"}
        acc_hp = g.new_acc()
        e_prev = None
        for t in range(g.T):
            e = D[t] - self.Y[t]
            ae = np.abs(e)
            g.add(acc["err_mae"], t, ae)
            g.add(acc_hp, t, highpass(D[t]) ** 2)
            if t > 0:
                occ = self.disocc[t]
                g.add(acc["flicker"], t, np.abs(e - self.warp_prev(e_prev, t)), mask=~occ)
                g.add(acc["err_disocc"], t, ae, mask=occ)
                g.add(acc["err_nondisocc"], t, ae, mask=~occ)
            e_prev = e
        out = {k: Grid.mean(a) for k, a in acc.items()}
        with np.errstate(invalid="ignore", divide="ignore"):
            out["hf_ratio"] = Grid.mean(acc_hp) / self.hp_energy
        return out


# ============================================================================
# PER SCENE
# ============================================================================

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _r(x):
    return "" if not np.isfinite(x) else round(float(x), 5)


def run_scene(scene_dir, preset, overwrite, limit=None):
    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    name = manifest["scene"]["name"]
    fps = manifest["render_settings"]["fps"]
    out_dir = os.path.join(scene_dir, "features")
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    ref_job = next(j for j in manifest["jobs"] if j["config"] == MAIN_REF)
    renders = [j for j in manifest["jobs"]
               if j["kind"] in ("taa", "legacy_check") and not j["config"].startswith("ssaa16_")]
    if limit:
        renders = renders[:limit]

    def cache_path(j):
        return os.path.join(cache_dir, f"{j['kind']}__{j['config']}.npz")

    todo = [j for j in renders if overwrite or not os.path.exists(cache_path(j))]
    content_cache = os.path.join(cache_dir, "content.npz")
    print(f"\n=== {name}: {len(renders)} renders, {len(todo)} to process ===")
    problems = []

    ref = None
    if todo or overwrite or not os.path.exists(content_cache):
        t0 = time.time()
        Y = load_luma(scene_dir, ref_job)
        print(f"  reference: {Y.shape[0]} frames {Y.shape[2]}x{Y.shape[1]} -- computing optical flow ...")
        ref = Reference(Y, fps, preset)
        np.savez_compressed(content_cache, **ref.content,
                            geometry=np.array([ref.grid.T, ref.grid.H, ref.grid.W, fps]))
        print(f"  reference done ({time.time() - t0:.0f}s)")

    for idx, j in enumerate(todo, 1):
        tag = f"{j['kind']}/{j['config']}"
        t0 = time.time()
        try:
            D = load_luma(scene_dir, j)
        except Exception as e:
            problems.append(f"{tag}: {e}")
            continue
        if D.shape != ref.Y.shape:
            problems.append(f"{tag}: shape {D.shape} != reference {ref.Y.shape} -- skipped")
            continue
        art = ref.artifacts(D)
        np.savez_compressed(cache_path(j), **art)
        print(f"  [{idx:3d}/{len(todo)}] {tag}  flicker={np.nanmean(art['flicker']):.3f} "
              f"hf_ratio={np.nanmean(art['hf_ratio']):.3f}  ({time.time() - t0:.0f}s)")

    # ---- assemble CSVs from caches ----------------------------------------
    cz = np.load(content_cache)
    T, H, W, fps_c = (int(x) for x in cz["geometry"])
    grid = Grid(T, H, W, fps_c)
    content = {k: cz[k] for k in CONTENT_KEYS}
    content_rows = []
    for c, r, w, base in grid.cells():
        content_rows.append({"scene": name, **base, **{k: _r(content[k][c, r, w]) for k in CONTENT_KEYS}})
    write_csv(os.path.join(out_dir, "content.csv"), content_rows)

    art_rows, table_rows, n_with_q = [], [], 0
    qdir = os.path.join(scene_dir, "scores", "cgvqm_emaps")
    for j in renders:
        if not os.path.exists(cache_path(j)):
            continue
        a = np.load(cache_path(j))
        qf = os.path.join(qdir, f"{j['kind']}__{j['config']}__{MAIN_REF}.npz")
        q = np.load(qf)["patch_quality"] if os.path.exists(qf) else None
        if q is not None and q.shape != (grid.nc, grid.nr, grid.nw):
            problems.append(f"{j['config']}: CGVQM patch grid {q.shape} != feature grid -- not joined")
            q = None
        n_with_q += q is not None
        p = j["params"]
        prm = {k: p.get(k, "") for k in ("sp", "aw", "hp", "ns", "fs")}
        for c, r, w, base in grid.cells():
            arow = {k: _r(a[k][c, r, w]) for k in ARTIFACT_KEYS}
            art_rows.append({"scene": name, "kind": j["kind"], "config": j["config"], **prm,
                             "clip": c, "row": r, "col": w, **arow})
            table_rows.append({"scene": name, "kind": j["kind"], "config": j["config"], **prm, **base,
                               "cgvqm_patch_quality": _r(q[c, r, w]) if q is not None else "",
                               **arow, **{k: _r(content[k][c, r, w]) for k in CONTENT_KEYS}})
    write_csv(os.path.join(out_dir, "artifacts.csv"), art_rows)
    write_csv(os.path.join(out_dir, "patch_table.csv"), table_rows)
    print(f"  wrote content.csv ({len(content_rows)} cells), artifacts.csv + patch_table.csv "
          f"({len(art_rows)} rows; CGVQM patch scores joined for {n_with_q} renders)")
    if problems:
        print(f"  !! {len(problems)} problem(s):")
        for pr in problems:
            print("     " + pr)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--flow-preset", default="medium", choices=list(DIS_PRESETS),
                    help="DIS optical flow quality/speed (default: medium)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N renders (quick test)")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    scenes = (sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "manifest.json")))
              if args.all else args.scene)
    bad = {}
    for s in scenes:
        p = run_scene(os.path.join(root, s), args.flow_preset, args.overwrite, args.limit)
        if p:
            bad[s] = len(p)
    print("\nDone." + (f" Problems in: {bad}" if bad else ""))


if __name__ == "__main__":
    main()