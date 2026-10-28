"""
encode_and_verify.py  (dataset v2)
----------------------------------
Turns every rendered PNG folder listed in a scene's manifest.json into a
LOSSLESS video, checks the video is bit-identical to the PNGs, and (optionally)
deletes the PNGs. Also checks that every job in a scene has the same number of
frames as the reference -- mismatches are what produce nonsense scores.

Codec: H.264 RGB at qp 0 (libx264rgb). Decoded frames are exactly the PNG
pixels (verified per frame with ffmpeg's framemd5), at about half the size.
Container (.mkv or .mp4) is set by VIDEO_EXT in generate_taa_jobs.py and stored
in manifest.json; both hold the same lossless stream. Either way, normal players
(QuickTime, browsers) can't play lossless RGB H.264 -- use VLC/mpv, or make
separate lossy preview copies for viewing / the user study.

USAGE
  python encode_and_verify.py --root ~/Documents/PerceptualTAA/data_v2 --scene oldmine resto-fwd
  python encode_and_verify.py --root ... --all                    # every scene under root
  python encode_and_verify.py --root ... --all --delete-pngs      # free space after verifying

Reading frames back in Python (for metrics), without extra dependencies:
  from encode_and_verify import load_video_rgb
  frames = load_video_rgb("videos/taa/sp050_aw0.040_hp100_ns08_fs1.00.mkv")  # (T, H, W, 3) uint8
"""

import os
import re
import json
import glob
import shutil
import time
import argparse
import subprocess

import numpy as np

ENCODE_ARGS = ["-c:v", "libx264rgb", "-qp", "0", "-preset", "slow"]


def _frame_hashes(src):
    """Per-frame MD5 of the decoded RGB pixels."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-pix_fmt", "rgb24", "-f", "framemd5", "-"],
                         check=True, capture_output=True, text=True).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def png_pattern(folder):
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not pngs:
        return None, 0, 0
    first = int(re.sub(r"\D", "", os.path.basename(pngs[0])))
    return os.path.join(folder, "%04d.png"), len(pngs), first


def encode(folder, video, fps):
    pattern, n, first = png_pattern(folder)
    os.makedirs(os.path.dirname(video), exist_ok=True)
    tmp = video + ".tmp" + os.path.splitext(video)[1]   # e.g. name.mkv.tmp.mkv (in-progress file)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-start_number", str(first),
                        "-i", pattern, *ENCODE_ARGS, tmp], check=True, capture_output=True, text=True)
        png_h = _frame_hashes_from_pngs(pattern, first)
        vid_h = _frame_hashes(tmp)
        if png_h != vid_h:
            raise RuntimeError(f"verification FAILED ({len(png_h)} png vs {len(vid_h)} video frames)")
        os.replace(tmp, video)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {(e.stderr or '').strip()[-500:]}")
    finally:
        if os.path.exists(tmp):          # never leave half-written temp files behind
            os.remove(tmp)
    return n


def _frame_hashes_from_pngs(pattern, first):
    out = subprocess.run(["ffmpeg", "-v", "error", "-start_number", str(first), "-i", pattern,
                          "-pix_fmt", "rgb24", "-f", "framemd5", "-"],
                         check=True, capture_output=True, text=True).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def _recently_modified(folder, seconds=120):
    """True if any file in the folder changed in the last `seconds` (i.e. Unreal is still writing it)."""
    try:
        newest = max(os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder))
    except ValueError:
        return False
    return time.time() - newest < seconds


def count_video_frames(video, fps):
    """Fast frame count from the container's stored duration (no decoding)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", video], check=True, capture_output=True, text=True).stdout
    return int(round(float(out.strip()) * fps))


def load_video_rgb(path):
    """Decode a video to a (T, H, W, 3) uint8 array via ffmpeg (no PyAV needed)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path],
                           check=True, capture_output=True, text=True).stdout.strip()
    w, h = map(int, probe.split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def process_scene(scene_dir, delete_pngs):
    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    fps = manifest["render_settings"]["fps"]
    jobs = [j for j in manifest["jobs"] if j["video"]]
    print(f"\n=== {manifest['scene']['name']}: {len(jobs)} video jobs ===")

    counts, problems = {}, []
    ledger_path = os.path.join(scene_dir, "videos", "verified.json")
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}

    def save_ledger():
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        with open(ledger_path, "w") as f:
            json.dump(ledger, f, indent=1)

    def video_frames(tag, video):
        """Known count from the ledger, else a fast duration-based count (then remembered)."""
        if tag not in ledger:
            ledger[tag] = count_video_frames(video, fps)
            save_ledger()
        return ledger[tag]
    stale = glob.glob(os.path.join(scene_dir, "videos", "**", "*.tmp.*"), recursive=True)
    for t in stale:                      # leftovers from an interrupted earlier run
        os.remove(t)
    if stale:
        print(f"  removed {len(stale)} leftover temp file(s) from an interrupted run")
    # Expected frame count = the reference's (it's rendered first). Used so that
    # renders still being written by Unreal are left alone -- safe to run mid-render.
    n_expected = None
    for j in jobs:
        if j["kind"] == "reference":
            v, d = os.path.join(scene_dir, j["video"]), os.path.join(scene_dir, j["png_dir"])
            if os.path.exists(v):
                n_expected = video_frames(f"reference/{j['config']}", v)
            elif os.path.isdir(d) and not _recently_modified(d):
                n_expected = png_pattern(d)[1] or None
    in_progress = []

    for j in jobs:
        png_dir = os.path.join(scene_dir, j["png_dir"])
        video = os.path.join(scene_dir, j["video"])
        tag = f"{j['kind']}/{j['config']}"
        try:
            if os.path.exists(video):
                counts[tag] = video_frames(tag, video)
                status = "exists"
            elif png_pattern(png_dir)[0]:
                n_png = png_pattern(png_dir)[1]
                if _recently_modified(png_dir) or (n_expected and n_png < n_expected):
                    in_progress.append(f"{tag} ({n_png}/{n_expected or '?'} frames)")
                    continue                     # still rendering (or incomplete): don't touch
                counts[tag] = encode(png_dir, video, fps)
                ledger[tag] = counts[tag]
                save_ledger()
                status = "encoded+verified"
            else:
                problems.append(f"MISSING  {tag}")
                continue
            if delete_pngs and os.path.isdir(png_dir) and os.path.exists(video):
                shutil.rmtree(png_dir)
                status += ", pngs deleted"
            print(f"  {tag}: {counts[tag]} frames ({status})")
        except Exception as e:
            problems.append(f"ERROR    {tag}: {e}")

    if in_progress:
        print(f"\n  skipped {len(in_progress)} render(s) still being written or incomplete "
              f"(run again later): {', '.join(in_progress[:3])}{' ...' if len(in_progress) > 3 else ''}")
        problems.extend(f"PENDING  {t}" for t in in_progress)

    ref_tag = next((t for t in counts if t.startswith("reference/")), None)
    if ref_tag:
        n_ref = counts[ref_tag]
        for tag, n in counts.items():
            if n != n_ref:
                problems.append(f"FRAMES   {tag}: {n} frames vs reference {n_ref}")
    else:
        problems.append("MISSING  reference video -- nothing can be scored")

    if problems:
        print(f"\n  !! {len(problems)} problem(s):")
        for p in problems:
            print("    " + p)
    else:
        print("  all jobs present, verified, frame counts match the reference")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root, e.g. ~/Documents/PerceptualTAA/data_v2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--delete-pngs", action="store_true",
                    help="delete each PNG folder after its video is verified bit-identical")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    scenes = (sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "manifest.json")))
              if args.all else args.scene)
    any_problems = False
    for s in scenes:
        any_problems |= bool(process_scene(os.path.join(root, s), args.delete_pngs))
    print("\nDone." + (" Some scenes have problems -- see above." if any_problems else ""))


if __name__ == "__main__":
    main()