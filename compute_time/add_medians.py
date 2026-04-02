"""
add_medians.py

Iterates through all raw parquet files, replicates the existing
pass-1 skip (iloc[150:]), computes per-frame medians across passes
2+3 (frames 150–end), and appends/updates a results_median.csv.

Usage:
    python add_medians.py --raw_dir ./raw --out ./results_median.csv

The script is non-destructive: it writes to a new file (results_median.csv)
rather than modifying your existing results.csv.
"""

import os
import re
import glob
import argparse
import pandas as pd

# ── Must match your existing METRICS list exactly ─────────────────────────────
METRICS = ['FrameTime', 'GameThreadTime', 'RenderThreadTime', 'GPUTime', 'RHIThreadTime', 'GPUMem/LocalUsedMB', 'GPU/TAA']


# ── Regex helpers (mirrors your parse_profile metadata extraction) ─────────────

def extract_meta_from_parquet_name(fname):
    """
    Attempts to recover scene/cvar/screen_pct from the parquet filename.
    This is a fallback — ideally you store metadata alongside the parquet.
    Expected filename pattern (from your existing code):
        <original_csv_stem>.parquet
    We can't recover metadata from the filename alone, so we rely on a
    sidecar CSV or the results.csv you already have.
    """
    return os.path.splitext(fname)[0]


def load_existing_results(results_path):
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return pd.DataFrame()


def build_parquet_to_meta_map(existing_results_path):
    """
    Builds a dict: csv_file_stem -> metadata row
    from your existing results.csv, which already has scene/cvar/screen_pct.
    This avoids re-parsing metadata from scratch.
    """
    df = load_existing_results(existing_results_path)
    if df.empty:
        print("⚠️  No existing results.csv found — metadata will be missing.")
        return {}
    mapping = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(row['csv_file'])[0]
        mapping[stem] = row.to_dict()
    return mapping


def compute_medians(parquet_path, skip_frames=150):
    """
    Loads a parquet file, drops the first `skip_frames` rows (pass 1),
    and returns per-column medians across the remaining frames (passes 2+3).
    """
    df = pd.read_parquet(parquet_path)

    if len(df) <= skip_frames:
        print(f"  ⚠️  Only {len(df)} frames in {os.path.basename(parquet_path)}, "
              f"not enough to skip {skip_frames}. Skipping.")
        return None, 0

    df_analysis = df.iloc[skip_frames:]
    medians = {}
    for col in METRICS:
        if col in df_analysis.columns:
            medians[f'median_{col}'] = round(df_analysis[col].median(), 4)

    return medians, len(df_analysis)


def main(raw_dir, existing_results_path, out_path, skip_frames=150):
    parquet_files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))

    if not parquet_files:
        print(f"❌ No parquet files found in {raw_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files.")

    meta_map = build_parquet_to_meta_map(existing_results_path)

    rows = []
    for pq_path in parquet_files:
        stem = os.path.splitext(os.path.basename(pq_path))[0]
        print(f"\n  Processing: {stem}")

        medians, n_frames = compute_medians(pq_path, skip_frames=skip_frames)
        if medians is None:
            continue

        # Base row from existing metadata if available
        meta = meta_map.get(stem, {})
        row = {
            'csv_file':        stem + '.csv',
            'scene':           meta.get('scene', ''),
            'screen_pct':      meta.get('screen_pct', ''),
            'cvar_name':       meta.get('cvar_name', ''),
            'cvar_value':      meta.get('cvar_value', ''),
            'frames_analysed': n_frames,
        }
        row.update(medians)
        rows.append(row)
        print(f"    ✅ {n_frames} frames analysed | "
              + " | ".join(f"{k}={v}" for k, v in medians.items()))

    if not rows:
        print("❌ No data to write.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Written {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add median metrics from parquet files.")
    parser.add_argument("--raw_dir",          default="./raw",
                        help="Directory containing .parquet files")
    parser.add_argument("--existing_results", default="./results.csv",
                        help="Your existing results.csv (used to recover metadata)")
    parser.add_argument("--out",              default="./results_median.csv",
                        help="Output CSV path")
    parser.add_argument("--skip_frames",      type=int, default=150,
                        help="Frames to skip (pass 1). Default: 150")
    args = parser.parse_args()

    main(
        raw_dir=args.raw_dir,
        existing_results_path=args.existing_results,
        out_path=args.out,
        skip_frames=args.skip_frames,
    )
