import yaml
import subprocess
import os
import time
import glob
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
YAML_FILE      = 'scenes.yaml'
SCENE_TO_RUN   = 'cubetest'
UE_EXE         = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"

SCREEN_PCT     = 100          # r.ScreenPercentage
TAA_CVAR       = "r.TemporalAACurrentFrameWeight"
TAA_VALUE      = 0.1

# Frames to capture. Needs to be larger than your sequence (150 frames) but
# the CSV profiler starts at level load, not at sequence start, so we pad
# generously. Frames captured during loading are filtered out in parsing.
CSV_FRAMES     = 500

# ── Output folder ────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
runtime_dir  = os.path.join(current_dir, "runtimes")
os.makedirs(runtime_dir, exist_ok=True)

# ── Load YAML ─────────────────────────────────────────────────────────────────
with open(YAML_FILE, 'r') as f:
    config = yaml.safe_load(f)

scene_info   = config['scenes'][SCENE_TO_RUN]
project_path = config['projects'][scene_info['project']]
level_path   = scene_info['level']

# Derive CSV directory from project path — UE always saves to Saved/Profiling/CSV
csv_dir = os.path.join(os.path.dirname(project_path), "Saved", "Profiling", "CSV")

# ── Build ExecCmds ────────────────────────────────────────────────────────────
# Note: csvprofile start is here, NOT in the Level Blueprint.
# The Level Blueprint only handles Begin Play → Play sequence.
# Load-time frames are filtered out during CSV parsing below.
exec_cmds = ",".join([
    "t.MaxFPS 30",
    "r.VSync 0",
    f"r.ScreenPercentage {SCREEN_PCT}",
    f"{TAA_CVAR} {TAA_VALUE}",
    "r.gpuCsvStatsEnabled 1",
    f"csvprofile frames={CSV_FRAMES}",
    "csvprofile start",
])

# ── Build command ─────────────────────────────────────────────────────────────
cmd = [
    UE_EXE,
    project_path,
    level_path,
    "-game",
    "-windowed", "-ResX=1920", "-ResY=1080",
    # "-deterministic",
    "-nosplash",
    "-novsync",
    "-log",
    f"-csvdir={runtime_dir}",
    f"-ExecCmds={exec_cmds}",
    "-ExitAfterCsvProfiling",
]

# ── Launch UE ─────────────────────────────────────────────────────────────────
print(f"Launching {SCENE_TO_RUN} | {TAA_CVAR}={TAA_VALUE} | ScreenPct={SCREEN_PCT}")
print(f"Command: {' '.join(cmd)}\n")
print(f"CSV will be read from: {csv_dir}\n")

# Record which CSVs exist before launch so we can find the new one after
existing_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))

subprocess.run(cmd)   # blocks until UE exits

# ── Find the new CSV ──────────────────────────────────────────────────────────
all_csvs    = set(glob.glob(os.path.join(csv_dir, "*.csv")))
new_csvs    = all_csvs - existing_csvs

if not new_csvs:
    print("ERROR: No new CSV file found. UE may have crashed or -ExitAfterCsvProfiling did not fire.")
    exit(1)

csv_path = max(new_csvs, key=os.path.getmtime)  # pick newest if somehow >1
print(f"CSV found: {csv_path}")

# ── Parse the CSV ─────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
print(f"\nColumns available: {list(df.columns)}")

# Find the TAA column — UE 5.6 may name it slightly differently.
# We search case-insensitively for 'temporal' to be safe.
taa_col = next(
    (c for c in df.columns if 'temporal' in c.lower()),
    None
)

if taa_col is None:
    print("WARNING: No TemporalAA column found in CSV. Check column names above.")
    print("The profiler is working but TAA may be named differently in UE 5.6.")
    exit(1)

print(f"TAA column found: '{taa_col}'")

# ── Filter out load-time frames ───────────────────────────────────────────────
# During level load, TAA ms will be 0 or very spiky. We drop the first N rows
# to exclude loading frames and keep only steady-state sequence frames.
# Adjust SKIP_FRAMES if your load is faster or slower than expected.
SKIP_FRAMES = 50   # conservative — increase if you still see spiky values
taa_values  = df[taa_col].dropna()
taa_steady  = taa_values.iloc[SKIP_FRAMES:]

taa_mean = taa_steady.mean()
taa_std  = taa_steady.std()
taa_min  = taa_steady.min()
taa_max  = taa_steady.max()

print(f"\nResults for {SCENE_TO_RUN} | {TAA_CVAR}={TAA_VALUE} | ScreenPct={SCREEN_PCT}")
print(f"  Frames analysed : {len(taa_steady)}")
print(f"  TAA mean        : {taa_mean:.4f} ms")
print(f"  TAA std         : {taa_std:.4f} ms")
print(f"  TAA min/max     : {taa_min:.4f} / {taa_max:.4f} ms")

# ── Append to master results file ────────────────────────────────────────────
results_path = os.path.join(current_dir, "results.csv")
write_header = not os.path.exists(results_path)

with open(results_path, 'a') as f:
    if write_header:
        f.write("scene,screen_pct,param_name,param_value,taa_ms_mean,taa_ms_std,taa_ms_min,taa_ms_max,frames_analysed,csv_file\n")
    f.write(
        f"{SCENE_TO_RUN},{SCREEN_PCT},{TAA_CVAR},{TAA_VALUE},"
        f"{taa_mean:.4f},{taa_std:.4f},{taa_min:.4f},{taa_max:.4f},"
        f"{len(taa_steady)},{os.path.basename(csv_path)}\n"
    )

print(f"\nAppended to {results_path}")