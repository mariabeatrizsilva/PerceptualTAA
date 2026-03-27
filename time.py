import yaml
import subprocess
import os
import glob
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
YAML_FILE      = 'scenes.yaml'
SCENE_TO_RUN   = 'cubetest'
UE_EXE         = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"

SCREEN_PCT     = 100
TAA_CVAR       = "r.TemporalAACurrentFrameWeight"
TAA_VALUE      = 0.1

CSV_FRAMES     = 500
SEQUENCE_LEN   = 150   # exact number of frames in your sequence

# ── Load YAML ─────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))

with open(YAML_FILE, 'r') as f:
    config = yaml.safe_load(f)

scene_info   = config['scenes'][SCENE_TO_RUN]
project_key  = scene_info['project']
project_info = config['projects'][project_key]
project_path = project_info['uproject']
csv_dir      = project_info['csv_dir']
level_path   = scene_info['level']

# ── Build ExecCmds ────────────────────────────────────────────────────────────
exec_cmds = " | ".join([
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
    "-nosplash",
    "-novsync",
    "-log",
    f"-ExecCmds={exec_cmds}",
    "-ExitAfterCsvProfiling",
]

# ── Launch UE ─────────────────────────────────────────────────────────────────
print(f"Launching {SCENE_TO_RUN} | {TAA_CVAR}={TAA_VALUE} | ScreenPct={SCREEN_PCT}")
print(f"CSV will be read from: {csv_dir}\n")

existing_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))

subprocess.run(cmd)  # blocks until UE exits

# ── Find the new CSV ──────────────────────────────────────────────────────────
all_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))
new_csvs = all_csvs - existing_csvs

if not new_csvs:
    print(f"ERROR: No new CSV file found in {csv_dir}")
    print("Check that UE launched correctly and the CSV profiler fired.")
    exit(1)

csv_path = max(new_csvs, key=os.path.getmtime)
print(f"CSV found: {csv_path}")

# ── Parse the CSV ─────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
print(f"\nColumns available: {list(df.columns)}")

taa_col = next(
    (c for c in df.columns if 'temporal' in c.lower()),
    None
)

if taa_col is None:
    print("WARNING: No TemporalAA column found in CSV. Check column names above.")
    print("The profiler is working but TAA may be named differently in UE 5.6.")
    exit(1)

print(f"TAA column found: '{taa_col}'")

# ── Select frames to analyse ──────────────────────────────────────────────────
# The Blueprint writes a csv.Event "SequenceStart" marker at the start of the
# second loop. We use that to find exactly which frames to measure.
# If the marker isn't found we fall back to using the last SEQUENCE_LEN frames.
taa_values = df[taa_col].dropna().reset_index(drop=True)

event_col = next(
    (c for c in df.columns if 'event' in c.lower()),
    None
)

if event_col is not None:
    marker_rows = df[df[event_col].astype(str).str.contains('SequenceStart', na=False)]
    if not marker_rows.empty:
        start_idx = marker_rows.index[0]
        print(f"SequenceStart marker found at frame {start_idx} — using next {SEQUENCE_LEN} frames")
        taa_steady = taa_values.iloc[start_idx:start_idx + SEQUENCE_LEN]
    else:
        print(f"No SequenceStart marker found — falling back to last {SEQUENCE_LEN} frames")
        taa_steady = taa_values.iloc[-SEQUENCE_LEN:]
else:
    print(f"No event column found — falling back to last {SEQUENCE_LEN} frames")
    taa_steady = taa_values.iloc[-SEQUENCE_LEN:]

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