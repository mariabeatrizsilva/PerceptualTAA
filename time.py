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
CSV_FRAMES     = 300

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

def save_results_to_master(csv_path, metrics):
    """
    Takes the output from get_key_metrics and appends it to a master results file.
    """
    results_path = os.path.join(os.getcwd(), "results.csv")
    write_header = not os.path.exists(results_path)

    # We map the function's output dictionary to your desired master CSV columns
    # We also use .get() to provide a fallback value if a metric is missing
    new_data = {
        'scene': SCENE_TO_RUN,
        'screen_pct': SCREEN_PCT,
        'taa_value': TAA_VALUE,  # Adding this to track your specific parameter
        'avg_frame_ms': metrics.get("Avg Frame Time (ms)", 0),
        'avg_gpu_ms': metrics.get("Avg GPU Time (ms)", 0),
        'avg_mem_mb': metrics.get("Avg VRAM Usage (MB)", 0),
        'peak_mem_mb': metrics.get("Peak VRAM Usage (MB)", 0),
        'frames_analysed': metrics.get("Total Frames", 0),
        'csv_file': os.path.basename(csv_path)
    }

    # Convert to DataFrame and append
    res_df = pd.DataFrame([new_data])
    res_df.to_csv(results_path, mode='a', index=False, header=write_header)

    print(f"\n✅ Results appended to: {results_path}")


# ── Parse the CSV ─────────────────────────────────────────────────────────────
def get_key_metrics(csv_path):
    """
    Reads the Unreal CSV and returns a dictionary of core performance stats
    including VRAM, FrameTime, and Thread timings.
    """
    try:
        # The exact column names from the Unreal CSV Profiler list
        needed = [
            'FrameTime', 
            'GameThreadTime', 
            'RenderThreadTime', 
            'GPUTime', 
            'RHIThreadTime', 
            'GPUMem/LocalUsedMB'
        ]
        
        # Read only the necessary columns to save memory and avoid "too many fields" errors
        df = pd.read_csv(
            csv_path, 
            usecols=lambda x: x in needed,
            low_memory=False
        )
        
        # 1. Strip metadata rows: Unreal CSVs often have header text rows at the start/end
        # 2. Convert to numeric: errors='coerce' turns non-numeric junk into NaN
        # 3. Dropna: Removes those NaN rows so calculations are accurate
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Helper to safely calculate stats even if a column was missing from the CSV
        def get_stats(col_name):
            if col_name in df.columns:
                return {
                    "avg": round(df[col_name].mean(), 2),
                    "max": round(df[col_name].max(), 2)
                }
            return {"avg": "N/A", "max": "N/A"}

        # Building the final metrics dictionary
        frame_stats = get_stats('FrameTime')
        game_stats = get_stats('GameThreadTime')
        render_stats = get_stats('RenderThreadTime')
        gpu_stats = get_stats('GPUTime')
        rhi_stats = get_stats('RHIThreadTime')
        vram_stats = get_stats('GPUMem/LocalUsedMB')

        metrics = {
            "Total Frames": len(df),
            "Avg Frame Time (ms)": frame_stats["avg"],
            "Max Frame Time (ms)": frame_stats["max"],
            "Avg Game Thread (ms)": game_stats["avg"],
            "Avg Render Thread (ms)": render_stats["avg"],
            "Avg GPU Time (ms)": gpu_stats["avg"],
            "Avg RHI Thread (ms)": rhi_stats["avg"],
            "Peak VRAM Usage (MB)": vram_stats["max"],
            "Avg VRAM Usage (MB)": vram_stats["avg"]
        }
        
        return metrics

    except Exception as e:
        return f"Error processing CSV: {e}"
metrics = get_key_metrics(csv_path=csv_path)

if isinstance(metrics, dict):
    save_results_to_master(csv_path, metrics)
else:
    print(metrics) # Print the error message if the function failed
