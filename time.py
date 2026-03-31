import yaml
import subprocess
import os
import time
import glob
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
YAML_FILE      = 'scenes.yaml'
SCENE_TO_RUN   = 'abandoned-demo'
UE_EXE         = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
RESOLUTIONS  = [100, 87, 71, 50]  # r.ScreenPercentage
CSV_FRAMES     = 300

TAA_VARIATIONS = {
    # Group 1: Alpha Weight (r.TemporalAACurrentFrameWeight)
    "alpha_weight": {
        "cvar": "r.TemporalAACurrentFrameWeight",
        "values": [0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.5, 0.6,0.7,0.8,0.9, 1.0]
    },
    # Group 2: TAA Num Samples (r.TemporalAASamples)
    "num_samples": {
        "cvar": "r.TemporalAASamples",
        "values": [4, 8, 16, 32, 64],
    },
    # Group 3: Filter Size (r.TemporalAAFilterSize)
    "filter_size": {
        "cvar": "r.TemporalAAFilterSize",
        "values": [0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 1.0],
    },
    # Group 4: History Screen Percentage (r.TemporalAA.HistoryScreenPercentage)
    "hist_percent": {
        "cvar": "r.TemporalAA.HistoryScreenPercentage",
        "values": [100, 125, 150, 200],
    }
}

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

def save_results_to_master(csv_path, metrics, current_res, current_cvar, current_val):
    """
    Appends the parsed metrics plus the current loop parameters to results.csv.
    """
    results_path = os.path.join(os.getcwd(), "results.csv")
    write_header = not os.path.exists(results_path)

    # Map all thread timings and loop parameters
    new_data = {
        'scene': SCENE_TO_RUN,
        'screen_pct': current_res,
        'cvar_name': current_cvar,
        'cvar_value': current_val,
        'avg_frame_ms': metrics.get("Avg Frame Time (ms)", 0),
        'avg_gpu_ms': metrics.get("Avg GPU Time (ms)", 0),
        'avg_game_ms': metrics.get("Avg Game Thread (ms)", 0),
        'avg_render_ms': metrics.get("Avg Render Thread (ms)", 0),
        'avg_rhi_ms': metrics.get("Avg RHI Thread (ms)", 0),
        'avg_mem_mb': metrics.get("Avg VRAM Usage (MB)", 0),
        'peak_mem_mb': metrics.get("Peak VRAM Usage (MB)", 0),
        'frames_analysed': metrics.get("Total Frames", 0),
        'csv_file': os.path.basename(csv_path)
    }

    res_df = pd.DataFrame([new_data])
    res_df.to_csv(results_path, mode='a', index=False, header=write_header)
    print(f"✅ Logged: Res={current_res}% | {current_cvar}={current_val}")


# ── Parse the CSV ─────────────────────────────────────────────────────────────
def get_key_metrics(csv_path):
    """
    Reads the Unreal CSV and returns a dictionary of core performance stats.
    """
    try:
        needed = [
            'FrameTime', 
            'GameThreadTime', 
            'RenderThreadTime', 
            'GPUTime', 
            'RHIThreadTime', 
            'GPUMem/LocalUsedMB'
        ]
        
        df = pd.read_csv(csv_path, usecols=lambda x: x in needed, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        def get_stats(col_name):
            if col_name in df.columns:
                return {
                    "avg": round(df[col_name].mean(), 4),
                    "max": round(df[col_name].max(), 4)
                }
            return {"avg": 0, "max": 0}

        vram = get_stats('GPUMem/LocalUsedMB')
        
        return {
            "Total Frames": len(df),
            "Avg Frame Time (ms)": get_stats('FrameTime')["avg"],
            "Avg Game Thread (ms)": get_stats('GameThreadTime')["avg"],
            "Avg Render Thread (ms)": get_stats('RenderThreadTime')["avg"],
            "Avg GPU Time (ms)": get_stats('GPUTime')["avg"],
            "Avg RHI Thread (ms)": get_stats('RHIThreadTime')["avg"],
            "Peak VRAM Usage (MB)": vram["max"],
            "Avg VRAM Usage (MB)": vram["avg"]
        }

    except Exception as e:
        print(f"⚠️ Error parsing {csv_path}: {e}")
        return None

# ── The Master Loop ──────────────────────────────────────────────────────────
for screen_pct in RESOLUTIONS:
    for group_name, var_config in TAA_VARIATIONS.items():
        cvar_name = var_config["cvar"]
        
        for val in var_config["values"]:
            print(f"\n{'='*70}")
            print(f"🎬 RUNNING: {SCENE_TO_RUN} | Res: {screen_pct}% | {cvar_name}: {val}")
            print(f"{'='*70}")

            # 1. Build ExecCmds dynamically
            exec_cmds = ",".join([
                "t.MaxFPS 30",
                "r.VSync 0",
                f"r.ScreenPercentage {screen_pct}",
                f"{cvar_name} {val}",
                "r.gpuCsvStatsEnabled 1",
                f"csvprofile frames={CSV_FRAMES}",
                "csvprofile start",
            ])

            # 2. Build the full command
            cmd = [
                UE_EXE, project_path, level_path,
                "-game", "-windowed", "-ResX=1920", "-ResY=1080",
                "-nosplash", "-novsync", "-log",
                f"-csvdir={runtime_dir}",
                f"-ExecCmds={exec_cmds}",
                "-ExitAfterCsvProfiling",
            ]

            # 3. Track existing CSVs to catch the new one
            existing_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))

            # 4. Launch and Wait
            subprocess.run(cmd)

            # 5. Find the new CSV
            all_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))
            new_csvs = all_csvs - existing_csvs

            if new_csvs:
                csv_path = max(new_csvs, key=os.path.getmtime)
                print(f"✅ CSV Captured: {os.path.basename(csv_path)}")

                # 6. Parse and Append to Results
                metrics = get_key_metrics(csv_path) # Using the function from earlier
                if isinstance(metrics, dict):
                    # Injecting the specific CVar info into the results row
                    save_results_to_master(csv_path, metrics, screen_pct, cvar_name, val)
            else:
                print(f"❌ ERROR: No CSV generated for {cvar_name}={val}")

            # Small cooldown to let OS cleanup memory/file locks
            time.sleep(2)

print("\n🚀 ALL VARIATIONS COMPLETE.")
