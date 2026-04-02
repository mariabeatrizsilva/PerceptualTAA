import yaml
import subprocess
import os
import time
import glob
import pandas as pd
import re

# ── Configuration ────────────────────────────────────────────────────────────
YAML_FILE      = '../scenes.yaml'
SCENE_TO_RUN   = 'cubetest'
UE_EXE         = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
RESOLUTIONS    = [100, 87, 71, 50]
CSV_FRAMES     = 450
PROCESS_PROFILES = True  # set to False to skip parsing/saving

METRICS = ['FrameTime', 'GameThreadTime', 'RenderThreadTime', 'GPUTime', 'RHIThreadTime', 'GPUMem/LocalUsedMB', 'GPU/TAA']

TAA_VARIATIONS = {
    "alpha_weight": {
        "cvar": "r.TemporalAACurrentFrameWeight",
        "values": [0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },
    "num_samples": {
        "cvar": "r.TemporalAASamples",
        "values": [4, 8, 16, 32, 64],
    },
    "filter_size": {
        "cvar": "r.TemporalAAFilterSize",
        "values": [0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 1.0],
    },
    "hist_percent": {
        "cvar": "r.TemporalAA.HistoryScreenPercentage",
        "values": [100, 125, 150, 200],
    }
}

# ── Output folders ────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
runtime_dir  = os.path.join(current_dir, "runtimes")
raw_dir      = os.path.join(current_dir, "raw")
os.makedirs(runtime_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)

# ── Load YAML ─────────────────────────────────────────────────────────────────
with open(YAML_FILE, 'r') as f:
    config = yaml.safe_load(f)

scene_info   = config['scenes'][SCENE_TO_RUN]
project_path = config['projects'][scene_info['project']]
level_path   = scene_info['level']

csv_dir = os.path.join(os.path.dirname(project_path), "Saved", "Profiling", "CSV")

# ── Parse a single profile ────────────────────────────────────────────────────
def parse_profile(csv_path, scene_name):
    """
    Reads a single Unreal CSV profile and returns:
      - meta: dict of run parameters from [HasHeaderRowAtEnd]
      - df:   DataFrame of raw per-frame metric values
    Returns (None, None) if parsing fails.
    """
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # ── Metadata from last line ───────────────────────────────────────────
        meta = {'csv_file': os.path.basename(csv_path), 'scene': scene_name}

        last_line = lines[-1]
        if last_line.startswith('[HasHeaderRowAtEnd]'):
            pairs = re.findall(r'\[([^\]]+)\],([^,\[]*)', last_line)
            raw_meta = {k: v.strip() for k, v in pairs}

            cmdline = raw_meta.get('commandline', '')

            # Screen percentage
            sp = re.search(r'r\.ScreenPercentage\s+(\d+)', cmdline)
            if sp:
                meta['screen_pct'] = int(sp.group(1))

            # Test cvar (exclude system ones)
            exec_match = re.search(r'ExecCmds="([^"]+)"', cmdline)
            if exec_match:
                exec_cmds = exec_match.group(1)
                system_cvars = {'t.MaxFPS', 'r.VSync', 'r.ScreenPercentage',
                                'r.gpuCsvStatsEnabled', 'csvprofile'}
                found = re.findall(r'(r\.[A-Za-z.]+)\s+([\d.]+)', exec_cmds)
                test_cvars = [(k, v) for k, v in found if k not in system_cvars]
                if test_cvars:
                    meta['cvar_name']  = test_cvars[-1][0]
                    meta['cvar_value'] = float(test_cvars[-1][1])

        # ── Per-frame data by column position ────────────────────────────────
        header = lines[0].strip().split(',')
        col_idx = {name: header.index(name) for name in METRICS if name in header}

        rows = []
        for line in lines[1:]:
            if line.startswith('[') or line.startswith('EVENTS'):
                continue
            fields = line.strip().split(',')
            row = {}
            for name, idx in col_idx.items():
                try:
                    row[name] = float(fields[idx])
                except (IndexError, ValueError):
                    pass
            if len(row) == len(col_idx):
                rows.append(row)

        df = pd.DataFrame(rows)
        return meta, df

    except Exception as e:
        print(f"⚠️  Error parsing {csv_path}: {e}")
        return None, None


# ── Process a single profile and append to results.csv ───────────────────────
def process_profile(csv_path, scene_name, screen_pct, cvar_name, cvar_value):
    parquet_name = os.path.splitext(os.path.basename(csv_path))[0] + '.parquet'
    parquet_path = os.path.join(raw_dir, parquet_name)

    if os.path.exists(parquet_path):
        print(f"  ⏭️  Already processed, skipping: {parquet_name}")
        return

    meta, df = parse_profile(csv_path, scene_name)

    if meta is None or df is None or df.empty:
        print(f"  ⚠️  No data extracted from {os.path.basename(csv_path)}, skipping.")
        return

    df.to_parquet(parquet_path, index=False)

    df_analysis = df.iloc[150:]

    row = {
        'scene':           scene_name,
        'screen_pct':      screen_pct,
        'cvar_name':       cvar_name,
        'cvar_value':      cvar_value,
        'frames_analysed': len(df_analysis),
        'csv_file':        os.path.basename(csv_path),
    }
    for col in METRICS:
        if col in df_analysis.columns:
            row[f'avg_{col}'] = round(df_analysis[col].mean(), 4)
            row[f'p95_{col}'] = round(df_analysis[col].quantile(0.95), 4)
            row[f'max_{col}'] = round(df_analysis[col].max(), 4)

    results_path = os.path.join(current_dir, 'results.csv')
    write_header = not os.path.exists(results_path)
    pd.DataFrame([row]).to_csv(results_path, mode='a', index=False, header=write_header)

    print(f"  ✅ Processed: {len(df)} frames | Analyzed: {len(df_analysis)} frames | {cvar_name}={cvar_value} @ {screen_pct}%")

# ── The Master Loop ──────────────────────────────────────────────────────────
for screen_pct in RESOLUTIONS:
    for group_name, var_config in TAA_VARIATIONS.items():
        cvar_name = var_config["cvar"]

        for val in var_config["values"]:
            print(f"\n{'='*70}")
            print(f"🎬 RUNNING: {SCENE_TO_RUN} | Res: {screen_pct}% | {cvar_name}: {val}")
            print(f"{'='*70}")

            exec_cmds = ",".join([
                "r.FixedFrameRate 30",          # Lock simulation delta time 
                "t.MaxFPS 0",                   # Remove real-time throttle
                "r.VSync 0",
                f"r.ScreenPercentage {screen_pct}",
                f"{cvar_name} {val}",
                "r.gpuCsvStatsEnabled 1",
                "Sleep 5",                      # Crucial: let TAA history buffers "warm up"
                f"csvprofile frames={CSV_FRAMES}", #  can i put delay=1 after {CSV_FRAMES}
                "csvprofile start",
            ])

            # exec_cmds = ",".join([
            #     "t.MaxFPS 30",
            #     "r.VSync 0",
            #     f"r.ScreenPercentage {screen_pct}",
            #     f"{cvar_name} {val}",
            #     "r.gpuCsvStatsEnabled 1",
            #     f"csvprofile frames={CSV_FRAMES}",
            #     "csvprofile start",
            # ])

            cmd = [
                UE_EXE, project_path, level_path,
                "-game", "-windowed", "-ResX=1920", "-ResY=1080",
                "-nosplash", "-novsync", "-log",
                # ADDED THESE
                "-Benching",              # Forces deterministic frame stepping
                "-NoTextureStreaming",     # Prevents IO spikes during the test
                "-NoSound",                # Disables audio thread noise
                "-NoVerifyGC",             # Disables expensive GC checks
                # END OF WHAT I ADDED
                f"-csvdir={runtime_dir}",
                f"-ExecCmds={exec_cmds}",
                "-ExitAfterCsvProfiling",
            ]

            existing_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))
            subprocess.run(cmd)
            all_csvs = set(glob.glob(os.path.join(csv_dir, "*.csv")))
            new_csvs = all_csvs - existing_csvs

            if new_csvs:
                csv_path = max(new_csvs, key=os.path.getmtime)
                print(f"  📄 CSV captured: {os.path.basename(csv_path)}")

            if PROCESS_PROFILES:
                process_profile(csv_path, SCENE_TO_RUN, screen_pct, cvar_name, val)
            else:
                print(f"  ❌ ERROR: No CSV generated for {cvar_name}={val}")

            time.sleep(2)

print("\n🚀 ALL VARIATIONS COMPLETE.")