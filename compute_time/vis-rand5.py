import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import random
import glob

# --- 1. Load Metadata ---
metadata_lookup = {}
csv_files = ['results-10reps.csv', 'results-4reps.csv']

print("--- Step 1: Loading CSV Metadata ---")
for csv_file in csv_files:
    if os.path.exists(csv_file):
        meta_df = pd.read_csv(csv_file)
        print(f"Found {csv_file} with {len(meta_df)} rows.")
        for _, row in meta_df.iterrows():
            # Standardize the name: ensure it's just the filename, not a path
            p_name = os.path.basename(row['csv_file']).replace('.csv', '.parquet')
            metadata_lookup[p_name] = {
                'cvar': row.get('cvar_name', 'N/A'),
                'val': row.get('cvar_value', 'N/A'),
                'res': f"{row.get('screen_pct', 'N/A')}%",
                'source': csv_file
            }
    else:
        print(f"⚠️ Warning: {csv_file} not found in current directory.")

# --- 2. Select Files to Plot ---
RAW_DIR = 'raw'
all_parquets = list(metadata_lookup.keys())
sampled_files = []

print("\n--- Step 2: Sampling Files ---")
# Filter to only files that actually exist in the 'raw' folder
existing_parquets = [p for p in all_parquets if os.path.exists(os.path.join(RAW_DIR, p))]

if not existing_parquets:
    print(f"❌ ERROR: No matching .parquet files found in '{RAW_DIR}' folder!")
    print(f"Check: Are your parquets named exactly like the CSV entries?")
else:
    # Try to take 5 from 10reps and 5 from 4reps if possible
    reps10 = [p for p in existing_parquets if '10reps' in metadata_lookup[p]['source']]
    reps4 = [p for p in existing_parquets if '4reps' in metadata_lookup[p]['source']]
    
    sampled_files = random.sample(reps10, min(len(reps10), 5)) + \
                    random.sample(reps4, min(len(reps4), 5))
    print(f"Successfully sampled {len(sampled_files)} files to plot.")

# --- 3. Plotting Function ---
def find_stable_frame(series, window=10, threshold=0.05):
    rolling = series.rolling(window=window).mean().dropna()
    steady_state = series.iloc[len(series)//2:].mean()
    stable_idx = rolling[abs(rolling - steady_state) / steady_state < threshold].index
    return stable_idx[0] if len(stable_idx) > 0 else None

def plot_with_reps(parquet_path, rep_length=150):
    df = pd.read_parquet(parquet_path)
    filename = os.path.basename(parquet_path)
    info = metadata_lookup.get(filename, {'cvar': 'Unknown', 'val': '?', 'res': '?', 'source': '?'})

    metrics = [('FrameTime', 'Frame (ms)'), ('GPUTime', 'GPU (ms)'), 
               ('GPU/TAA', 'TAA (ms)'), ('GPUMem/LocalUsedMB', 'VRAM (MB)')]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    
    # Metadata Text Box
    info_text = f"Param: {info['cvar']}\nValue: {info['val']}\nRes: {info['res']}\nFile: {filename}"
    fig.text(0.75, 0.92, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    for ax, (col, label) in zip(axes, metrics):
        if col in df.columns:
            print(f"Plotting {col}: Mean value is {df[col].mean()}")
            ax.plot(df[col], linewidth=0.7, color='#2a7db5')
            for m in range(0, len(df), rep_length):
                ax.axvline(x=m, color='red', linestyle='--', linewidth=0.6, alpha=0.3)
            
            stable = find_stable_frame(df[col])
            if stable:
                ax.axvline(x=stable, color='green', linestyle=':', label=f'Stable@{stable}')
        ax.set_ylabel(label)

    plt.tight_layout(rect=[0, 0, 0.73, 0.95])
    return fig

# --- 4. Final Run ---
os.makedirs('rep_trends', exist_ok=True)
print("\n--- Step 3: Generating Plots ---")
for p_name in sampled_files:
    full_path = os.path.join(RAW_DIR, p_name)
    print(f"Drawing: {p_name}...")
    fig = plot_with_reps(full_path)
    fig.savefig(f'rep_trends/{p_name.replace(".parquet", ".png")}')
    plt.close(fig)

print(f"\n✅ Done! Files should be in: {os.path.abspath('rep_trends')}")