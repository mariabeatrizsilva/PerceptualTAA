import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import glob 

def find_stable_frame(series, window=10, threshold=0.05):
    """
    Returns the first frame where the rolling mean stays within
    `threshold` (fractional) of the overall steady-state mean.
    """
    rolling = series.rolling(window=window).mean().dropna()
    steady_state = series.iloc[len(series)//2:].mean()  # use second half as reference
    stable_idx = rolling[abs(rolling - steady_state) / steady_state < threshold].index
    return stable_idx[0] if len(stable_idx) > 0 else None


def plot_with_stabilisation(parquet_path, loop_length=150, window=10, threshold=0.05):
    df = pd.read_parquet(parquet_path)
    df = df.reset_index(drop=True)
    df.index.name = 'frame'

    metrics = [
        ('FrameTime',          'Frame Time (ms)'),
        ('GPUTime',            'GPU Time (ms)'),
        ('RenderThreadTime',   'Render Thread (ms)'),
        ('GameThreadTime',     'Game Thread (ms)'),
        ('RHIThreadTime',      'RHI Thread (ms)'),
        ('GPUMem/LocalUsedMB', 'VRAM (MB)'),
        ('GPU/TAA',            'TAA GPU cost (ms)'),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 18), sharex=True)

    stable_frames = {}

    for ax, (col, label) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_ylabel(label, fontsize=9)
            ax.text(0.5, 0.5, 'not in data', transform=ax.transAxes, ha='center')
            continue

        ax.plot(df[col], linewidth=0.8, color='steelblue')

        # loop boundary
        ax.axvline(x=loop_length, color='red', linestyle='--', linewidth=1, label='loop boundary')

        # find stabilisation point
        stable = find_stable_frame(df[col], window=window, threshold=threshold)
        if stable is not None:
            stable_frames[col] = stable
            ax.axvline(x=stable, color='green', linestyle=':', linewidth=1.5, label=f'stable @ frame {stable}')
            ax.legend(fontsize=7)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    # overall recommendation — take the latest stabilisation point across all metrics
    if stable_frames:
        recommended = max(stable_frames.values())
        fig.suptitle(f'per-frame timings — recommended skip: first {recommended} frames', fontsize=12)
        print(f"\nStabilisation frames per metric:")
        for col, frame in sorted(stable_frames.items(), key=lambda x: x[1]):
            print(f"  {col:<25} stable @ frame {frame}")
        print(f"\n  → recommended WARMUP_FRAMES = {recommended}")
    else:
        fig.suptitle('per-frame timings')

    axes[-1].set_xlabel('frame')
    plt.tight_layout()
    # plt.show()

    return fig,stable_frames


# stable = plot_with_stabilisation('../raw/Profile(20260401_165247).parquet', loop_length=150)

os.makedirs('stable_plots', exist_ok=True)

for path in sorted(glob.glob('raw/*.parquet')):
    name = os.path.splitext(os.path.basename(path))[0]
    fig,stable = plot_with_stabilisation(path, loop_length=150)
    fig.savefig(f'stable_plots/{name}.png', dpi=150)
    plt.close(fig)