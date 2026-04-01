"""
plot_video_stats.py
Generates visualizations for video parameterization stats (SI, TI, CF, TP, MV, DTP).
Works with whatever columns are present — skips MV/DTP if not yet computed.

Usage:
    python plot_video_stats.py --csv video_stats.csv

Outputs (saved to ./plots/):
    1. table.png              — styled table of all videos and scores
    2. histograms.png         — one histogram per stat (log x-axis)
    3. scatter_matrix.png     — pairplot of all stats
    4. radar.png              — spider chart, one polygon per video
    5. dendrogram.png         — hierarchical clustering
    6. clustermap.png         — heatmap + dendrogram combined
    7. flow_<video>.png       — optical flow HSV visualization (if ./flow/ exists)

Dependencies:
    pip install pandas matplotlib scipy seaborn scikit-learn opencv-python numpy
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import cv2
from pathlib import Path

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

OUTPUT_DIR = Path("./plots")
FLOW_DIR   = Path("./flow")
VIDEO_DIR  = Path("./16SSAA-vids")

ALL_FEATURES = ["SI", "TI", "CF", "TP", "MV", "DTP"]
PARAM_LABELS = {
    "SI":  "Spatial Information",
    "TI":  "Temporal Information",
    "CF":  "Colorfulness",
    "TP":  "Texture Parameter",
    "MV":  "Motion Vector",
    "DTP": "Dynamic Texture",
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def short_name(filename):
    return os.path.splitext(filename)[0]

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Only use features that are present and fully populated
    features = [f for f in ALL_FEATURES if f in df.columns and df[f].notna().all() and (df[f] != "").all()]
    df = df[df[features].notna().all(axis=1)].copy()
    df["label"] = df["filename"].apply(short_name)
    print(f"  Loaded {len(df)} videos with features: {features}")
    return df, features

def save(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ──────────────────────────────────────────────
# 1. Table
# ──────────────────────────────────────────────

def plot_table(df, features):
    n_rows = len(df)
    n_cols = len(features) + 1  # +1 for filename

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 2), max(6, n_rows * 0.45 + 1.5)))
    ax.axis("off")

    col_labels = ["Video"] + [PARAM_LABELS[f] for f in features]
    table_data = [[row["label"]] + [f"{row[f]:.4f}" for f in features]
                  for _, row in df.iterrows()]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(n_cols):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            table[i, j].set_facecolor("#f2f2f2" if i % 2 == 0 else "white")
            if j == 0:
                table[i, j].set_text_props(fontweight="bold", ha="left")

    ax.set_title("Video Parameterization Statistics", fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────
# 2. Histograms (log x-axis, one per stat)
# ──────────────────────────────────────────────

def plot_histograms(df, features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns=features, index=df.index)

    n = len(features)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(df)))

    for i, feat in enumerate(features):
        ax = axes[i]
        values = scaled_df[feat].values
        labels = df["label"].values

        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_labels = labels[order]
        sorted_colors = colors[order]

        ax.barh(sorted_labels, sorted_vals, color=sorted_colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")  # zero line
        ax.set_xlabel("z-score", fontsize=9)
        ax.set_title(PARAM_LABELS[feat], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Video Statistics (z-scored)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
# ──────────────────────────────────────────────
# 3. Scatter matrix (pairplot)
# ──────────────────────────────────────────────

def plot_scatter_matrix(df, features):
    plot_df = df[features].copy()
    plot_df.index = df["label"]

    g = sns.pairplot(
        plot_df.reset_index(),
        vars=features,
        diag_kind="kde",
        plot_kws={"alpha": 0.7, "s": 40, "color": "#378ADD"},
        diag_kws={"color": "#378ADD", "fill": True, "alpha": 0.4},
    )
    for ax in g.axes.flatten():
        if ax is not None:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel in PARAM_LABELS:
                ax.set_xlabel(PARAM_LABELS[xlabel], fontsize=8)
            if ylabel in PARAM_LABELS:
                ax.set_ylabel(PARAM_LABELS[ylabel], fontsize=8)

    g.figure.suptitle("Scatter Matrix of Video Statistics", y=1.01, fontsize=13, fontweight="bold")
    return g.figure

# ──────────────────────────────────────────────
# 4. Radar / spider chart
# ──────────────────────────────────────────────

def plot_radar(df, features):
    # Normalize each feature to [0, 1] for comparability
    normed = df[features].copy()
    for f in features:
        mn, mx = normed[f].min(), normed[f].max()
        normed[f] = (normed[f] - mn) / (mx - mn + 1e-12)

    n_feat = len(features)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    labels = [PARAM_LABELS[f] for f in features]

    n_videos = len(df)
    colors = cm.tab20(np.linspace(0, 1, n_videos))

    ncols = 4
    nrows = int(np.ceil(n_videos / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3),
                              subplot_kw=dict(polar=True))
    axes = np.array(axes).flatten()

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        values = [normed.loc[row.name, f] for f in features]
        values += values[:1]

        ax.plot(angles, values, color=colors[idx], linewidth=1.5)
        ax.fill(angles, values, color=colors[idx], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=0)
        ax.set_title(df.loc[row.name, "label"], fontsize=7, pad=8, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Video Parameter Profiles (normalised)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────
# 5. Dendrogram
# ──────────────────────────────────────────────

def plot_dendrogram(df, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    Z = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, labels=df["label"].values, ax=ax,
               leaf_rotation=45, leaf_font_size=9,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────
# 6. Clustermap
# ──────────────────────────────────────────────

def plot_clustermap(df, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(X, columns=[PARAM_LABELS[f] for f in features], index=df["label"].values)

    g = sns.clustermap(
        scaled_df,
        method="ward",
        metric="euclidean",
        cmap="RdYlBu_r",
        figsize=(9, 10),
        linewidths=0.4,
        yticklabels=True,
        xticklabels=True,
        cbar_kws={"label": "z-score"},
    )
    g.fig.suptitle("Video Clustermap (z-scored features)", y=1.01, fontsize=12, fontweight="bold")
    return g.fig

# ──────────────────────────────────────────────
# 7. Optical flow visualization
# ──────────────────────────────────────────────

def visualize_flow(flow_path, video_path):
    """
    Takes the first non-zero flow frame, encodes as HSV:
      hue    = direction of motion
      value  = magnitude of motion
    Returns an RGB image.
    """
    motion_vid = np.load(flow_path)  # (T, 2, H, W)

    # Find first frame with non-zero flow (frame 0 is always zero)
    frame_idx = 1
    for i in range(1, motion_vid.shape[0]):
        if np.any(motion_vid[i] != 0):
            frame_idx = i
            break

    dx = motion_vid[frame_idx, 0, :, :]
    dy = motion_vid[frame_idx, 1, :, :]

    mag, ang = cv2.cartToPolar(dx, dy)

    hsv = np.zeros((dx.shape[0], dx.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # full saturation
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)   # hue = direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # value = magnitude

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_all_flows():
    """Plot flow visualizations for all available .npy files in a grid."""
    flow_files = sorted(FLOW_DIR.glob("*_flow.npy"))
    if not flow_files:
        print("  No flow files found, skipping optical flow visualization.")
        return None

    n = len(flow_files)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for idx, flow_path in enumerate(flow_files):
        stem = flow_path.stem.replace("_flow", "")
        video_path = None
        for ext in [".mp4", ".avi", ".mov"]:
            candidate = VIDEO_DIR / f"{stem}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        ax = axes[idx]
        try:
            rgb = visualize_flow(flow_path, video_path)
            ax.imshow(rgb)
            ax.set_title(stem, fontsize=7, fontweight="bold")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        ax.axis("off")

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    # Legend note
    fig.text(0.5, 0.01, "Hue = direction of motion   |   Brightness = magnitude of motion",
             ha="center", fontsize=9, style="italic", color="#555")
    fig.suptitle("Optical Flow Visualization (Farneback)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(csv_path):
    print(f"Loading {csv_path} ...")
    df, features = load_data(csv_path)

    print("\nGenerating plots ...")
    save(plot_table(df, features),           "table.png")
    save(plot_histograms(df, features),      "histograms.png")
    save(plot_scatter_matrix(df, features),  "scatter_matrix.png")
    save(plot_radar(df, features),           "radar.png")
    save(plot_dendrogram(df, features),      "dendrogram.png")
    save(plot_clustermap(df, features),      "clustermap.png")

    if FLOW_DIR.exists():
        flow_fig = plot_all_flows()
        if flow_fig:
            save(flow_fig, "optical_flow.png")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="video_stats.csv")
    args = parser.parse_args()
    main(args.csv)