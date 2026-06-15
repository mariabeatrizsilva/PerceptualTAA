"""
plot_factorial.py
-----------------
Generates diagnostic and publication-quality plots for the full factorial
TAA parameter sweep independence analysis.

Place in tests/ alongside analyze_independence.py.

USAGE:
  python plot_factorial.py --scene oldmine
  python plot_factorial.py --scene oldmine village-day subway-turn --pool

OUTPUT (saved to tests/{scene}/plots/):
  01_main_effects.png       — score vs each parameter (averaged over others)
  02_interaction_aw_hp.png  — key interaction: aw × hp lines
  03_eta_sq_bar.png         — effect sizes for all ANOVA terms
  04_heatmap_aw_hp.png      — 2D heatmap of aw vs hp (averaged over ns, fs)

  If --pool: also saved to tests/plots_pooled/
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
TESTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_DIR)

PARAM_COLS  = ["alpha_weight", "num_samples", "filter_size", "hist_percent"]
PARAM_LABELS = {
    "alpha_weight": "Alpha Weight (aw)",
    "num_samples":  "Num Samples (ns)",
    "filter_size":  "Filter Size (fs)",
    "hist_percent": "History % (hp)",
}
VALID_SCORE_MIN = 0.0
VALID_SCORE_MAX = 100.0

# Publication style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("colorblind")

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_scene(scene_name):
    path = os.path.join(TESTS_DIR, scene_name, "scores.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"scores.json not found for '{scene_name}'. Run run_factorial_metrics.py first.")
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for combo, data in raw.items():
        row = {"combo": combo, "score": data["score"], "scene": scene_name}
        row.update(data["params"])
        rows.append(row)
    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df[(df["score"] >= VALID_SCORE_MIN) & (df["score"] <= VALID_SCORE_MAX)].copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"  [{scene_name}] Excluded {n_before - n_after} outlier(s) outside [{VALID_SCORE_MIN}, {VALID_SCORE_MAX}]")
    return df


def ensure_plots_dir(scene_name):
    d = os.path.join(TESTS_DIR, scene_name, "plots")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# PLOT 1: MAIN EFFECTS
# ---------------------------------------------------------------------------

def plot_main_effects(df, plots_dir, label):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    fig.suptitle(f"Main Effects — {label}", fontsize=13, fontweight="bold", y=1.02)

    for ax, param in zip(axes, PARAM_COLS):
        means = df.groupby(param)["score"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            means[param], means["mean"],
            yerr=means["sem"] * 1.96,   # 95% CI
            fmt="o-", color=PALETTE[0],
            capsize=4, linewidth=2, markersize=6,
        )
        ax.set_xlabel(PARAM_LABELS[param], fontsize=10)
        ax.set_ylabel("CGVQM Score" if param == "alpha_weight" else "", fontsize=10)
        ax.set_title(PARAM_LABELS[param].split(" (")[0], fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(plots_dir, "01_main_effects.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 2: KEY INTERACTION — aw × hp
# ---------------------------------------------------------------------------

def plot_interaction_aw_hp(df, plots_dir, label):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    hp_levels = sorted(df["hist_percent"].unique())
    aw_levels = sorted(df["alpha_weight"].unique())

    for i, hp in enumerate(hp_levels):
        sub = df[df["hist_percent"] == hp]
        means = sub.groupby("alpha_weight")["score"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            means["alpha_weight"], means["mean"],
            yerr=means["sem"] * 1.96,
            fmt="o-", color=PALETTE[i],
            capsize=4, linewidth=2, markersize=6,
            label=f"hp = {int(hp)}%",
        )

    ax.set_xlabel("Alpha Weight (aw)", fontsize=11)
    ax.set_ylabel("CGVQM Score", fontsize=11)
    ax.set_title(f"aw × hp Interaction — {label}", fontsize=12, fontweight="bold")
    ax.set_xticks(aw_levels)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.legend(title="History %", fontsize=9, title_fontsize=9)
    sns.despine(ax=ax)

    # Annotate: non-parallel lines = interaction
    ax.text(
        0.98, 0.04,
        "Non-parallel lines\nindicate interaction",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="gray", style="italic",
    )

    plt.tight_layout()
    path = os.path.join(plots_dir, "02_interaction_aw_hp.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 3: ETA-SQUARED BAR CHART
# ---------------------------------------------------------------------------

def compute_eta_sq(df):
    """Recompute eta-squared for all terms without statsmodels dependency."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    ALIAS = {"alpha_weight": "aw", "num_samples": "ns",
             "filter_size": "fs", "hist_percent": "hp"}
    df2 = df.rename(columns=ALIAS).copy()
    for col in ALIAS.values():
        df2[col] = df2[col].astype(float) - df2[col].astype(float).mean()

    aliases = list(ALIAS.values())
    pairs   = [f"{a}:{b}" for i, a in enumerate(aliases) for b in aliases[i+1:]]
    formula = "score ~ " + " + ".join(aliases + pairs)
    model   = smf.ols(formula, data=df2).fit()
    anova_B = anova_lm(model, typ=2)
    ss_total = anova_B["sum_sq"].sum()
    anova_B["eta_sq"] = anova_B["sum_sq"] / ss_total
    return anova_B


def plot_eta_sq(df, plots_dir, label):
    anova = compute_eta_sq(df)
    anova_terms = anova.drop("Residual")

    # Color: main effects vs interactions
    colors = [PALETTE[0] if ":" not in idx else PALETTE[1]
              for idx in anova_terms.index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(anova_terms.index[::-1], anova_terms["eta_sq"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Effect Size (η²)", fontsize=11)
    ax.set_title(f"ANOVA Effect Sizes — {label}", fontsize=12, fontweight="bold")

    # Reference lines
    for x, ls, lbl in [(0.01, ":", "small"), (0.06, "--", "medium"), (0.14, "-", "large")]:
        ax.axvline(x, color="gray", linestyle=ls, linewidth=0.8, alpha=0.7)
        ax.text(x + 0.002, 0.5, lbl, transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray", va="center")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE[0], label="Main effect"),
                       Patch(facecolor=PALETTE[1], label="2-way interaction")]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()
    path = os.path.join(plots_dir, "03_eta_sq_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 4: HEATMAP aw × hp
# ---------------------------------------------------------------------------

def plot_heatmap_aw_hp(df, plots_dir, label):
    pivot = df.groupby(["alpha_weight", "hist_percent"])["score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True, fmt=".2f",
        cmap="RdYlGn",
        vmin=max(VALID_SCORE_MIN, df["score"].min()),
        vmax=df["score"].max(),
        linewidths=0.5,
        cbar_kws={"label": "CGVQM Score"},
    )
    ax.set_xlabel("History % (hp)", fontsize=11)
    ax.set_ylabel("Alpha Weight (aw)", fontsize=11)
    ax.set_title(f"aw × hp Mean Score — {label}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(plots_dir, "04_heatmap_aw_hp.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PER-SCENE RUNNER
# ---------------------------------------------------------------------------

def run_scene(scene_name):
    print(f"\n{'='*60}")
    print(f"PLOTTING: {scene_name}")
    print(f"{'='*60}")

    df        = load_scene(scene_name)
    plots_dir = ensure_plots_dir(scene_name)

    plot_main_effects(df, plots_dir, label=scene_name)
    plot_interaction_aw_hp(df, plots_dir, label=scene_name)
    plot_eta_sq(df, plots_dir, label=scene_name)
    plot_heatmap_aw_hp(df, plots_dir, label=scene_name)

    print(f"  All plots saved to: {plots_dir}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for the TAA factorial sweep."
    )
    parser.add_argument(
        "--scene", nargs="+", required=True,
        help="Scene folder name(s) inside tests/"
    )
    parser.add_argument(
        "--pool", action="store_true",
        help="Also generate pooled plots combining all scenes"
    )
    args = parser.parse_args()

    dfs = {}
    for scene in args.scene:
        try:
            dfs[scene] = run_scene(scene)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if args.pool and len(dfs) > 1:
        print(f"\n{'='*60}")
        print("PLOTTING: POOLED")
        print(f"{'='*60}")
        pooled     = pd.concat(list(dfs.values()), ignore_index=True)
        plots_dir  = os.path.join(TESTS_DIR, "plots_pooled")
        os.makedirs(plots_dir, exist_ok=True)
        label      = "Pooled (" + ", ".join(dfs.keys()) + ")"

        plot_main_effects(pooled, plots_dir, label=label)
        plot_interaction_aw_hp(pooled, plots_dir, label=label)
        plot_eta_sq(pooled, plots_dir, label=label)
        plot_heatmap_aw_hp(pooled, plots_dir, label=label)
        print(f"  Pooled plots saved to: {plots_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()