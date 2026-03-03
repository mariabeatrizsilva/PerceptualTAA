"""
analyze_dataset.py

Analyzes dataset.json to understand how parameters (alpha_weight, filter_size,
hist_percent, num_samples) affect scene quality across resolutions.

Produces:
    1. Grouped bar charts — effect size per parameter per resolution
       (one averaged across scenes + one per scene)
    2. Score vs parameter value curves — averaged across scenes, one plot per parameter
    3. Heatmaps — effect size across scenes x resolutions, one per parameter
    4. Correlation scatter — resolution vs effect size, all (scene, param) points

Usage:
    python analyze_dataset.py --dataset dataset.json --output plots/
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
PARAMS       = ['alpha_weight', 'filter_size', 'hist_percent', 'num_samples']
RESOLUTIONS  = [50, 71, 87, 100]
HIST_MIN_VAL = 100.0   # only use hist_percent values >= 100
REF_KEY_RE   = 'ref-'  # we only want ref-{base_scene}, not ref-{}-screen-per-

COLORS = {
    50:  '#e74c3c',
    71:  '#e67e22',
    87:  '#3498db',
    100: '#2ecc71',
}

# ============================================================================
# DATA LOADING & FILTERING
# ============================================================================

def load_dataset(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def is_cross_scene_ref(ref_key: str) -> bool:
    """Returns True if ref_key contains 'screen-per' (cross-resolution ref, skip)."""
    return 'screen-per' in ref_key


def get_ref_key(scene_data: dict, resolution: str) -> str | None:
    """
    For a given resolution, find the correct ref key:
    - resolution 100: use self-ref (ref-{scene})
    - other resolutions: use cross-scene ref (ref-{base_scene}, no screen-per)
    Returns None if not found.
    """
    pct_data = scene_data.get(resolution, {})
    for ref_key in pct_data:
        if resolution == '100':
            # self-ref: no screen-per in key
            if not is_cross_scene_ref(ref_key):
                return ref_key
        else:
            # cross-scene ref: no screen-per in key
            if not is_cross_scene_ref(ref_key):
                return ref_key
    return None


def filter_entries(param: str, entries: list) -> list:
    """Apply param-specific filters to entries."""
    if param == 'hist_percent':
        entries = [e for e in entries if e['value'] >= HIST_MIN_VAL]
    return entries


def extract_scores(dataset: dict) -> dict:
    """
    Extract scores into a clean structure:
    {scene: {resolution: {param: [(value, score), ...]}}}

    Only includes scenes that have all 4 resolutions and all params available.
    """
    result = {}

    for scene, scene_data in dataset.items():
        if scene == 'metric':
            continue

        scene_scores = {}
        valid = True

        for res in RESOLUTIONS:
            res_str  = str(res)
            ref_key  = get_ref_key(scene_data, res_str)

            if ref_key is None:
                valid = False
                break

            ref_data = scene_data[res_str][ref_key]
            scene_scores[res] = {}

            for param in PARAMS:
                if param not in ref_data:
                    # missing param at this resolution — still ok, just empty
                    scene_scores[res][param] = []
                    continue
                entries = filter_entries(param, ref_data[param])
                scene_scores[res][param] = [(e['value'], e['score']) for e in entries]

        if valid:
            result[scene] = scene_scores

    return result


def effect_size(scores: list) -> float:
    """Max - min score for a list of (value, score) tuples."""
    if len(scores) < 2:
        return 0.0
    vals = [s for _, s in scores]
    return max(vals) - min(vals)


# ============================================================================
# FIGURE 1: Grouped bar chart — effect size per param per resolution
# ============================================================================

def plot_effect_size_bars(scores: dict, output_dir: str):
    """One averaged plot + one per scene."""
    scenes = list(scores.keys())

    # ── Averaged across scenes ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(PARAMS))
    width = 0.18

    for i, res in enumerate(RESOLUTIONS):
        means = []
        for param in PARAMS:
            sizes = [effect_size(scores[s][res][param]) for s in scenes
                     if scores[s][res][param]]
            means.append(np.mean(sizes) if sizes else 0)
        ax.bar(x + i * width, means, width, label=f'{res}%', color=COLORS[res], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([p.replace('_', ' ') for p in PARAMS])
    ax.set_ylabel('Effect size (max − min score)')
    ax.set_title('Parameter sensitivity by resolution — averaged across scenes')
    ax.legend(title='Resolution')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_effect_size_avg.png'), dpi=150)
    plt.close()
    print('  Saved fig1_effect_size_avg.png')

    # ── Per scene ───────────────────────────────────────────────
    for scene in scenes:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, res in enumerate(RESOLUTIONS):
            vals = [effect_size(scores[scene][res][param]) for param in PARAMS]
            ax.bar(x + i * width, vals, width, label=f'{res}%', color=COLORS[res], alpha=0.85)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([p.replace('_', ' ') for p in PARAMS])
        ax.set_ylabel('Effect size (max − min score)')
        ax.set_title(f'Parameter sensitivity by resolution — {scene}')
        ax.legend(title='Resolution')
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig1_effect_size_{scene}.png'), dpi=150)
        plt.close()
    print(f'  Saved fig1_effect_size_{{scene}}.png for {len(scenes)} scenes')


# ============================================================================
# FIGURE 2: Score vs param value — averaged across scenes
# ============================================================================

def plot_score_curves(scores: dict, output_dir: str):
    """One plot per parameter, 4 lines (one per resolution), averaged across scenes."""
    scenes = list(scores.keys())

    for param in PARAMS:
        fig, ax = plt.subplots(figsize=(9, 5))

        for res in RESOLUTIONS:
            # Collect all unique x values across scenes
            all_values = sorted(set(
                v for s in scenes
                for v, _ in scores[s][res][param]
            ))
            if not all_values:
                continue

            # For each x value, average score across scenes that have it
            avg_scores = []
            for val in all_values:
                scene_scores = [
                    sc for s in scenes
                    for v, sc in scores[s][res][param]
                    if abs(v - val) < 1e-9
                ]
                if scene_scores:
                    avg_scores.append(np.mean(scene_scores))
                else:
                    avg_scores.append(None)

            valid = [(v, s) for v, s in zip(all_values, avg_scores) if s is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, 'o-', color=COLORS[res], label=f'{res}%', linewidth=2, markersize=5)

        ax.set_xlabel(param.replace('_', ' '))
        ax.set_ylabel('CGVQM score')
        ax.set_title(f'Score vs {param.replace("_", " ")} — averaged across scenes')
        ax.legend(title='Resolution')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f'fig2_curves_{param}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f'  Saved {fname}')


# ============================================================================
# FIGURE 3: Heatmap — effect size across scenes x resolutions
# ============================================================================

def plot_heatmaps(scores: dict, output_dir: str):
    """One heatmap per parameter: scenes on y-axis, resolutions on x-axis."""
    scenes = sorted(scores.keys())
    cmap   = LinearSegmentedColormap.from_list('sens', ['#f0f4ff', '#1a3a6b'])

    for param in PARAMS:
        data = np.array([
            [effect_size(scores[s][res][param]) for res in RESOLUTIONS]
            for s in scenes
        ])

        fig, ax = plt.subplots(figsize=(7, max(4, len(scenes) * 0.55 + 1.5)))
        im = ax.imshow(data, aspect='auto', cmap=cmap)

        ax.set_xticks(range(len(RESOLUTIONS)))
        ax.set_xticklabels([f'{r}%' for r in RESOLUTIONS])
        ax.set_yticks(range(len(scenes)))
        ax.set_yticklabels(scenes)
        ax.set_xlabel('Resolution')
        ax.set_title(f'Effect size heatmap — {param.replace("_", " ")}')

        # annotate cells
        for i in range(len(scenes)):
            for j in range(len(RESOLUTIONS)):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if data[i,j] > data.max()*0.6 else 'black')

        plt.colorbar(im, ax=ax, label='Effect size')
        plt.tight_layout()
        fname = f'fig3_heatmap_{param}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f'  Saved {fname}')


# ============================================================================
# FIGURE 4: Correlation scatter — resolution vs effect size
# ============================================================================

def plot_correlation(scores: dict, output_dir: str):
    """Scatter of resolution vs effect size, all (scene, param) points, colored by param."""
    scenes = list(scores.keys())
    param_colors = {
        'alpha_weight': '#e74c3c',
        'filter_size':  '#3498db',
        'hist_percent': '#2ecc71',
        'num_samples':  '#9b59b6',
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for param in PARAMS:
        xs, ys = [], []
        for scene in scenes:
            for res in RESOLUTIONS:
                es = effect_size(scores[scene][res][param])
                xs.append(res)
                ys.append(es)

        # jitter x slightly so points don't overlap
        jitter = np.random.uniform(-1.5, 1.5, len(xs))
        ax.scatter(np.array(xs) + jitter, ys,
                   color=param_colors[param], alpha=0.6, s=40,
                   label=param.replace('_', ' '))

        # trend line per param
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xline = np.linspace(min(RESOLUTIONS), max(RESOLUTIONS), 100)
        ax.plot(xline, p(xline), color=param_colors[param], linewidth=2, alpha=0.9)

    ax.set_xlabel('Resolution (%)')
    ax.set_ylabel('Effect size (max − min score)')
    ax.set_title('Resolution vs parameter sensitivity — all scenes')
    ax.set_xticks(RESOLUTIONS)
    ax.legend(title='Parameter')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_correlation.png'), dpi=150)
    plt.close()
    print('  Saved fig4_correlation.png')


# ============================================================================
# PRINTED SUMMARY
# ============================================================================

def print_summary(scores: dict):
    scenes = list(scores.keys())
    print(f'\n{"="*65}')
    print(f'  ANALYSIS SUMMARY — {len(scenes)} scenes')
    print(f'{"="*65}')

    print(f'\n{"─"*65}')
    print(f'  Effect size (max−min score) per parameter per resolution')
    print(f'  Averaged across {len(scenes)} scenes')
    print(f'{"─"*65}')
    header = f'  {"Parameter":<20}' + ''.join(f'  {r}%'.rjust(8) for r in RESOLUTIONS)
    print(header)
    print(f'  {"─"*60}')

    for param in PARAMS:
        row = f'  {param:<20}'
        for res in RESOLUTIONS:
            sizes = [effect_size(scores[s][res][param]) for s in scenes
                     if scores[s][res][param]]
            mean  = np.mean(sizes) if sizes else 0
            row  += f'  {mean:>6.3f}'
        print(row)

    print(f'\n{"─"*65}')
    print(f'  Hypothesis check: does effect size increase at lower resolution?')
    print(f'{"─"*65}')
    for param in PARAMS:
        sizes_by_res = {}
        for res in RESOLUTIONS:
            sizes = [effect_size(scores[s][res][param]) for s in scenes
                     if scores[s][res][param]]
            sizes_by_res[res] = np.mean(sizes) if sizes else 0

        # check if monotonically increasing as resolution decreases
        vals = [sizes_by_res[r] for r in sorted(RESOLUTIONS, reverse=True)]  # 100->50
        monotonic = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        corr = np.corrcoef(RESOLUTIONS, [sizes_by_res[r] for r in RESOLUTIONS])[0,1]
        print(f'  {param:<20}  corr(res, effect)={corr:+.3f}  {"✓ monotonic" if monotonic else "✗ not monotonic"}')

    print(f'\n{"─"*65}')
    print(f'  Best parameter value per resolution (averaged across scenes)')
    print(f'{"─"*65}')
    for param in PARAMS:
        print(f'\n  {param}:')
        for res in RESOLUTIONS:
            # find value with highest average score across scenes
            all_values = sorted(set(
                v for s in scenes
                for v, _ in scores[s][res][param]
            ))
            if not all_values:
                continue
            best_val, best_score = None, -np.inf
            for val in all_values:
                avg = np.mean([
                    sc for s in scenes
                    for v, sc in scores[s][res][param]
                    if abs(v - val) < 1e-9
                ])
                if avg > best_score:
                    best_score, best_val = avg, val
            print(f'    {res}%  →  best value = {best_val}  (avg score = {best_score:.3f})')

    print(f'\n{"="*65}\n')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset.json and produce plots.')
    parser.add_argument('--dataset', type=str, default='dataset.json',
                        help='Path to dataset.json')
    parser.add_argument('--output', type=str, default='plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading {args.dataset}...')
    dataset = load_dataset(args.dataset)

    print('Extracting scores...')
    scores = extract_scores(dataset)
    print(f'Found {len(scores)} scenes with all 4 resolutions: {", ".join(sorted(scores.keys()))}')

    print('\nGenerating Figure 1 — effect size bar charts...')
    plot_effect_size_bars(scores, args.output)

    print('\nGenerating Figure 2 — score vs param value curves...')
    plot_score_curves(scores, args.output)

    print('\nGenerating Figure 3 — heatmaps...')
    plot_heatmaps(scores, args.output)

    print('\nGenerating Figure 4 — correlation scatter...')
    plot_correlation(scores, args.output)

    print_summary(scores)
    print(f'All plots saved to: {args.output}/')


if __name__ == '__main__':
    main()