#!/usr/bin/env python3
"""
TAA Parameter Analysis Script
Analyzes temporal anti-aliasing parameter scores across multiple videos
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_video_scores(video_path: Path) -> Dict[str, Dict]:
    """Load all parameter score files for a single video."""
    scores_dir = video_path / "scores_cgvqm"
    
    if not scores_dir.exists():
        return None
    
    parameter_files = {
        'alpha_weight': 'vary_alpha_weight_scores.json',
        'filter_size': 'vary_filter_size_scores.json',
        'hist_percent': 'vary_hist_percent_scores.json',
        'num_samples': 'vary_num_samples_scores.json'
    }
    
    video_data = {}
    for param_name, filename in parameter_files.items():
        filepath = scores_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                video_data[param_name] = json.load(f)
    
    return video_data if video_data else None


def extract_parameter_value(key: str) -> float:
    """Extract numeric parameter value from key like 'vary_alpha_weight_0.01'."""
    parts = key.split('_')
    try:
        return float(parts[-1])
    except ValueError:
        return None


def validate_and_clean_data(all_data: Dict) -> Dict:
    """Validate scores and remove invalid entries (outside 0-100 range)."""
    cleaned_data = {}
    total_invalid = 0
    
    for video_name, video_data in all_data.items():
        cleaned_video = {}
        
        for param_name, param_data in video_data.items():
            cleaned_param = {}
            
            for key, entry in param_data.items():
                score = entry['score']
                
                # Check if score is valid (between 0 and 100)
                if 0 <= score <= 100:
                    cleaned_param[key] = entry
                else:
                    total_invalid += 1
                    param_val = extract_parameter_value(key)
                    print(f"⚠️  WARNING: Invalid score {score:.2f} for {video_name} / {param_name} / value={param_val} (skipping)")
            
            if cleaned_param:
                cleaned_video[param_name] = cleaned_param
        
        if cleaned_video:
            cleaned_data[video_name] = cleaned_video
    
    if total_invalid > 0:
        print(f"\n⚠️  Total invalid scores filtered: {total_invalid}")
    
    return cleaned_data


def collect_all_data(outputs_dir: str = "outputs") -> Dict:
    """Collect data from all videos."""
    outputs_path = Path(outputs_dir)
    
    if not outputs_path.exists():
        raise FileNotFoundError(f"Directory '{outputs_dir}' not found")
    
    all_data = {}
    
    # Iterate through all subdirectories in outputs
    for video_dir in sorted(outputs_path.iterdir()):
        if video_dir.is_dir():
            video_name = video_dir.name
            video_scores = load_video_scores(video_dir)
            
            if video_scores:
                all_data[video_name] = video_scores
                print(f"✓ Loaded data for: {video_name}")
            else:
                print(f"✗ No scores found for: {video_name}")
    
    if not all_data:
        raise ValueError("No valid video data found!")
    
    print(f"\nTotal videos loaded: {len(all_data)}")
    
    # Validate and clean data
    print("\nValidating scores...")
    all_data = validate_and_clean_data(all_data)
    
    return all_data


def create_summary_table(all_data: Dict) -> pd.DataFrame:
    """Create summary table with min, max, and average scores per video and parameter."""
    rows = []
    
    for video_name, video_data in all_data.items():
        row = {'Video': video_name}
        
        # Collect all scores across all parameters for this video
        all_scores = []
        
        for param_name, param_data in video_data.items():
            scores = [entry['score'] for entry in param_data.values()]
            all_scores.extend(scores)
            
            # Average score for this parameter
            row[f'{param_name}_avg'] = np.mean(scores)
        
        # Overall min and max across all parameters
        row['min_score'] = min(all_scores)
        row['max_score'] = max(all_scores)
        row['score_range'] = row['max_score'] - row['min_score']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    cols = ['Video', 'min_score', 'max_score', 'score_range']
    param_cols = [col for col in df.columns if col.endswith('_avg')]
    cols.extend(sorted(param_cols))
    
    return df[cols]


def create_main_effects_plot(all_data: Dict, output_path: str = "main_effects_plot.png"):
    """Create line plots showing how each parameter affects score, averaged across videos.
    Uses normalized scores (deviation from video mean) to account for baseline differences.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    param_names = ['alpha_weight', 'filter_size', 'hist_percent', 'num_samples']
    param_labels = ['Alpha Weight', 'Filter Size', 'Hist Percent', 'Num Samples']
    
    for idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        
        # Collect NORMALIZED data for this parameter across all videos
        param_values = []
        normalized_scores = []
        
        for video_name, video_data in all_data.items():
            if param_name in video_data:
                # Calculate this video's mean score across all parameters
                video_scores_all = []
                for param in video_data.values():
                    video_scores_all.extend([e['score'] for e in param.values()])
                video_mean = np.mean(video_scores_all)
                
                # Normalize scores for this parameter
                for key, entry in video_data[param_name].items():
                    param_val = extract_parameter_value(key)
                    if param_val is not None:
                        param_values.append(param_val)
                        normalized_scores.append(entry['score'] - video_mean)
        
        if param_values:
            # Group by parameter value and calculate mean and std
            df = pd.DataFrame({'param': param_values, 'score': normalized_scores})
            grouped = df.groupby('param')['score'].agg(['mean', 'std', 'count'])
            grouped = grouped.sort_index()
            
            # Plot mean with error bars
            ax.errorbar(grouped.index, grouped['mean'], 
                       yerr=grouped['std'], 
                       marker='o', linewidth=2, markersize=8,
                       capsize=5, capthick=2, label='Mean ± Std')
            
            # Plot individual video traces (lighter) - normalized
            for video_name, video_data in all_data.items():
                if param_name in video_data:
                    # Calculate video mean
                    video_scores_all = []
                    for param in video_data.values():
                        video_scores_all.extend([e['score'] for e in param.values()])
                    video_mean = np.mean(video_scores_all)
                    
                    vals = []
                    video_scores = []
                    for key, entry in video_data[param_name].items():
                        param_val = extract_parameter_value(key)
                        if param_val is not None:
                            vals.append(param_val)
                            video_scores.append(entry['score'] - video_mean)
                    
                    if vals:
                        sorted_pairs = sorted(zip(vals, video_scores))
                        vals, video_scores = zip(*sorted_pairs)
                        ax.plot(vals, video_scores, alpha=0.15, linewidth=1, color='gray')
            
            # Add horizontal line at 0 (video mean)
            ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
            
            ax.set_xlabel(param_label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score Deviation from Video Mean', fontsize=12, fontweight='bold')
            ax.set_title(f'Effect of {param_label} on Score\n(Normalized by Video Baseline)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Find and annotate optimal value
            optimal_idx = grouped['mean'].idxmax()
            optimal_score = grouped.loc[optimal_idx, 'mean']
            ax.axvline(optimal_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.text(optimal_idx, ax.get_ylim()[1] * 0.90, 
                   f'Optimal: {optimal_idx:.2f}\n(+{optimal_score:.2f} vs mean)',
                   ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved main effects plot to: {output_path}")
    plt.close()


def create_heatmap(all_data: Dict, output_path: str = "parameter_heatmap.png"):
    """Create heatmap showing scores across videos and parameter values."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    param_names = ['alpha_weight', 'filter_size', 'hist_percent', 'num_samples']
    param_labels = ['Alpha Weight', 'Filter Size', 'Hist Percent', 'Num Samples']
    
    for idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        
        # Collect data for heatmap
        video_names = []
        param_values_set = set()
        
        # First pass: collect all unique parameter values
        for video_name, video_data in all_data.items():
            if param_name in video_data:
                for key in video_data[param_name].keys():
                    param_val = extract_parameter_value(key)
                    if param_val is not None:
                        param_values_set.add(param_val)
        
        param_values_sorted = sorted(param_values_set)
        
        # Second pass: build matrix
        matrix_data = []
        for video_name, video_data in all_data.items():
            if param_name in video_data:
                video_names.append(video_name)
                row = []
                
                for param_val in param_values_sorted:
                    # Find matching entry
                    score = None
                    for key, entry in video_data[param_name].items():
                        if abs(extract_parameter_value(key) - param_val) < 0.001:
                            score = entry['score']
                            break
                    row.append(score if score is not None else np.nan)
                
                matrix_data.append(row)
        
        if matrix_data:
            df_heatmap = pd.DataFrame(matrix_data, 
                                     index=video_names,
                                     columns=[f'{v:.2f}' for v in param_values_sorted])
            
            sns.heatmap(df_heatmap, annot=False, fmt='.1f', cmap='RdYlGn',
                       cbar_kws={'label': 'Score'}, ax=ax, vmin=85, vmax=95)
            ax.set_title(f'{param_label} - Scores by Video', fontsize=14, fontweight='bold')
            ax.set_xlabel('Parameter Value', fontsize=11)
            ax.set_ylabel('Video', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {output_path}")
    plt.close()


def create_sensitivity_ranking(all_data: Dict, output_path: str = "sensitivity_ranking.png"):
    """Create bar chart showing parameter sensitivity using normalized scores.
    Measures the range of average effects (not absolute score range).
    """
    param_names = ['alpha_weight', 'filter_size', 'hist_percent', 'num_samples']
    param_labels = ['Alpha Weight', 'Filter Size', 'Hist Percent', 'Num Samples']
    
    # Calculate effect range for each parameter (using normalized scores)
    ranges = []
    
    for param_name in param_names:
        # Collect normalized scores grouped by parameter value
        param_effects = {}
        
        for video_name, video_data in all_data.items():
            if param_name in video_data:
                # Calculate video mean
                video_scores_all = []
                for param in video_data.values():
                    video_scores_all.extend([e['score'] for e in param.values()])
                video_mean = np.mean(video_scores_all)
                
                # Normalize and group by parameter value
                for key, entry in video_data[param_name].items():
                    param_val = extract_parameter_value(key)
                    if param_val is not None:
                        if param_val not in param_effects:
                            param_effects[param_val] = []
                        param_effects[param_val].append(entry['score'] - video_mean)
        
        # Calculate mean effect for each parameter value, then find range
        if param_effects:
            mean_effects = [np.mean(scores) for scores in param_effects.values()]
            effect_range = max(mean_effects) - min(mean_effects)
            ranges.append(effect_range)
        else:
            ranges.append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(param_labels, ranges, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, range_val in zip(bars, ranges):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{range_val:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Effect Range (Normalized)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Sensitivity Rankings\n(Higher = More Impactful, Normalized by Video)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved sensitivity ranking to: {output_path}")
    plt.close()


def generate_optimal_values_table(all_data: Dict) -> pd.DataFrame:
    """Generate table showing optimal parameter values using normalized scores."""
    param_names = ['alpha_weight', 'filter_size', 'hist_percent', 'num_samples']
    param_labels = ['Alpha Weight', 'Filter Size', 'Hist Percent', 'Num Samples']
    
    results = []
    
    for param_name, param_label in zip(param_names, param_labels):
        # Collect normalized data for this parameter
        param_effects = {}
        
        for video_name, video_data in all_data.items():
            if param_name in video_data:
                # Calculate video mean
                video_scores_all = []
                for param in video_data.values():
                    video_scores_all.extend([e['score'] for e in param.values()])
                video_mean = np.mean(video_scores_all)
                
                # Normalize and group
                for key, entry in video_data[param_name].items():
                    param_val = extract_parameter_value(key)
                    if param_val is not None:
                        if param_val not in param_effects:
                            param_effects[param_val] = []
                        param_effects[param_val].append(entry['score'] - video_mean)
        
        # Find optimal value
        if param_effects:
            avg_effects = {val: np.mean(scores) for val, scores in param_effects.items()}
            optimal_val = max(avg_effects, key=avg_effects.get)
            optimal_effect = avg_effects[optimal_val]
            effect_range = max(avg_effects.values()) - min(avg_effects.values())
            
            results.append({
                'Parameter': param_label,
                'Optimal Value': optimal_val,
                'Avg Effect at Optimal': f'{optimal_effect:+.2f}',
                'Effect Range': f'{effect_range:.2f}'
            })
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("TAA Parameter Analysis")
    print("=" * 60)
    print()
    
    # Collect all data
    print("Loading data...")
    all_data = collect_all_data("outputs")
    print()
    
    # Create summary table
    print("Generating summary table...")
    summary_df = create_summary_table(all_data)
    summary_df.to_csv("summary_table.csv", index=False, float_format='%.2f')
    print("✓ Saved summary table to: summary_table.csv")
    print()
    print(summary_df.to_string(index=False))
    print()
    
    # Create optimal values table
    print("\nGenerating optimal parameter values table...")
    optimal_df = generate_optimal_values_table(all_data)
    optimal_df.to_csv("optimal_parameters.csv", index=False)
    print("✓ Saved optimal parameters to: optimal_parameters.csv")
    print()
    print(optimal_df.to_string(index=False))
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    create_main_effects_plot(all_data)
    create_heatmap(all_data)
    create_sensitivity_ranking(all_data)
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - summary_table.csv")
    print("  - optimal_parameters.csv")
    print("  - main_effects_plot.png")
    print("  - parameter_heatmap.png")
    print("  - sensitivity_ranking.png")


if __name__ == "__main__":
    main()