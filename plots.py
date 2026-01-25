#!/usr/bin/env python3
"""
CGVQM Score Analysis and Visualization
Compares rendering quality metrics across two scenes with varying parameters
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
SCENE1_NAME = "plantshow"
SCENE2_NAME = "plantshow"
SCENE1_PATH = f"outputs/{SCENE1_NAME}/scores_cgvqm"
SCENE2_PATH = f"outputs/{SCENE2_NAME}/scores_cgvqm"
OUTPUT_DIR = "plots_plantshow"

PARAMETERS = ["alpha_weight", "num_samples", "filter_size", "hist_percent"]

# Add these to your configuration section at the top
SCENE1_CVVDP_PATH = f"outputs/{SCENE1_NAME}/scores_cvvdp"
SCENE2_CVVDP_PATH = f"outputs/{SCENE2_NAME}/scores_cvvdp"

def plot_metric_comparison(cgvqm_scene1, cgvqm_scene2, cvvdp_scene1, cvvdp_scene2, parameter_name, output_dir):
    """Create side-by-side comparison of CGVQM and CVVDP scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract CGVQM data
    cgvqm_s1_params = [d['param_value'] for d in cgvqm_scene1]
    cgvqm_s1_scores = [d['score'] for d in cgvqm_scene1]
    cgvqm_s2_params = [d['param_value'] for d in cgvqm_scene2]
    cgvqm_s2_scores = [d['score'] for d in cgvqm_scene2]
    
    # Extract CVVDP data
    cvvdp_s1_params = [d['param_value'] for d in cvvdp_scene1]
    cvvdp_s1_scores = [d['score'] for d in cvvdp_scene1]
    cvvdp_s2_params = [d['param_value'] for d in cvvdp_scene2]
    cvvdp_s2_scores = [d['score'] for d in cvvdp_scene2]
    
    # Plot CGVQM
    ax1.plot(cgvqm_s1_params, cgvqm_s1_scores, marker='o', linewidth=2, 
            markersize=8, label=SCENE1_NAME, color='#2E86AB')
    ax1.plot(cgvqm_s2_params, cgvqm_s2_scores, marker='s', linewidth=2, 
            markersize=8, label=SCENE2_NAME, color='#A23B72')
    ax1.set_xlabel(f'{parameter_name.replace("_", " ").title()}', fontsize=12)
    ax1.set_ylabel('CGVQM Score', fontsize=12)
    ax1.set_title('CGVQM', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot CVVDP
    ax2.plot(cvvdp_s1_params, cvvdp_s1_scores, marker='o', linewidth=2, 
            markersize=8, label=SCENE1_NAME, color='#2E86AB')
    ax2.plot(cvvdp_s2_params, cvvdp_s2_scores, marker='s', linewidth=2, 
            markersize=8, label=SCENE2_NAME, color='#A23B72')
    ax2.set_xlabel(f'{parameter_name.replace("_", " ").title()}', fontsize=12)
    ax2.set_ylabel('CVVDP Score', fontsize=12)
    ax2.set_title('CVVDP', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Metric Comparison: {parameter_name.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f'metric_comparison_{parameter_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def load_json_scores(filepath):
    """Load scores from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_parameter_values_and_scores(data):
    """Extract parameter values and their corresponding scores from JSON data"""
    results = []
    for key, value in data.items():
        # Extract parameter value from key (e.g., "vary_filter_size_0.1" -> 0.1)
        param_value = key.split('_')[-1]
        try:
            param_value = float(param_value)
        except ValueError:
            continue
        
        score = value['score']
        per_frame_errors = value['per_frame_errors']
        
        results.append({
            'param_value': param_value,
            'score': score,
            'per_frame_errors': per_frame_errors
        })
    
    # Sort by parameter value
    results.sort(key=lambda x: x['param_value'])
    return results

def plot_overall_scores(scene1_data, scene2_data, parameter_name, output_dir):
    """Create comparison plot of overall scores for a parameter across both scenes"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data for scene 1
    scene1_params = [d['param_value'] for d in scene1_data]
    scene1_scores = [d['score'] for d in scene1_data]
    
    # Extract data for scene 2
    scene2_params = [d['param_value'] for d in scene2_data]
    scene2_scores = [d['score'] for d in scene2_data]
    
    # Plot both scenes
    ax.plot(scene1_params, scene1_scores, marker='o', linewidth=2, 
            markersize=8, label=SCENE1_NAME, color='#2E86AB')
    ax.plot(scene2_params, scene2_scores, marker='s', linewidth=2, 
            markersize=8, label=SCENE2_NAME, color='#A23B72')
    
    ax.set_xlabel(f'{parameter_name.replace("_", " ").title()}', fontsize=12)
    ax.set_ylabel('CGVQM Score', fontsize=12)
    ax.set_title(f'Overall CGVQM Score vs {parameter_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'overall_score_{parameter_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_per_frame_comparison(scene1_data, scene2_data, parameter_name, output_dir):
    """Create per-frame error comparison plots"""
    n_configs = len(scene1_data)
    
    # Create subplots for each parameter configuration
    fig, axes = plt.subplots(n_configs, 1, figsize=(14, 4*n_configs))
    if n_configs == 1:
        axes = [axes]
    
    for idx, (s1_config, s2_config) in enumerate(zip(scene1_data, scene2_data)):
        ax = axes[idx]
        param_val = s1_config['param_value']
        
        # Plot per-frame errors
        frames1 = range(len(s1_config['per_frame_errors']))
        frames2 = range(len(s2_config['per_frame_errors']))
        
        ax.plot(frames1, s1_config['per_frame_errors'], 
                alpha=0.7, linewidth=1, label=f'{SCENE1_NAME} (score: {s1_config["score"]:.2f})',
                color='#2E86AB')
        ax.plot(frames2, s2_config['per_frame_errors'], 
                alpha=0.7, linewidth=1, label=f'{SCENE2_NAME} (score: {s2_config["score"]:.2f})',
                color='#A23B72')
        
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Per-frame Error', fontsize=10)
        ax.set_title(f'{parameter_name.replace("_", " ").title()} = {param_val}', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'per_frame_{parameter_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_per_frame_statistics(scene1_data, scene2_data, parameter_name, output_dir):
    """Plot statistics of per-frame errors (mean, std, min, max) across parameter values"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scene1_params = [d['param_value'] for d in scene1_data]
    scene2_params = [d['param_value'] for d in scene2_data]
    
    # Calculate statistics
    scene1_means = [np.mean(d['per_frame_errors']) for d in scene1_data]
    scene1_stds = [np.std(d['per_frame_errors']) for d in scene1_data]
    scene1_mins = [np.min(d['per_frame_errors']) for d in scene1_data]
    scene1_maxs = [np.max(d['per_frame_errors']) for d in scene1_data]
    
    scene2_means = [np.mean(d['per_frame_errors']) for d in scene2_data]
    scene2_stds = [np.std(d['per_frame_errors']) for d in scene2_data]
    scene2_mins = [np.min(d['per_frame_errors']) for d in scene2_data]
    scene2_maxs = [np.max(d['per_frame_errors']) for d in scene2_data]
    
    # Mean
    axes[0, 0].plot(scene1_params, scene1_means, marker='o', label=SCENE1_NAME, color='#2E86AB')
    axes[0, 0].plot(scene2_params, scene2_means, marker='s', label=SCENE2_NAME, color='#A23B72')
    axes[0, 0].set_ylabel('Mean Per-frame Error', fontsize=10)
    axes[0, 0].set_title('Mean', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std Dev
    axes[0, 1].plot(scene1_params, scene1_stds, marker='o', label=SCENE1_NAME, color='#2E86AB')
    axes[0, 1].plot(scene2_params, scene2_stds, marker='s', label=SCENE2_NAME, color='#A23B72')
    axes[0, 1].set_ylabel('Std Dev of Per-frame Error', fontsize=10)
    axes[0, 1].set_title('Standard Deviation (Temporal Stability)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min
    axes[1, 0].plot(scene1_params, scene1_mins, marker='o', label=SCENE1_NAME, color='#2E86AB')
    axes[1, 0].plot(scene2_params, scene2_mins, marker='s', label=SCENE2_NAME, color='#A23B72')
    axes[1, 0].set_xlabel(f'{parameter_name.replace("_", " ").title()}', fontsize=10)
    axes[1, 0].set_ylabel('Min Per-frame Error', fontsize=10)
    axes[1, 0].set_title('Minimum (Best Frame)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max
    axes[1, 1].plot(scene1_params, scene1_maxs, marker='o', label=SCENE1_NAME, color='#2E86AB')
    axes[1, 1].plot(scene2_params, scene2_maxs, marker='s', label=SCENE2_NAME, color='#A23B72')
    axes[1, 1].set_xlabel(f'{parameter_name.replace("_", " ").title()}', fontsize=10)
    axes[1, 1].set_ylabel('Max Per-frame Error', fontsize=10)
    axes[1, 1].set_title('Maximum (Worst Frame)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Per-frame Error Statistics vs {parameter_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    output_path = output_dir / f'statistics_{parameter_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_plot(all_scene1_data, all_scene2_data, output_dir):
    """Create a summary plot showing all parameters side by side"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, param in enumerate(PARAMETERS):
        ax = axes[idx]
        
        scene1_data = all_scene1_data[param]
        scene2_data = all_scene2_data[param]
        
        scene1_params = [d['param_value'] for d in scene1_data]
        scene1_scores = [d['score'] for d in scene1_data]
        
        scene2_params = [d['param_value'] for d in scene2_data]
        scene2_scores = [d['score'] for d in scene2_data]
        
        ax.plot(scene1_params, scene1_scores, marker='o', linewidth=2, 
                markersize=8, label=SCENE1_NAME, color='#2E86AB')
        ax.plot(scene2_params, scene2_scores, marker='s', linewidth=2, 
                markersize=8, label=SCENE2_NAME, color='#A23B72')
        
        ax.set_xlabel(f'{param.replace("_", " ").title()}', fontsize=11)
        ax.set_ylabel('CGVQM Score', fontsize=11)
        ax.set_title(f'{param.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('CGVQM Score Comparison Across All Parameters', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = output_dir / 'summary_all_parameters.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def extract_parameter_values_and_scores_cvvdp(data):
    """Extract parameter values and scores from CVVDP JSON data (no per-frame errors)"""
    results = []
    for key, value in data.items():
        # Extract parameter value from key (e.g., "vary_filter_size_0.1" -> 0.1)
        param_value = key.split('_')[-1]
        try:
            param_value = float(param_value)
        except ValueError:
            continue
        
        # CVVDP data structure is just a float score, not a dict
        score = value if isinstance(value, (int, float)) else value.get('score', value)
        
        results.append({
            'param_value': param_value,
            'score': score
        })
    
    # Sort by parameter value
    results.sort(key=lambda x: x['param_value'])
    return results


def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting CGVQM analysis...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Store all data for summary plot
    all_scene1_data = {}
    all_scene2_data = {}
    
    # Process each parameter
    for param in PARAMETERS:
        print(f"\nProcessing parameter: {param}")
        
        # Load CGVQM data
        cgvqm_scene1_file = Path(SCENE1_PATH) / f"vary_{param}_scores.json"
        cgvqm_scene2_file = Path(SCENE2_PATH) / f"vary_{param}_scores.json"
        
        # Load CVVDP data
        cvvdp_scene1_file = Path(SCENE1_CVVDP_PATH) / f"vary_{param}_scores.json"
        cvvdp_scene2_file = Path(SCENE2_CVVDP_PATH) / f"vary_{param}_scores.json"
        
        try:
            # Load CGVQM
            cgvqm_scene1_json = load_json_scores(cgvqm_scene1_file)
            cgvqm_scene2_json = load_json_scores(cgvqm_scene2_file)
            
            cgvqm_scene1_data = extract_parameter_values_and_scores(cgvqm_scene1_json)
            cgvqm_scene2_data = extract_parameter_values_and_scores(cgvqm_scene2_json)
            
            # Load CVVDP
            cvvdp_scene1_json = load_json_scores(cvvdp_scene1_file)
            cvvdp_scene2_json = load_json_scores(cvvdp_scene2_file)
            
            # Use the new function for CVVDP (no per-frame data)
            cvvdp_scene1_data = extract_parameter_values_and_scores_cvvdp(cvvdp_scene1_json)
            cvvdp_scene2_data = extract_parameter_values_and_scores_cvvdp(cvvdp_scene2_json)
            
            # Store for summary
            all_scene1_data[param] = cgvqm_scene1_data
            all_scene2_data[param] = cgvqm_scene2_data
            
            # Generate plots
            plot_overall_scores(cgvqm_scene1_data, cgvqm_scene2_data, param, output_dir)
            plot_per_frame_statistics(cgvqm_scene1_data, cgvqm_scene2_data, param, output_dir)
            plot_per_frame_comparison(cgvqm_scene1_data, cgvqm_scene2_data, param, output_dir)
            
            # NEW: Generate metric comparison plot
            plot_metric_comparison(cgvqm_scene1_data, cgvqm_scene2_data, 
                                 cvvdp_scene1_data, cvvdp_scene2_data, 
                                 param, output_dir)
            
        except FileNotFoundError as e:
            print(f"  WARNING: Could not find file - {e}")
            continue
        except Exception as e:
            print(f"  ERROR processing {param}: {e}")
            continue
    
    # Create summary plot
    if all_scene1_data and all_scene2_data:
        print(f"\nCreating summary plot...")
        create_summary_plot(all_scene1_data, all_scene2_data, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! All plots saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()