"""
Compute scene-level quality metrics from master scores.
"""
import pandas as pd
import numpy as np

def compute_scene_metrics(scores_csv="master_scores_cleaned.csv"):
    """
    Compute aggregate quality metrics for each scene.
    
    Returns DataFrame with one row per scene containing:
    - mean_cgvqm_all: Average quality across all TAA variations
    - std_cgvqm_all: Standard deviation (variability)
    - min_cgvqm_all: Worst quality achieved
    - max_cgvqm_all: Best quality achieved
    - range_cgvqm_all: max - min (sensitivity to parameters)
    - mean_cgvqm_per_group: Average for each parameter group
    """
    
    df = pd.read_csv(scores_csv)
    
    scene_metrics = []
    
    for scene in df['scene'].unique():
        scene_data = df[df['scene'] == scene]
        
        metrics = {
            'scene': scene,
            'n_variations': len(scene_data),
            
            # Overall statistics
            'mean_cgvqm_all': scene_data['cgvqm_score'].mean(),
            'std_cgvqm_all': scene_data['cgvqm_score'].std(),
            'min_cgvqm_all': scene_data['cgvqm_score'].min(),
            'max_cgvqm_all': scene_data['cgvqm_score'].max(),
            'range_cgvqm_all': scene_data['cgvqm_score'].max() - scene_data['cgvqm_score'].min(),
            
            # Median (robust to outliers)
            'median_cgvqm_all': scene_data['cgvqm_score'].median(),
        }
        
        # Per parameter group statistics
        for param_group in df['param_group'].unique():
            group_data = scene_data[scene_data['param_group'] == param_group]
            if len(group_data) > 0:
                metrics[f'mean_cgvqm_{param_group}'] = group_data['cgvqm_score'].mean()
                metrics[f'std_cgvqm_{param_group}'] = group_data['cgvqm_score'].std()
                metrics[f'range_cgvqm_{param_group}'] = (
                    group_data['cgvqm_score'].max() - group_data['cgvqm_score'].min()
                )
        
        # Frame-level error statistics
        metrics['mean_frame_error_all'] = scene_data['mean_frame_error'].mean()
        metrics['max_frame_error_all'] = scene_data['max_frame_error'].max()
        
        scene_metrics.append(metrics)
    
    scene_df = pd.DataFrame(scene_metrics)
    
    # Sort by mean quality (worst to best)
    scene_df = scene_df.sort_values('mean_cgvqm_all')
    
    return scene_df

if __name__ == "__main__":
    # Compute scene metrics
    scene_df = compute_scene_metrics()
    
    # Save to CSV
    output_file = "scene_level_metrics.csv"
    scene_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    # Display results
    print("\nScene Quality Rankings (worst to best):")
    print("="*80)
    display_cols = ['scene', 'mean_cgvqm_all', 'std_cgvqm_all', 'range_cgvqm_all', 
                    'min_cgvqm_all', 'max_cgvqm_all']
    print(scene_df[display_cols].to_string(index=False))
    
    print("\n\nParameter Group Sensitivity:")
    print("="*80)
    print("Which parameter groups cause the most quality variation per scene?\n")
    sensitivity_cols = ['scene', 'range_cgvqm_alpha_weight', 'range_cgvqm_num_samples',
                       'range_cgvqm_filter_size', 'range_cgvqm_hist_percent']
    if all(col in scene_df.columns for col in sensitivity_cols):
        print(scene_df[sensitivity_cols].to_string(index=False))
    
    print("\n\nKey Insights:")
    print(f"Scene with lowest average quality: {scene_df.iloc[0]['scene']} "
          f"(mean CGVQM: {scene_df.iloc[0]['mean_cgvqm_all']:.2f})")
    print(f"Scene with highest average quality: {scene_df.iloc[-1]['scene']} "
          f"(mean CGVQM: {scene_df.iloc[-1]['mean_cgvqm_all']:.2f})")
    print(f"Most parameter-sensitive scene: "
          f"{scene_df.loc[scene_df['range_cgvqm_all'].idxmax(), 'scene']} "
          f"(range: {scene_df['range_cgvqm_all'].max():.2f})")