"""
Parameter Analysis: Understand which TAA parameters matter most and 
find optimal settings for different scene types.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

# =============================================================================
# ANALYSIS 1: PARAMETER IMPORTANCE
# =============================================================================

def analyze_parameter_importance(scores_df):
    """
    Across all scenes and videos, which parameter group has the biggest 
    impact on quality?
    
    Calculates relative impact: (range / mean) for each scene, then averages.
    This accounts for the fact that different scenes have different baseline quality.
    
    Returns:
        DataFrame with relative impact statistics per parameter group
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: PARAMETER IMPORTANCE")
    print("="*80)
    
    param_groups = sorted(scores_df['param_group'].unique())
    scenes = scores_df['scene'].unique()
    
    importance_results = []
    
    for param_group in param_groups:
        # For each scene, calculate: (max - min) / mean for this parameter
        relative_impacts = []
        
        for scene in scenes:
            scene_param_data = scores_df[
                (scores_df['scene'] == scene) & 
                (scores_df['param_group'] == param_group)
            ]
            
            if len(scene_param_data) == 0:
                continue
            
            score_range = scene_param_data['cgvqm_score'].max() - scene_param_data['cgvqm_score'].min()
            score_mean = scene_param_data['cgvqm_score'].mean()
            
            if score_mean > 0:  # Avoid division by zero
                relative_impact = (score_range / score_mean) * 100  # Express as percentage
                relative_impacts.append(relative_impact)
        
        # Calculate statistics across all scenes
        subset = scores_df[scores_df['param_group'] == param_group]
        
        importance_results.append({
            'param_group': param_group,
            'relative_impact_mean': np.mean(relative_impacts),
            'relative_impact_median': np.median(relative_impacts),
            'relative_impact_std': np.std(relative_impacts),
            'absolute_range': subset['cgvqm_score'].max() - subset['cgvqm_score'].min(),
            'mean_score': subset['cgvqm_score'].mean(),
            'n_videos': len(subset)
        })
    
    importance_df = pd.DataFrame(importance_results)
    importance_df = importance_df.sort_values('relative_impact_mean', ascending=False)
    
    print("\nParameter Importance Rankings (by relative impact %):")
    print("-" * 80)
    print("Relative Impact = (max_score - min_score) / mean_score for each scene, averaged")
    print("-" * 80)
    print(importance_df[['param_group', 'relative_impact_mean', 'relative_impact_median', 
                         'absolute_range', 'mean_score']].to_string(index=False))
    
    print("\n\nInterpretation:")
    print(f"  • {importance_df.iloc[0]['param_group']}: Changes quality by ~{importance_df.iloc[0]['relative_impact_mean']:.1f}% on average")
    print(f"  • {importance_df.iloc[-1]['param_group']}: Changes quality by ~{importance_df.iloc[-1]['relative_impact_mean']:.1f}% on average")
    print(f"  • {importance_df.iloc[0]['param_group']} is {importance_df.iloc[0]['relative_impact_mean'] / importance_df.iloc[-1]['relative_impact_mean']:.1f}x more important than {importance_df.iloc[-1]['param_group']}")
    
    # Create visualization
    fig = go.Figure()
    
    # Add bars for relative impact
    fig.add_trace(go.Bar(
        x=importance_df['param_group'],
        y=importance_df['relative_impact_mean'],
        marker_color='#3498db',
        text=[f"{val:.2f}%" for val in importance_df['relative_impact_mean']],
        textposition='outside',
        error_y=dict(
            type='data',
            array=importance_df['relative_impact_std'],
            visible=True
        )
    ))
    
    fig.update_layout(
        title='Parameter Importance: Which Parameter Group Affects Quality Most?<br>' +
              '<sub>Relative Impact = (Range / Mean Score) × 100% · Higher = More important to tune · Averaged across 23 scenes</sub>',
        xaxis_title='Parameter Group',
        yaxis_title='Average Quality Impact (%)',
        height=500,
        showlegend=False
    )
    
    fig.write_html('analysis_1_parameter_importance.html')
    print("\n✓ Created analysis_1_parameter_importance.html")
    
    # Also create comparison: relative vs absolute
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Relative Impact (%)', 'Absolute Range (Points)')
    )
    
    fig2.add_trace(
        go.Bar(x=importance_df['param_group'], 
               y=importance_df['relative_impact_mean'],
               marker_color='#3498db',
               text=[f"{val:.1f}%" for val in importance_df['relative_impact_mean']],
               textposition='outside',
               showlegend=False),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Bar(x=importance_df['param_group'], 
               y=importance_df['absolute_range'],
               marker_color='#e74c3c',
               text=importance_df['absolute_range'].round(1),
               textposition='outside',
               showlegend=False),
        row=1, col=2
    )
    
    fig2.update_xaxes(title_text="Parameter Group", row=1, col=1)
    fig2.update_xaxes(title_text="Parameter Group", row=1, col=2)
    fig2.update_yaxes(title_text="Impact (%)", row=1, col=1)
    fig2.update_yaxes(title_text="Score Range", row=1, col=2)
    
    fig2.update_layout(
        title_text='Parameter Importance: Relative vs Absolute Impact<br>' +
                   '<sub>Left: Normalized by baseline quality · Right: Raw score differences</sub>',
        height=500
    )
    
    fig2.write_html('analysis_1_parameter_importance_detailed.html')
    print("✓ Created analysis_1_parameter_importance_detailed.html")
    
    return importance_df


# =============================================================================
# ANALYSIS 2: OPTIMAL PARAMETERS BY SCENE TYPE
# =============================================================================

def analyze_optimal_params_by_scene_type(scores_df, features_df):
    """
    Group scenes by their characteristics, then find optimal parameters 
    for each group.
    
    Scene Type Categorization:
    --------------------------
    Uses 2×2 matrix based on MEDIAN SPLITS of key features:
    
    1. Texture Complexity (texture_variance_avg):
       - High: Above median (detailed surfaces, foliage, complex materials)
       - Low: Below median (smooth surfaces, simple geometry)
    
    2. Motion Magnitude (flow_mean_avg):
       - High: Above median (fast camera pans, lots of movement)
       - Low: Below median (slow pans, near-static camera)
    
    This creates 4 scene types:
       High Texture + High Motion  → "Complex Moving"   (hardest for TAA)
       High Texture + Low Motion   → "Complex Static"   (static but detailed)
       Low Texture + High Motion   → "Simple Moving"    (fast but simple)
       Low Texture + Low Motion    → "Simple Static"    (easiest for TAA)
    
    Example scenes:
       Complex Moving:  lightfoliage-close (leaves + camera movement)
       Complex Static:  oldmine-close (brick walls, slow pan)
       Simple Moving:   subway-turn (smooth surfaces, fast rotation)
       Simple Static:   cubetest (simple geometry, minimal motion)
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: OPTIMAL PARAMETERS BY SCENE TYPE")
    print("="*80)
    
    # Merge scores with scene features
    merged = scores_df.merge(features_df, on='scene', how='inner')
    
    # Define scene groups based on key features
    # Use median splits for texture and motion
    texture_median = features_df['texture_variance_avg'].median()
    motion_median = features_df['flow_mean_avg'].median()
    
    print(f"\nTexture variance median: {texture_median:.3f}")
    print(f"Motion magnitude median: {motion_median:.3f}")
    
    # Create scene type categories
    def categorize_scene(row):
        high_texture = row['texture_variance_avg'] > texture_median
        high_motion = row['flow_mean_avg'] > motion_median
        
        if high_texture and high_motion:
            return 'Complex Moving'
        elif high_texture and not high_motion:
            return 'Complex Static'
        elif not high_texture and high_motion:
            return 'Simple Moving'
        else:
            return 'Simple Static'
    
    merged['scene_type'] = merged.apply(categorize_scene, axis=1)
    
    # Print scene type breakdown
    print("\nScene Type Breakdown:")
    print("-" * 60)
    scene_type_counts = features_df.merge(
        merged[['scene', 'scene_type']].drop_duplicates(), 
        on='scene'
    )['scene_type'].value_counts().sort_index()
    
    for scene_type, count in scene_type_counts.items():
        scenes_in_type = scene_type_counts.loc[scene_type]
        example_scenes = merged[merged['scene_type'] == scene_type]['scene'].unique()[:3]
        print(f"  {scene_type:20s}: {count:2d} scenes  (e.g., {', '.join(example_scenes)})")
    
    # For each scene type, find optimal parameter values
    scene_types = merged['scene_type'].unique()
    param_groups = merged['param_group'].unique()
    
    recommendations = []
    
    for scene_type in scene_types:
        scene_subset = merged[merged['scene_type'] == scene_type]
        
        print(f"\n{scene_type} Scenes ({len(scene_subset['scene'].unique())} scenes):")
        print("-" * 60)
        
        for param_group in param_groups:
            param_subset = scene_subset[scene_subset['param_group'] == param_group]
            
            if len(param_subset) == 0:
                continue
            
            # Find best parameter value
            best_idx = param_subset['cgvqm_score'].idxmax()
            best_row = param_subset.loc[best_idx]
            
            # Get statistics for this parameter in this scene type
            param_mean = param_subset['cgvqm_score'].mean()
            param_std = param_subset['cgvqm_score'].std()
            param_range = param_subset['cgvqm_score'].max() - param_subset['cgvqm_score'].min()
            
            recommendations.append({
                'scene_type': scene_type,
                'param_group': param_group,
                'optimal_value': best_row['param_value'],
                'best_score': best_row['cgvqm_score'],
                'mean_score': param_mean,
                'score_std': param_std,
                'score_range': param_range,
                'sensitivity': param_range  # How much params matter
            })
            
            print(f"  {param_group:20s}: optimal = {best_row['param_value']:6.2f} "
                  f"(score: {best_row['cgvqm_score']:5.2f}, "
                  f"range: {param_range:5.2f})")
    
    rec_df = pd.DataFrame(recommendations)
    
    # Create heatmap: scene type × parameter group → optimal value
    pivot_optimal = rec_df.pivot(index='scene_type', 
                                  columns='param_group', 
                                  values='optimal_value')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_optimal.values,
        x=pivot_optimal.columns,
        y=pivot_optimal.index,
        text=np.round(pivot_optimal.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='Viridis',
        colorbar=dict(title="Optimal<br>Parameter<br>Value")
    ))
    
    fig.update_layout(
        title='Optimal Parameter Values by Scene Type<br>' +
              '<sub>Shows which parameter values work best for each scene category</sub>',
        xaxis_title='Parameter Group',
        yaxis_title='Scene Type',
        height=500,
        width=800
    )
    
    fig.write_html('analysis_2_optimal_params_by_scene_type.html')
    print("\n✓ Created analysis_2_optimal_params_by_scene_type.html")
    
    # Create sensitivity heatmap: which scene types are most parameter-sensitive?
    pivot_sensitivity = rec_df.pivot(index='scene_type',
                                      columns='param_group',
                                      values='score_range')
    
    fig2 = go.Figure(data=go.Heatmap(
        z=pivot_sensitivity.values,
        x=pivot_sensitivity.columns,
        y=pivot_sensitivity.index,
        text=np.round(pivot_sensitivity.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='Reds',
        colorbar=dict(title="Score<br>Range")
    ))
    
    fig2.update_layout(
        title='Parameter Sensitivity by Scene Type<br>' +
              '<sub>Higher values = More sensitive to parameter choice · Red = Needs careful tuning</sub>',
        xaxis_title='Parameter Group',
        yaxis_title='Scene Type',
        height=500,
        width=800
    )
    
    fig2.write_html('analysis_2_parameter_sensitivity_by_scene_type.html')
    print("✓ Created analysis_2_parameter_sensitivity_by_scene_type.html")
    
    # Save recommendations table
    rec_df.to_csv('analysis_2_recommendations_table.csv', index=False)
    print("✓ Created analysis_2_recommendations_table.csv")
    
    return rec_df, merged


# =============================================================================
# ANALYSIS 3: PARAMETER CURVES
# =============================================================================

def create_parameter_curves_by_scene_type(scores_df, features_df):
    """
    Show how quality changes as each parameter varies, 
    grouped by scene type.
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: PARAMETER CURVES BY SCENE TYPE")
    print("="*80)
    
    # Merge with features
    merged = scores_df.merge(features_df, on='scene', how='inner')
    
    # Categorize scenes
    texture_median = features_df['texture_variance_avg'].median()
    motion_median = features_df['flow_mean_avg'].median()
    
    def categorize_scene(row):
        high_texture = row['texture_variance_avg'] > texture_median
        high_motion = row['flow_mean_avg'] > motion_median
        
        if high_texture and high_motion:
            return 'Complex Moving'
        elif high_texture and not high_motion:
            return 'Complex Static'
        elif not high_texture and high_motion:
            return 'Simple Moving'
        else:
            return 'Simple Static'
    
    merged['scene_type'] = merged.apply(categorize_scene, axis=1)
    
    param_groups = sorted(merged['param_group'].unique())
    
    # Create 2x2 subplot for 4 parameter groups
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[pg.replace('_', ' ').title() for pg in param_groups],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = {
        'Complex Moving': '#e74c3c',
        'Complex Static': '#e67e22', 
        'Simple Moving': '#3498db',
        'Simple Static': '#2ecc71'
    }
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (param_group, (row, col)) in zip(param_groups, positions):
        param_data = merged[merged['param_group'] == param_group]
        
        # Track which scene types we've added to legend
        legend_added = set()
        
        for scene_type in sorted(param_data['scene_type'].unique()):
            subset = param_data[param_data['scene_type'] == scene_type]
            
            # Aggregate: mean score per parameter value for this scene type
            agg = subset.groupby('param_value')['cgvqm_score'].agg(['mean', 'std']).reset_index()
            
            show_legend = scene_type not in legend_added
            if show_legend:
                legend_added.add(scene_type)
            
            fig.add_trace(
                go.Scatter(
                    x=agg['param_value'],
                    y=agg['mean'],
                    mode='lines+markers',
                    name=scene_type,
                    line=dict(color=colors[scene_type], width=2),
                    marker=dict(size=8),
                    showlegend=show_legend,
                    legendgroup=scene_type,
                    error_y=dict(
                        type='data',
                        array=agg['std'],
                        visible=True
                    )
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text='Parameter Value', row=row, col=col)
        fig.update_yaxes(title_text='Quality Score' if col == 1 else '', row=row, col=col)
    
    fig.update_layout(
        title_text='How Quality Changes with Parameter Values<br>' +
                   '<sub>Each line = one scene type · Error bars show variability within scene type</sub>',
        height=800,
        hovermode='x unified',
        legend=dict(
            title='Scene Type',
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        )
    )
    
    fig.write_html('analysis_3_parameter_curves_by_scene_type.html')
    print("\n✓ Created analysis_3_parameter_curves_by_scene_type.html")


# =============================================================================
# ANALYSIS 4: CASE STUDY - WORST SCENE
# =============================================================================

def create_case_study_worst_scene(scores_df, features_df):
    """
    Deep dive into the scene with worst average quality.
    Show all parameter curves for this specific scene.
    """
    print("\n" + "="*80)
    print("ANALYSIS 4: CASE STUDY - WORST PERFORMING SCENE")
    print("="*80)
    
    # Find worst scene by mean quality
    worst_scene = features_df.loc[features_df['mean_cgvqm_all'].idxmin(), 'scene']
    worst_score = features_df.loc[features_df['mean_cgvqm_all'].idxmin(), 'mean_cgvqm_all']
    
    print(f"\nWorst performing scene: {worst_scene}")
    print(f"Average quality score: {worst_score:.2f}")
    
    # Get this scene's features
    scene_features = features_df[features_df['scene'] == worst_scene].iloc[0]
    print(f"Texture variance: {scene_features['texture_variance_avg']:.3f}")
    print(f"Motion magnitude: {scene_features['flow_mean_avg']:.3f}")
    print(f"Edge density: {scene_features['edge_density_avg']:.3f}")
    
    # Get all scores for this scene
    scene_scores = scores_df[scores_df['scene'] == worst_scene]
    
    param_groups = sorted(scene_scores['param_group'].unique())
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[pg.replace('_', ' ').title() for pg in param_groups],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (param_group, (row, col)) in zip(param_groups, positions):
        param_data = scene_scores[scene_scores['param_group'] == param_group]
        param_data = param_data.sort_values('param_value')
        
        # Find optimal
        best_idx = param_data['cgvqm_score'].idxmax()
        best_value = param_data.loc[best_idx, 'param_value']
        best_score = param_data.loc[best_idx, 'cgvqm_score']
        
        # Plot curve
        fig.add_trace(
            go.Scatter(
                x=param_data['param_value'],
                y=param_data['cgvqm_score'],
                mode='lines+markers',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10),
                name=param_group,
                showlegend=False,
                hovertemplate='Value: %{x:.2f}<br>Score: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Mark optimal point
        fig.add_trace(
            go.Scatter(
                x=[best_value],
                y=[best_score],
                mode='markers',
                marker=dict(size=15, color='#e74c3c', symbol='star'),
                name='Optimal',
                showlegend=(row == 1 and col == 1),
                hovertemplate=f'OPTIMAL<br>Value: {best_value:.2f}<br>Score: {best_score:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text='Parameter Value', row=row, col=col)
        fig.update_yaxes(title_text='Quality Score' if col == 1 else '', row=row, col=col)
    
    fig.update_layout(
        title_text=f'Case Study: {worst_scene} (Worst Performing Scene)<br>' +
                   f'<sub>Average score: {worst_score:.2f} · Red star = optimal parameter value</sub>',
        height=800,
        hovermode='closest'
    )
    
    fig.write_html('analysis_4_case_study_worst_scene.html')
    print(f"\n✓ Created analysis_4_case_study_worst_scene.html")
    
    # Print optimal parameters for this scene
    print(f"\nOptimal parameters for {worst_scene}:")
    print("-" * 60)
    for param_group in param_groups:
        param_data = scene_scores[scene_scores['param_group'] == param_group]
        best_idx = param_data['cgvqm_score'].idxmax()
        best_value = param_data.loc[best_idx, 'param_value']
        best_score = param_data.loc[best_idx, 'cgvqm_score']
        worst_score_param = param_data['cgvqm_score'].min()
        improvement = best_score - worst_score_param
        
        print(f"  {param_group:20s}: {best_value:6.2f} "
              f"(score: {best_score:5.2f}, improvement: +{improvement:5.2f})")


# =============================================================================
# ANALYSIS 5: SCENE-SPECIFIC SENSITIVITY
# =============================================================================

def analyze_scene_sensitivity(scores_df, features_df):
    """
    Which scenes are most/least sensitive to parameter choices?
    Correlate sensitivity with scene features.
    """
    print("\n" + "="*80)
    print("ANALYSIS 5: SCENE SENSITIVITY TO PARAMETERS")
    print("="*80)
    
    # For each scene, calculate how much scores vary across all parameters
    scene_sensitivity = []
    
    for scene in scores_df['scene'].unique():
        scene_data = scores_df[scores_df['scene'] == scene]
        
        sensitivity = {
            'scene': scene,
            'overall_std': scene_data['cgvqm_score'].std(),
            'overall_range': scene_data['cgvqm_score'].max() - scene_data['cgvqm_score'].min(),
            'mean_score': scene_data['cgvqm_score'].mean()
        }
        
        # Per-parameter sensitivity
        for param_group in scene_data['param_group'].unique():
            param_data = scene_data[scene_data['param_group'] == param_group]
            sensitivity[f'{param_group}_range'] = param_data['cgvqm_score'].max() - param_data['cgvqm_score'].min()
        
        scene_sensitivity.append(sensitivity)
    
    sens_df = pd.DataFrame(scene_sensitivity)
    
    # Merge with features
    sens_merged = sens_df.merge(features_df, on='scene', how='inner')
    
    # Sort by overall sensitivity
    sens_merged = sens_merged.sort_values('overall_range', ascending=False)
    
    print("\nMost Parameter-Sensitive Scenes:")
    print("-" * 80)
    print(sens_merged[['scene', 'overall_range', 'mean_score', 
                       'texture_variance_avg', 'flow_mean_avg']].head(10).to_string(index=False))
    
    print("\n\nLeast Parameter-Sensitive Scenes:")
    print("-" * 80)
    print(sens_merged[['scene', 'overall_range', 'mean_score',
                       'texture_variance_avg', 'flow_mean_avg']].tail(10).to_string(index=False))
    
    # Correlate sensitivity with features
    print("\n\nCorrelation: Scene Features vs Parameter Sensitivity")
    print("-" * 80)
    
    feature_cols = ['texture_variance_avg', 'edge_density_avg', 'high_freq_energy_avg',
                    'flow_mean_avg', 'temporal_activity_avg']
    
    correlations = []
    for feat in feature_cols:
        if feat in sens_merged.columns:
            rho, p = spearmanr(sens_merged[feat], sens_merged['overall_range'])
            correlations.append({
                'Feature': feat,
                'Spearman ρ': rho,
                'p-value': p,
                'Significant': 'Yes' if p < 0.05 else 'No'
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Spearman ρ', key=abs, ascending=False)
    print(corr_df.to_string(index=False))
    
    # Create scatter plot: texture variance vs sensitivity
    fig = px.scatter(
        sens_merged,
        x='texture_variance_avg',
        y='overall_range',
        size='mean_score',
        color='flow_mean_avg',
        hover_name='scene',
        hover_data=['mean_score', 'edge_density_avg'],
        title='Scene Feature vs Parameter Sensitivity<br>' +
              '<sub>Do complex scenes require more careful parameter tuning?</sub>',
        labels={
            'texture_variance_avg': 'Texture Variance (Complexity)',
            'overall_range': 'Parameter Sensitivity (Score Range)',
            'flow_mean_avg': 'Motion Magnitude',
            'mean_score': 'Avg Quality'
        }
    )
    
    fig.update_layout(height=600)
    fig.write_html('analysis_5_scene_sensitivity.html')
    print("\n✓ Created analysis_5_scene_sensitivity.html")
    
    sens_merged.to_csv('analysis_5_scene_sensitivity_table.csv', index=False)
    print("✓ Created analysis_5_scene_sensitivity_table.csv")
    
    return sens_merged


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TAA PARAMETER ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    scores_df = pd.read_csv('master_scores_cleaned.csv')  # Your per-video scores
    features_df = pd.read_csv('scene_level_metrics.csv')  # Your scene features
    
    # Also load video features if available
    try:
        video_features = pd.read_csv('video_features.csv')
        features_df = features_df.merge(video_features, on='scene', how='left')
        print("✓ Loaded video_features.csv")
    except FileNotFoundError:
        print("⚠ video_features.csv not found, using only scene_level_metrics")
    
    print(f"✓ Loaded {len(scores_df)} video scores across {len(scores_df['scene'].unique())} scenes")
    print(f"✓ Loaded features for {len(features_df)} scenes")
    
    # Run all analyses
    print("\n" + "="*80)
    print("RUNNING ANALYSES")
    print("="*80)
    
    # Analysis 1: Parameter Importance
    importance_df = analyze_parameter_importance(scores_df)
    
    # Analysis 2: Optimal parameters by scene type
    recommendations_df, merged_data = analyze_optimal_params_by_scene_type(scores_df, features_df)
    
    # Analysis 3: Parameter curves by scene type
    create_parameter_curves_by_scene_type(scores_df, features_df)
    
    # Analysis 4: Case study of worst scene
    create_case_study_worst_scene(scores_df, features_df)
    
    # Analysis 5: Scene sensitivity analysis
    sensitivity_df = analyze_scene_sensitivity(scores_df, features_df)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  Analysis 1 - Parameter Importance:")
    print("    • analysis_1_parameter_importance.html")
    print("    • analysis_1_parameter_importance_detailed.html")
    print("\n  Analysis 2 - Optimal Parameters by Scene Type:")
    print("    • analysis_2_optimal_params_by_scene_type.html")
    print("    • analysis_2_parameter_sensitivity_by_scene_type.html")
    print("    • analysis_2_recommendations_table.csv")
    print("\n  Analysis 3 - Parameter Curves:")
    print("    • analysis_3_parameter_curves_by_scene_type.html")
    print("\n  Analysis 4 - Case Study:")
    print("    • analysis_4_case_study_worst_scene.html")
    print("\n  Analysis 5 - Scene Sensitivity:")
    print("    • analysis_5_scene_sensitivity.html")
    print("    • analysis_5_scene_sensitivity_table.csv")
    print("\n" + "="*80)