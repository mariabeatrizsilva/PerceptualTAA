"""
Phase 3: Interactive Analysis Dashboard (Updated)
Merge all data and create interactive visualizations including manual labels.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import pearsonr, spearmanr

def load_all_data():
    """Load and merge all data sources."""
    print("Loading data...")
    
    # Load cleaned scores and metrics
    try:
        metrics = pd.read_csv('scene_level_metrics_cleaned.csv')
        print("  ✓ Loaded scene_level_metrics_cleaned.csv")
    except FileNotFoundError:
        metrics = pd.read_csv('scene_level_metrics.csv')
        print("  ✓ Loaded scene_level_metrics.csv")
    
    # Load manual labels
    labels = pd.read_csv('manual_labels_template.csv')
    print("  ✓ Loaded manual_labels_template.csv")
    
    # Load video features
    try:
        features = pd.read_csv('video_features.csv')
        print("  ✓ Loaded video_features.csv")
    except FileNotFoundError:
        print("  ✗ video_features.csv not found - run Phase 2 first!")
        return None
    
    # Merge everything
    df = metrics.merge(labels, on='scene', how='inner')
    df = df.merge(features, on='scene', how='inner')
    
    print(f"\n✓ Merged data: {len(df)} scenes with all features")
    
    return df


def create_feature_vs_quality_plot(df, feature_col, feature_name, color_by='motion_speed'):
    """Create interactive scatter plot of feature vs quality."""
    
    # Calculate correlation
    valid_data = df[[feature_col, 'mean_cgvqm_all']].dropna()
    if len(valid_data) > 2:
        r_pearson, p_pearson = pearsonr(valid_data[feature_col], valid_data['mean_cgvqm_all'])
        r_spearman, p_spearman = spearmanr(valid_data[feature_col], valid_data['mean_cgvqm_all'])
        corr_text = f"Pearson r={r_pearson:.3f} (p={p_pearson:.3f})<br>Spearman r={r_spearman:.3f} (p={p_spearman:.3f})"
    else:
        corr_text = "Insufficient data"
    
    fig = px.scatter(
        df, 
        x=feature_col, 
        y='mean_cgvqm_all',
        color=color_by,
        hover_name='scene',
        hover_data={
            'scene': False,
            feature_col: ':.2f',
            'mean_cgvqm_all': ':.2f',
            'motion_speed': True,
            'lighting': True,
            'has_vegetation': True,
            'visual_complexity': True,
            color_by: False
        },
        title=f'{feature_name} vs Quality Score<br><sub>{corr_text}</sub>',
        labels={
            feature_col: feature_name,
            'mean_cgvqm_all': 'Mean CGVQM Score (Higher = Better Quality)'
        },
        size_max=15
    )
    
    # Add trend line
    if len(valid_data) > 2:
        z = np.polyfit(valid_data[feature_col], valid_data['mean_cgvqm_all'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[feature_col].min(), df[feature_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash', width=2),
            showlegend=True
        ))
    
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
    fig.update_layout(height=600, hovermode='closest')
    
    return fig


def create_manual_label_plots(df):
    """Create plots for manually labeled categorical variables."""
    
    print("\nCreating manual label plots...")
    
    # Motion Speed
    if 'motion_speed' in df.columns:
        motion_order = ['None', 'Slow', 'Medium', 'Medium/Fast', 'Fast']
        motion_data = df[df['motion_speed'].isin(motion_order)]
        if len(motion_data) > 0:
            fig = px.box(
                motion_data, 
                x='motion_speed', 
                y='mean_cgvqm_all',
                color='motion_speed',
                category_orders={'motion_speed': motion_order},
                title='Quality by Motion Speed<br><sub>Shows how camera motion speed affects TAA quality</sub>',
                labels={
                    'motion_speed': 'Motion Speed (Manual Label)',
                    'mean_cgvqm_all': 'Mean CGVQM Score'
                },
                hover_data=['scene', 'environment', 'has_vegetation']
            )
            fig.add_scatter(
                x=motion_data['motion_speed'],
                y=motion_data['mean_cgvqm_all'],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Scenes',
                text=motion_data['scene'],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<extra></extra>'
            )
            fig.update_layout(height=600, showlegend=True)
            fig.write_html('interactive_manual_motion_speed.html')
            print(f"  ✓ Created interactive_manual_motion_speed.html")
    
    # Lighting
    if 'lighting' in df.columns:
        lighting_order = ['Dark', 'Medium', 'Light', 'High']
        lighting_data = df[df['lighting'].isin(lighting_order)]
        if len(lighting_data) > 0:
            fig = px.box(
                lighting_data,
                x='lighting',
                y='mean_cgvqm_all',
                color='lighting',
                category_orders={'lighting': lighting_order},
                title='Quality by Lighting Level<br><sub>Do darker scenes hide TAA artifacts?</sub>',
                labels={
                    'lighting': 'Lighting Level (Manual Label)',
                    'mean_cgvqm_all': 'Mean CGVQM Score'
                },
                hover_data=['scene', 'environment', 'motion_speed']
            )
            fig.add_scatter(
                x=lighting_data['lighting'],
                y=lighting_data['mean_cgvqm_all'],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='diamond'),
                name='Scenes',
                text=lighting_data['scene'],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<extra></extra>'
            )
            fig.update_layout(height=600, showlegend=True)
            fig.write_html('interactive_manual_lighting.html')
            print(f"  ✓ Created interactive_manual_lighting.html")
    
    # Vegetation
    if 'has_vegetation' in df.columns:
        veg_data = df[df['has_vegetation'].notna()]
        if len(veg_data) > 0:
            fig = px.box(
                veg_data,
                x='has_vegetation',
                y='mean_cgvqm_all',
                color='has_vegetation',
                title='Quality by Vegetation Level<br><sub>Does foliage reduce TAA quality?</sub>',
                labels={
                    'has_vegetation': 'Vegetation Level (Manual Label)',
                    'mean_cgvqm_all': 'Mean CGVQM Score'
                },
                hover_data=['scene', 'environment', 'motion_speed']
            )
            fig.add_scatter(
                x=veg_data['has_vegetation'],
                y=veg_data['mean_cgvqm_all'],
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond'),
                name='Scenes',
                text=veg_data['scene'],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<extra></extra>'
            )
            fig.update_layout(height=600, showlegend=True)
            fig.write_html('interactive_manual_vegetation.html')
            print(f"  ✓ Created interactive_manual_vegetation.html")
    
    # Visual Complexity (numeric)
    if 'visual_complexity' in df.columns:
        df['complexity_numeric'] = pd.to_numeric(df['visual_complexity'], errors='coerce')
        complexity_data = df.dropna(subset=['complexity_numeric'])
        if len(complexity_data) > 0:
            fig = px.scatter(
                complexity_data,
                x='complexity_numeric',
                y='mean_cgvqm_all',
                color='motion_speed',
                size='range_cgvqm_all',
                title='Quality by Visual Complexity<br><sub>Manual complexity rating (1-7 scale) vs quality</sub>',
                labels={
                    'complexity_numeric': 'Visual Complexity (Manual Label, 1-7)',
                    'mean_cgvqm_all': 'Mean CGVQM Score',
                    'range_cgvqm_all': 'Parameter Sensitivity'
                },
                hover_data=['scene', 'environment', 'has_vegetation', 'lighting']
            )
            
            # Add trend line
            if len(complexity_data) > 2:
                z = np.polyfit(complexity_data['complexity_numeric'], 
                             complexity_data['mean_cgvqm_all'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(complexity_data['complexity_numeric'].min(),
                                    complexity_data['complexity_numeric'].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend, y=p(x_trend),
                    mode='lines', name='Trend',
                    line=dict(color='red', dash='dash', width=2)
                ))
            
            fig.update_layout(height=600)
            fig.write_html('interactive_manual_complexity.html')
            print(f"  ✓ Created interactive_manual_complexity.html")
    
    # Texture Detail (numeric)
    if 'texture_detail' in df.columns:
        df['texture_numeric'] = pd.to_numeric(df['texture_detail'], errors='coerce')
        texture_data = df.dropna(subset=['texture_numeric'])
        if len(texture_data) > 0:
            fig = px.scatter(
                texture_data,
                x='texture_numeric',
                y='mean_cgvqm_all',
                color='has_vegetation',
                size='range_cgvqm_all',
                title='Quality by Texture Detail<br><sub>Manual texture rating (1-7 scale) vs quality</sub>',
                labels={
                    'texture_numeric': 'Texture Detail (Manual Label, 1-7)',
                    'mean_cgvqm_all': 'Mean CGVQM Score',
                    'range_cgvqm_all': 'Parameter Sensitivity'
                },
                hover_data=['scene', 'environment', 'motion_speed', 'lighting']
            )
            
            # Add trend line
            if len(texture_data) > 2:
                z = np.polyfit(texture_data['texture_numeric'], 
                             texture_data['mean_cgvqm_all'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(texture_data['texture_numeric'].min(),
                                    texture_data['texture_numeric'].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend, y=p(x_trend),
                    mode='lines', name='Trend',
                    line=dict(color='red', dash='dash', width=2)
                ))
            
            fig.update_layout(height=600)
            fig.write_html('interactive_manual_texture.html')
            print(f"  ✓ Created interactive_manual_texture.html")


def create_comparison_dashboard(df):
    """Create comprehensive interactive dashboard."""
    
    # Key computed features to analyze
    key_features = [
        ('flow_mean_avg', 'Average Motion Magnitude (Optical Flow)'),
        ('flow_std_avg', 'Motion Variability'),
        ('flow_max_avg', 'Peak Motion Magnitude'),
        ('edge_density_avg', 'Edge Density (Complexity)'),
        ('texture_variance_avg', 'Texture Variance (Detail)'),
        ('temporal_activity_avg', 'Temporal Activity (Scene Change)'),
        ('high_freq_energy_avg', 'High-Frequency Energy (Fine Detail)'),
        ('brightness_avg', 'Brightness'),
        ('contrast_avg', 'Contrast'),
        ('color_entropy_avg', 'Color Diversity'),
        ('saturation_mean', 'Color Saturation')
    ]
    
    print("\nCreating computed feature plots...")
    
    # Create individual HTML files for each computed feature
    for feature_col, feature_name in key_features:
        if feature_col in df.columns:
            fig = create_feature_vs_quality_plot(df, feature_col, feature_name)
            filename = f"interactive_{feature_col}.html"
            fig.write_html(filename)
            print(f"  ✓ Created {filename}")
        else:
            print(f"  ⚠ Skipped {feature_col} (not in data)")
    
    # Create manual label plots
    create_manual_label_plots(df)
    
    # Create combined correlation heatmap
    create_correlation_heatmap(df)
    
    # Create multi-feature comparison
    create_multi_feature_plot(df, key_features)


def create_correlation_heatmap(df):
    """Create interactive correlation heatmap."""
    
    # Select numeric columns for correlation
    feature_cols = [
        'flow_mean_avg', 'flow_std_avg', 'flow_max_avg',
        'edge_density_avg', 'texture_variance_avg', 'high_freq_energy_avg',
        'temporal_activity_avg', 'brightness_avg', 'contrast_avg',
        'color_entropy_avg', 'saturation_mean'
    ]
    
    # Add complexity/texture if numeric
    if 'visual_complexity' in df.columns:
        df['complexity_numeric'] = pd.to_numeric(df['visual_complexity'], errors='coerce')
        feature_cols.append('complexity_numeric')
    if 'texture_detail' in df.columns:
        df['texture_numeric'] = pd.to_numeric(df['texture_detail'], errors='coerce')
        feature_cols.append('texture_numeric')
    
    # Quality metrics
    quality_cols = ['mean_cgvqm_all', 'std_cgvqm_all', 'range_cgvqm_all']
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    quality_cols = [col for col in quality_cols if col in df.columns]
    
    # Calculate correlations
    corr_data = df[feature_cols + quality_cols].corr()
    
    # Focus on correlations with quality metrics
    corr_with_quality = corr_data.loc[feature_cols, quality_cols]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_with_quality.values,
        x=corr_with_quality.columns,
        y=corr_with_quality.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_with_quality.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlations with Quality Metrics<br><sub>Red = Negative, Blue = Positive</sub>',
        xaxis_title='Quality Metrics',
        yaxis_title='Video Features',
        height=800,
        width=600
    )
    
    filename = "interactive_correlation_heatmap.html"
    fig.write_html(filename)
    print(f"  ✓ Created {filename}")


def create_multi_feature_plot(df, key_features):
    """Create side-by-side comparison of multiple features."""
    
    # Select top 4 most correlated features
    correlations = {}
    for feature_col, _ in key_features:
        if feature_col in df.columns:
            valid_data = df[[feature_col, 'mean_cgvqm_all']].dropna()
            if len(valid_data) > 2:
                r, _ = spearmanr(valid_data[feature_col], valid_data['mean_cgvqm_all'])
                correlations[feature_col] = abs(r)
    
    # Sort by absolute correlation
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:4]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[dict(key_features)[feat[0]] for feat in top_features],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (feature_col, corr_val), (row, col) in zip(top_features, positions):
        feature_name = dict(key_features)[feature_col]
        
        # Add scatter points
        for motion_speed in df['motion_speed'].unique():
            subset = df[df['motion_speed'] == motion_speed]
            fig.add_trace(
                go.Scatter(
                    x=subset[feature_col],
                    y=subset['mean_cgvqm_all'],
                    mode='markers',
                    name=str(motion_speed),
                    text=subset['scene'],
                    hovertemplate='<b>%{text}</b><br>' +
                                f'{feature_name}: %{{x:.2f}}<br>' +
                                'CGVQM: %{y:.2f}<extra></extra>',
                    marker=dict(size=10, line=dict(width=1, color='white')),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
        
        # Add trend line
        valid_data = df[[feature_col, 'mean_cgvqm_all']].dropna()
        if len(valid_data) > 2:
            z = np.polyfit(valid_data[feature_col], valid_data['mean_cgvqm_all'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[feature_col].min(), df[feature_col].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Update axes
        fig.update_xaxes(title_text=feature_name, row=row, col=col)
        fig.update_yaxes(title_text='CGVQM Score' if col == 1 else '', row=row, col=col)
    
    fig.update_layout(
        title_text='Top 4 Most Correlated Features with Quality<br><sub>Colored by Motion Speed</sub>',
        height=800,
        hovermode='closest',
        legend=dict(title='Motion Speed')
    )
    
    filename = "interactive_top_features.html"
    fig.write_html(filename)
    print(f"  ✓ Created {filename}")


def print_correlation_summary(df):
    """Print correlation summary statistics."""
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Features to analyze
    features = {
        'flow_mean_avg': 'Motion Magnitude',
        'flow_std_avg': 'Motion Variability',
        'edge_density_avg': 'Edge Density',
        'texture_variance_avg': 'Texture Variance',
        'temporal_activity_avg': 'Temporal Activity',
        'high_freq_energy_avg': 'High-Freq Energy',
        'brightness_avg': 'Brightness',
        'contrast_avg': 'Contrast'
    }
    
    correlations = []
    
    for feature_col, feature_name in features.items():
        if feature_col in df.columns:
            valid_data = df[[feature_col, 'mean_cgvqm_all']].dropna()
            if len(valid_data) > 2:
                r_spearman, p_spearman = spearmanr(valid_data[feature_col], 
                                                   valid_data['mean_cgvqm_all'])
                correlations.append({
                    'Feature': feature_name,
                    'Spearman r': r_spearman,
                    'p-value': p_spearman,
                    'Significant': 'Yes' if p_spearman < 0.05 else 'No'
                })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Spearman r', key=abs, ascending=False)
    
    print("\nCorrelations with Mean CGVQM Score:")
    print("(Negative = Feature increases → Quality decreases)")
    print("(Positive = Feature increases → Quality increases)")
    print("-" * 80)
    print(corr_df.to_string(index=False))
    
    # Significant findings
    significant = corr_df[corr_df['Significant'] == 'Yes']
    if len(significant) > 0:
        print("\n" + "="*80)
        print("KEY FINDINGS (Statistically Significant):")
        print("="*80)
        for _, row in significant.iterrows():
            direction = "NEGATIVELY" if row['Spearman r'] < 0 else "POSITIVELY"
            strength = "strongly" if abs(row['Spearman r']) > 0.6 else "moderately"
            print(f"  • {row['Feature']} is {strength} {direction} correlated with quality")
            print(f"    (r = {row['Spearman r']:.3f}, p = {row['p-value']:.3f})")


if __name__ == "__main__":
    print("="*80)
    print("PHASE 3: INTERACTIVE ANALYSIS WITH ALL FEATURES")
    print("="*80)
    
    # Load all data
    df = load_all_data()
    
    if df is None:
        print("\nError: Could not load all required data files.")
        print("Make sure you have run:")
        print("  1. Phase 1: parse_scores.py, compute_scene_metrics.py")
        print("  2. Phase 2: extract_features.py")
        print("  3. Data cleaning: clean_data.py")
        exit(1)
    
    # Print correlation summary
    print_correlation_summary(df)
    
    # Create interactive visualizations
    create_comparison_dashboard(df)
    
    print("\n" + "="*80)
    print("INTERACTIVE VISUALIZATIONS CREATED!")
    print("="*80)
    print("\nComputed Feature Plots:")
    print("  • interactive_flow_mean_avg.html - Motion magnitude vs quality")
    print("  • interactive_edge_density_avg.html - Complexity vs quality")
    print("  • interactive_texture_variance_avg.html - Texture vs quality")
    print("  • interactive_high_freq_energy_avg.html - Fine detail vs quality")
    print("  • ... and more")
    print("\nManual Label Plots:")
    print("  • interactive_manual_motion_speed.html - Motion speed categories")
    print("  • interactive_manual_lighting.html - Lighting levels")
    print("  • interactive_manual_vegetation.html - Vegetation presence")
    print("  • interactive_manual_complexity.html - Visual complexity rating")
    print("  • interactive_manual_texture.html - Texture detail rating")
    print("\nSummary Plots:")
    print("  • interactive_correlation_heatmap.html - All correlations")
    print("  • interactive_top_features.html - Top 4 features side-by-side")
    print("\nFeatures:")
    print("  ✓ Hover over points to see scene names and details")
    print("  ✓ Click legend to toggle categories")
    print("  ✓ Zoom, pan, and explore interactively")