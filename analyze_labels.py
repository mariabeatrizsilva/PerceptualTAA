"""
Quick analysis of manual labels + quality scores.
Shows immediate insights while Phase 2 is running.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_data():
    """Load manual labels and scene-level metrics."""
    labels = pd.read_csv('manual_labels_template.csv')
    
    # Try cleaned data first, fall back to original
    try:
        metrics = pd.read_csv('scene_level_metrics_cleaned.csv')
        print("Using cleaned scene metrics")
    except FileNotFoundError:
        metrics = pd.read_csv('scene_level_metrics.csv')
        print("Using original scene metrics (run clean_data.py to clean)")
    
    # Merge (inner join will automatically exclude scenes without metrics)
    df = labels.merge(metrics, on='scene', how='inner')
    
    # Report if any scenes were excluded
    excluded = set(labels['scene']) - set(df['scene'])
    if excluded:
        print(f"Note: {len(excluded)} scene(s) excluded (no quality metrics): {excluded}")
    
    return df

def create_visualizations(df):
    """Create quick exploratory visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TAA Quality Analysis: Manual Labels vs Quality Scores', 
                 fontsize=16, fontweight='bold')
    
    # 1. Quality by Motion Speed
    ax = axes[0, 0]
    motion_order = ['None', 'Slow', 'Medium', 'Medium/Fast', 'Fast']
    motion_data = df[df['motion_speed'].isin(motion_order)]
    if len(motion_data) > 0:
        sns.boxplot(data=motion_data, x='motion_speed', y='mean_cgvqm_all', 
                   order=motion_order, ax=ax, palette='viridis')
        ax.set_title('Quality vs Motion Speed', fontweight='bold')
        ax.set_xlabel('Motion Speed')
        ax.set_ylabel('Mean CGVQM Score')
        ax.tick_params(axis='x', rotation=45)
    
    # 2. Quality by Lighting
    ax = axes[0, 1]
    lighting_order = ['Dark', 'Medium', 'Light', 'High']
    lighting_data = df[df['lighting'].isin(lighting_order)]
    if len(lighting_data) > 0:
        sns.boxplot(data=lighting_data, x='lighting', y='mean_cgvqm_all',
                   order=lighting_order, ax=ax, palette='rocket')
        ax.set_title('Quality vs Lighting', fontweight='bold')
        ax.set_xlabel('Lighting Level')
        ax.set_ylabel('Mean CGVQM Score')
    
    # 3. Quality by Texture Detail (numeric)
    ax = axes[0, 2]
    # Convert texture_detail to numeric if it's not already
    df['texture_numeric'] = pd.to_numeric(df['texture_detail'], errors='coerce')
    texture_data = df.dropna(subset=['texture_numeric'])
    if len(texture_data) > 0:
        ax.scatter(texture_data['texture_numeric'], texture_data['mean_cgvqm_all'], 
                  s=100, alpha=0.6, c=texture_data['mean_cgvqm_all'], cmap='RdYlGn')
        ax.set_title('Quality vs Texture Detail', fontweight='bold')
        ax.set_xlabel('Texture Detail (1-7 scale)')
        ax.set_ylabel('Mean CGVQM Score')
        
        # Add trend line
        z = np.polyfit(texture_data['texture_numeric'], texture_data['mean_cgvqm_all'], 1)
        p = np.poly1d(z)
        ax.plot(texture_data['texture_numeric'].sort_values(), 
               p(texture_data['texture_numeric'].sort_values()), 
               "r--", alpha=0.8, linewidth=2)
    
    # 4. Quality by Vegetation
    ax = axes[1, 0]
    veg_order = ['No', 'Little', 'Lots', 'Yes']
    veg_data = df[df['has_vegetation'].isin(veg_order)]
    if len(veg_data) > 0:
        sns.boxplot(data=veg_data, x='has_vegetation', y='mean_cgvqm_all',
                   order=veg_order, ax=ax, palette='Greens')
        ax.set_title('Quality vs Vegetation', fontweight='bold')
        ax.set_xlabel('Vegetation Level')
        ax.set_ylabel('Mean CGVQM Score')
    
    # 5. Quality by Visual Complexity (numeric)
    ax = axes[1, 1]
    df['complexity_numeric'] = pd.to_numeric(df['visual_complexity'], errors='coerce')
    complexity_data = df.dropna(subset=['complexity_numeric'])
    if len(complexity_data) > 0:
        ax.scatter(complexity_data['complexity_numeric'], complexity_data['mean_cgvqm_all'],
                  s=100, alpha=0.6, c=complexity_data['mean_cgvqm_all'], cmap='RdYlGn')
        ax.set_title('Quality vs Visual Complexity', fontweight='bold')
        ax.set_xlabel('Visual Complexity (1-7 scale)')
        ax.set_ylabel('Mean CGVQM Score')
        
        # Add trend line
        z = np.polyfit(complexity_data['complexity_numeric'], complexity_data['mean_cgvqm_all'], 1)
        p = np.poly1d(z)
        ax.plot(complexity_data['complexity_numeric'].sort_values(),
               p(complexity_data['complexity_numeric'].sort_values()),
               "r--", alpha=0.8, linewidth=2)
    
    # 6. Parameter Sensitivity by Motion Type
    ax = axes[1, 2]
    motion_types = df['motion_type'].value_counts().head(5).index
    motion_data = df[df['motion_type'].isin(motion_types)]
    if len(motion_data) > 0:
        sns.boxplot(data=motion_data, x='motion_type', y='range_cgvqm_all',
                   ax=ax, palette='coolwarm')
        ax.set_title('Parameter Sensitivity vs Motion Type', fontweight='bold')
        ax.set_xlabel('Motion Type')
        ax.set_ylabel('Quality Range (Sensitivity)')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('manual_labels_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: manual_labels_analysis.png")
    
    return fig

def print_insights(df):
    """Print key insights from the data."""
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM MANUAL LABELS")
    print("="*80)
    
    # Convert to numeric
    df['texture_numeric'] = pd.to_numeric(df['texture_detail'], errors='coerce')
    df['complexity_numeric'] = pd.to_numeric(df['visual_complexity'], errors='coerce')
    
    # 1. Motion speed effect
    print("\n1. MOTION SPEED vs QUALITY:")
    motion_quality = df.groupby('motion_speed')['mean_cgvqm_all'].agg(['mean', 'std', 'count'])
    print(motion_quality.sort_values('mean', ascending=False))
    
    # 2. Lighting effect
    print("\n2. LIGHTING vs QUALITY:")
    lighting_quality = df.groupby('lighting')['mean_cgvqm_all'].agg(['mean', 'std', 'count'])
    print(lighting_quality.sort_values('mean', ascending=False))
    
    # 3. Vegetation effect
    print("\n3. VEGETATION vs QUALITY:")
    veg_quality = df.groupby('has_vegetation')['mean_cgvqm_all'].agg(['mean', 'std', 'count'])
    print(veg_quality.sort_values('mean', ascending=False))
    
    # 4. Texture correlation
    if 'texture_numeric' in df.columns:
        from scipy.stats import pearsonr, spearmanr
        texture_clean = df.dropna(subset=['texture_numeric', 'mean_cgvqm_all'])
        if len(texture_clean) > 2:
            r_pearson, p_pearson = pearsonr(texture_clean['texture_numeric'], 
                                           texture_clean['mean_cgvqm_all'])
            r_spearman, p_spearman = spearmanr(texture_clean['texture_numeric'],
                                              texture_clean['mean_cgvqm_all'])
            print(f"\n4. TEXTURE DETAIL CORRELATION:")
            print(f"   Pearson r = {r_pearson:.3f} (p = {p_pearson:.3f})")
            print(f"   Spearman r = {r_spearman:.3f} (p = {p_spearman:.3f})")
            if p_spearman < 0.05:
                direction = "NEGATIVE" if r_spearman < 0 else "POSITIVE"
                print(f"   → {direction} correlation (significant!)")
    
    # 5. Complexity correlation
    if 'complexity_numeric' in df.columns:
        complexity_clean = df.dropna(subset=['complexity_numeric', 'mean_cgvqm_all'])
        if len(complexity_clean) > 2:
            r_pearson, p_pearson = pearsonr(complexity_clean['complexity_numeric'],
                                           complexity_clean['mean_cgvqm_all'])
            r_spearman, p_spearman = spearmanr(complexity_clean['complexity_numeric'],
                                              complexity_clean['mean_cgvqm_all'])
            print(f"\n5. VISUAL COMPLEXITY CORRELATION:")
            print(f"   Pearson r = {r_pearson:.3f} (p = {p_pearson:.3f})")
            print(f"   Spearman r = {r_spearman:.3f} (p = {p_spearman:.3f})")
            if p_spearman < 0.05:
                direction = "NEGATIVE" if r_spearman < 0 else "POSITIVE"
                print(f"   → {direction} correlation (significant!)")
    
    # 6. Best and worst scenes with labels
    print("\n6. SCENE RANKINGS:")
    print("\nTop 5 Best Quality Scenes:")
    top5 = df.nlargest(5, 'mean_cgvqm_all')[['scene', 'mean_cgvqm_all', 'motion_speed', 
                                               'visual_complexity', 'texture_detail', 'lighting']]
    print(top5.to_string(index=False))
    
    print("\nTop 5 Worst Quality Scenes:")
    bottom5 = df.nsmallest(5, 'mean_cgvqm_all')[['scene', 'mean_cgvqm_all', 'motion_speed',
                                                   'visual_complexity', 'texture_detail', 'lighting']]
    print(bottom5.to_string(index=False))
    
    # 7. Parameter sensitivity insights
    print("\n7. PARAMETER SENSITIVITY:")
    print("\nMost Parameter-Sensitive Scenes:")
    sensitive = df.nlargest(5, 'range_cgvqm_all')[['scene', 'range_cgvqm_all', 'motion_speed',
                                                     'has_vegetation', 'texture_detail']]
    print(sensitive.to_string(index=False))
    
    print("\nLeast Parameter-Sensitive (Most Robust) Scenes:")
    robust = df.nsmallest(5, 'range_cgvqm_all')[['scene', 'range_cgvqm_all', 'motion_speed',
                                                   'visual_complexity', 'lighting']]
    print(robust.to_string(index=False))

if __name__ == "__main__":
    print("="*80)
    print("ANALYZING MANUAL LABELS + QUALITY SCORES")
    print("="*80)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} scenes with labels and quality metrics")
    
    # Print insights
    print_insights(df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS...")
    print("="*80)
    create_visualizations(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nCheck out 'manual_labels_analysis.png' for visualizations")
    print("\nThese are preliminary insights. Once Phase 2 completes,")
    print("we'll have even more detailed correlations with optical flow,")
    print("edge density, and other computed features!")