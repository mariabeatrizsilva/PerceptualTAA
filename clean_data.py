"""
Clean the master scores data by removing invalid entries.

Removes:
1. All scores from oldmine-warm (inconsistent)
2. Invalid scores from fantasticvillage-open (hist_percent 125 and 150)
"""
import pandas as pd

def clean_scores(input_file="master_scores.csv", output_file="master_scores_cleaned.csv"):
    """
    Clean master scores by removing invalid entries.
    """
    # Load data
    df = pd.read_csv(input_file)
    print(f"Original data: {len(df)} records")
    
    # Track removals
    removed = []
    
    # 1. Remove all oldmine-warm scores
    oldmine_warm_count = len(df[df['scene'] == 'oldmine-warm'])
    df = df[df['scene'] != 'oldmine-warm']
    removed.append(f"Removed {oldmine_warm_count} records from oldmine-warm (inconsistent scores)")
    
    # 2. Remove invalid fantasticvillage-open scores
    # hist_percent values 125 and 150 have scores outside valid range (0-100)
    invalid_fantasy = df[
        (df['scene'] == 'fantasticvillage-open') & 
        (df['param_group'] == 'hist_percent') & 
        (df['param_value'].isin(['125', '150']))
    ]
    invalid_count = len(invalid_fantasy)
    
    df = df[~(
        (df['scene'] == 'fantasticvillage-open') & 
        (df['param_group'] == 'hist_percent') & 
        (df['param_value'].isin(['125', '150']))
    )]
    removed.append(f"Removed {invalid_count} records from fantasticvillage-open (invalid scores: hist_percent 125, 150)")
    
    # 3. Additional sanity check: remove any scores outside 0-100 range
    invalid_range = df[(df['cgvqm_score'] < 0) | (df['cgvqm_score'] > 100)]
    if len(invalid_range) > 0:
        print("\nWarning: Found additional invalid scores:")
        print(invalid_range[['scene', 'param_group', 'param_value', 'cgvqm_score']])
        df = df[(df['cgvqm_score'] >= 0) & (df['cgvqm_score'] <= 100)]
        removed.append(f"Removed {len(invalid_range)} additional records with scores outside 0-100 range")
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("DATA CLEANING SUMMARY")
    print("="*80)
    for item in removed:
        print(f"  - {item}")
    print(f"\nCleaned data: {len(df)} records")
    print(f"Removed: {len(pd.read_csv(input_file)) - len(df)} records total")
    print(f"Saved to: {output_file}")
    
    # Summary by scene
    print("\n" + "="*80)
    print("RECORDS PER SCENE (after cleaning):")
    print("="*80)
    scene_counts = df.groupby('scene').size().sort_values(ascending=False)
    print(scene_counts)
    
    return df

if __name__ == "__main__":
    print("="*80)
    print("CLEANING MASTER SCORES DATA")
    print("="*80)
    print("\nRemoving:")
    print("  1. All oldmine-warm records (inconsistent scores)")
    print("  2. fantasticvillage-open hist_percent 125 and 150 (invalid scores)")
    print()
    
    df_cleaned = clean_scores()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Use 'master_scores_cleaned.csv' for all future analysis")
    print("2. Re-run compute_scene_metrics.py with cleaned data")
    print("3. Re-run analyze_labels.py with cleaned data")