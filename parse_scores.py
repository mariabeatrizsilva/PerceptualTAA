"""
Parse all CGVQM score JSON files into a master CSV dataset.
"""
import json
import pandas as pd
from pathlib import Path

# Scene names
SCENE_NAMES = [
    "abandoned", "abandoned-demo", "abandoned-flipped", "cubetest", 
    "fantasticvillage-open", "lightfoliage", "lightfoliage-close", 
    "oldmine", "oldmine-close", "oldmine-warm", "quarry-all", 
    "quarry-rocksonly", "resto-close", "resto-fwd", "resto-pan", 
    "scifi", "subway-lookdown", "subway-turn", "wildwest-bar", 
    "wildwest-barzoom", "wildwest-behindcounter", "wildwest-store", 
    "wildwest-town"
]

# Parameter groups
PARAM_GROUPS = [
    "alpha_weight",
    "num_samples", 
    "filter_size",
    "hist_percent"
]

def parse_all_scores(base_path="outputs"):
    """Parse all JSON score files into a single DataFrame."""
    
    all_data = []
    
    for scene in SCENE_NAMES:
        scene_path = Path(base_path) / scene / "scores_cgvqm"
        
        if not scene_path.exists():
            print(f"Warning: {scene_path} does not exist, skipping {scene}")
            continue
            
        print(f"Processing {scene}...")
        
        for param_group in PARAM_GROUPS:
            json_file = scene_path / f"vary_{param_group}_scores.json"
            
            if not json_file.exists():
                print(f"  Warning: {json_file} does not exist")
                continue
            
            # Load JSON
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Parse each variation in this parameter group
            for variation_key, scores in data.items():
                # Extract parameter value from key
                # Format: "vary_alpha_weight_0.01" -> param_value = "0.01"
                parts = variation_key.split('_')
                param_value = '_'.join(parts[3:])  # Everything after "vary_{param_name}_"
                
                # Get score and per-frame errors
                cgvqm_score = scores.get('score', None)
                per_frame_errors = scores.get('per_frame_errors', [])
                
                # Calculate statistics on per-frame errors
                if per_frame_errors:
                    mean_frame_error = sum(per_frame_errors) / len(per_frame_errors)
                    max_frame_error = max(per_frame_errors)
                    min_frame_error = min(per_frame_errors)
                    std_frame_error = pd.Series(per_frame_errors).std()
                else:
                    mean_frame_error = max_frame_error = min_frame_error = std_frame_error = None
                
                all_data.append({
                    'scene': scene,
                    'param_group': param_group,
                    'param_value': param_value,
                    'cgvqm_score': cgvqm_score,
                    'mean_frame_error': mean_frame_error,
                    'max_frame_error': max_frame_error,
                    'min_frame_error': min_frame_error,
                    'std_frame_error': std_frame_error,
                    'num_frames': len(per_frame_errors)
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Scenes processed: {df['scene'].nunique()}")
    print(f"Parameter groups: {df['param_group'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Parse all scores
    df = parse_all_scores()
    
    # Save to CSV
    output_file = "master_scores.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Show preview
    print("\nPreview:")
    print(df.head(10))
    
    # Summary statistics
    print("\nSummary by parameter group:")
    print(df.groupby('param_group')['cgvqm_score'].describe())