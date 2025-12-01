import json
import numpy as np
import os

# --- Configuration ---
# Define the quality labels globally
QLABELS = ['very annoying', 'annoying', 'slightly annoying', 'perceptible but not annoying', 'imperceptible']
INPUT_JSON_PATH = 'outputs/scores_cgvqm_frames/vary_hist_percent_scores.json'
OUTPUT_JSON_PATH = 'outputs/scores_cgvqm_frames/labeled/vary_hist_percent_scores.json'

# --- Helper Function ---
def get_quality_label(score):
    """
    Maps the 0-100 CGVQM score to a perceptual quality label.
    """
    clamped_score = max(0, min(score, 100))
    label_index = int(np.round(clamped_score / 25))
    final_index = min(label_index, len(QLABELS) - 1)
    return QLABELS[final_index]

# --- Main Labeling Logic ---
def add_labels_to_json(input_path, output_path):
    print(f"Reading scores from: {input_path}")
    
    # 1. Load the data
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {input_path}")
        return
    
    # 2. Process and add labels
    labeled_data = {}
    print("Processing scores and generating labels...")
    
    for key, value in data.items():
        # Check if value is a number (int or float)
        if isinstance(value, (int, float)):
            score = value
            label = get_quality_label(score)
            # Create a dictionary entry with score and label
            labeled_data[key] = {
                'score': score,
                'label': label
            }
        # Check if value is already a dictionary with 'score' key
        elif isinstance(value, dict) and 'score' in value:
            score = value['score']
            label = get_quality_label(score)
            labeled_data[key] = value.copy()
            labeled_data[key]['label'] = label
        else:
            # Handle unexpected data formats
            labeled_data[key] = {
                'original_value': value,
                'label': 'N/A (Invalid Format)'
            }
    
    # 3. Save the updated data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(labeled_data, f, indent=4)
    
    print(f"✅ Labeling complete. Updated data saved to: {output_path}")

if __name__ == '__main__':
    add_labels_to_json(INPUT_JSON_PATH, OUTPUT_JSON_PATH)