import json
import numpy as np
import os

# --- Configuration ---
# Define the quality labels globally
QLABELS = ['very annoying', 'annoying', 'slightly annoying', 'perceptible but not annoying', 'imperceptible']
INPUT_JSON_PATH = 'outputs/scores/vary_num_samples_scores.json'
OUTPUT_JSON_PATH = 'outputs/scores/vary_num_samples_scores_labeled.json'

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
        if 'score' in value:
            score = value['score']
            label = get_quality_label(score)
            
            # Create a new entry, adding the 'label' key
            labeled_data[key] = value.copy()
            labeled_data[key]['label'] = label
        else:
            # Handle cases where the score wasn't calculated (e.g., error entry)
            labeled_data[key] = value.copy()
            labeled_data[key]['label'] = 'N/A (Error)'

    # 3. Save the updated data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(labeled_data, f, indent=4)
        
    print(f"✅ Labeling complete. Updated data saved to: {output_path}")

if __name__ == '__main__':
    # Adjust paths if you are running this script from a different directory
    # If running from the project root, the paths above should work.
    add_labels_to_json(INPUT_JSON_PATH, OUTPUT_JSON_PATH)