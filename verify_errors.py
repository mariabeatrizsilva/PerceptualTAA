import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def pad_arrays(arrays):
    """Pads smaller arrays with NaN values to match the length of the longest array."""
    max_len = max(len(arr) for arr in arrays)
    padded_arrays = []
    for arr in arrays:
        # Pad with NaN values (for correct average calculation)
        padding = np.full(max_len - len(arr), np.nan)
        padded_arr = np.concatenate([arr, padding])
        padded_arrays.append(padded_arr)
    return padded_arrays, max_len

def analyze_errors(json_file_path):
    """Analyze per-frame errors from CGVQM results JSON file and save plots 
    to a structured directory.
    """
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return
    
    # Load the JSON file
    print(f"Loading: {json_filepath}\n")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all scores and per-frame errors
    video_names = []
    overall_scores = []
    all_per_frame_errors_np = []
    
    for video_name, video_data in data.items():
        video_names.append(video_name)
        overall_scores.append(video_data['score'])
        all_per_frame_errors_np.append(np.array(video_data['per_frame_errors']))
    
    overall_scores = np.array(overall_scores)

    # ============================================================================
    # (A) Data Transformation and Padding
    # ============================================================================
    
    padded_errors, max_frames = pad_arrays(all_per_frame_errors_np)
    padded_errors_matrix = np.stack(padded_errors)
    
    # Calculate per-frame quality (100 - error)
    all_per_frame_quality = [100 - errors for errors in all_per_frame_errors_np]
    
    padded_quality, _ = pad_arrays(all_per_frame_quality)
    padded_quality_matrix = np.stack(padded_quality)
    
    avg_per_frame_error = np.nanmean(padded_errors_matrix, axis=0)
    avg_per_frame_quality = np.nanmean(padded_quality_matrix, axis=0)
    
    all_errors_flat = np.concatenate(all_per_frame_errors_np)
    all_quality_flat = np.concatenate(all_per_frame_quality)

    # ============================================================================
    # (B) Extreme Frames Data Collection (For Plot 8 and Structured Output)
    # ============================================================================
    
    extreme_frames_data = [] # Stores data for all 5 top and 5 bottom frames
    structured_frame_list = {} # For printing detailed list

    print("\n" + "="*60)
    print("EXTREME FRAME ANALYSIS: Top 5 / Bottom 5 Frame Quality Per Video")
    print("="*60)
    
    for i, video_name in enumerate(video_names):
        quality = all_per_frame_quality[i]
        structured_frame_list[video_name] = {'Top 5 Frames': [], 'Bottom 5 Frames': []}
        
        # --- Best Frames (Top 5) ---
        top_indices = np.argsort(quality)[::-1][:5]
        top_scores = quality[top_indices]
        
        top_frame_indices_str = " | ".join([f'F{index}' for index in top_indices])

        # Data for Plot 8 scatter & annotation
        extreme_frames_data.append({
            'video': video_name,
            'quality': top_scores.tolist(), # Store scores list for context
            'mean_quality': np.mean(top_scores), # Calculate mean for horizontal centering
            'frame_string': top_frame_indices_str,
            'type': 'Top 5',
        })
        
        # Data for structured list
        for rank, (score, index) in enumerate(zip(top_scores, top_indices)):
             structured_frame_list[video_name]['Top 5 Frames'].append(f'F{index} ({score:.2f})')
            
        # --- Worst Frames (Bottom 5) ---
        bottom_indices = np.argsort(quality)[:5]
        bottom_scores = quality[bottom_indices]
        
        bottom_frame_indices_str = " | ".join([f'F{index}' for index in bottom_indices])

        # Data for Plot 8 scatter & annotation
        extreme_frames_data.append({
            'video': video_name,
            'quality': bottom_scores.tolist(), # Store scores list for context
            'mean_quality': np.mean(bottom_scores), # Calculate mean for horizontal centering
            'frame_string': bottom_frame_indices_str,
            'type': 'Bottom 5',
        })
        
        # Data for structured list
        for rank, (score, index) in enumerate(zip(bottom_scores, bottom_indices)):
             structured_frame_list[video_name]['Bottom 5 Frames'].append(f'F{index} ({score:.2f})')
        
        print(f"\n{video_name} Frames:\n  Top 5: {top_frame_indices_str}\n  Bottom 5: {bottom_frame_indices_str}")

    # ============================================================================
    # (C) Verification and Statistics (Omitted for brevity)
    # ============================================================================
    # ... (Code for verification and statistics) ...

    # ============================================================================
    # (D) Setup Output Directory
    # ============================================================================
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    json_dir = os.path.dirname(os.path.abspath(json_file_path))
    parent_dir = os.path.dirname(json_dir)
    output_dir = os.path.join(parent_dir, 'error_plots', base_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {output_dir}")

    # ============================================================================
    # (E) Plotting (All 8 Plots)
    # ============================================================================
    
    # --- Plots 1 through 7 (Saving and closing figures) ---
    
    # Plot 1
    plt.figure(figsize=(14, 6))
    for i, video_name in enumerate(video_names):
        plt.plot(all_per_frame_errors_np[i], label=video_name, alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('Per-Frame Error')
    plt.title('Plot 1: Per-Frame Errors Across All Videos (Lower is Better)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'1_per_frame_errors.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') 

    # Plot 2
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors_flat, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Per-Frame Error')
    plt.ylabel('Frequency (Frame Count)')
    plt.title('Plot 2: Distribution of Per-Frame Errors (All Videos)')
    plt.axvline(all_errors_flat.mean(), color='red', linestyle='--', label=f'Mean Error: {all_errors_flat.mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'2_distribution_error_histogram.png')
    plt.savefig(plot_path, dpi=150)
    plt.close('all') 
    
    # Plot 3
    plt.figure(figsize=(12, 6))
    plt.boxplot(all_per_frame_errors_np, tick_labels=video_names)
    plt.xlabel('Video Name')
    plt.ylabel('Per-Frame Error')
    plt.title('Plot 3: Per-Frame Error Box Plot by Video')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'3_boxplot_error.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') 
    
    # Plot 4
    plt.figure(figsize=(14, 6))
    for i, video_name in enumerate(video_names):
        plt.plot(all_per_frame_quality[i], label=video_name, alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('Per-Frame Quality')
    plt.title('Plot 4: Per-Frame Quality Across All Videos (Higher is Better)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'4_per_frame_quality.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') 
    
    # Plot 5
    plt.figure(figsize=(12, 6))
    plt.violinplot(all_per_frame_quality, showmeans=True)
    plt.xticks(ticks=np.arange(1, len(video_names) + 1), labels=video_names, rotation=45, ha='right')
    plt.xlabel('Video Name')
    plt.ylabel('Per-Frame Quality')
    plt.title('Plot 5: Per-Frame Quality Violin Plot by Video')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'5_violinplot_quality.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') 
    
    # Plot 6
    plt.figure(figsize=(10, 6))
    mean_errors = [arr.mean() for arr in all_per_frame_errors_np]
    plt.scatter(mean_errors, overall_scores, s=100, alpha=0.6)
    for i, video_name in enumerate(video_names):
        plt.annotate(video_name, (mean_errors[i], overall_scores[i]), fontsize=8, alpha=0.7)
    x_range = np.array([min(mean_errors), max(mean_errors)])
    plt.plot(x_range, 100 - x_range, 'r--', label='100 - error', alpha=0.5)
    plt.xlabel('Mean Per-Frame Error')
    plt.ylabel('Overall Score')
    plt.title('Plot 6: Overall Score vs Mean Per-Frame Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'6_score_vs_error.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') 
    
    # Plot 7
    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(max_frames), avg_per_frame_quality, label='Average Quality Across All Videos', color='purple')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Per-Frame Quality')
    plt.title('Plot 7: Average Per-Frame Quality Across All Videos (Identifying Weak Moments)')
    plt.axhline(avg_per_frame_quality.mean(), color='red', linestyle='--', label=f'Overall Mean Quality: {avg_per_frame_quality.mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'7_average_per_frame_quality.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') 


    # Plot 8: Extreme Frame Quality Comparison (Using single strings with fixed vertical offset)
    plt.figure(figsize=(14, 8)) 
    
    # Map frame types to colors
    colors = {'Top 5': 'green', 'Bottom 5': 'red'}
    
    # Define vertical offset magnitudes
    VERT_OFFSET_TOP = 0.10 # Closer to the video line
    VERT_OFFSET_BOTTOM = 0.25 # Further from the video line
    
    # Map categorical video names to numerical y-values for plotting
    y_tick_values = np.arange(len(video_names))
    video_name_to_y = {name: y_tick_values[i] for i, name in enumerate(video_names)}
    
    # Find the global min and max quality for X-axis limits
    all_scores_flat = [score for d in extreme_frames_data for score in d['quality']]
    x_min = min(all_scores_flat) if all_scores_flat else 0
    x_max = max(all_scores_flat) if all_scores_flat else 100

    # Scatter points and create annotation strings
    for video in video_names:
        y_center = video_name_to_y[video]
        video_data = [d for d in extreme_frames_data if d['video'] == video]
        
        # Plot Scatter Points
        for d in video_data:
            # Plot each score individually
            for score in d['quality']:
                plt.scatter(score, y_center, 
                            color=colors[d['type']], 
                            marker='o', s=100, alpha=0.7, 
                            label=d['type'] if video == video_names[0] else None) 

        # Annotate with the single string:
        
        # 1. Top 5 Frames (Centered on mean, offset BELOW the line)
        top_data = [d for d in video_data if d['type'] == 'Top 5'][0]
        
        plt.annotate(f'TOP: {top_data["frame_string"]}', 
                     (top_data['mean_quality'], y_center - VERT_OFFSET_TOP), 
                     fontsize=7, va='center', ha='center', 
                     color='darkgreen')

        # 2. Bottom 5 Frames (Centered on mean, offset BELOW the line, with greater offset)
        bottom_data = [d for d in video_data if d['type'] == 'Bottom 5'][0]
        
        plt.annotate(f'BOTTOM: {bottom_data["frame_string"]}', 
                     (bottom_data['mean_quality'], y_center - VERT_OFFSET_BOTTOM), 
                     fontsize=7, va='center', ha='center', 
                     color='darkred')

    # FIX: Explicitly set Y-axis limits to ensure bottom labels are not clipped
    # Y-axis spans from 0 (lowest video) to N-1 (highest video)
    # We need padding below 0 and above N-1
    y_min = -VERT_OFFSET_BOTTOM * 1.5 # Ensure enough space for the lowest bottom label
    y_max = len(video_names) - 1 + 0.5 # Add padding above the highest video
    
    plt.ylim(y_min, y_max)
    
    # Set X-axis limits
    x_range_padding = (x_max - x_min) * 0.01 
    plt.xlim(x_min - x_range_padding, x_max + x_range_padding)

    # Set Y-axis ticks back to categorical video names
    plt.yticks(y_tick_values, video_names)

    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Frame Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Per-Frame Quality Score')
    plt.ylabel('Video Name')
    plt.title('Plot 8: Extreme Frames (Dots) with Grouped Frame Index Labels (Below)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    
    plot_path = os.path.join(output_dir, f'8_extreme_frames_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all') # Close plot
    print(f"Saved plot: {plot_path}")
    
    # ============================================================================
    # (F) Structured Frame List Output (Displayed Below Plot)
    # ============================================================================
    print("\n" + "="*60)
    print("FRAME INDEX REFERENCE (Full Detail)")
    print("Frames are listed by rank (1-5), with Quality Score in parentheses.")
    print("="*60)
    
    for video in video_names:
        print(f"\n### Video: {video} ###")
        
        # Top 5 Frames
        top_frames_str = [f'Rank {i+1}: {frame}' for i, frame in enumerate(structured_frame_list[video]['Top 5 Frames'])]
        print("* **Top 5 Quality Frames (Green Dots):**")
        print(f"  > {', '.join(top_frames_str)}")
        
        # Bottom 5 Frames
        bottom_frames_str = [f'Rank {i+1}: {frame}' for i, frame in enumerate(structured_frame_list[video]['Bottom 5 Frames'])]
        print("* **Bottom 5 Quality Frames (Red Dots):**")
        print(f"  > {', '.join(bottom_frames_str)}")


    print("\nAll plots have been generated and closed.")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze per-frame errors from CGVQM results JSON file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_errors.py vary_num_samples.json
  python analyze_errors.py /path/to/scores.json
  
  # On Mac/Linux, you can drag and drop:
  python analyze_errors.py [drag file here]
        """
    )
    
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to the JSON file containing CGVQM scores and per-frame errors'
    )
    
    args = parser.parse_args()
    
    analyze_errors(args.json_file)


if __name__ == '__main__':
    main()