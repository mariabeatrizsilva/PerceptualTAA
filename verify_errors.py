import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def analyze_errors(json_file_path):
    """Analyze per-frame errors from CGVQM results JSON file."""
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return
    
    # Load the JSON file
    print(f"Loading: {json_file_path}\n")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all scores and per-frame errors
    video_names = []
    overall_scores = []
    all_per_frame_errors = []
    
    for video_name, video_data in data.items():
        video_names.append(video_name)
        overall_scores.append(video_data['score'])
        all_per_frame_errors.append(video_data['per_frame_errors'])
    
    # Convert to numpy arrays for easier manipulation
    overall_scores = np.array(overall_scores)
    
    # ============================================================================
    # (2) Verify that 100 - mean(errors) ≈ overall score
    # ============================================================================
    print("="*60)
    print("VERIFICATION: Does 100 - mean(per_frame_errors) = overall_score?")
    print("="*60)
    
    for i, video_name in enumerate(video_names):
        per_frame_errors = np.array(all_per_frame_errors[i])
        mean_error = per_frame_errors.mean()
        computed_score = 100 - mean_error
        actual_score = overall_scores[i]
        
        print(f"\n{video_name}:")
        print(f"  Overall score (from JSON):     {actual_score:.4f}")
        print(f"  Mean per-frame error:          {mean_error:.4f}")
        print(f"  Computed score (100 - error):  {computed_score:.4f}")
        print(f"  Difference:                    {abs(actual_score - computed_score):.6f}")
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    all_errors_flat = np.concatenate(all_per_frame_errors)
    print(f"Total frames across all videos: {len(all_errors_flat)}")
    print(f"Mean error across ALL frames:   {all_errors_flat.mean():.4f}")
    print(f"Std error across ALL frames:    {all_errors_flat.std():.4f}")
    print(f"Min error:                      {all_errors_flat.min():.4f}")
    print(f"Max error:                      {all_errors_flat.max():.4f}")
    
    mean_of_overall_scores = overall_scores.mean()
    print(f"\nMean of overall scores:         {mean_of_overall_scores:.4f}")
    print(f"100 - mean(all errors):         {100 - all_errors_flat.mean():.4f}")
    
    # ============================================================================
    # (1) Plot all the errors
    # ============================================================================
    
    # Get output directory (same as input file)
    output_dir = os.path.dirname(json_file_path)
    if not output_dir:
        output_dir = '.'
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Plot 1: Per-frame errors for each video (separate lines)
    plt.figure(figsize=(14, 6))
    for i, video_name in enumerate(video_names):
        per_frame_errors = np.array(all_per_frame_errors[i])
        plt.plot(per_frame_errors, label=video_name, alpha=0.7)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Per-Frame Error')
    plt.title('Per-Frame Errors Across All Videos')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{base_name}_per_frame_errors.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")
    
    # Plot 2: Distribution of errors (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors_flat, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Per-Frame Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Per-Frame Errors (All Videos)')
    plt.axvline(all_errors_flat.mean(), color='red', linestyle='--', 
                label=f'Mean: {all_errors_flat.mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{base_name}_error_distribution.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    
    # Plot 3: Box plot comparing videos
    plt.figure(figsize=(12, 6))
    plt.boxplot(all_per_frame_errors, labels=video_names)
    plt.xlabel('Video Name')
    plt.ylabel('Per-Frame Error')
    plt.title('Per-Frame Error Distribution by Video')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{base_name}_boxplot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    # Plot 4: Overall scores vs mean per-frame error
    plt.figure(figsize=(10, 6))
    mean_errors = [np.array(errors).mean() for errors in all_per_frame_errors]
    plt.scatter(mean_errors, overall_scores, s=100, alpha=0.6)
    
    # Add video names as labels
    for i, video_name in enumerate(video_names):
        plt.annotate(video_name, (mean_errors[i], overall_scores[i]), 
                    fontsize=8, alpha=0.7)
    
    # Add diagonal line showing 100 - error relationship
    x_range = np.array([min(mean_errors), max(mean_errors)])
    plt.plot(x_range, 100 - x_range, 'r--', label='100 - error', alpha=0.5)
    
    plt.xlabel('Mean Per-Frame Error')
    plt.ylabel('Overall Score')
    plt.title('Overall Score vs Mean Per-Frame Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{base_name}_score_vs_error.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.show()
    print("\nAll plots displayed!")


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