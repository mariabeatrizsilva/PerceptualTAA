import pandas as pd
import matplotlib.pyplot as plt
import math
import json 
import os
from typing import Optional, Dict, Any, List, Tuple

def get_dict(file_path: str) -> Optional[Dict[str, float]]:
    """Loads a JSON file whose top level is a dictionary."""
    try:
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        
        # Ensure the loaded data is a dictionary as expected
        if isinstance(data_dict, dict):
            return data_dict
        else:
            print(f"Error: JSON file content is not a dictionary: {file_path}")
            return None
            
    except FileNotFoundError:
        print(f"'{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def _process_render_data_dict(data_dict: Dict[str, float]) -> Optional[pd.DataFrame]:
    """
    Loads JSON-style dictionary data, extracts the numerical parameter value 
    from the key (assuming key format is 'prefix_value'), and returns the processed DataFrame.
    """
    if not data_dict:
        return None
        
    try:
        # Create DataFrame from the dictionary, keys become 'exp_name', values become 'score'
        df = pd.DataFrame(list(data_dict.items()), columns=['exp_name', 'score'])
        
        # Extract the numerical value from the experiment name (the part after the last underscore)
        # Note: This assumes all experiment keys follow the 'prefix_value' pattern.
        df['num_val'] = df['exp_name'].str.split('_').str[-1].astype(float)
        
        # Sort the DataFrame by the numerical value for smooth plotting
        df = df.sort_values(by='num_val').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error occurred during processing dictionary: {e}")
        return None

def plot_dual_experiment_grid(
    frames_data: Dict[str, Dict[str, float]], 
    regular_data: Dict[str, Dict[str, float]], 
    score_metric: str = "CGVQM Score", 
    space_even: bool = False
):
    """
    Plots a grid where each subplot compares the 'frames' and 'regular' data 
    for a single common experiment/parameter.
    
    Args:
        frames_data: Dictionary of loaded data for the 'frames' runs.
        regular_data: Dictionary of loaded data for the 'regular' runs.
        score_metric: The name of the metric being plotted (for labels).
        space_even: Whether to use evenly spaced x-ticks instead of numerical values.
    """
    # Find common keys (experiments) present in both datasets
    common_params = sorted(list(frames_data.keys() & regular_data.keys()))
    num_plots = len(common_params)
    
    if num_plots == 0:
        print("No common experiments found between 'frames' and 'regular' datasets.")
        return

    # Determine the grid layout (4x4 or smaller if less than 16 plots)
    cols = min(4, num_plots)
    rows = math.ceil(num_plots / cols)
    
    # Adjust for very wide plots if rows=1
    if rows == 1 and cols > 2:
        figsize = (5 * cols, 6)
    else:
        figsize = (5 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten the axes array for easy iteration
    axes = [axes] if num_plots == 1 else axes.flatten()

    ax_index = 0
    
    # --- Loop through common parameters and plot both lines ---
    for parameter_name in common_params:
        ax = axes[ax_index]
        
        # 1. Process Data
        df_frames = _process_render_data_dict(frames_data[parameter_name])
        df_regular = _process_render_data_dict(regular_data[parameter_name])
        
        # 2. Plot Frames Data (Blue)
        if df_frames is not None:
            _plot_single_experiment(
                df=df_frames, 
                ax=ax, 
                parameter_name=parameter_name, 
                score_metric=score_metric, 
                x_evenly_spaced=space_even,
                label='Frames Run',
                color='#1f77b4' # Muted Blue
            )

        # 3. Plot Regular Data (Orange)
        if df_regular is not None:
            _plot_single_experiment(
                df=df_regular, 
                ax=ax, 
                parameter_name=parameter_name, 
                score_metric=score_metric, 
                x_evenly_spaced=space_even,
                label='Regular Run',
                color='#ff7f0e' # Muted Orange
            )

        # 4. Final Subplot Customization
        ax.set_title(f'{parameter_name}', fontsize=12) # Simplified title for grid
        ax.legend(loc='lower right', fontsize=8)
        
        # Combine all unique numerical values for consistent ticks
        if not space_even and df_frames is not None and df_regular is not None:
            all_num_vals = pd.concat([df_frames['num_val'], df_regular['num_val']]).unique()
            all_num_vals.sort()
            
            tick_labels = [f'{val:g}' for val in all_num_vals.tolist()]
            ax.set_xticks(all_num_vals)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        
        ax_index += 1

    # Hide any unused subplots
    for i in range(ax_index, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(f'Comparative Grid Analysis: Frames vs. Regular ({score_metric})', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to make space for suptitle
    plt.show()


def load_multiple_experiments(file_paths_dict):
    """
    Loads data dictionaries from a dictionary of file paths.
    
    Args:
        file_paths_dict: Dictionary where keys are labels (e.g., 'Alpha Weight') 
                         and values are the file paths (str).
                         
    Returns:
        A dictionary where keys are labels and values are the loaded data dictionaries.
    """
    loaded_data = {}
    for label, file_path in file_paths_dict.items():
        data_dict = get_dict(file_path)
        if data_dict is not None:
            loaded_data[label] = data_dict
        else:
            print(f"Skipping '{label}' due to file loading error.")
    return loaded_data

def _plot_single_experiment(
    df: pd.DataFrame, 
    ax: plt.Axes, 
    parameter_name: str, 
    score_metric: str, 
    x_evenly_spaced: bool = False,
    label: str = 'Score', # Added label parameter for dual plotting
    color: str = 'b' # Added color parameter for dual plotting
):
    """
    Plots the score vs. parameter for a single experiment    
    """
    
    # 1. Determine X-axis data and labels
    # Use 'g' format specifier to suppress trailing zeros when unnecessary (e.g., 1.0 -> 1)
    tick_labels = [f'{val:g}' for val in df['num_val'].tolist()] 

    if x_evenly_spaced:
        # Use a range for even spacing but label with the actual values
        x_data = range(len(df))
        x_label = f'{parameter_name} (Evenly Spaced)'
    else:
        # Use the actual numerical values for correct scaling
        x_data = df['num_val']
        x_label = f'{parameter_name}'

    # 2. Plotting
    ax.plot(x_data, df['score'], marker='o', linestyle='-', color=color, label=label)

    # 3. Set Ticks and Labels (only set once per subplot for the primary data)
    # To prevent tick overlap on dual plots, we check if this is the first plot (label is default)
    if label == 'Score':
        if x_evenly_spaced:
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        else:
            # Only set major ticks, will re-set in the dual function if needed
            ax.set_xticks(df['num_val'])
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
        ax.set_title(f'{parameter_name} vs. {score_metric}', fontsize=12)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(score_metric, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

# --- 4. PRIMARY PLOTTING FUNCTIONS ---

def plot_render_scores_from_dict(
    data_dict: Dict[str, float], 
    parameter_name: str,
    score_metric: str = "CGVQM Score",
    x_evenly_spaced: bool = False
) -> Optional[pd.DataFrame]:
    """
    Processes dictionary data and generates a single score vs. parameter plot.
    """
    df = _process_render_data_dict(data_dict)
    
    if df is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    
    _plot_single_experiment(
        df=df, 
        ax=ax, 
        parameter_name=parameter_name, 
        score_metric=score_metric, 
        x_evenly_spaced=x_evenly_spaced
    )
    
    plt.tight_layout()
    plt.show()
    
    return df

def compare_render_experiments(data_sets: Dict[str, Dict[str, float]], score_metric: str = "CGVQM Score", plot = False):
    """
    Loads multiple experiment data sets (dictionaries) and plots data on a single graph.
    
    Args:
        data_sets: Dictionary where keys are labels (e.g., 'Alpha Test') 
                   and values are the data dictionaries (e.g., DATASET).
    """
    if plot:
        fig, ax = plt.subplots(figsize=(12, 7))
    all_data = []

    for label, data_dict in data_sets.items():
        # Use the dictionary processing helper
        df = _process_render_data_dict(data_dict)
        
        if df is None:
            continue

        df['experiment_type'] = label
        
        # Plotting against the numerical value
        if plot:
            ax.plot(
                df['num_val'], 
                df['score'], 
                marker='o', 
                linestyle='-', 
                label=f'{label}'
            )
        
        all_data.append(df)
        print(f"✅ Loaded and sorted data for: {label}")

    # --- Final Plot Customization ---
    if plot:
        ax.set_title(f'Comparison of {score_metric} vs. Varied Parameters', fontsize=16)
        ax.set_xlabel('Parameter Value', fontsize=14)
        ax.set_ylabel(score_metric, fontsize=14)
        ax.set_xscale('log') 
        ax.legend(title='Experiment Type')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def plot_experiment_grid(data_sets: Dict[str, Dict[str, float]], score_metric: str = "CGVQM Score", space_even: bool = False):
    """
    Loads data from multiple dictionaries and plots each experiment's score vs. its 
    unique parameter value in a separate subplot within a grid.
    
    Args:
        data_sets: Dictionary where keys are the parameter names (e.g., 'Alpha Weight')
                   and values are the data dictionaries (e.g., DATASET).
    """
    
    num_plots = len(data_sets)
    if num_plots == 0:
        print("No data sets provided to plot.")
        return

    # Determine the grid layout
    cols = min(2, num_plots)
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    # Flatten the axes array for easy iteration, handling the case of a single plot
    axes = [axes] if num_plots == 1 else axes.flatten()

    ax_index = 0
    
    # --- Loop through each file and generate a subplot ---
    for parameter_name, data_dict in data_sets.items():
        ax = axes[ax_index]
        
        # Use the dictionary processing helper
        df = _process_render_data_dict(data_dict)
        
        if df is None:
            ax.set_title(f"{parameter_name} (No Data)", color='red')
        else:
            _plot_single_experiment(
                df=df, 
                ax=ax, 
                parameter_name=parameter_name, 
                score_metric=score_metric, 
                x_evenly_spaced=space_even 
            )

        ax_index += 1

    # Hide any unused subplots
    for i in range(ax_index, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(f'Comparative Analysis of Render Parameters', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def summarize_performance_table(all_data: pd.DataFrame):
    """
    Analyzes the combined experiment data to generate a summary table showing 
    the best and worst scores achieved by each experiment type and the 
    corresponding numerical parameter values (num_val).
    """
    
    if all_data.empty:
        print("Input DataFrame is empty. Cannot generate summary.")
        return None

    # Find the row index for the max and min scores within each experiment_type group
    idx_max = all_data.groupby('experiment_type')['score'].idxmax()
    idx_min = all_data.groupby('experiment_type')['score'].idxmin()

    # Get the actual score corresponding to the best and worst index.
    best_performance = all_data.loc[idx_max].set_index('experiment_type')
    worst_performance = all_data.loc[idx_min].set_index('experiment_type')
    
    summary_df = pd.DataFrame({
        'best_score': best_performance['score'],
        'best_score_value': best_performance['num_val'],
        'worst_score': worst_performance['score'],
        'worst_score_value': worst_performance['num_val'],
    })
    
    # --- 2. Calculate Pearson Correlation ---
    # Correlation between numerical parameter value and score
    correlation_series = all_data.groupby('experiment_type')[['num_val', 'score']].corr().unstack().iloc[:, 1]
    correlation_series.name = 'pearson_correlation'
    
    summary_df = summary_df.merge(
        correlation_series,
        left_index=True,
        right_index=True
    )

    # --- 3. Final Formatting ---
    summary_df = summary_df.reset_index()

    summary_df = summary_df[[
        'experiment_type', 
        'best_score', 
        'best_score_value', 
        'worst_score', 
        'worst_score_value',
        'pearson_correlation'
    ]]
    
    return summary_df

# --- 5. EXECUTION BLOCK EXAMPLE ---

if __name__ == '__main__':
    # --- Example Mock Data (To demonstrate functionality since real files are absent) ---
    # IMPORTANT: You will replace the following mock data with actual file loading calls.
    
    # Mock data for 'Alpha Weight'
    alpha_weight_data = {
      "vary_alpha_weight_0.01": 53.71,
      "vary_alpha_weight_0.02": 54.29,
      "vary_alpha_weight_0.04": 55.30,
      "vary_alpha_weight_0.06": 55.99,
      "vary_alpha_weight_0.1": 56.90,
      "vary_alpha_weight_0.2": 57.94,
      "vary_alpha_weight_0.5": 58.82,
      "vary_alpha_weight_1.0": 58.95
    }
    
    # Mock data for a second experiment (e.g., 'Filter Size')
    filter_size_data = {
      "vary_filter_size_1": 54.0,
      "vary_filter_size_3": 55.5,
      "vary_filter_size_5": 57.0,
      "vary_filter_size_7": 56.5
    }
    
    # --- 1. Example Usage for a Single Plot (using the new function structure) ---
    print("\n--- Example 1: Single Plot from Dictionary ---")
    
    # How you would typically use this in a notebook:
    # 1. Load the data dictionary
    # alpha_dict = get_dict(os.path.join(score_directory_frames, file_path_alpha))
    # 2. Plot
    # plot_render_scores_from_dict(alpha_dict, parameter_name="Alpha Weight", score_metric="PSNR")
    
    # Using mock data for demonstration:
    df_alpha = plot_render_scores_from_dict(
        data_dict=alpha_weight_data, 
        parameter_name="Alpha Weight", 
        score_metric="CGVQM Score"
    )

    # --- 2. Example Usage for a Grid Plot (multiple parameters) ---
    print("\n--- Example 2: Grid Plot (Multiple Parameters) ---")
    
    comparison_data_sets = {
        "Alpha Weight": alpha_weight_data,
        "Filter Size": filter_size_data,
        # You would load more files here:
        # "Hist Percent": get_dict(os.path.join(score_directory_frames, 'vary_hist_percent_scores.json')),
    }
    
    plot_experiment_grid(
        data_sets=comparison_data_sets,
        score_metric="CGVQM Score",
        space_even=False # Use actual numerical X-axis scaling
    )
    
    # --- 3. Example Usage for a Comparison Plot (multiple experiments on one graph) ---
    print("\n--- Example 3: Comparison Plot (Multiple Experiments) ---")
    
    # This assumes the two experiments vary the SAME parameter (e.g., Alpha Weight),
    # but the scores come from different runs (e.g., 'frames' vs 'regular').
    
    # Mock data for 'Regular' directory comparison (same parameter values, different scores)
    alpha_regular_data = {
      "vary_alpha_weight_0.01": 51.71,
      "vary_alpha_weight_0.02": 52.29,
      "vary_alpha_weight_0.1": 54.90,
      "vary_alpha_weight_1.0": 56.95
    }

    comparison_plot_data = {
        "Frames Run": alpha_weight_data,
        "Regular Run": alpha_regular_data,
    }
    
    df_combined = compare_render_experiments(
        data_sets=comparison_plot_data,
        score_metric="CGVQM Score"
    )

    # --- 4. Example Usage for Summary Table ---
    if df_combined is not None and not df_combined.empty:
        print("\n--- Example 4: Performance Summary Table ---")
        summary = summarize_performance_table(df_combined)
        print(summary)