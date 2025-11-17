import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple, List

# --- Helper Function for Data Processing ---

def _process_render_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads JSON render data, extracts the numerical parameter value from exp_name,
    and returns the processed DataFrame.
    """
    try:
        # Load JSON data, treating top-level keys as the index (maintains order)
        df = pd.read_json(file_path, orient='index')
        df = df.reset_index(names='exp_name')
        
        # Add a column for the clean numerical value (assuming it's the last part)
        df['num_val'] = df['exp_name'].str.split('_').str[-1].astype(float)
        
        # Sort the DataFrame by the numerical value for smooth plotting
        df = df.sort_values(by='num_val').reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        print(f"'{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error occurred during processing {file_path}: {e}")
        return None


def _plot_single_experiment(
    df: pd.DataFrame, 
    ax: plt.Axes, 
    parameter_name: str, 
    score_metric: str, 
    x_evenly_spaced: bool = False
):
    """
    Plots the score vs. parameter for a single experiment    
    Args:
        df: The processed pandas DataFrame containing 'num_val' and 'score'.
        ax: The matplotlib Axes object to plot on.
        parameter_name: The label for the x-axis (e.g., 'Alpha Weight').
        score_metric: The label for the y-axis (e.g., 'CGVQM Score').
        x_evenly_spaced: If True, uses categorical plotting; False for numerical.
    """
    
    # 1. Determine X-axis data and labels
    tick_labels = df['num_val'].tolist()

    if x_evenly_spaced:
        x_data = df['exp_name']
        x_label = f'{parameter_name} (Evenly Spaced)'
    else:
        x_data = df['num_val']
        x_label = f'{parameter_name}'

    # 2. Plotting
    ax.plot(x_data, df['score'], marker='o', linestyle='-', color='b')

    if x_evenly_spaced:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    else:
        ax.set_xticks(df['num_val'])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
    ax.set_title(f'{parameter_name} vs. {score_metric}', fontsize=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(score_metric, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_render_scores_from_json(
    file_path: str,
    parameter_name: str,
    score_metric: str = "CGVQM Score",
    x_evenly_spaced: bool = True
) -> Optional[pd.DataFrame]:
    """
    Loads JSON data, processes it, and generates a score vs. parameter plot.
    """
    df = _process_render_data(file_path)
    
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


def compare_render_experiments(file_paths: dict[str, str], score_metric: str = "CGVQM Score"):
    """
    Loads multiple experiment files and plots data
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    all_data = []

    for label, file_path in file_paths.items():
        df = _process_render_data(file_path)
        
        if df is None:
            continue

        df['experiment_type'] = label
        
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
    ax.set_title(f'Comparison of {score_metric} vs. Varied Parameters', fontsize=16)
    ax.set_xlabel('Parameter Value', fontsize=14)
    ax.set_ylabel(score_metric, fontsize=14)
    
    # Optional: Use log scale if values span multiple orders of magnitude
    ax.set_xscale('log') 
    
    ax.legend(title='Experiment Type')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def plot_experiment_grid(file_paths: dict[str, str], score_metric: str = "CGVQM Score", space_even: bool = False):
    """
    Loads data from multiple files and plots each experiment's score vs. its 
    unique parameter value in a separate subplot within a grid.
    
    The plot type is set to x_evenly_spaced=False (numerical scale) by default.
    """
    
    num_plots = len(file_paths)
    if num_plots == 0:
        print("No files provided to plot.")
        return

    # Determine the grid layout
    cols = min(2, num_plots)
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    # Flatten the axes array for easy iteration, handling the case of a single plot
    axes = [axes] if num_plots == 1 else axes.flatten()

    ax_index = 0
    
    # --- Loop through each file and generate a subplot ---
    for parameter_name, file_path in file_paths.items():
        ax = axes[ax_index]
        
        df = _process_render_data(file_path)
        
        if df is None:
            ax.set_title(f"{parameter_name} (File Not Found)", color='red')
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

    Args:
        all_data: A pandas DataFrame containing all experiment results, 
                  expected to have columns: 'experiment_type', 'score', and 'num_val'.

    Returns:
        A pandas DataFrame summarizing the performance, or None if the input is empty.
    """
    
    if all_data.empty:
        print("Input DataFrame is empty. Cannot generate summary.")
        return None

    # Find the row index for the max and min scores within each experiment_type group
    idx_max = all_data.groupby('experiment_type')['score'].idxmax()
    idx_min = all_data.groupby('experiment_type')['score'].idxmin()

    # Get the actual score corresponding to the best and worst index.
    # .set_index('experiment_type') makes sure pandas aligns the values (not make new row)
    best_performance = all_data.loc[idx_max].set_index('experiment_type')
    worst_performance = all_data.loc[idx_min].set_index('experiment_type')
    
    summary_df = pd.DataFrame({
        'best_score': best_performance['score'],
        'best_score_value': best_performance['num_val'],
        'worst_score': worst_performance['score'],
        'worst_score_value': worst_performance['num_val'],
    })
    
    # --- 2. Calculate Pearson Correlation ---
    # Calculate the Pearson correlation (R-value) between 'num_val' and 'score'
    correlation_series = all_data.groupby('experiment_type')[['num_val', 'score']].corr().unstack().iloc[:, 1]
    correlation_series.name = 'pearson_correlation'
    
    # Merge the correlation data onto the summary DataFrame using the index ('experiment_type')
    summary_df = summary_df.merge(
        correlation_series,
        left_index=True,
        right_index=True
    )

    # --- 3. Final Formatting ---
    # Move the 'experiment_type' index back to a regular column
    summary_df = summary_df.reset_index()

    # Reorder columns to include the new correlation column
    summary_df = summary_df[[
        'experiment_type', 
        'best_score', 
        'best_score_value', 
        'worst_score', 
        'worst_score_value',
        'pearson_correlation'
    ]]
    
    return summary_df