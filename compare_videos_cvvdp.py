import subprocess
import shlex
import os
import json
import re # Added for robust score extraction

# --- Configuration ---
# The main executable command
CVVDP_EXECUTABLE = 'cvvdp'

# The reference path, which is constant across all runs
REF_PATH = '/home/bia/PTAA/data/frames/16SSAA/%04d.png'
BASE_PATH ='/home/bia/PTAA/data/frames/'

# Other constant parameters
DISPLAY_MODE = 'standard_4k'
FPS_VALUE = 30

def extract_file_name(test_path: str) -> str:
    """
    Extracts the 'file name' (the directory name before the frame pattern)
    from the test path.
    Example: '/.../vary_alpha_weight_0.1/%04d.png' -> 'vary_alpha_weight_0.1'
    """
    # Remove the constant frame pattern '/%04d.png'
    if test_path.endswith('/%04d.png'):
        base_dir = test_path[:-len('/%04d.png')]
    else:
        # Fallback if the path format is slightly different
        base_dir = os.path.dirname(test_path)
    
    # Extract the last directory component
    return os.path.basename(base_dir)

def extract_score_from_output(stdout: str) -> float | None:
    """
    Parses the score from the cvvdp command's standard output, 
    based on the observed format 'cvvdp=X.XXXX [JOD]'.
    """
    score_pattern = re.compile(r"cvvdp=(\d+\.?\d*)")
    
    match = score_pattern.search(stdout)
    
    if match:
        try:
            # Return the captured floating-point number from group 1
            return float(match.group(1))
        except ValueError:
            print("Warning: Could not convert extracted score to a number.")
            return None
    
    print("Warning: Score not found in command output using the pattern 'cvvdp=(\d+\.?\d*)'.")
    return None

def run_cvvdp_command(test_path: str) -> dict:
    """
    Constructs and runs the cvvdp command, returning the result in the 
    requested JSON format structure.
    """
    file_name = extract_file_name(test_path)
    print(f"--- Preparing to run command for file: {file_name} ---")

    # Construct the full command as a list of arguments
    command = [
        CVVDP_EXECUTABLE,
        '--test', test_path,
        '--ref', REF_PATH,
        '--display', DISPLAY_MODE,
        '--fps', str(FPS_VALUE)
    ]

    full_command_str = shlex.join(command)
    print(f"Executing: {full_command_str}")

    # Initialize result dictionary with a default error score
    result_dict = {file_name: {"score": "ERROR_NOT_RUN"}}

    try:
        # Execute the command
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            shell=False
        )

        print("\n--- Command Execution Succeeded (Parsing Output) ---")
        
        # 1. Extract the numeric score from the standard output
        score = extract_score_from_output(result.stdout)
        
        # 2. Update the result dictionary
        result_dict[file_name]["score"] = score if score is not None else "ERROR_SCORE_NOT_PARSED"

        # Optional: Print raw output for debugging
        # print("RAW STDOUT (for reference):")
        # print(result.stdout if result.stdout else "No output.")
        
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR: Command Failed ---")
        print(f"Exit Code: {e.returncode}")
        print(f"STDERR:\n{e.stderr}")
        result_dict[file_name]["score"] = f"ERROR_CODE_{e.returncode}"
        
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"The executable '{CVVDP_EXECUTABLE}' was not found.")
        print("Ensure the cvvdp tool is correctly installed and accessible.")
        result_dict[file_name]["score"] = "ERROR_EXECUTABLE_NOT_FOUND"
        
    return result_dict

def run_multiple_directories(base_folder_name: str) -> dict:
    """
    Loops through all subdirectories within the given base_folder_name
    and runs the cvvdp command for each one.
    """
    full_base_path = os.path.join(BASE_PATH , base_folder_name)
    all_results = {}
    
    if not os.path.isdir(full_base_path):
        print(f"Error: Base path not found: {full_base_path}")
        return {}

    print(f"\n--- Starting batch run in: {full_base_path} ---")
    
    # Iterate through all entries in the base directory
    for item_name in os.listdir(full_base_path):
        sub_folder_path = os.path.join(full_base_path, item_name)
        
        # We only want to process actual directories
        if os.path.isdir(sub_folder_path):
            # Construct the full test path for the command
            # Example: /.../vary_alpha_weight_0.1/%04d.png
            test_path = os.path.join(sub_folder_path, '%04d.png')
            
            # Run the command for this subfolder and merge the result
            try:
                single_result = run_cvvdp_command(test_path)
                all_results.update(single_result)
            except Exception as e:
                # Catch any unexpected errors during the process
                print(f"An unexpected error occurred processing {item_name}: {e}")
                # Log the error in the results
                all_results[item_name] = {"score": f"UNEXPECTED_ERROR_{type(e).__name__}"}
                
    return all_results

def batch_process_cvvdp(target_folder):
    output_dir = "outputs/scores_cvdp"
    err_scores_path = os.path.join(output_dir, f"{target_folder}_scores.json")
    final_output = run_multiple_directories(target_folder)
    os.makedirs(output_dir, exist_ok=True)
    with open(err_scores_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print("\n--- Final JSON Output for Batch Run ---")
    print(json.dumps(final_output, indent=4))
    print(f"Results successfully saved to: {err_scores_path}")

# --- Example Usage ---
if __name__ == "__main__":
    folders_to_process = [
        'vary_filter_size',
        'vary_num_samples',
        'vary_hist_percent'
    ]

    for folder_name in folders_to_process:
        batch_process_cvvdp(folder_name)