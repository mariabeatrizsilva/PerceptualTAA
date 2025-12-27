import unreal
import os

# --- Configuration Constants ---
# !!! IMPORTANT: UPDATE THESE PATHS TO MATCH YOUR PROJECT !!!
LEVEL_PATH = "/Game/AbandonedPowerPlant/Maps/PowerPlant" 
SEQUENCE_PATH = "/Game/AbandonedPowerPlant/plantwalk2" 
# !!! IMPORTANT: Set the path to your desired Post-Process Material (e.g., MaterialInstanceConstant'/Game/Materials/MyPostProcessMat.MyPostProcessMat')
POST_PROCESS_MATERIAL_PATH = "/MovieRenderPipeline/Materials/MovieRenderQueue_MotionVectors"
# Base directory for output. The full path will be appended with the job name structure.
OUTPUT_BASE_DIRECTORY = os.path.expanduser("~/Documents/PerceptualTAA/data/abandoned1")

# Define all variation groups, their console variables (CVars), and the values to test
TAA_VARIATIONS = {
    # Group 1: Alpha Weight (r.TemporalAACurrentFrameWeight)
    "alpha_weight": {
        "cvar": "r.TemporalAACurrentFrameWeight",
        "values": [0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.5, 1.0],
    },
    # Group 2: TAA Num Samples (r.TemporalAASamples)
    "num_samples": {
        "cvar": "r.TemporalAASamples",
        "values": [4, 8, 16, 32, 64],
    },
    # Group 3: Filter Size (r.TemporalAAFilterSize)
    "filter_size": {
        "cvar": "r.TemporalAAFilterSize",
        "values": [0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 1.0],
    },
    # Group 4: History Screen Percentage (r.TemporalAA.HistoryScreenPercentage)
    "hist_percent": {
        "cvar": "r.TemporalAA.HistoryScreenPercentage",
        "values": [50, 75, 100, 125, 150, 200],
    }
}


def create_movie_render_job(parameter_name, cvar_name, value):
    """
    Creates a single Movie Render Queue job with specified settings and CVar variation.
    
    Args:
        parameter_name (str): The name of the parameter group (for folder structure).
        cvar_name (str): The console variable name to set.
        value (float/int): The value to set for the console variable.
    """
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()

    # Create the job name based on the required folder structure:
    # vary_{parametername}/vary_paramname_{subvalue}
    value_str = str(value) #.replace('.', '_')
    job_sub_path = f"vary_{parameter_name}/vary_{parameter_name}_{value_str}"
    
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = job_sub_path
    
    # Set the map/sequence
    job.map = unreal.SoftObjectPath(LEVEL_PATH)
    job.sequence = unreal.SoftObjectPath(SEQUENCE_PATH)
    
    config = job.get_configuration()
    
    # --- 1. Deferred Render Pass Configuration ---
        
    
    # Get the Deferred Pass Setting to ensure deferred rendering is active.
    # We removed the PostProcessSetting code to resolve the "no attribute" error.
    deferred_pass_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    # This property ensures the deferred pass is active (often required to avoid black frames).
    deferred_pass_setting.disable_multisample_effects = False 
    deferred_pass_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

    # deferred_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    # deferred_setting.disable_multisample_effects = False
    
    # # Crucial: Ensure exactly one Post-Process Material element is in the array.
    # # First, clear the array (optional, but ensures only one is present)
    # deferred_setting.post_process_materials.clear()
    
    # # Add the single required Post-Process Material
    # post_process_material = unreal.SoftObjectPath(POST_PROCESS_MATERIAL_PATH)
    # if not post_process_material.is_valid():
    #     unreal.log_warning(f"Warning: Post Process Material path '{POST_PROCESS_MATERIAL_PATH}' is invalid!")
    
    # deferred_setting.post_process_materials.append(post_process_material)
    
    # --- 2. TAA Configuration ---
    aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa_setting.spatial_sample_count = 1  # 1 spatial sample for TAA
    aa_setting.override_anti_aliasing = True
    aa_setting.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_TEMPORAL_AA
    # Note: Temporal sample count is often overridden by the CVar r.TemporalAASamples, 
    # but we set a default here just in case, while Group 2 explicitly controls it via CVar.
    # aa_setting.temporal_sample_count = 8
    
    # --- 3. CVar Variation ---
    console_var_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    console_var_setting.add_or_update_console_variable(cvar_name, value)

    # --- 4. Output Configuration (PNG, 30 FPS, File Structure) ---
    output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output_setting.output_directory = unreal.DirectoryPath(OUTPUT_BASE_DIRECTORY)
    
    output_setting.file_name_format = "{job_name}/{frame_number}"
    
    output_setting.output_resolution = unreal.IntPoint(1920, 1080) # Default resolution
    output_setting.output_frame_rate = unreal.FrameRate(30, 1) # 30 FPS
    output_setting.zero_pad_frame_numbers = 4
    
    # --- 5. PNG Output Setting ---
    # Remove any unwanted output types (like the JPG from your original scripts)
    # config.remove_setting_by_class(unreal.MoviePipelineImageSequenceOutput_JPG)
    # Add the required PNG output setting
    png_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    png_setting.write_alpha = True # Optional, but common for PNG

    print(f"Created job: {job_sub_path} | CVar: {cvar_name}={value}")
    return job

def create_movie_render_job_TAA(parameter_name, cvar_name, value):
    """
    Creates a single Movie Render Queue job with specified settings and CVar variation.
    
    Args:
        parameter_name (str): The name of the parameter group (for folder structure).
        cvar_name (str): The console variable name to set.
        value (float/int): The value to set for the console variable.
    """
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()

    # Create the job name based on the required folder structure:
    # vary_{parametername}/vary_paramname_{subvalue}
    value_str = str(value) #.replace('.', '_')
    job_sub_path = f"vary_{parameter_name}_TAA/vary_{parameter_name}_TAA_{value_str}"
    
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = job_sub_path
    
    # Set the map/sequence
    job.map = unreal.SoftObjectPath(LEVEL_PATH)
    job.sequence = unreal.SoftObjectPath(SEQUENCE_PATH)
    
    config = job.get_configuration()
    
    # --- 1. Deferred Render Pass Configuration ---
        
    
    # Get the Deferred Pass Setting to ensure deferred rendering is active.
    # We removed the PostProcessSetting code to resolve the "no attribute" error.
    deferred_pass_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    # This property ensures the deferred pass is active (often required to avoid black frames).
    deferred_pass_setting.disable_multisample_effects = False 
    deferred_pass_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    
    # --- 2. TAA Configuration ---
    aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa_setting.spatial_sample_count = 1  # 1 spatial sample for TAA
    aa_setting.override_anti_aliasing = True
    aa_setting.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_TEMPORAL_AA
    # Note: Temporal sample count is often overridden by the CVar r.TemporalAASamples, 
    # but we set a default here just in case, while Group 2 explicitly controls it via CVar.
    # aa_setting.temporal_sample_count = 8
    
    # --- 3. Sample Count Variation ---
    aa_setting.temporal_sample_count = value

    # --- 4. Output Configuration (PNG, 30 FPS, File Structure) ---
    output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output_setting.output_directory = unreal.DirectoryPath(OUTPUT_BASE_DIRECTORY)
    
    output_setting.file_name_format = "{job_name}/{frame_number}"
    
    output_setting.output_resolution = unreal.IntPoint(1920, 1080) # Default resolution
    output_setting.output_frame_rate = unreal.FrameRate(30, 1) # 30 FPS
    output_setting.zero_pad_frame_numbers = 4
    
    # --- 5. PNG Output Setting ---
    # Add the required PNG output setting
    png_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    png_setting.write_alpha = True # Optional, but common for PNG

    print(f"Created job: {job_sub_path} | CVar: {cvar_name}={value}")
    return job

def create_supersample_render_job():
    """
    Creates a single Movie Render Queue job with specified settings and CVar variation.
    
    Args:
        parameter_name (str): The name of the parameter group (for folder structure).
        cvar_name (str): The console variable name to set.
        value (float/int): The value to set for the console variable.
    """
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    job_sub_path = f"16SSAA"
    
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = job_sub_path
    
    # Set the map/sequence
    job.map = unreal.SoftObjectPath(LEVEL_PATH)
    job.sequence = unreal.SoftObjectPath(SEQUENCE_PATH)
    
    config = job.get_configuration()
    
    # --- 1. Deferred Render Pass Configuration ---
    deferred_pass_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    # This property ensures the deferred pass is active (often required to avoid black frames).
    deferred_pass_setting.disable_multisample_effects = False 
    deferred_pass_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    
    # --- 2. TAA Configuration ---
    aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa_setting.spatial_sample_count = 16  # 1 spatial sample for TAA
    aa_setting.override_anti_aliasing = True
    aa_setting.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE
    
    # --- 4. Output Configuration (PNG, 30 FPS, File Structure) ---
    output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output_setting.output_directory = unreal.DirectoryPath(OUTPUT_BASE_DIRECTORY)
    
    output_setting.file_name_format = "{job_name}/{frame_number}"
    
    output_setting.output_resolution = unreal.IntPoint(1920, 1080) # Default resolution
    output_setting.output_frame_rate = unreal.FrameRate(30, 1) # 30 FPS
    output_setting.zero_pad_frame_numbers = 4
    
    # --- 5. PNG Output Setting ---
    # Add the required PNG output setting
    png_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    png_setting.write_alpha = True # Optional, but common for PNG

    print(f"Created job: {job_sub_path} | 16SSAA")
    return job

def main():
    """Main function to create all render jobs across all variation groups."""
    
    print("--- Starting MRQ Job Creation for TAA Variations ---")
    
    # Clear any existing jobs in the queue for a fresh start (optional)
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    subsystem.get_queue().delete_all_jobs()
    
    total_jobs_created = 0

    for param_name, config_data in TAA_VARIATIONS.items():
        cvar = config_data["cvar"]
        values = config_data["values"]
        
        print(f"\nProcessing variation group: {param_name} ({cvar})")
        
        for value in values:
            try:
                create_movie_render_job(param_name, cvar, value)
                total_jobs_created += 1
                if cvar=="r.TemporalAASamples":
                    create_movie_render_job_TAA(param_name, cvar, value)
            except Exception as e:
                unreal.log_error(f"Failed to create job for {param_name} with value {value}: {e}")

    create_supersample_render_job()
    total_jobs_created += 1
    print(f"\n--- Job Creation Complete: {total_jobs_created} Jobs Added to MRQ ---")
    print("Please check the Movie Render Queue window to confirm and start rendering.")


if __name__ == "__main__":
    main()
