"""
generate_factorial_jobs.py
--------------------------
Unreal Engine Movie Render Queue script for a full 3^4 factorial TAA sweep.
Outputs frames to: tests/{scene_name}/full_factorial/{combo_name}/0001.png ...
Also renders the 16SSAA reference to: tests/{scene_name}/16SSAA/

HOW TO USE:
  1. Set SCENE_CONFIG to your scene (level path, sequence path, output name).
  2. Paste/run this script in the Unreal Engine Python console or editor utility.
  3. Open Movie Render Queue and render all jobs.

OUTPUT STRUCTURE:
  tests/
    {scene_name}/
      16SSAA/
        0001.png ...
      full_factorial/
        aw0.04_ns4_fs0.1_hp100/
          0001.png ...
        aw0.04_ns4_fs0.1_hp150/
          ...
"""

import unreal
import os
import itertools

# ============================================================================
# SCENE CONFIGURATION — fill in your scene details here
# ============================================================================
SCENE_CONFIG = {
    "scene_name": "quarry-rocksonly",          # used for folder naming
    "level_path": "/Game/Scene_QuarrySlate/Maps/Quarry_Slate",
    "sequence_path": "/Game/Scene_QuarrySlate/Sequences/flythrough",
    "screen_percentage": 87,
}

# Base output root — two levels up from tests/{scene_name}/
# Assumes this project lives at e.g. ~/Documents/PerceptualTAA/
OUTPUT_ROOT = os.path.expanduser("~/Documents/PerceptualTAA/tests")

# ============================================================================
# FACTORIAL PARAMETER GRID — 3 levels each => 3^4 = 81 combinations
# ============================================================================
FACTORIAL_PARAMS = {
    "alpha_weight": {
        "cvar": "r.TemporalAACurrentFrameWeight",
        "values": [0.04, 0.5, 1],
        "short": "aw",
    },
    "num_samples": {
        "cvar": "r.TemporalAASamples",
        "values": [4, 16, 64],
        "short": "ns",
    },
    "filter_size": {
        "cvar": "r.TemporalAAFilterSize",
        "values": [0.1, 0.5, 1.0],
        "short": "fs",
    },
    "hist_percent": {
        "cvar": "r.TemporalAA.HistoryScreenPercentage",
        "values": [100, 150, 200],
        "short": "hp",
    },
}

# ============================================================================
# HELPERS
# ============================================================================

def get_output_dir(scene_name):
    return os.path.join(OUTPUT_ROOT, scene_name)


def base_job_config(queue, level_path, sequence_path):
    """Allocate a new job and return (job, config) with shared render settings."""
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.map = unreal.SoftObjectPath(level_path)
    job.sequence = unreal.SoftObjectPath(sequence_path)

    config = job.get_configuration()

    # Deferred render pass
    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    deferred.disable_multisample_effects = False

    # Output: PNG, 1080p, 30fps
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_resolution = unreal.IntPoint(1920, 1080)
    output.output_frame_rate = unreal.FrameRate(30, 1)
    output.zero_pad_frame_numbers = 4
    output.file_name_format = "{frame_number}"

    png = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    png.write_alpha = True

    return job, config


def create_reference_job(scene_name, level_path, sequence_path):
    """16SSAA reference render — no TAA, 16 spatial samples."""
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()

    job, config = base_job_config(queue, level_path, sequence_path)
    job.job_name = "16SSAA reference"

    # Anti-aliasing: no TAA, 16 spatial samples
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count = 16
    aa.override_anti_aliasing = True
    aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE

    # Screen percentage
    cvars = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    cvars.add_or_update_console_variable("r.ScreenPercentage", SCENE_CONFIG["screen_percentage"])

    # Output path
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(
        os.path.join(get_output_dir(scene_name), "16SSAA")
    )

    print(f"Created reference job: 16SSAA -> {output.output_directory.path}")
    return job


def combo_name(param_names, values):
    """Build a short readable name like aw0.04_ns4_fs0.1_hp100."""
    parts = []
    for pname, val in zip(param_names, values):
        short = FACTORIAL_PARAMS[pname]["short"]
        parts.append(f"{short}{val}")
    return "_".join(parts)


def create_factorial_job(scene_name, level_path, sequence_path, param_names, values):
    """One job for a single parameter combination."""
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()

    job, config = base_job_config(queue, level_path, sequence_path)

    name = combo_name(param_names, values)
    job.job_name = f"factorial_{name}"

    # TAA anti-aliasing
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count = 1
    aa.override_anti_aliasing = True
    aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_TEMPORAL_AA

    # CVars: all 4 parameters + screen percentage
    cvars = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    cvars.add_or_update_console_variable("r.ScreenPercentage", SCENE_CONFIG["screen_percentage"])
    for pname, val in zip(param_names, values):
        cvar = FACTORIAL_PARAMS[pname]["cvar"]
        cvars.add_or_update_console_variable(cvar, val)

    # Output path
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(
        os.path.join(get_output_dir(scene_name), "full_factorial", name)
    )

    return job, name


# ============================================================================
# MAIN
# ============================================================================

def main():
    scene_name    = SCENE_CONFIG["scene_name"]
    level_path    = SCENE_CONFIG["level_path"]
    sequence_path = SCENE_CONFIG["sequence_path"]

    print(f"\n=== Generating full factorial TAA jobs for scene: {scene_name} ===")
    print(f"Output root: {get_output_dir(scene_name)}\n")

    # 1. Reference job
    create_reference_job(scene_name, level_path, sequence_path)

    # 2. All 3^4 = 81 factorial combinations
    param_names = list(FACTORIAL_PARAMS.keys())
    value_lists = [FACTORIAL_PARAMS[p]["values"] for p in param_names]

    total = 0
    for combo in itertools.product(*value_lists):
        try:
            job, name = create_factorial_job(scene_name, level_path, sequence_path, param_names, combo)
            print(f"  Created: {name}")
            total += 1
        except Exception as e:
            unreal.log_error(f"Failed to create job for {combo}: {e}")

    print(f"\n=== Done: {total} factorial jobs + 1 reference job = {total + 1} total ===")
    print("Open Movie Render Queue and press Render to start.")


if __name__ == "__main__":
    main()
