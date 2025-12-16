# PerceptualTAA

Code used for project investigating the perceptual foundations of Temporal Anti-Aliasing (TAA) through systematic parameter analysis and quality metric evaluation.

## Overview

This project examines how varying TAA parameters affects perceptually-based quality metrics. Using Unreal Engine 5.6, scenes with diverse visual characteristics are rendered with different TAA configurations and evaluated using state-of-the-art perceptual metrics. The insights gained provide a foundation for optimizing not only TAA but also video generation methods where comparable temporal artifacts occur.

**Key contributions:**
1. Automated scene rendering with varying TAA parameters in Unreal Engine (`generate_mrq.py`)
2. Perceptual metric computation tools for CVVDP and CGVQM (`compute_metrics.py`)
3. Analysis and visualization of metric results across multiple scenes and parameters

## Features

- **Automated Rendering Pipeline**: Generate video sequences with systematic TAA parameter variations using Unreal Engine's Movie Render Queue (MRQ)
- **Video Evaluation with Two Reference-Based Perceptual Quality Metrics**: [CGVQM: Computer Graphics Video Quality Metric](https://github.com/IntelLabs/cgvqm) and [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP/tree/main)
- **Batch Processing**: Compute metrics across multiple parameter folders efficiently
- **Visualization Tools**: Generate plots analyzing parameter impact on perceptual quality

## Tested Scenes

This toolkit has been validated with:
- **City Park Environment** - Organic, nature-heavy scene ([City Park Collection](https://www.unrealengine.com/en-US/blog/free-city-park-environment-collection-now-available))
- **Factory Environment** - Rigid, industrial setting ([Free Factory Collection](https://www.unrealengine.com/en-US/blog/free-factory-environment-collection-now-available))

The framework can be extended to other Unreal Scenes.

## Repository Structure

```
PerceptualTAA/
├── data/                          # User-provided data directory (not in repo)
│   └── {scene_name}/
│       ├── 16SSAA/                # Reference renders (16x supersampling)
│       │   └── %04d.png
│       └── {parameter_folder}/    # Renders for specific parameter variations
│           └── {param_value}/
│               └── %04d.png
├── src/
│   └── cgvqm/                     # CGVQM metric implementation (cloned)
├── outputs/
│   └── {scene_name}/
│       ├── scores_cgvqm/          # CGVQM metric results (JSON)
│       ├── scores_cvvdp/          # CVVDP metric results (JSON)
│       └── error_plots/           # Visualization plots
│           └── {parameter_name}/
├── generate_mrq.py                # Unreal Engine MRQ automation
├── compute_metrics.py             # Main metric computation script
├── verify_errors.py               # Generates per-frame error plots
├── ffmpeg.py                      # Converts generated frames to mp4 in batches using FFMPEG
```

## Installation

### Prerequisites

- **Unreal Engine 5.6** with:
  - Movie Render Queue (MRQ) plugin enabled
  - Python scripting enabled
- **Python 3.x**
- **FFmpeg**

### Python Dependencies

```bash
# Core dependencies
pip install numpy opencv-python matplotlib

# CVVDP metric
pip install pycvvdp

# CGVQM metric - clone into src/
cd src/
git clone [CGVQM_REPO_URL] cgvqm
cd ..
```

### FFmpeg Installation

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Data Setup

### Directory Structure

Create a `data/` directory in the project root with the following structure:

```
data/
└── {scene_name}/              # e.g., 'parkenv', 'factory'
    ├── 16SSAA/                # Reference render (required)
    │   └── %04d.png           # Frame sequence: 0001.png, 0002.png, ...
    └── {param_folder}/        # e.g., 'vary_alpha_weight'
        └── {param_value}/     # e.g., 'vary_alpha_weight_0.04'
            └── %04d.png       # Frame sequence
```

### Example Structure

```
data/
└── parkenv/
    ├── 16SSAA/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── vary_alpha_weight/
    │   ├── vary_alpha_weight_0.04/
    │   │   ├── 0001.png
    │   │   └── ...
    │   └── vary_alpha_weight_0.08/
    │       ├── 0001.png
    │       └── ...
    └── vary_filter_size/
        ├── vary_filter_size_1.0/
        │   └── ...
        └── vary_filter_size_2.0/
            └── ...
```

### Frame Format Requirements

- **Format**: PNG (recommended for lossless quality) or JPG
- **Naming**: Sequential frames with 4-digit padding (`%04d.png`)
- **Reference**: Must have a `16SSAA` folder with reference renders
  - 16xSSAA (16x supersampling anti-aliasing) provides ground truth quality to compare parameter variations using the metrics.

## Usage

### 1. Generate Renders in Unreal Engine

Use `generate_mrq.py` to automate rendering with varying TAA parameters:

```bash
python generate_mrq.py
```

This script interfaces with Unreal Engine's Movie Render Queue to systematically render scenes with different TAA configurations.

### 2. Configure Scene Settings

Edit the configuration at the top of `compute_metrics.py`:

```python
SCENE_NAME = 'parkenv'      # Change for different scenes
REF_NAME = '16SSAA'         # Reference folder name (keep as 16SSAA)
BASE_MP4 = 'data/'
BASE_FRAMES = f'data/{SCENE_NAME}/'
FRAMES_SUFFIX = '%04d.png'  # Frame naming pattern
```

### 3. Compute Perceptual Metrics

#### Compute CGVQM for single parameter folder:
```bash
python compute_metrics.py --metric CGVQM --folders vary_alpha_weight
```

#### Compute CVVDP for multiple parameter folders:
```bash
python compute_metrics.py --metric CVVDP --folders vary_filter_size vary_num_samples
```

#### Test single video (debugging):
```bash
python compute_metrics.py --metric CGVQM --folders vary_alpha_weight --single video_name
```

#### Using short flags:
```bash
python compute_metrics.py -m CGVQM -f vary_alpha_weight
```

**Command-line Arguments:**
- `--metric` / `-m`: Metric type (`CGVQM` or `CVVDP`)
- `--folders` / `-f`: Space-separated list of parameter folders to process
- `--single` / `-s`: (Optional) Process only a specific video for testing

### 4. Generate Visualization Plots

After computing metrics, generate analysis plots:

```bash
python verify_errors.py
```

This reads the JSON files from `outputs/{scene_name}/scores_{metric}/` and generates plots in `outputs/{scene_name}/error_plots/{parameter_name}/`.

## Output Structure

### Metric Scores

Computed metrics are saved as JSON files:

```
outputs/
└── {scene_name}/           # e.g., 'parkenv'
    ├── scores_cgvqm/
    │   ├── vary_alpha_weight_scores.json
    │   ├── vary_filter_size_scores.json
    │   ├── vary_hist_percent_scores.json
    │   └── vary_num_samples_scores.json
    └── scores_cvvdp/
        ├── vary_alpha_weight_scores.json
        └── ...
```

Each JSON contains metric values for all videos in that parameter folder.

### Visualization Plots

Plots are organized by parameter:

```
outputs/
└── {scene_name}/
    └── error_plots/
        ├── vary_alpha_weight/
        │   └── [plots analyzing alpha weight impact]
        ├── vary_filter_size/
        │   └── [plots analyzing filter size impact]
        └── vary_num_samples/
            └── [plots analyzing sample count impact]
```

## Workflow Summary

1. **Scene Preparation**: Set up Unreal Engine scene with MRQ
2. **Automated Rendering**: Run `generate_mrq.py` to populate queue with parameter variations, and click render in MRQ to obtain the render grames.
3. **Data Organization**: Ensure renders are in correct `data/{scene_name}/` structure
4. **Configuration**: Update `SCENE_NAME` in `compute_metrics.py`
5. **Metric Computation**: Run `compute_metrics.py` for desired metrics and folders
6. **Visualization**: Run `verify_errors.py` to generate analysis plots
7. **Analysis**: Review JSON scores and plots to understand parameter impacts

## Research Context

Temporal Anti-Aliasing (TAA) is critical for reducing flickering and edge crawling in real-time rendering. However, TAA introduces tradeoffs between sharpness, ghosting, and temporal stability. This project provides:

- **Quantitative Analysis**: Perceptual metrics quantify how parameter changes affect visual quality
- **Parameter Optimization**: Data-driven insights for TAA tuning
- **Broader Applications**: Techniques applicable to video generation and temporal artifact reduction

By analyzing diverse scenes (organic vs. rigid, motion-heavy vs. static), this work reveals how TAA performance varies across visual contexts and provides guidelines for adaptive anti-aliasing strategies.

## Acknowledgments

- Unreal Engine for providing high-quality rendering capabilities
- CVVDP and CGVQM authors for perceptual metric implementations
- Epic Games for the City Park and Factory environment collections

