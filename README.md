# PerceptualTAA

Code for investigating the perceptual foundations of Temporal Anti-Aliasing (TAA) through systematic parameter analysis and quality metric evaluation.

Scenes are rendered in Unreal Engine 5.6 at varying TAA configurations and screen resolutions, then evaluated using perceptual video quality metrics. Results are aggregated into an interactive visualization tool for cross-scene analysis.

## Repository Structure

```
PerceptualTAA/
├── data/                          # User-provided (not in repo)
│   └── {scene_name}/
│       ├── 16SSAA/                # Reference renders (16x supersampling)
│       └── {param_folder}/        # e.g. vary_alpha_weight/
│           └── {param_value}/     # e.g. vary_alpha_weight_0.04/
│               └── %04d.png
├── outputs/
│   └── {scene_name}/
│       ├── scores_cgvqm/          # CGVQM results (JSON)
│       ├── scores_cvvdp/          # CVVDP results (JSON)
│       └── error_plots/
├── src/
│   └── cgvqm/                     # CGVQM implementation (cloned)
├── generate_mrq.py                # Unreal Engine MRQ automation (populates queue with renders at parameter configs)
├── compute_metrics.py             # Metric computation (CGVQM + CVVDP)
├── build_dataset.py               # Aggregates JSON outputs into dataset.json
├── d3_viewer.html                 # Interactive D3 quality explorer
└── ffmpeg.py                      # Frame sequence → mp4 conversion
```

---

## Installation

### Prerequisites

- **Unreal Engine 5.6** with Movie Render Queue and Python scripting enabled
- **Python 3.x**
- **FFmpeg**

### Python Dependencies

```bash
pip install numpy opencv-python matplotlib torch torchvision pillow

# CVVDP
pip install pycvvdp

# CGVQM — clone into src/
cd src/ && git clone [CGVQM_REPO_URL] cgvqm && cd ..
```

---

## Data Setup

Renders for each scene live under `data/`. Screen resolution variants use the naming convention `{scene_name}-screen-per-{pct}`:

```
data/
└── oldmine/                        # full resolution
│   ├── 16SSAA/                     # reference frames
│   └── vary_alpha_weight/
│       ├── vary_alpha_weight_0.04/
│       └── vary_alpha_weight_0.08/
├── oldmine-screen-per-25/          # 25% screen resolution variant
│   ├── 16SSAA/
│   └── vary_alpha_weight/
└── oldmine-screen-per-75/
    └── ...
```

Frame files should be PNG, named sequentially (`0001.png`, `0002.png`, ...).

---

## Workflow

### 1. Render in Unreal Engine

```bash
python generate_mrq.py
```

Populates the Movie Render Queue with parameter variations. Click Render in MRQ to produce the frame sequences.

### 2. Compute Metrics

```bash
# CGVQM for one or more parameter folders, one or more scenes
python compute_metrics.py -m CGVQM -f vary_alpha_weight vary_filter_size --scenes oldmine quarry

# All parameter folders in a scene
python compute_metrics.py -m CGVQM --all --scenes oldmine

# Use a different scene's 16SSAA as reference (e.g. compare degraded renders to full-quality ref)
python compute_metrics.py -m CGVQM --all --scenes oldmine-screen-per-25 --ref-scene oldmine

# Test a single video
python compute_metrics.py -m CGVQM -f vary_alpha_weight --single vary_alpha_weight_0.04 --scenes oldmine
```

Results are saved to `outputs/{scene_name}/scores_cgvqm/{folder}_scores[_ref-{ref_scene}].json`.

Each JSON entry follows this structure:

```json
{
  "_meta": {
    "reference_scene": "oldmine",
    "metric": "CGVQM",
    "source_file": "vary_alpha_weight_scores.json"
  },
  "vary_alpha_weight_0.04": {
    "score": 96.61,
    "per_frame_errors": [4.87, 11.87, ...]
  }
}
```

### 3. Build the Dataset

After computing metrics, run `build_dataset.py` to merge all JSON outputs into a single `dataset.json` used by the plotter. This is incremental — only new entries are added on each run.

```bash
python build_dataset.py              # incremental update (safe to re-run)
python build_dataset.py --dry-run    # preview what would be added
python build_dataset.py --rebuild    # full rebuild from scratch
```

See [Visualization](#visualization) below for the full `dataset.json` schema.

### 4. Visualize

```bash
# Serve locally (required — D3 cannot load JSON from file://)
python -m http.server 8000
# Open http://localhost:8000/d3_viewer.html
```

---

## Visualization

`build_dataset.py` + `d3_viewer.html` form an interactive quality explorer.

### Dataset Schema

`dataset.json` is structured as `base_scene → screen_pct → ref_scene → param → [{value, score, per_frame_errors}]`:

```json
{
  "oldmine": {
    "100": {
      "ref-oldmine": {
        "_meta": {
          "reference_scene": "oldmine",
          "metric": "CGVQM"
        },
        "alpha_weight": [
          {"value": 0.01, "score": 95.59, "per_frame_errors": [...]},
          {"value": 0.02, "score": 95.76, "per_frame_errors": [...]}
        ],
        "filter_size": [...]
      }
    },
    "25": {
      "ref-oldmine": { ... },
      "ref-oldmine-screen-per-25": { ... }
    }
  }
}
```

The `ref_scene` level captures cases where a degraded resolution variant (e.g. `screen-per-25`) is evaluated against either its own `16SSAA` or the full-resolution scene's `16SSAA`, enabling direct comparison of both reference strategies.

`_meta` is optional for older JSON files — `build_dataset.py` reconstructs it from the filename when absent.

### plot.html (currently unavailable -- has been replaced by d3_viewer.html )

`plot.html` loads `dataset.json` and provides an interactive line plot of parameter value vs. CGVQM score:

- **Scene** and **parameter** dropdowns to navigate the data
- **One line per `(screen_pct, ref_scene)` combination** — if a screen percentage was evaluated against two different references, both appear as separate lines
- **Click legend items** to toggle individual lines on/off
- **Hover dots** for exact score, parameter value, reference scene, and frame count

---

## Acknowledgments

- Unreal Engine and Epic Games for rendering infrastructure and environment assets
- [CVVDP](https://github.com/gfxdisp/ColorVideoVDP) and [CGVQM](https://github.com/IntelLabs/cgvqm) authors for perceptual metric implementations