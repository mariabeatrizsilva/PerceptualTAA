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
  "oldmine": {
    "metric": "CGVQM",
    "100": {
      "ref-oldmine": {
        "alpha_weight": [...]
      }
    },
    "25": {
      "ref-oldmine-screen-per-25": { ... },
      "ref-oldmine": { ... }
    }
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

### d3_viewer.html

- **Scene** and **parameter** dropdowns
- **Series legend** — one entry per `(screen_pct, ref_scene)` combination, grouped by resolution with dividers; click to toggle lines on/off
- **Chart tab** — line plot of parameter value vs. CGVQM score; hover dots for exact score, value, reference scene, and frame count
- **Summary tab** — statistics table for all visible series: min, max, range, mean, std dev, best parameter value, and Δ from first to best

---

## Extending the Visualization

`plot.html` is a self-contained file (~400 lines). The key concepts for extending it are documented inline; this section gives a higher-level map.

### State model

Three globals drive the entire view:

| Variable | Type | Description |
|---|---|---|
| `dataset` | object | The full parsed `dataset.json` |
| `activeScene` | string | Currently selected base scene |
| `activeParam` | string | Currently selected parameter name |
| `seriesVisibility` | `{[id]: bool}` | Toggle state per series, persisted across redraws |

After changing any of these, call `buildLegend()` then `draw()` to update the view.

### getSeries() — the core accessor

`getSeries()` is the single function that translates the nested `dataset` structure into a flat array of series objects for the current selection. Everything else (legend, chart, summary table) reads from its output. Its shape:

```js
{
  id:       "25|ref-oldmine",   // seriesVisibility key
  pct:      "25",               // screen percentage
  ref:      "ref-oldmine",      // ref_key from dataset
  refScene: "oldmine",          // human-readable from _meta
  color:    "var(--c3)",        // CSS variable, resolved before SVG use
  entries:  [{value, score, per_frame_errors}, ...],
  label:    "25%",
}
```

To filter series by an additional dimension (e.g. only show a specific ref scene), add a condition inside `getSeries()`'s inner loop.

### Adding a new control

1. Add a `.control-group` + `<select>` (or checkboxes) in the sidebar HTML
2. Add a state variable (e.g. `let activeRef = null`)
3. Wire a change event that updates the variable, then calls `buildLegend()` + `draw()`
4. Filter inside `getSeries()` using the new variable

### Adding a new visualisation tab

1. Add a `<button class="tab-btn" data-tab="mytab">` to `.tabs` in the HTML
2. Add `<div class="tab-panel" id="tab-mytab">` to `<main>`
3. Write a `drawMyThing()` function that reads `getSeries()` and renders into `#tab-mytab`
4. Add `if (btn.dataset.tab === 'mytab') drawMyThing()` to the tab click handler
5. Call `drawMyThing()` at the end of `draw()` and `buildLegend()` so it stays in sync

### Adding a new summary statistic

In `drawSummary()`, add your computation to the `rows.map(...)` stats block, add a `<th>` to the header string, and a `<td>` to the row template string. Follow the existing `isBestX` pattern if you want the best value highlighted.

### Colour assignment

Colours cycle through `COLOR_VARS` (`--c0`…`--c5`) in the order `getSeries()` returns series (pct descending, then ref alphabetically). To add more colours, extend the CSS variables in `:root` and add them to `COLOR_VARS`.

---

## Acknowledgments

- Unreal Engine and Epic Games for rendering infrastructure and environment assets
- [CVVDP](https://github.com/gfxdisp/ColorVideoVDP) and [CGVQM](https://github.com/IntelLabs/cgvqm) authors for perceptual metric implementations