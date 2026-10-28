"""
generate_taa_jobs.py  (dataset v2)
----------------------------------
Populates Unreal's Movie Render Queue with a full-grid TAA sweep for one or
more scenes. Run inside the UE Python console / editor utility of the project
that contains the scenes (scenes from other projects are skipped automatically).

DESIGN (see notes at bottom of file for rationale)
  Varied (full grid):   screen percentage (sp) x alpha weight (aw) x history % (hp)
  Fixed at UE default:  num samples (ns) = 8, filter size (fs) = 1.0
  Per scene also:       1 x 16-SSAA reference at 100% screen percentage
                        1 x aux job writing motion vectors + depth (EXR)
                        (optional) legacy-check jobs that reproduce the old
                        factorial setup, to verify the pipeline

OUTPUT LAYOUT  (everything for a scene lives under one folder)
  {OUTPUT_ROOT}/{scene}/
      manifest.json                         <- every job + its exact params
      reference/ssaa16_sp100/0001.png ...
      taa/sp050_aw0.040_hp100_ns08_fs1.00/0001.png ...
      aux/motion_depth/{render_pass}/0001.exr ...
      legacy_check/...                      (only if enabled)
  Later, encode_and_verify.py turns every PNG folder into a lossless video:
      videos/reference/ssaa16_sp100.mkv
      videos/taa/sp050_aw0.040_hp100_ns08_fs1.00.mkv

NAMING
  Config names contain EVERY registered TAA parameter (varied or fixed), in a
  fixed order, zero-padded to fixed width. So:
    - names sort correctly in file browsers
    - adding a new varied parameter later never makes old names ambiguous
      (just register it in PARAMS; old names stay valid because the fixed
      value was always written)
  Don't parse names in analysis code -- read manifest.json instead.

USAGE
  1. Edit SCENES / SWEEP below if needed.
  2. Open the UE project, run this script in the Python console.
  3. Movie Render Queue -> Render (Local).
  4. Repeat in each project that holds pilot scenes.
"""

import os
import json
import itertools
import datetime

import unreal

# ============================================================================
# 1. GLOBAL SETTINGS
# ============================================================================

DATASET_VERSION = "v2"
OUTPUT_ROOT = os.path.expanduser(f"~/Documents/PerceptualTAA/data_{DATASET_VERSION}")

OUTPUT_RESOLUTION = (1920, 1080)
FPS = 30

# Warm-up: TAA needs a filled history buffer. Without warm-up, the first
# ~1/alpha frames are unconverged (at alpha=0.02 that's ~50 frames = 1.7 s of a
# 5 s clip), which would unfairly penalise low alpha values.
WARMUP_FRAMES = 60

CLEAR_QUEUE_FIRST = False   # True = wipe whatever is already in MRQ
SKIP_EXISTING = True        # skip jobs whose PNGs or encoded video already exist

# ============================================================================
# 2. SCENES  (pilot = 5; add the rest from scenes.yaml to scale up)
# ============================================================================
# "project" must match the .uproject name; only scenes belonging to the
# currently open project are queued.

SCENES = [
    # Newproj -- known high / medium sensitivity from the factorial study
    {"name": "fantasticvillage-open", "project": "Newproj",
     "level": "/Game/Fantastic_Village_Pack/maps/map_village_day",
     "sequence": "/Game/Fantastic_Village_Pack/animations/Cam_LevelSequence"},
    {"name": "oldmine", "project": "Newproj",
     "level": "/Game/Scene_OldMine/Maps/Industrial_OldMine",
     "sequence": "/Game/Scene_OldMine/Sequences/seq1-reg"},
    # Wildwest -- known flat / insensitive
    {"name": "resto-fwd", "project": "Wildwest",
     "level": "/Game/RestaurantScene/Maps/Showcase",
     "sequence": "/Game/RestaurantScene/Cinematics/Flythrough2"},
    # New content types: dense foliage, and rotation-heavy camera motion
    {"name": "lightfoliage", "project": "NatureEnvs",
     "level": "/Game/Light_Foliage/Maps/Showcase",
     "sequence": "/Game/Light_Foliage/Flythrough"},
    {"name": "subway-turn", "project": "SubwayTrain",
     "level": "/Game/SubwayTrain/Maps/Demonstration",
     "sequence": "/Game/SubwayTrain/Flythrough"},
]

# ============================================================================
# 3. PARAMETER REGISTRY  (every parameter that can appear in a config name)
# ============================================================================
# Order here = order in the config name. To vary a new parameter later, add it
# here (with its default in FIXED) and then move it into SWEEP.

PARAMS = {
    "sp": {"cvar": "r.ScreenPercentage",                    "fmt": "{:03d}",   "cast": int},
    "aw": {"cvar": "r.TemporalAACurrentFrameWeight",        "fmt": "{:.3f}",   "cast": float},
    "hp": {"cvar": "r.TemporalAA.HistoryScreenPercentage",  "fmt": "{:03d}",   "cast": int},
    "ns": {"cvar": "r.TemporalAASamples",                   "fmt": "{:02d}",   "cast": int},
    "fs": {"cvar": "r.TemporalAAFilterSize",                "fmt": "{:.2f}",   "cast": float},
}

SWEEP = {
    # 100/87/50 % per axis = 100/75/25 % of pixels. Add 71 (=50 % pixels) for the full run.
    "sp": [100, 87, 50],
    # Dense near the UE default (0.04), fills the old 0.2-0.5 gap, keeps the flicker end.
    "aw": [0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    # Same 4 levels as the original paper sweep.
    "hp": [100, 125, 150, 200],
}

FIXED = {"ns": 8, "fs": 1.0}   # UE defaults; negligible effect in the factorial study

# ============================================================================
# 4. OPTIONAL LEGACY CHECK
# ============================================================================
# Re-renders a few combos exactly as the old factorial test did (87 % screen
# percentage, ns=16, fs=1.0, no warm-up, reference ALSO at 87 %). Scoring these
# against the legacy reference should reproduce the old scores.json values;
# scoring them against the new 100 % reference shows how much the old
# reference choice mattered. Only useful for scenes in the old factorial study.
LEGACY_CHECK_SCENES = ["fantasticvillage-open", "oldmine", "resto-fwd"]
LEGACY_COMBOS = [  # (aw, hp) at sp=87, ns=16, fs=1.0
    (0.04, 100), (0.04, 200), (1.0, 100), (1.0, 200),
]

# Aux passes (built into MRQ's plugin content)
AUX_MATERIALS = [
    "/MovieRenderPipeline/Materials/MovieRenderQueue_MotionVectors",
    "/MovieRenderPipeline/Materials/MovieRenderQueue_WorldDepth",
]


# ============================================================================
# HELPERS
# ============================================================================

def config_name(values):
    """{'sp':50,'aw':0.04,...} -> 'sp050_aw0.040_hp100_ns08_fs1.00' (all params, fixed order)."""
    parts = []
    for key, spec in PARAMS.items():
        parts.append(key + spec["fmt"].format(spec["cast"](values[key])))
    return "_".join(parts)


def _try_set(obj, prop, value):
    """Set an editor property, warning (not crashing) if this UE version lacks it."""
    try:
        obj.set_editor_property(prop, value)
        return True
    except Exception as e:
        unreal.log_warning(f"[generate_taa_jobs] could not set {type(obj).__name__}.{prop}: {e}")
        return False


def already_rendered(png_dir, video_path):
    if os.path.exists(video_path):
        return True
    return os.path.isdir(png_dir) and any(f.endswith(".png") for f in os.listdir(png_dir))


def new_job(queue, scene, job_name, out_dir, file_fmt="{frame_number}"):
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = job_name
    job.map = unreal.SoftObjectPath(scene["level"])
    job.sequence = unreal.SoftObjectPath(scene["sequence"])
    config = job.get_configuration()

    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    deferred.disable_multisample_effects = False

    out = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.output_resolution = unreal.IntPoint(*OUTPUT_RESOLUTION)
    out.output_frame_rate = unreal.FrameRate(FPS, 1)
    out.zero_pad_frame_numbers = 4
    out.file_name_format = file_fmt
    out.output_directory = unreal.DirectoryPath(out_dir)
    return job, config


def set_aa(config, method, spatial_samples, warmup):
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.override_anti_aliasing = True
    aa.anti_aliasing_method = method
    aa.spatial_sample_count = spatial_samples
    if warmup > 0:
        _try_set(aa, "engine_warm_up_count", warmup)
        _try_set(aa, "render_warm_up_frames", True)
        _try_set(aa, "render_warm_up_count", warmup)
    return aa


def set_cvars(config, values):
    cv = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for key, val in values.items():
        cv.add_or_update_console_variable(PARAMS[key]["cvar"], PARAMS[key]["cast"](val))


def add_png(config):
    png = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    png.write_alpha = False   # RGB only: smaller files, and metrics ignore alpha


def enable_aux_passes(config):
    """Turn on MRQ's built-in motion-vector and depth post-process passes."""
    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    try:
        passes = list(deferred.get_editor_property("additional_post_process_materials"))
    except Exception as e:
        unreal.log_warning(f"[generate_taa_jobs] no additional_post_process_materials on this UE version: {e}")
        return False
    wanted = set(AUX_MATERIALS)
    found = set()
    for p in passes:
        mat = p.get_editor_property("material")
        path = mat.get_path_name().split(".")[0] if mat else ""
        if path in wanted:
            _try_set(p, "enabled", True)
            found.add(path)
    for path in wanted - found:
        p = unreal.MoviePipelinePostProcessPass()
        _try_set(p, "enabled", True)
        _try_set(p, "material", unreal.load_asset(path))
        passes.append(p)
    deferred.set_editor_property("additional_post_process_materials", passes)
    _try_set(deferred, "render_main_pass", False)   # only the aux passes are needed here
    return True


# ============================================================================
# JOB BUILDERS
# ============================================================================

def build_scene(queue, scene):
    name = scene["name"]
    root = os.path.join(OUTPUT_ROOT, name)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "scene": scene,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "render_settings": {"output_resolution": OUTPUT_RESOLUTION, "fps": FPS,
                            "warmup_frames": WARMUP_FRAMES,
                            "engine_version": unreal.SystemLibrary.get_engine_version()},
        "sweep": SWEEP, "fixed": FIXED,
        "jobs": [],
    }
    n_new = n_skip = 0

    def add(kind, cfg, params, build_fn):
        nonlocal n_new, n_skip
        png_dir = os.path.join(root, kind, cfg)
        video = os.path.join(root, "videos", kind, cfg + ".mkv")
        manifest["jobs"].append({"kind": kind, "config": cfg, "params": params,
                                 "png_dir": os.path.relpath(png_dir, root),
                                 "video": os.path.relpath(video, root)})
        if SKIP_EXISTING and already_rendered(png_dir, video):
            n_skip += 1
            return
        build_fn(png_dir)
        n_new += 1

    # --- reference: 16x SSAA, no TAA, ALWAYS 100 % screen percentage
    def ref_fn(out_dir):
        _, c = new_job(queue, scene, f"{name}/reference/ssaa16_sp100", out_dir)
        set_aa(c, unreal.AntiAliasingMethod.AAM_NONE, 16, WARMUP_FRAMES)
        set_cvars(c, {"sp": 100})
        add_png(c)
    add("reference", "ssaa16_sp100", {"sp": 100, "spatial_samples": 16, "aa": "none"}, ref_fn)

    # --- aux: motion vectors + depth, 100 %, 1 sample, EXR
    def aux_fn(out_dir):
        _, c = new_job(queue, scene, f"{name}/aux/motion_depth", out_dir,
                       file_fmt="{render_pass}/{frame_number}")
        set_aa(c, unreal.AntiAliasingMethod.AAM_NONE, 1, WARMUP_FRAMES)
        set_cvars(c, {"sp": 100})
        enable_aux_passes(c)
        exr = c.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
        _try_set(exr, "compression", unreal.EXRCompressionFormat.ZIP)
    aux_dir = os.path.join(root, "aux", "motion_depth")
    manifest["jobs"].append({"kind": "aux", "config": "motion_depth",
                             "params": {"passes": AUX_MATERIALS},
                             "png_dir": os.path.relpath(aux_dir, root), "video": None})
    if not (SKIP_EXISTING and os.path.isdir(aux_dir) and os.listdir(aux_dir)):
        aux_fn(aux_dir); n_new += 1
    else:
        n_skip += 1

    # --- main TAA grid
    keys = list(SWEEP.keys())
    for combo in itertools.product(*[SWEEP[k] for k in keys]):
        values = dict(FIXED)
        values.update(dict(zip(keys, combo)))
        cfg = config_name(values)

        def taa_fn(out_dir, values=values, cfg=cfg):
            _, c = new_job(queue, scene, f"{name}/taa/{cfg}", out_dir)
            set_aa(c, unreal.AntiAliasingMethod.AAM_TEMPORAL_AA, 1, WARMUP_FRAMES)
            set_cvars(c, values)
            add_png(c)
        add("taa", cfg, {k: PARAMS[k]["cast"](v) for k, v in values.items()}, taa_fn)

    # --- optional legacy check (old factorial settings: sp87 everywhere, no warm-up)
    if name in LEGACY_CHECK_SCENES:
        def legacy_ref_fn(out_dir):
            _, c = new_job(queue, scene, f"{name}/legacy_check/ssaa16_sp087", out_dir)
            set_aa(c, unreal.AntiAliasingMethod.AAM_NONE, 16, 0)
            set_cvars(c, {"sp": 87})
            add_png(c)
        add("legacy_check", "ssaa16_sp087", {"sp": 87, "spatial_samples": 16, "aa": "none",
                                             "warmup_frames": 0}, legacy_ref_fn)
        for aw, hp in LEGACY_COMBOS:
            values = {"sp": 87, "aw": aw, "hp": hp, "ns": 16, "fs": 1.0}
            cfg = config_name(values)

            def legacy_fn(out_dir, values=values, cfg=cfg):
                _, c = new_job(queue, scene, f"{name}/legacy_check/{cfg}", out_dir)
                set_aa(c, unreal.AntiAliasingMethod.AAM_TEMPORAL_AA, 1, 0)
                set_cvars(c, values)
                add_png(c)
            p = {k: PARAMS[k]["cast"](v) for k, v in values.items()}
            p["warmup_frames"] = 0
            p["old_factorial_name"] = f"aw{aw:g}_ns16_fs1.0_hp{hp}"
            add("legacy_check", cfg, p, legacy_fn)

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return n_new, n_skip


# ============================================================================
# MAIN
# ============================================================================

def main():
    project = unreal.SystemLibrary.get_game_name()
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    if CLEAR_QUEUE_FIRST:
        queue.delete_all_jobs()

    mine = [s for s in SCENES if s["project"] == project]
    others = [s["name"] for s in SCENES if s["project"] != project]
    print(f"\n=== TAA dataset {DATASET_VERSION} | project '{project}' ===")
    print(f"Output root: {OUTPUT_ROOT}")
    if not mine:
        print(f"No scenes configured for project '{project}'. Check the 'project' fields in SCENES.")
        return
    if others:
        print(f"Skipping scenes from other projects (run there): {others}")

    grid = 1
    for v in SWEEP.values():
        grid *= len(v)
    print(f"Grid: {' x '.join(f'{k}({len(v)})' for k, v in SWEEP.items())} = {grid} TAA jobs/scene "
          f"+ reference + aux{' + legacy check' if LEGACY_CHECK_SCENES else ''}\n")

    total_new = total_skip = 0
    for scene in mine:
        n_new, n_skip = build_scene(queue, scene)
        total_new += n_new
        total_skip += n_skip
        print(f"  {scene['name']}: {n_new} jobs queued, {n_skip} skipped (already rendered)")

    print(f"\n=== Done: {total_new} jobs queued, {total_skip} skipped ===")
    print("Open Movie Render Queue -> Render (Local). Then run encode_and_verify.py.")


if __name__ == "__main__":
    main()


# ============================================================================
# RATIONALE (for the paper's methods section / future you)
# ============================================================================
# - ns and fs fixed: factorial study showed main effects < 0.3 CGVQM points.
# - aw and hp crossed: they interact (factorial), so they must be varied jointly.
# - sp crossed with both: resolution dependence is the paper's central finding.
# - Reference at 100 % for every resolution: quality is measured against the
#   ideal full-resolution image. (The old factorial test rendered its reference
#   at 87 %, i.e. it measured distance to a degraded target.)
# - Warm-up: so low-alpha configs aren't scored on an unconverged history.
# - PNG without alpha, then lossless video (encode_and_verify.py): bit-identical
#   to PNG, roughly half the size. Lossy H.264 (crf 10-15, yuv420p) is NOT
#   identical and its artifacts would contaminate the metric.