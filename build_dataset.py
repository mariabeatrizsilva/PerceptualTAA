"""
build_dataset.py

Crawls outputs/ directory, merges any new CGVQM scores into dataset.json.
Run this every time you generate new metrics — it only adds new data,
never recomputes what's already there.

Usage:
    python build_dataset.py                  # incremental update
    python build_dataset.py --rebuild        # full rebuild from scratch
    python build_dataset.py --dry-run        # print what would be added without writing
"""

import os
import json
import re
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset.json')
SCORES_SUBDIR = 'scores_cgvqm'

# Parses folder names:
#   oldmine                  -> base_scene=oldmine, screen_pct=100
#   oldmine-screen-per-25    -> base_scene=oldmine, screen_pct=25
SCENE_RE = re.compile(r'^(.+?)(?:-screen-per-(\d+))?$')

# Parses JSON entry keys:
#   vary_alpha_weight_0.01   -> param=alpha_weight, value=0.01
#   vary_num_samples_16      -> param=num_samples,  value=16
KEY_RE = re.compile(r'^vary_(.+?)_([\d.]+)$')


def parse_scene_folder(folder_name: str) -> tuple:
    """
    Parse a scene folder name into (base_scene, screen_pct).

    Examples:
        'oldmine'               -> ('oldmine', '100')
        'oldmine-screen-per-25' -> ('oldmine', '25')
        'quarry-all'            -> ('quarry-all', '100')
    """
    m = SCENE_RE.match(folder_name)
    if not m:
        return folder_name, '100'
    base_scene = m.group(1)
    screen_pct = m.group(2) if m.group(2) else '100'
    return base_scene, screen_pct


def parse_ref_scene_from_filename(json_filename: str, fallback_scene: str) -> str:
    """
    Parse the reference scene from a scores JSON filename.

    Examples:
        'vary_alpha_weight_scores.json'             -> fallback_scene
        'vary_alpha_weight_scores_ref-oldmine.json' -> 'oldmine'
    """
    m = re.search(r'_ref-(.+)\.json$', json_filename)
    if m:
        return m.group(1)
    return fallback_scene


def parse_param_key(key: str):
    """
    Parse a JSON entry key into (param_name, param_value), or None if not parseable.

    Examples:
        'vary_alpha_weight_0.01' -> ('alpha_weight', 0.01)
        'vary_num_samples_16'    -> ('num_samples', 16.0)
        '_meta'                  -> None
    """
    m = KEY_RE.match(key)
    if not m:
        return None
    return m.group(1), float(m.group(2))


def load_dataset() -> dict:
    """Load existing dataset.json, or return empty dict if not found."""
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_dataset(dataset: dict):
    """Save dataset to dataset.json."""
    with open(DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"\n✓ Dataset saved to: {DATASET_PATH}")


def entry_exists(dataset: dict, base_scene: str, screen_pct: str,
                 ref_key: str, param: str, value: float) -> bool:
    """Check if a specific (scene, pct, ref, param, value) point already exists."""
    try:
        existing = dataset[base_scene][screen_pct][ref_key][param]
        return any(abs(e['value'] - value) < 1e-9 for e in existing)
    except KeyError:
        return False


def crawl_outputs(outputs_dir: str, dataset: dict, dry_run: bool = False) -> tuple:
    """
    Crawl outputs/ and merge any new data into dataset.

    Returns:
        (updated_dataset, n_added, n_skipped)
    """
    n_added = 0
    n_skipped = 0

    if not os.path.exists(outputs_dir):
        print(f"ERROR: outputs/ directory not found at {outputs_dir}")
        return dataset, 0, 0

    scene_folders = sorted([
        d for d in os.listdir(outputs_dir)
        if os.path.isdir(os.path.join(outputs_dir, d))
    ])

    for scene_folder in scene_folders:
        base_scene, screen_pct = parse_scene_folder(scene_folder)
        scores_dir = os.path.join(outputs_dir, scene_folder, SCORES_SUBDIR)

        if not os.path.exists(scores_dir):
            continue

        json_files = sorted([f for f in os.listdir(scores_dir) if f.endswith('.json')])

        for json_file in json_files:
            json_path = os.path.join(scores_dir, json_file)
            ref_scene = parse_ref_scene_from_filename(json_file, base_scene)
            ref_key = f"ref-{ref_scene}"

            try:
                with open(json_path, 'r') as f:
                    raw = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Could not parse {json_path}: {e}")
                continue

            # Build metadata — prefer file's _meta if present, fill in gaps
            file_meta = raw.get('_meta', {})

            # If _meta has reference_scene, trust it over filename parsing
            ref_scene = file_meta.get('reference_scene') or parse_ref_scene_from_filename(json_file, base_scene)
            ref_key = f"ref-{ref_scene}"

            resolved_meta = {
                "reference_scene": ref_scene,
                "metric": file_meta.get('metric', 'CGVQM'),
                "source_file": json_file,
            }
            # Merge any remaining extra fields from _meta
            for k, v in file_meta.items():
                if k not in resolved_meta:
                    resolved_meta[k] = v

            # Ensure nested structure exists in dataset
            dataset.setdefault(base_scene, {})
            dataset[base_scene].setdefault(screen_pct, {})
            dataset[base_scene][screen_pct].setdefault(ref_key, {"_meta": resolved_meta})

            # Always keep _meta fresh
            dataset[base_scene][screen_pct][ref_key]["_meta"] = resolved_meta

            # Process each score entry
            for key, entry in raw.items():
                if key.startswith('_'):
                    continue  # skip _meta and any future meta keys

                parsed = parse_param_key(key)
                if parsed is None:
                    print(f"  WARNING: Unrecognised key '{key}' in {json_file}, skipping")
                    continue

                param, value = parsed

                if entry_exists(dataset, base_scene, screen_pct, ref_key, param, value):
                    n_skipped += 1
                    continue

                # Extract score and per_frame_errors — handle both old and new JSON formats
                if isinstance(entry, dict):
                    score = entry.get('score')
                    per_frame_errors = entry.get('per_frame_errors', [])
                else:
                    # Old format: entry is just a raw score float
                    score = entry
                    per_frame_errors = []

                new_point = {
                    "value": value,
                    "score": score,
                    "per_frame_errors": per_frame_errors
                }

                if dry_run:
                    print(f"  [DRY RUN] {base_scene} | pct={screen_pct} | {ref_key} | {param}={value} -> score={score:.4f}")
                else:
                    dataset[base_scene][screen_pct][ref_key].setdefault(param, [])
                    dataset[base_scene][screen_pct][ref_key][param].append(new_point)

                n_added += 1

    # Sort all param arrays by value so plots are always ordered
    if not dry_run:
        for base_scene in dataset:
            for screen_pct in dataset[base_scene]:
                for ref_key in dataset[base_scene][screen_pct]:
                    for param, entries in dataset[base_scene][screen_pct][ref_key].items():
                        if param.startswith('_') or not isinstance(entries, list):
                            continue
                        entries.sort(key=lambda x: x['value'])

    return dataset, n_added, n_skipped


def print_summary(dataset: dict):
    """Print a human-readable summary of the dataset contents."""
    print("\n── Dataset Summary ──────────────────────────────────────────────")
    total_points = 0
    for base_scene in sorted(dataset):
        print(f"\n  {base_scene}")
        for screen_pct in sorted(dataset[base_scene], key=lambda x: int(x)):
            for ref_key in sorted(dataset[base_scene][screen_pct]):
                block = dataset[base_scene][screen_pct][ref_key]
                params = {k: len(v) for k, v in block.items()
                          if not k.startswith('_') and isinstance(v, list)}
                n = sum(params.values())
                total_points += n
                ref_scene = block.get('_meta', {}).get('reference_scene', ref_key)
                param_str = ', '.join(f"{k}({v})" for k, v in params.items())
                print(f"    pct={screen_pct:>4} | ref={ref_scene:<40} | {param_str}")
    print(f"\n  Total data points: {total_points}")
    print("─────────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description='Build/update dataset.json from outputs/ directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_dataset.py                        # incremental update
  python build_dataset.py --rebuild              # full rebuild from scratch
  python build_dataset.py --dry-run              # preview without writing
  python build_dataset.py --outputs-dir /path    # custom outputs directory
        """
    )
    parser.add_argument('--rebuild', action='store_true',
                        help='Ignore existing dataset.json and rebuild from scratch.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be added without writing anything.')
    parser.add_argument('--outputs-dir', type=str, default=OUTPUTS_DIR,
                        help=f'Path to outputs directory (default: {OUTPUTS_DIR})')
    args = parser.parse_args()

    print("── PerceptualTAA Dataset Builder ────────────────────────────────")
    print(f"  Outputs dir : {args.outputs_dir}")
    print(f"  Dataset file: {DATASET_PATH}")
    print(f"  Mode        : {'REBUILD' if args.rebuild else 'DRY RUN' if args.dry_run else 'INCREMENTAL'}")
    print("─────────────────────────────────────────────────────────────────\n")

    if args.rebuild or args.dry_run:
        dataset = {}
        if args.rebuild:
            print("Rebuilding dataset from scratch...\n")
    else:
        dataset = load_dataset()
        if dataset:
            print(f"Loaded existing dataset. Checking for new entries...\n")
        else:
            print("No existing dataset found. Building from scratch...\n")

    dataset, n_added, n_skipped = crawl_outputs(args.outputs_dir, dataset, dry_run=args.dry_run)

    print(f"\n── Results ──────────────────────────────────────────────────────")
    print(f"  New entries added : {n_added}")
    print(f"  Entries skipped   : {n_skipped} (already in dataset)")

    if args.dry_run:
        print("\n  [DRY RUN] Nothing written.")
    else:
        save_dataset(dataset)
        print_summary(dataset)


if __name__ == '__main__':
    main()