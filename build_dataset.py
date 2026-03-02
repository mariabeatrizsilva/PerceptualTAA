"""
build_dataset.py

Crawls outputs/ directory, merges any new CGVQM scores into dataset.json.
Run this every time you generate new metrics — it only adds new data,
never recomputes what's already there.

Output schema:
    {
        "oldmine": {
            "metric": "CGVQM",
            "100": {
                "ref-oldmine": {
                    "alpha_weight": [{"value": 0.01, "score": 95.59, "per_frame_errors": [...]}, ...]
                }
            },
            "25": {
                "ref-oldmine-screen-per-25": { ... },  # no _ref- suffix in filename
                "ref-oldmine": { ... }                  # _ref-oldmine suffix in filename
            }
        }
    }

ref key is always:
    - ref-{folder_name}       when filename has no _ref- suffix (self-reference)
    - ref-{whatever}          when filename has _ref-{whatever} suffix (cross-scene)

metric is stored once per base_scene, not repeated on every block.

Usage:
    python build_dataset.py                  # incremental update
    python build_dataset.py --rebuild        # full rebuild from scratch
    python build_dataset.py --dry-run        # preview without writing
"""

import os
import json
import re
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset.json')
SCORES_SUBDIR = 'scores_cgvqm'
METRIC_NAME   = 'CGVQM'

# Parses folder names:
#   oldmine                  -> base_scene=oldmine,   screen_pct=100
#   oldmine-screen-per-25    -> base_scene=oldmine,   screen_pct=25
#   quarry-all               -> base_scene=quarry-all, screen_pct=100
SCENE_RE = re.compile(r'^(.+?)(?:-screen-per-(\d+))?$')

# Parses JSON entry keys:
#   vary_alpha_weight_0.01   -> param=alpha_weight, value=0.01
#   vary_num_samples_16      -> param=num_samples,  value=16.0
KEY_RE = re.compile(r'^vary_(.+?)_([\d.]+)$')


def parse_scene_folder(folder_name: str) -> tuple:
    """
    Parse a scene folder name into (base_scene, screen_pct).

    The folder name is also used as the default ref when no _ref- suffix
    is present in the JSON filename — i.e. self-reference.

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


def parse_ref_from_filename(json_filename: str) -> str | None:
    """
    Extract explicit cross-scene reference from a JSON filename.
    Returns None if no _ref- suffix is present (meaning self-reference).

    Examples:
        'vary_alpha_weight_scores.json'             -> None
        'vary_alpha_weight_scores_ref-oldmine.json' -> 'oldmine'
    """
    m = re.search(r'_ref-(.+)\.json$', json_filename)
    return m.group(1) if m else None


def parse_param_key(key: str) -> tuple | None:
    """
    Parse a JSON entry key into (param_name, param_value).
    Returns None for keys that aren't param entries (e.g. '_meta').

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
    n_added   = 0
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

            try:
                with open(json_path, 'r') as f:
                    raw = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Could not parse {json_path}: {e}")
                continue

            # ref is determined solely from the filename:
            #   no _ref- suffix → self-reference → use the full folder name
            #                     (e.g. oldmine-screen-per-25, not just oldmine)
            #   _ref-X suffix   → explicit cross-scene reference → use X
            # _meta is ignored entirely for ref resolution.
            explicit_ref = parse_ref_from_filename(json_file)
            ref_scene    = explicit_ref if explicit_ref else scene_folder
            ref_key      = f"ref-{ref_scene}"

            # Ensure nested structure: scene → metric, screen_pct → ref_key
            if not dry_run:
                dataset.setdefault(base_scene, {"metric": METRIC_NAME})
                dataset[base_scene].setdefault(screen_pct, {})
                dataset[base_scene][screen_pct].setdefault(ref_key, {})

            # Process each score entry, skipping meta keys
            for key, entry in raw.items():
                if key.startswith('_'):
                    continue  # skip _meta and any other meta keys

                parsed = parse_param_key(key)
                if parsed is None:
                    print(f"  WARNING: Unrecognised key '{key}' in {json_file}, skipping")
                    continue

                param, value = parsed

                if entry_exists(dataset, base_scene, screen_pct, ref_key, param, value):
                    n_skipped += 1
                    continue

                # Support both old format (raw float) and new format (dict with score + per_frame_errors)
                if isinstance(entry, dict):
                    score            = entry.get('score')
                    per_frame_errors = entry.get('per_frame_errors', [])
                else:
                    score            = entry
                    per_frame_errors = []

                new_point = {
                    "value":            value,
                    "score":            score,
                    "per_frame_errors": per_frame_errors,
                }

                if dry_run:
                    print(f"  [DRY RUN] {base_scene} | pct={screen_pct} | {ref_key} | {param}={value} -> score={score:.4f}")
                else:
                    dataset[base_scene][screen_pct][ref_key].setdefault(param, [])
                    dataset[base_scene][screen_pct][ref_key][param].append(new_point)

                n_added += 1

    # Sort all param arrays by value so plots are always in order
    if not dry_run:
        for base_scene, scene_data in dataset.items():
            for screen_pct, pct_data in scene_data.items():
                if screen_pct == 'metric':
                    continue
                for ref_key, ref_data in pct_data.items():
                    for param, entries in ref_data.items():
                        if isinstance(entries, list):
                            entries.sort(key=lambda x: x['value'])

    return dataset, n_added, n_skipped


def print_summary(dataset: dict):
    """Print a human-readable summary of the dataset contents."""
    print("\n── Dataset Summary ──────────────────────────────────────────────")
    total_points = 0
    for base_scene in sorted(dataset):
        metric = dataset[base_scene].get('metric', '?')
        print(f"\n  {base_scene}  [{metric}]")
        for screen_pct, pct_data in sorted(dataset[base_scene].items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
            if screen_pct == 'metric':
                continue
            for ref_key, ref_data in sorted(pct_data.items()):
                params = {k: len(v) for k, v in ref_data.items() if isinstance(v, list)}
                n = sum(params.values())
                total_points += n
                param_str = ', '.join(f"{k}({v})" for k, v in params.items())
                print(f"    pct={screen_pct:>4} | {ref_key:<45} | {param_str}")
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
    parser.add_argument('--rebuild',  action='store_true',
                        help='Ignore existing dataset.json and rebuild from scratch.')
    parser.add_argument('--dry-run',  action='store_true',
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
            print("Loaded existing dataset. Checking for new entries...\n")
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