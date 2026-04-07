"""
dataloader.py
--------------
Central loader for the CGVQM TAA dataset.

Usage in any notebook:
    from dataloader import load_df, load_sensitivity

    df = load_df("../dataset.json")                  # flat, centered
    sensitivity = load_sensitivity("path/to/dataset.json") # one row per scene x resolution x parameter
"""

import json
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

RESOLUTIONS = ['100', '87', '71','50']
PARAMS = ['alpha_weight', 'filter_size', 'hist_percent', 'num_samples']

EXCLUDE_SCENES = [
    'junkyard-mound1', 'junkyard-mound2',
    'oldmine-speed-18', 'oldmine-speed-35',
    'oldmine-speed-75', 'oldmine-speed-9',
    'oldmine-warm', 'wildwest-barzoom',
]

PARAM_LABELS = {
    'alpha_weight':  'Alpha weight',
    'hist_percent':  'History %',
    'filter_size':   'Filter size',
    'num_samples':   'Num samples',
}

RESOLUTION_LABELS = {50: '25%', 71: '50%', 87: '75%', 100: '100%'}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_records(raw: dict) -> pd.DataFrame:
    """Parse raw JSON into a flat DataFrame."""
    records = []
    for scene, scene_data in raw.items():
        if not isinstance(scene_data, dict):
            continue
        for res in RESOLUTIONS:
            if res not in scene_data:
                continue
            for ref, ref_data in scene_data[res].items():
                if ref != f'ref-{scene}':
                    continue
                for param in PARAMS:
                    if param not in ref_data:
                        continue
                    for entry in ref_data[param]:
                        records.append({
                            'scene':      scene,
                            'resolution': int(res),
                            'parameter':  param,
                            'value':      entry['value'],
                            'score':      entry['score'],
                            'per_frame_errors': entry.get('per_frame_errors', []),
                        })
    return pd.DataFrame(records)


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove excluded scenes and invalid hist_percent values."""
    df = df[~df['scene'].isin(EXCLUDE_SCENES)].copy()
    df = df[~((df['parameter'] == 'hist_percent') & (df['value'] < 100))]
    return df.reset_index(drop=True)


def _center_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add score_centered column.

    For each (scene, parameter, resolution) group, subtract the mean score
    computed over the parameter values that are shared across *all* scenes
    at that (parameter, resolution). This removes scene-level and
    resolution-level baseline differences so sensitivity reflects only
    responsiveness to the TAA parameter.
    """
    def _center_group(group):
        param = group['parameter'].iloc[0]
        res   = group['resolution'].iloc[0]

        # values shared across every scene for this (param, res)
        common = df[(df['parameter'] == param) & (df['resolution'] == res)]
        n_scenes    = common['scene'].nunique()
        shared_vals = (
            common.groupby('value')['scene'].nunique()
            .pipe(lambda s: s[s == n_scenes].index)
        )

        mask = group['value'].isin(shared_vals)
        mean = group.loc[mask, 'score'].mean()
        return group['score'] - mean

    df = df.copy()
    df['score_centered'] = (
        df.groupby(['scene', 'parameter', 'resolution'], group_keys=False)
        .apply(_center_group)
    )
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def load_df(json_path: str, center: bool = True) -> pd.DataFrame:
    """
    Load the dataset into a flat DataFrame.

    Parameters
    ----------
    json_path : str
        Path to dataset.json.
    center : bool
        If True (default), add a score_centered column.

    Returns
    -------
    DataFrame with columns:
        scene, resolution, parameter, value, score, [score_centered]
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    df = _parse_records(raw)
    df = _apply_filters(df)
    if center:
        df = _center_scores(df)
    return df


def load_sensitivity(
    json_path: str,
    score_col: str = 'score_centered',
) -> pd.DataFrame:
    """
    Load the dataset and compute per-(scene, resolution, parameter)
    sensitivity scores.

    Parameters
    ----------
    json_path : str
        Path to dataset.json.
    score_col : str
        Column to use for sensitivity computation.
        'score_centered' (default) or 'score'.

    Returns
    -------
    DataFrame with columns:
        scene, resolution, parameter,
        sensitivity_std, sensitivity_range, n_values
    """
    df = load_df(json_path, center=(score_col == 'score_centered'))

    sensitivity = (
        df.groupby(['scene', 'resolution', 'parameter'])[score_col]
        .agg(
            sensitivity_std='std',
            sensitivity_range=lambda x: x.max() - x.min(),
            n_values='count',
        )
        .reset_index()
    )
    return sensitivity


def load_perframe_pervalue(json_path: str, warmup_frames: int = 10, cooldown_frames: int = 10) -> pd.DataFrame:
    df = load_df(json_path, center=False)
    
    EXCLUDE_FRAMES = lambda f: (f % 30) > 5 and (f % 30) < 25

    records = []
    for (scene, res, param), group in df.groupby(['scene', 'resolution', 'parameter']):
        for _, row in group.iterrows():
            frames = [
                {'frame': frame_idx, 'error': error}
                for frame_idx, error in enumerate(row['per_frame_errors'])
            ]
            df_frames = pd.DataFrame(frames)
            df_frames = df_frames[df_frames['frame'].apply(EXCLUDE_FRAMES)]

            # trim warmup and cooldown
            total = len(df_frames)
            df_frames = df_frames[
                (df_frames['frame'] >= warmup_frames) & 
                (df_frames['frame'] < total - cooldown_frames)
            ]

            p5  = df_frames['error'].quantile(0.5)
            p95 = df_frames['error'].quantile(0.95)

            records.append({
                'scene':         scene,
                'resolution':    res,
                'parameter':     param,
                'value':         row['value'],
                'quality_best':  100 - p5,
                'quality_worst': 100 - p95,
            })

    return pd.DataFrame(records)

def load_sensitivity_perframe(json_path: str) -> pd.DataFrame:
    df = load_perframe_pervalue(json_path)
    
    def agg_sensitivity(group):
        quality_best  = group['quality_best']
        quality_worst = group['quality_worst']
        return pd.Series({
            'sensitivity_pct_range': (quality_best - quality_worst).mean(),
            'quality_best':          quality_best.mean(),
            'quality_worst':         quality_worst.mean(),
            'sensitivity_best_std':  quality_best.std(),
            'sensitivity_worst_std': quality_worst.std(),
        })

    return (
        df.groupby(['scene', 'resolution', 'parameter'])
        .apply(agg_sensitivity)
        .reset_index()
    )