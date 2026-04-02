import json
import pandas as pd

RESOLUTIONS = ['50', '71', '87', '100']
PARAMS = ['alpha_weight', 'hist_percent', 'filter_size', 'num_samples']

# Scenes to exclude (outliers)
EXCLUDE_SCENES = [
    'junkyard-mound1', 'junkyard-mound2',
    'oldmine-speed-18', 'oldmine-speed-35',
    'oldmine-speed-75', 'oldmine-speed-9', 'oldmine-warm', 
    'wildwest-barzoom'
]

with open('dataset.json', 'r') as f:
    raw = json.load(f)

records = []
for scene, scene_data in raw.items():
    if not isinstance(scene_data, dict):
        continue
    for res in RESOLUTIONS:
        if res not in scene_data:
            continue
        for ref, ref_data in scene_data[res].items():
            if ref != f'ref-{scene}': # we only want data when lower resolutions are compared to full-quality 16xSSAA
                continue
            for param in PARAMS:
                if param not in ref_data:
                    continue
                for entry in ref_data[param]:
                    records.append({
                        'scene':            scene,
                        'resolution':       int(res),
                        'parameter':        param,
                        'value':            entry['value'],
                        'score':            entry['score'],
                        'per_frame_errors': entry.get('per_frame_errors', []), # this is optional, can ignore
                    })

df = pd.DataFrame(records)
df = df[~df['scene'].isin(EXCLUDE_SCENES)]

# Exclude hist_percent values below 100 (must do because Unreal treats all sub 100 values as 100)
df = df[~((df['parameter'] == 'hist_percent') & (df['value'] < 100))]


### ________ Optional: Center Data _________
# find which values are shared across all scenes for each (parameter, resolution) 
common_vals_count = df.groupby(['parameter', 'resolution', 'value'])['scene'].nunique()
n_scenes_per_param_res = df.groupby(['parameter','resolution'])['scene'].nunique()

def center_score(group):
	# group is a subset of df for one specific (scene, parameter, resolution) combination 
	# e.g. all rows for "abandoned" + "alpha_weight" + resolution 50
    param, res = group['parameter'].iloc[0], group['resolution'].iloc[0]
    
    # get the values tested for this (parameter, resolution) across all scenes
    common = df[(df['parameter'] == param) & (df['resolution'] == res)]
    common_vals = common.groupby('value')['scene'].nunique()
    n_scenes = common['scene'].nunique()
    shared_vals = common_vals[common_vals == n_scenes].index
    
    # compute mean score over shared values only, then subtract from all scores
    mask = group['value'].isin(shared_vals)
    mean = group.loc[mask, 'score'].mean()
    return group['score'] - mean

df['score_centered'] = df.groupby(
    ['scene', 'parameter', 'resolution'], group_keys=False
).apply(center_score)