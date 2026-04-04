import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def get_data(data_pth, RESOLUTIONS, PARAMS, EXCLUDE_SCENES):
    with open(data_pth, 'r') as f:
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

    # Find missing params (ideally none)
    for param in PARAMS:
        sub = df[df['parameter'] == param]
        pivot = sub.groupby(['scene', 'resolution', 'value'])['score'].count().unstack('value')
        missing = pivot[pivot.isna().any(axis=1)]
        if not missing.empty:
            print(f"\n=== {param} ===")
            print(missing.to_string())
            
    return df

def quadratic_model(X):
    X_linear = X
    X_log = np.log(X)
    X_interaction = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X_linear, X_log, X_interaction])

def my_model(xy, a, b, c, d, e, f):
    res, param = xy
    
    return a + b * np.log(res) + c * np.log(param) + d * param * res + e * param + f * res
    # return (a - b * (1/res)**.5) * (param ** c) - d
    # return a + (b * (np.log(res * c) * param)) ** d

def infer(X, b, ms):
    """
    Manually predict using the fitted coefficients
    X shape: (n_samples, n_features) - original features
    """
    X_poly = quadratic_model(X)
    y_pred = b + np.dot(X_poly, ms)
    return y_pred

if __name__ == '__main__':
    RESOLUTIONS = ['50', '71', '87', '100']
    PARAMS = ['alpha_weight', 'hist_percent', 'filter_size', 'num_samples']

    # Scenes to exclude (outliers)
    EXCLUDE_SCENES = [
        'junkyard-mound1', 'junkyard-mound2',
        'oldmine-speed-18', 'oldmine-speed-35',
        'oldmine-speed-75', 'oldmine-speed-9', 'oldmine-warm', 
        'wildwest-barzoom'
    ]

    data_pth = '../dataset.json'
    df = get_data(data_pth, RESOLUTIONS, PARAMS, EXCLUDE_SCENES)
            
    agg = df.groupby(['parameter', 'resolution', 'value'])['score'].agg(['mean', 'std', 'count']).reset_index()
    agg['ci95'] = 1.96 * (agg['std'] / np.sqrt(agg['count']))
    agg['resolution'] = agg['resolution'].astype(str) + '%'
    
    modelparams = []

    for param in PARAMS:
        # if param != 'alpha_weight':
        #     continue
        sub = agg[agg['parameter'] == param]
        resolution = np.asarray([int(x[:-1]) for x in sub['resolution']])
        value = sub['value']
        mean = sub['mean']
        
        X = np.column_stack([resolution, value])
        X_poly = quadratic_model(X)
        # model = LinearRegression()
        # model.fit(X_poly, mean)
        # b = model.intercept_
        # ms = model.coef_
        popt, _ = curve_fit(my_model, (resolution, value), mean, maxfev=100000) 
        
        modelparams += [[param] + list(popt)]
        
        # plt.scatter(value, mean)
        
        # for res in np.unique(resolution):
        #     ress = np.ones(100) * res
        #     vals = np.linspace(np.min(value), np.max(value), 100)
        #     Xs = np.column_stack([ress, vals])
        #     # y_pred = infer(Xs, b, ms)
        #     y_pred = my_model((ress, vals), *popt)
        #     plt.plot(vals, y_pred)
        
    # plt.show()
    
    print(modelparams)
    df = pd.DataFrame(modelparams, columns=["TAAParam", "k1", "k2", "k3", "k4", "k5", "k6"])
    df.to_csv('modelparams.csv', index=False)