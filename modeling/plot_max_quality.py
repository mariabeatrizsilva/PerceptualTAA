import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps

df = pd.read_csv('modelparams.csv')

TAAParam = "alpha_weight"

params = df[df["TAAParam"] == TAAParam][["k1", "k2", "k3", "k4", "k5", "k6"]].to_numpy()[0]
a, b, c, d, e, f = params

print(a, b, c)

# ============= EXAMPLE USAGE =============

# Example 1: Simple quadratic function
def my_model(param, res):
    return a + b * np.log(res) + c * np.log(param) + d * param * res + e * param + f * res

cmap = colormaps['viridis']
resvals = np.logspace(np.log10(50), np.log10(100), 25)
for res in resvals:
    N = 100
    ress = np.ones(N) * res
    if TAAParam == "alpha_weight":
        vals = np.linspace(0, 1, N)
        
    y_pred = my_model(vals, ress)
    
    plt.plot(vals, y_pred, c=cmap((res-50)/50), linewidth=.5)

xs = np.linspace(50, 100, 100)
ps = c / (-d * xs - e)
ys = my_model(ps, xs)
plt.plot(ps, ys, '--', c='red')

plt.xlabel("alpha_weight")
plt.ylabel("Quality")

plt.xlim([.01, 1])
# plt.ylim([82, 100])
plt.show()