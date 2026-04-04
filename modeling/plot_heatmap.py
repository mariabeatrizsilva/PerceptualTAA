import json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.optimize import fsolve

font_path = "/Users/kennychen/Library/Fonts/LinBiolinum_R.otf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

fontsize = 8

mpl.rcParams.update({
    "font.family": prop.get_name(),
    "font.size": fontsize,           # match font size in your paper
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
})

mpl.rcParams['text.antialiased'] = True

fig_width_in = 3.33
fig_height_in = fig_width_in / 1.5
# fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))


def create_heatmap(func, x_range, y_range, x_label='x', y_label='y', 
                   title='Function Heatmap', num_points=100, figsize=(10, 8),
                   cmap='viridis', show_values=False, vmin=None, vmax=None):
    """
    Create a heatmap visualization of a 2-parameter function.
    
    Parameters:
    -----------
    func : callable
        Function that takes two arguments (x, y) and returns a scalar z
    x_range : tuple
        (min, max) for x-axis parameter
    y_range : tuple
        (min, max) for y-axis parameter
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str
        Plot title
    num_points : int
        Number of points along each axis (resolution)
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name (e.g., 'viridis', 'plasma', 'coolwarm', 'RdYlBu_r')
    show_values : bool
        Whether to annotate cells with values (only for small grids)
    vmin, vmax : float
        Min and max values for color scale (optional)
    
    Returns:
    --------
    fig, ax, Z : matplotlib figure, axes, and the computed values matrix
    """
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute function values
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            try:
                Z[i, j] = func(X[i, j], Y[i, j])
            except:
                Z[i, j] = np.nan
    
    # Create figure
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    
    if y_label == "Quality":
        cbarlabel = "Resolution"
    else:
        cbarlabel = "Quality"
    
    # Create heatmap
    if show_values and num_points <= 20:
        sns.heatmap(Z, 
                    xticklabels=np.round(x, 2), 
                    yticklabels=np.round(y, 2),
                    annot=True, 
                    fmt='.2f', 
                    cmap=cmap,
                    cbar_kws={'label': cbarlabel},
                    vmin=vmin,
                    vmax=vmax,
                    ax=ax)
    else:
        im = ax.imshow(Z, 
                       extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                       origin='lower', 
                       aspect='auto',
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        
    # cbar.set_label(cbarlabel, rotation=270, labelpad=10)
    # cbar.set_label(cbarlabel, rotation=270, labelpad=15, loc='bottom', fontsize=10)
    # cbar.ax.set_xlabel(cbarlabel, fontsize=fontsize/2, labelpad=5)
    xlabel = cbar.ax.set_xlabel(cbarlabel, fontsize=fontsize/1.5)
    xlabel.set_horizontalalignment('left')
    cbar.ax.xaxis.set_label_coords(-.5, -.025)
    
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax, Z

df = pd.read_csv('modelparams.csv')

TAAParam = "alpha_weight"

params = df[df["TAAParam"] == TAAParam][["k1", "k2", "k3", "k4", "k5", "k6"]].to_numpy()[0]
a, b, c, d, e, f = params

print(a, b, c)

# ============= EXAMPLE USAGE =============

# Example 1: Simple quadratic function
def my_model(param, res):
    return a + b * np.log(res) + c * np.log(param) + d * param * res + e * param + f * res

def inverse_model(param, quality):
    """Find resolution that gives target quality for given param"""
    def equation(res):
        if res <= 0:
            return 1e10  # penalize invalid resolution
        return my_model(param, res) - quality
    
    # Initial guess - try middle of reasonable range
    initial_guess = 50
    
    try:
        solution = fsolve(equation, initial_guess, full_output=True)
        res = solution[0][0]
        info = solution[1]
        
        # Check if solution converged and is valid
        if info['fvec'][0]**2 < 1e-6 and res <= 100:
            return res
        else:
            return np.nan
    except:
        return np.nan

# fig, ax, Z = create_heatmap(
#     func=my_model,
#     x_range=(0, 1),
#     y_range=(50, 100),
#     x_label='X Parameter',
#     y_label='Y Parameter',
#     title='Example: $z = x^2 + y^2$',
#     num_points=100,
#     cmap='viridis'
# )
# plt.show()


def create_heatmap_with_contours(func, x_range, y_range, num_points=100, 
                                  num_contours=10, mark_min_y=True, **kwargs):
    """Extended version with contour lines and optional minimum y markers"""
    fig, ax, Z = create_heatmap(func, x_range, y_range, num_points=num_points, **kwargs)
    
    # Add contour lines
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    contours = ax.contour(X, Y, Z, levels=num_contours, colors='white', 
                          alpha=0.4, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=fontsize/1.5)
    
    # Mark minimum y points on each contour
    if mark_min_y and kwargs["y_label"] != "Quality":
        min_y_points = []
        
        for level_idx, level in enumerate(contours.levels):
            
            if level < 78:
                continue
            # print(level)
            # Get the contour paths for this level
            paths = contours.collections[level_idx].get_paths()
            
            # Track overall minimum y across all segments
            overall_min_y = np.inf
            overall_min_x = None
            touches_boundary = False
            
            for path in paths:
                vertices = path.vertices
                
                if len(vertices) == 0:
                    continue
                
                # Check if this contour segment touches the boundaries
                x_coords = vertices[:, 0]
                y_coords = vertices[:, 1]
                
                # Tolerance for floating point comparison
                tol = (x_range[1] - x_range[0]) * 1e-6
                
                # Check if contour touches any boundary
                touches_left = np.any(np.abs(x_coords - x_range[0]) < tol)
                touches_right = np.any(np.abs(x_coords - x_range[1]) < tol)
                touches_bottom = np.any(np.abs(y_coords - y_range[0]) < tol)
                touches_top = np.any(np.abs(y_coords - y_range[1]) < tol)
                
                segment_touches_boundary = (touches_left or touches_right or 
                                           touches_bottom or touches_top)
                
                # Find minimum y in this segment
                min_idx = np.argmin(y_coords)
                min_y = y_coords[min_idx]
                min_x = x_coords[min_idx]
                
                # Check if the minimum point itself is on the bottom boundary
                min_on_bottom = np.abs(min_y - y_range[0]) < tol
                
                if min_y < overall_min_y:
                    overall_min_y = min_y
                    overall_min_x = min_x
                    # Only mark as touching boundary if min is on bottom 
                    # OR if contour touches bottom (suggesting it continues lower)
                    touches_boundary = min_on_bottom or touches_bottom
            
            # Only add point if it's a true minimum (not clamped to boundary)
            if overall_min_x is not None and not touches_boundary:
                min_y_points.append((overall_min_x, overall_min_y, level))
        
        # Plot the minimum y points
        if min_y_points:
            min_x_vals = [p[0] for p in min_y_points]
            min_y_vals = [p[1] for p in min_y_points]
            
            ax.scatter(min_x_vals, min_y_vals, 
                      c='red', s=10, marker='o', 
                      edgecolors='darkred', linewidths=.5,
                      label='Minimum resolution', zorder=5)
            
            # Connect points with a dashed line to show the locus
            if len(min_x_vals) > 1:
                ax.plot(min_x_vals, min_y_vals, 'r--', 
                       linewidth=1, alpha=0.7, zorder=4)
            
            # ax.legend(loc='best', fontsize=9)
    
    return fig, ax, Z
# Example with contours

yaxis = "quality"
if yaxis == "resolution":

    fig, ax, Z = create_heatmap_with_contours(
        func=my_model,
        x_range=(0.01, 1),
        y_range=(30, 100),
        x_label='Alpha Weight',
        y_label='Resolution',
        title='',
        num_points=1000,
        cmap='plasma',
        num_contours=10
    )
    plt.savefig('../figs/application-lossless.pdf', bbox_inches='tight', pad_inches=0.01)
elif yaxis == "quality":
    fig, ax, Z = create_heatmap_with_contours(
        func=inverse_model,
        x_range=(0.01, 1),
        y_range=(75, 100),
        x_label='Alpha Weight',
        y_label='Quality',
        title='',
        num_points=200,
        cmap='viridis',
        num_contours=5
    )
    
    N = 250
    ress = np.ones(N) * 99
    vals = np.linspace(0.001, 1, N)
    y_pred = my_model(vals, ress)
    
    ax.plot(vals, y_pred, c='white', linewidth=4)
    
    for resolution in [30, 45, 60, 75, 90]:
        p = c / (-d * resolution - e)
        y = my_model(p, resolution)
        ax.scatter(p, y, 
                      c='red', s=10, marker='o', 
                      edgecolors='darkred', linewidths=.5, zorder=5)
    
    xs = np.linspace(0, 90, N)
    ps = c / (-d * xs - e)
    ys = my_model(ps, xs)
    ax.plot(ps, ys, '--', c='red', linewidth=1)
    
    plt.xlim([.01, 1])
    plt.ylim([75, 95])
    
    # plt.show()
    plt.savefig('../figs/application-scaling.pdf', bbox_inches='tight', pad_inches=0.01)