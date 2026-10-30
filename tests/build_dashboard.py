"""
build_dashboard.py  (dataset v2)
--------------------------------
Builds ONE self-contained interactive HTML page from everything the pipeline has
produced so far. Re-run it whenever you've scored / featurised more scenes, then
refresh the page.

Reads (whatever exists -- missing pieces just leave their section empty):
  {root}/scores_all.csv                       from score_taa.py
  {root}/{scene}/features/patch_table.csv     from compute_features.py
Writes:
  {root}/dashboard.html                       (or --out)

Sections
  Overview     data checks: scores per scene, flagged renders, possible outlier
               renders, legacy check vs. the old factorial scores
  Parameters   quality vs. alpha (lines = history %), one panel per resolution;
               alpha x history heatmaps; best config per scene & resolution
  Artifacts    CGVQM, flicker and detail ratio vs. alpha -- why the curve peaks;
               patch-level correlations between CGVQM and each artifact measure
  Content      per patch: which alpha scored best, and how much alpha mattered,
               against the patch's motion / detail / disocclusion
  Interactions how alpha, history %, resolution, motion and detail act together
               (patch level): alpha curves split by content, best alpha per
               motion x detail, history gain by content, and the held-out gain
               from content-aware alpha vs. UE default / one alpha per resolution
  Metrics      CGVQM vs. CVVDP agreement (if CVVDP was scored)

USAGE
  pip install pandas numpy plotly     # plotly is only used to embed its JS (works offline)
  python build_dashboard.py --root ~/Documents/PerceptualTAA/data_v2
  python build_dashboard.py --root ... --old-scores oldmine=tests/oldmine/scores.json
  then open data_v2/dashboard.html in a browser
"""

import os
import json
import glob
import argparse
import datetime

import numpy as np
import pandas as pd

MAIN_REF = "ssaa16_sp100"
LEGACY_REF = "ssaa16_sp087"
DEFAULT_AW, DEFAULT_HP = 0.04, 100
CONTENT_KEYS = ["motion_mean", "disocc_frac", "detail_hf", "detail_grad",
                "contrast", "luma_mean", "temporal_change"]
ARTIFACT_KEYS = ["err_mae", "flicker", "hf_ratio", "err_disocc", "err_nondisocc"]
PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    return x


def records(df):
    return [_clean(r) for r in df.to_dict("records")]


# ============================================================================
# LOAD
# ============================================================================

def load_scores(root):
    path = os.path.join(root, "scores_all.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_patches(root):
    frames = []
    for p in sorted(glob.glob(os.path.join(root, "*", "features", "patch_table.csv"))):
        df = pd.read_csv(p)
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================================
# DERIVED TABLES
# ============================================================================

def render_table(scores, patches):
    """One row per TAA render (vs. the 100 % reference), metrics as columns."""
    if scores.empty:
        return pd.DataFrame()
    s = scores[(scores["ref"] == MAIN_REF) & (scores["kind"] == "taa")]
    keys = ["scene", "config", "sp", "aw", "hp", "ns", "fs"]
    wide = s.pivot_table(index=keys, columns="metric", values="score", aggfunc="first").reset_index()
    wide.columns.name = None
    flags = s[s["metric"] == "cgvqm"][["scene", "config", "flag"]]
    wide = wide.merge(flags, on=["scene", "config"], how="left")
    wide["flag"] = wide["flag"].fillna("")
    if not patches.empty:
        art = (patches[patches["kind"] == "taa"].groupby(["scene", "config"])[ARTIFACT_KEYS]
               .mean().reset_index())
        wide = wide.merge(art, on=["scene", "config"], how="left")
    return wide


def mark_outliers(rt, metric="cgvqm"):
    """A render is a 'possible outlier' if it sits far from its two alpha-neighbours
    on the same (scene, resolution, history) curve: |score - mean(neighbours)| >
    max(0.75, 6 x the scene's median absolute such deviation)."""
    rt = rt.copy()
    rt["neighbor_dev"] = np.nan
    if metric not in rt:
        rt["outlier"] = False
        return rt
    for _, g in rt.groupby(["scene", "sp", "hp"]):
        g = g.sort_values("aw")
        v = g[metric].to_numpy()
        if len(v) < 3:
            continue
        dev = np.full(len(v), np.nan)
        dev[1:-1] = v[1:-1] - (v[:-2] + v[2:]) / 2
        rt.loc[g.index, "neighbor_dev"] = dev
    thr = rt.groupby("scene")["neighbor_dev"].transform(
        lambda d: max(0.75, 6 * np.nanmedian(np.abs(d))) if d.notna().any() else np.inf)
    big = rt["neighbor_dev"].abs() > thr
    # one bad render also pushes its two neighbours off; only flag the worst of each run
    local_max = pd.Series(False, index=rt.index)
    for _, g in rt.groupby(["scene", "sp", "hp"]):
        g = g.sort_values("aw")
        a = g["neighbor_dev"].abs().fillna(0).to_numpy()
        prev = np.r_[0, a[:-1]]
        nxt = np.r_[a[1:], 0]
        local_max.loc[g.index] = (a >= prev) & (a >= nxt)
    rt["outlier"] = big & local_max
    return rt


def best_configs(rt, metric="cgvqm"):
    rows = []
    if metric not in rt:
        return rows
    for (scene, sp), g in rt.groupby(["scene", "sp"]):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        b = g.loc[g[metric].idxmax()]
        d = g[(np.isclose(g["aw"], DEFAULT_AW)) & (g["hp"] == DEFAULT_HP)]
        dv = d[metric].iloc[0] if len(d) else np.nan
        rows.append({"scene": scene, "sp": sp, "best_aw": b["aw"], "best_hp": b["hp"],
                     "best": b[metric], "default": dv, "gain": b[metric] - dv,
                     "range": g[metric].max() - g[metric].min()})
    return rows


def cell_table(patches):
    """Per (scene, resolution, history, patch-clip): best alpha, sensitivity, content."""
    if patches.empty or "cgvqm_patch_quality" not in patches:
        return pd.DataFrame()
    p = patches[(patches["kind"] == "taa")].dropna(subset=["cgvqm_patch_quality"])
    if p.empty:
        return pd.DataFrame()
    keys = ["scene", "sp", "hp", "clip", "row", "col"]
    idx = p.groupby(keys)["cgvqm_patch_quality"].idxmax()
    best = p.loc[idx, keys + ["aw", "cgvqm_patch_quality"]].rename(
        columns={"aw": "best_aw", "cgvqm_patch_quality": "best_q"})
    agg = p.groupby(keys)["cgvqm_patch_quality"].agg(["min", "max", "count"]).reset_index()
    agg["sensitivity"] = agg["max"] - agg["min"]
    dflt = p[np.isclose(p["aw"], DEFAULT_AW)][keys + ["cgvqm_patch_quality"]].rename(
        columns={"cgvqm_patch_quality": "default_q"})
    content = p.groupby(keys)[[k for k in CONTENT_KEYS if k in p]].first().reset_index()
    out = (best.merge(agg[keys + ["sensitivity", "count"]], on=keys)
               .merge(dflt, on=keys, how="left").merge(content, on=keys))
    out["gain_vs_default"] = out["best_q"] - out["default_q"]
    return out


def patch_correlations(patches):
    """Spearman correlation between CGVQM patch quality and each artifact measure, per scene."""
    rows = []
    if patches.empty or "cgvqm_patch_quality" not in patches:
        return rows
    p = patches[patches["kind"] == "taa"].dropna(subset=["cgvqm_patch_quality"]).copy()
    if p.empty:
        return rows
    p["detail_change"] = np.abs(np.log2(p["hf_ratio"].clip(lower=1e-6)))
    measures = ["err_mae", "flicker", "detail_change", "err_disocc"]
    for scene, g in list(p.groupby("scene")) + [("All scenes", p)]:
        r = {"scene": scene, "n": int(len(g))}
        for m in measures:
            sub = g[["cgvqm_patch_quality", m]].dropna()
            r[m] = sub.corr(method="spearman").iloc[0, 1] if len(sub) > 10 else np.nan
        rows.append(r)
    return rows


def curve_payload(patches):
    """Compact per-patch alpha curves for the interaction plots.
    content: one entry per (scene, clip, row, col) with its content features.
    curves:  one entry per (patch, resolution, history) with arrays over alpha
             of CGVQM patch quality (q), flicker (f) and detail ratio (h)."""
    if patches.empty or "cgvqm_patch_quality" not in patches:
        return [], [], []
    p = patches[patches["kind"] == "taa"].dropna(subset=["cgvqm_patch_quality"])
    if p.empty:
        return [], [], []
    aws = sorted(p["aw"].unique().tolist())
    ck = ["scene", "clip", "row", "col"]
    feats = [k for k in CONTENT_KEYS if k in p]
    content = p.groupby(ck)[feats].first().reset_index()
    content["id"] = np.arange(len(content))
    p = p.merge(content[ck + ["id"]], on=ck)
    curves = []
    for (cid, sp, hp), g in p.groupby(["id", "sp", "hp"]):
        g = g.set_index("aw")
        arr = lambda col, d: [None if (a not in g.index or not np.isfinite(g.at[a, col])) else round(float(g.at[a, col]), d)
                              for a in aws]
        curves.append({"c": int(cid), "sp": int(sp), "hp": int(hp),
                       "q": arr("cgvqm_patch_quality", 3), "f": arr("flicker", 3), "h": arr("hf_ratio", 4)})
    return aws, records(content), curves


def legacy_rows(scores, old_scores):
    if scores.empty:
        return []
    s = scores[(scores["kind"] == "legacy_check") & (scores["metric"] == "cgvqm")]
    if s.empty:
        return []
    rows = []
    for (scene, cfg), g in s.groupby(["scene", "config"]):
        r = {"scene": scene, "config": cfg,
             "new_ref87": g.loc[g["ref"] == LEGACY_REF, "score"].mean(),
             "new_ref100": g.loc[g["ref"] == MAIN_REF, "score"].mean(),
             "old": np.nan}
        old = old_scores.get(scene)
        if old:
            parts = {x[:2]: x[2:] for x in cfg.split("_")}      # sp087_aw0.040_... -> {'sp':'087',...}
            name = f"aw{float(parts['aw']):g}_ns{int(parts['ns'])}_fs{float(parts['fs']):.1f}_hp{int(parts['hp'])}"
            r["old"] = old.get(name, {}).get("score", np.nan)
            r["old_name"] = name
        rows.append(r)
    return rows


# ============================================================================
# MAIN
# ============================================================================

def build(root, out, old_scores_args):
    scores = load_scores(root)
    patches = load_patches(root)
    rt = render_table(scores, patches)
    if not rt.empty:
        rt = mark_outliers(rt)
    old_scores = {}
    for item in old_scores_args:
        scene, path = item.split("=", 1)
        with open(path) as f:
            old_scores[scene] = json.load(f)

    cells = cell_table(patches)
    curve_aws, content_cells, curves = curve_payload(patches)
    metrics = [m for m in ("cgvqm", "cvvdp") if m in rt]
    scenes = sorted(set(rt["scene"]) if not rt.empty else set()
                    | (set(patches["scene"]) if not patches.empty else set()))

    overview = []
    for sc in scenes:
        g = rt[rt["scene"] == sc] if not rt.empty else pd.DataFrame()
        c = cells[cells["scene"] == sc] if not cells.empty else pd.DataFrame()
        overview.append({
            "scene": sc,
            "renders": int(len(g)),
            "cgvqm_min": g["cgvqm"].min() if "cgvqm" in g else np.nan,
            "cgvqm_median": g["cgvqm"].median() if "cgvqm" in g else np.nan,
            "cgvqm_max": g["cgvqm"].max() if "cgvqm" in g else np.nan,
            "flagged": int((g["flag"] != "").sum()) if "flag" in g else 0,
            "outliers": int(g["outlier"].sum()) if "outlier" in g else 0,
            "has_cvvdp": bool("cvvdp" in g and g["cvvdp"].notna().any()),
            "patch_cells": int(len(c)),
        })

    data = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "root": root,
        "default": {"aw": DEFAULT_AW, "hp": DEFAULT_HP},
        "scenes": scenes,
        "metrics": metrics,
        "content_keys": [k for k in CONTENT_KEYS if not cells.empty and k in cells],
        "overview": [_clean(r) for r in overview],
        "renders": records(rt.drop(columns=["ns", "fs"], errors="ignore")) if not rt.empty else [],
        "best": [_clean(r) for r in best_configs(rt)] if not rt.empty else [],
        "cells": records(cells) if not cells.empty else [],
        "corr": [_clean(r) for r in patch_correlations(patches)],
        "legacy": [_clean(r) for r in legacy_rows(scores, old_scores)],
        "curve_aws": curve_aws,
        "content": content_cells,
        "curves": curves,
    }

    try:
        import plotly
        js_path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
        plotly_tag = "<script>" + open(js_path, encoding="utf-8").read() + "</script>"
        how = "embedded (works offline)"
    except Exception:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'
        how = "loaded from CDN (needs internet; `pip install plotly` to embed instead)"

    tmpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")
    html = open(tmpl_path, encoding="utf-8").read()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("<!--PLOTLY-->", plotly_tag).replace("/*DATA*/null", payload)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Scenes: {', '.join(scenes) or '(none yet)'}")
    print(f"Renders: {len(data['renders'])}  patch cells: {len(data['cells'])}  metrics: {metrics}")
    print(f"Plotly {how}")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root, e.g. ~/Documents/PerceptualTAA/data_v2")
    ap.add_argument("--out", default=None, help="output HTML (default: {root}/dashboard.html)")
    ap.add_argument("--old-scores", nargs="*", default=[], metavar="SCENE=PATH",
                    help="old factorial scores.json per scene, for the legacy check table")
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    build(root, args.out or os.path.join(root, "dashboard.html"), args.old_scores)


if __name__ == "__main__":
    main()