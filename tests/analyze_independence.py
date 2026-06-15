"""
analyze_independence.py
-----------------------
Tests whether the 4 TAA parameters can be assumed independent, using the
scores produced by run_factorial_metrics.py.

METHOD:
  Fit two linear models to the CGVQM scores:
    Model A (main effects only):  score ~ aw + ns + fs + hp
    Model B (+ 2-way interactions): score ~ aw + ns + fs + hp
                                          + aw*ns + aw*fs + aw*hp
                                          + ns*fs + ns*hp + fs*hp

  If Model B is not significantly better than Model A (F-test, p > 0.05),
  the independence assumption holds.

  Also reports effect sizes (eta-squared) for every term so you can show
  reviewers which interactions, if any, matter.

USAGE:
  # single scene
  python analyze_independence.py --scene quarry-rocksonly

  # pool all scenes together for more power
  python analyze_independence.py --scene quarry-rocksonly village-day subway-turn --pool

OUTPUT (printed + saved to tests/{scene}/independence_report.txt):
  - ANOVA table for Model B
  - F-test: main-effects vs full interaction model
  - Verdict on independence assumption

DEPENDENCIES:
  pip install pandas scipy statsmodels
"""

import os
import sys
import json
import argparse
import textwrap

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
except ImportError:
    sys.exit("Please install statsmodels:  pip install statsmodels")

# ---------------------------------------------------------------------------
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_COLS = ["alpha_weight", "num_samples", "filter_size", "hist_percent"]
# Short R-style names (no dots, no spaces) for the formula
PARAM_ALIAS = {
    "alpha_weight": "aw",
    "num_samples":  "ns",
    "filter_size":  "fs",
    "hist_percent": "hp",
}
# ---------------------------------------------------------------------------


def load_scene(scene_name):
    """Load scores.json for one scene and return a DataFrame."""
    path = os.path.join(TESTS_DIR, scene_name, "scores.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"scores.json not found for scene '{scene_name}'. "
                                f"Run run_factorial_metrics.py first.")
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for combo, data in raw.items():
        row = {"combo": combo, "score": data["score"], "scene": scene_name}
        row.update(data["params"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_formula(terms):
    """Build an OLS formula string with the given list of term strings."""
    return "score ~ " + " + ".join(terms)


def analyze(df, label, out_lines):
    """Run the independence analysis on a DataFrame, append report to out_lines."""

    def h(text):
        out_lines.append("\n" + "="*70)
        out_lines.append(text)
        out_lines.append("="*70)

    h(f"INDEPENDENCE ANALYSIS — {label}")
    out_lines.append(f"N observations: {len(df)}")
    out_lines.append(f"Score mean: {df['score'].mean():.4f}  std: {df['score'].std():.4f}")

    # Rename columns to short aliases so statsmodels formula parser is happy
    df2 = df.rename(columns=PARAM_ALIAS)
    aliases = list(PARAM_ALIAS.values())   # ["aw", "ns", "fs", "hp"]

    # Treat parameters as continuous (they are numeric levels)
    # Center them so interaction coefficients are interpretable
    for col in aliases:
        df2[col] = df2[col].astype(float)
        df2[col] = df2[col] - df2[col].mean()

    # -----------------------------------------------------------------------
    # Model A: main effects only
    # -----------------------------------------------------------------------
    formula_A = build_formula(aliases)
    model_A   = smf.ols(formula_A, data=df2).fit()

    # -----------------------------------------------------------------------
    # Model B: main effects + all 6 pairwise interactions
    # -----------------------------------------------------------------------
    pairs = [f"{a}:{b}" for i, a in enumerate(aliases) for b in aliases[i+1:]]
    formula_B = build_formula(aliases + pairs)
    model_B   = smf.ols(formula_B, data=df2).fit()

    # -----------------------------------------------------------------------
    # ANOVA table for Model B (type II SS)
    # -----------------------------------------------------------------------
    h("ANOVA TABLE (Model B — main effects + 2-way interactions)")
    anova_B = anova_lm(model_B, typ=2)

    # Compute eta-squared = SS_effect / SS_total
    ss_total = anova_B["sum_sq"].sum()
    anova_B["eta_sq"] = anova_B["sum_sq"] / ss_total

    out_lines.append(anova_B.to_string(float_format=lambda x: f"{x:.4f}"))

    # -----------------------------------------------------------------------
    # F-test: does adding interactions significantly improve fit?
    # -----------------------------------------------------------------------
    h("F-TEST: Main Effects Only (A) vs + Interactions (B)")
    f_result = model_A.compare_f_test(model_B)
    F_stat, p_val, df_diff = f_result
    out_lines.append(f"F({int(df_diff)}, {int(model_B.df_resid)}) = {F_stat:.4f},  p = {p_val:.4f}")
    out_lines.append(f"R² (main effects only):   {model_A.rsquared:.4f}")
    out_lines.append(f"R² (+ interactions):      {model_B.rsquared:.4f}")
    out_lines.append(f"ΔR²:                      {model_B.rsquared - model_A.rsquared:.4f}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    h("VERDICT")
    alpha = 0.05
    if p_val > alpha:
        verdict = (
            f"p = {p_val:.4f} > {alpha} — adding interaction terms does NOT "
            f"significantly improve the model.\n"
            f"=> The independence assumption appears VALID for this dataset."
        )
    else:
        verdict = (
            f"p = {p_val:.4f} <= {alpha} — interactions are statistically significant.\n"
            f"=> The independence assumption may NOT hold. "
            f"Check the ANOVA table for which pairs drive the effect."
        )
    out_lines.append(verdict)

    # -----------------------------------------------------------------------
    # Largest interaction effects (by eta-squared)
    # -----------------------------------------------------------------------
    h("TOP INTERACTION TERMS BY EFFECT SIZE (eta²)")
    interaction_rows = anova_B.loc[anova_B.index.str.contains(":")]
    top = interaction_rows.sort_values("eta_sq", ascending=False)
    out_lines.append(top[["sum_sq", "F", "PR(>F)", "eta_sq"]].to_string(
        float_format=lambda x: f"{x:.4f}"
    ))

    out_lines.append("")
    return model_A, model_B


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test TAA parameter independence from full factorial scores."
    )
    parser.add_argument(
        "--scene", nargs="+", required=True,
        help="Scene name(s) matching folders in tests/"
    )
    parser.add_argument(
        "--pool", action="store_true",
        help="Pool all scenes into one combined analysis (more statistical power)"
    )
    args = parser.parse_args()

    dfs = {}
    for scene in args.scene:
        try:
            dfs[scene] = load_scene(scene)
            print(f"Loaded {len(dfs[scene])} combos for scene '{scene}'")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if not dfs:
        sys.exit("No data loaded. Exiting.")

    all_lines = []

    # Per-scene analysis
    for scene, df in dfs.items():
        out_lines = []
        analyze(df, label=scene, out_lines=out_lines)
        all_lines += out_lines

        report_path = os.path.join(TESTS_DIR, scene, "independence_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(out_lines))
        print(f"\nReport saved: {report_path}")

    # Pooled analysis
    if args.pool and len(dfs) > 1:
        pooled = pd.concat(list(dfs.values()), ignore_index=True)
        out_lines = []
        analyze(pooled, label="POOLED (" + ", ".join(dfs.keys()) + ")", out_lines=out_lines)
        all_lines += out_lines

        report_path = os.path.join(TESTS_DIR, "independence_report_pooled.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(out_lines))
        print(f"\nPooled report saved: {report_path}")

    # Print everything to console too
    print("\n" + "\n".join(all_lines))


if __name__ == "__main__":
    main()
