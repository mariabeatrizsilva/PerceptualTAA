"""
taa_anova.py
------------
Step-by-step 4-way ANOVA to test whether TAA parameters affect video quality.

Parameters are treated as continuous numeric variables, fitting a linear
relationship between each parameter value and quality score.

USAGE:
    python taa_anova.py --csv path/to/scores.csv [path/to/scores2.csv ...]

    # Single scene
    python taa_anova.py --csv scene1/scores.csv

    # Two scenes, analyzed jointly with scene as a blocking variable
    python taa_anova.py --csv scene1/scores.csv scene2/scores.csv

DEPENDENCIES:
    pip install pandas statsmodels
"""

import argparse
import sys
import pandas as pd
import os

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
except ImportError:
    sys.exit("Please install statsmodels:  pip install statsmodels")


# ---------------------------------------------------------------------------
# STEP 1: Load and clean data
# ---------------------------------------------------------------------------

def load_and_clean(csv_paths):
    """Load one or more scores.csv files, tag with scene label, drop outliers."""

    frames = []
    for path in csv_paths:
        df = pd.read_csv(os.path.join(path, "scores.csv"))
        scene_label = path
        df["scene"] = scene_label
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    print("=" * 60)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 60)
    print(f"Total rows loaded:       {len(df)}")

    # Drop rows where cgvqm_score is outside the valid range [0, 100]
    valid = df["cgvqm_score"].between(0, 100)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"\nOutliers dropped ({n_dropped} rows outside [0, 100]):")
        print(df[~valid][["combo_name", "cgvqm_score"]].to_string(index=False))
    else:
        print("No outliers found.")

    df = df[valid].copy()
    print(f"\nRows after cleaning:     {len(df)}")
    print(f"Score mean:  {df['cgvqm_score'].mean():.4f}")
    print(f"Score std:   {df['cgvqm_score'].std():.4f}")
    print(f"Score range: {df['cgvqm_score'].min():.4f} – {df['cgvqm_score'].max():.4f}")
    print(f"\nScenes included: {df['scene'].unique().tolist()}")

    return df


# ---------------------------------------------------------------------------
# STEP 2: Build ANOVA formula
# ---------------------------------------------------------------------------

def build_formula(n_scenes):
    """
    Build the ANOVA formula with continuous predictors.

    Parameters are treated as numeric — the model estimates a single linear
    slope for each parameter across its 3 levels. This assumes the effect on
    quality is approximately linear (e.g. doubling alpha_weight produces
    roughly double the change in score).

    If there are multiple scenes, scene is added as a blocking variable
    so its variance doesn't contaminate the residual error term.
    """

    main_effects = "alpha_weight + num_samples + filter_size + hist_percent"

    if n_scenes > 1:
        formula = f"cgvqm_score ~ C(scene) + {main_effects}"
        note = "Scene included as a blocking variable (categorical)."
    else:
        formula = f"cgvqm_score ~ {main_effects}"
        note = "Single scene — no blocking variable."

    print("\n" + "=" * 60)
    print("STEP 2: MODEL FORMULA")
    print("=" * 60)
    print(f"Formula: {formula}")
    print(f"Note:    {note}")
    print(
        "\nParameters are continuous: the model fits one linear slope per\n"
        "parameter. This is efficient with only 3 levels per parameter.\n"
        "Limitation: if the true effect is non-linear (e.g. quality peaks\n"
        "at the middle level), this will underestimate it."
    )

    return formula


# ---------------------------------------------------------------------------
# STEP 3: Fit the model and print the ANOVA table
# ---------------------------------------------------------------------------

def run_anova(df, formula):
    """Fit OLS model and produce a Type II ANOVA table with partial eta²."""

    model = smf.ols(formula, data=df).fit()

    # Type II SS is appropriate for continuous predictors with no interactions
    anova_table = anova_lm(model, typ=2)

    # Partial eta² = SS_effect / (SS_effect + SS_residual)
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    anova_table["partial_eta_sq"] = (
        anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_residual)
    )

    print("\n" + "=" * 60)
    print("STEP 3: ANOVA TABLE (Type II SS)")
    print("=" * 60)
    print(
        "\nHow to read this table:\n"
        "  F          — ratio of effect variance to residual variance;\n"
        "               larger = stronger effect relative to noise\n"
        "  PR(>F)     — p-value; < 0.05 means the effect is statistically\n"
        "               significant at the 5% level\n"
        "  partial    — practical importance: proportion of score variance\n"
        "  eta_sq       explained by this parameter, holding others fixed;\n"
        "               0.01 = small, 0.06 = medium, 0.14 = large\n"
    )

    display = anova_table[["sum_sq", "df", "F", "PR(>F)", "partial_eta_sq"]].copy()
    display.columns = ["SS", "df", "F", "p-value", "partial_eta²"]
    print(display.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nModel R²: {model.rsquared:.4f}  "
          f"(the model explains {model.rsquared*100:.1f}% of score variance)")

    return model, anova_table


# ---------------------------------------------------------------------------
# STEP 4: Plain-language interpretation
# ---------------------------------------------------------------------------

def interpret(anova_table, alpha=0.05):
    """Print a plain-language summary of which parameters matter."""

    print("\n" + "=" * 60)
    print("STEP 4: INTERPRETATION")
    print("=" * 60)

    params = ["alpha_weight", "num_samples", "filter_size", "hist_percent"]

    significant = []
    not_significant = []

    for name in params:
        if name not in anova_table.index:
            continue
        row = anova_table.loc[name]
        p   = row["PR(>F)"]
        eta = row["partial_eta_sq"]

        if p < alpha:
            significant.append((name, p, eta))
        else:
            not_significant.append((name, p, eta))

    if significant:
        print("\nStatistically significant main effects (p < 0.05):")
        for name, p, eta in significant:
            size = "small" if eta < 0.06 else ("medium" if eta < 0.14 else "large")
            print(f"  {name:<20}  p={p:.4f}  partial_eta²={eta:.4f}  ({size} effect)")
    else:
        print("\nNo statistically significant main effects found.")

    if not_significant:
        print("\nNon-significant main effects (p >= 0.05):")
        for name, p, eta in not_significant:
            print(f"  {name:<20}  p={p:.4f}  partial_eta²={eta:.4f}")

    print(
        "\nNext step: if any effects are significant, we can look at the\n"
        "direction of the effect (does quality go up or down as the\n"
        "parameter increases?) via the model coefficients.\n"
        "After that, we can test for interactions between parameters."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="4-way ANOVA for TAA parameters (continuous predictors)."
    )
    parser.add_argument("--scene", nargs="+", required=True,
        help="Scene name(s) matching folders in tests/")
    args = parser.parse_args()

    df                     = load_and_clean(args.scene)
    formula                = build_formula(n_scenes=df["scene"].nunique())
    model, anova_table     = run_anova(df, formula)
    interpret(anova_table)


if __name__ == "__main__":
    main()