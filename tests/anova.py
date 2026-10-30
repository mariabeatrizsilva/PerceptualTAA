"""
anova.py
------------
Tests TAA parameter main effects and interactions on video quality scores.

Runs two models and compares them:
  Model A (main effects only):    score ~ aw + ns + fs + hp
  Model B (+ 2-way interactions): score ~ aw + ns + fs + hp
                                        + aw:ns + aw:fs + aw:hp
                                        + ns:fs + ns:hp + fs:hp

If Model B is not significantly better than Model A (F-test, p > 0.05),
the parameters can be treated as independent.

USAGE:
    python taa_anova.py --scene scene1
    python taa_anova.py --scene scene1 scene2

DEPENDENCIES:
    pip install pandas statsmodels
"""

import os
import argparse
import sys
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
except ImportError:
    sys.exit("Please install statsmodels:  pip install statsmodels")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05

INTERACTIONS = [
    "alpha_weight:num_samples",
    "alpha_weight:filter_size",
    "alpha_weight:hist_percent",
    "num_samples:filter_size",
    "num_samples:hist_percent",
    "filter_size:hist_percent",
]


# ---------------------------------------------------------------------------
# STEP 1: Load and clean data
# ---------------------------------------------------------------------------

def load_and_clean(scene_names):
    frames = []
    for scene in scene_names:
        path = os.path.join(TESTS_DIR, scene, "scores.csv")
        df = pd.read_csv(path)
        df["scene"] = scene
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    print("=" * 60)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 60)
    print(f"Total rows loaded:       {len(df)}")

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
# STEP 2: Fit Model A — main effects only
# ---------------------------------------------------------------------------

def run_model_a(df, n_scenes):
    main_effects = "alpha_weight + num_samples + filter_size + hist_percent"
    if n_scenes > 1:
        formula = f"cgvqm_score ~ C(scene) + {main_effects}"
    else:
        formula = f"cgvqm_score ~ {main_effects}"

    model = smf.ols(formula, data=df).fit()

    # Type II SS: correct for continuous predictors, no interactions
    anova_table = anova_lm(model, typ=2)
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    anova_table["partial_eta_sq"] = (
        anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_residual)
    )

    print("\n" + "=" * 60)
    print("STEP 2: MODEL A — MAIN EFFECTS ONLY (Type II SS)")
    print("=" * 60)
    print(f"Formula: {formula}\n")
    print(
        "How to read this table:\n"
        "  F        — larger = stronger effect relative to noise\n"
        "  p-value  — < 0.05 means statistically significant\n"
        "  partial  — proportion of score variance explained by this\n"
        "  eta²       parameter; 0.01=small, 0.06=medium, 0.14=large\n"
    )

    display = anova_table[["sum_sq", "df", "F", "PR(>F)", "partial_eta_sq"]].copy()
    display.columns = ["SS", "df", "F", "p-value", "partial_eta²"]
    print(display.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nModel A R²: {model.rsquared:.4f}")

    # Plain-language summary
    params = ["alpha_weight", "num_samples", "filter_size", "hist_percent"]
    print("\nMain effects summary:")
    for name in params:
        if name not in anova_table.index:
            continue
        p   = anova_table.loc[name, "PR(>F)"]
        eta = anova_table.loc[name, "partial_eta_sq"]
        sig = "SIGNIFICANT" if p < ALPHA else "not significant"
        print(f"  {name:<20}  p={p:.4f}  partial_eta²={eta:.4f}  [{sig}]")

    return model


# ---------------------------------------------------------------------------
# STEP 3: Fit Model B — main effects + 2-way interactions
# ---------------------------------------------------------------------------

def run_model_b(df, n_scenes):
    main_effects  = "alpha_weight + num_samples + filter_size + hist_percent"
    interactions  = " + ".join(INTERACTIONS)
    if n_scenes > 1:
        formula = f"cgvqm_score ~ C(scene) + {main_effects} + {interactions}"
    else:
        formula = f"cgvqm_score ~ {main_effects} + {interactions}"

    model = smf.ols(formula, data=df).fit()

    # Type III SS: correct when interactions are present
    anova_table = anova_lm(model, typ=3)
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    anova_table["partial_eta_sq"] = (
        anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_residual)
    )

    print("\n" + "=" * 60)
    print("STEP 3: MODEL B — MAIN EFFECTS + 2-WAY INTERACTIONS (Type III SS)")
    print("=" * 60)
    print(f"Formula: {formula}\n")

    display = anova_table[["sum_sq", "df", "F", "PR(>F)", "partial_eta_sq"]].copy()
    display.columns = ["SS", "df", "F", "p-value", "partial_eta²"]
    print(display.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nModel B R²: {model.rsquared:.4f}")

    # Plain-language summary of interaction terms only
    print("\nInteraction terms summary:")
    for term in INTERACTIONS:
        if term not in anova_table.index:
            continue
        p   = anova_table.loc[term, "PR(>F)"]
        eta = anova_table.loc[term, "partial_eta_sq"]
        sig = "SIGNIFICANT" if p < ALPHA else "not significant"
        print(f"  {term:<35}  p={p:.4f}  partial_eta²={eta:.4f}  [{sig}]")

    return model

def run_per_scene(df):
    print("\n" + "=" * 60)
    print("STEP 3b: MODEL B PER SCENE — INTERACTION COMPARISON")
    print("=" * 60)
    print(
        "\nChecking whether the alpha_weight:hist_percent interaction\n"
        "is consistent across scenes, or specific to one.\n"
    )

    for scene in df["scene"].unique():
        scene_df = df[df["scene"] == scene].copy()
        main_effects = "alpha_weight + num_samples + filter_size + hist_percent"
        interactions = " + ".join(INTERACTIONS)
        formula = f"cgvqm_score ~ {main_effects} + {interactions}"

        model = smf.ols(formula, data=scene_df).fit()
        anova_table = anova_lm(model, typ=3)
        ss_residual = anova_table.loc["Residual", "sum_sq"]
        anova_table["partial_eta_sq"] = (
            anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_residual)
        )

        print(f"Scene: {scene}")
        for term in INTERACTIONS:
            if term not in anova_table.index:
                continue
            p   = anova_table.loc[term, "PR(>F)"]
            eta = anova_table.loc[term, "partial_eta_sq"]
            sig = "SIGNIFICANT" if p < ALPHA else "not significant"
            print(f"  {term:<35}  p={p:.4f}  partial_eta²={eta:.4f}  [{sig}]")
        print()

# ---------------------------------------------------------------------------
# STEP 4: F-test — does adding interactions improve the model?
# ---------------------------------------------------------------------------

def run_ftest(model_a, model_b):
    print("\n" + "=" * 60)
    print("STEP 4: F-TEST — MODEL A vs MODEL B")
    print("=" * 60)
    print(
        "\nThis tests whether adding interaction terms significantly improves\n"
        "the model. If not (p > 0.05), parameters can be treated as independent.\n"
    )

    F_stat, p_val, df_diff = model_b.compare_f_test(model_a)
    delta_r2 = model_b.rsquared - model_a.rsquared

    print(f"F({int(abs(df_diff))}, {int(model_b.df_resid)}) = {F_stat:.4f}")
    print(f"p-value  = {p_val:.4f}")
    print(f"R² Model A (main effects):    {model_a.rsquared:.4f}")
    print(f"R² Model B (+ interactions):  {model_b.rsquared:.4f}")
    print(f"ΔR²:                          {delta_r2:.4f}")

    print("\n--- VERDICT ---")
    if p_val > ALPHA:
        print(
            f"p = {p_val:.4f} > {ALPHA}: adding interactions does NOT significantly\n"
            f"improve the model.\n"
            f"=> Parameters appear INDEPENDENT. The main effects model is sufficient."
        )
    else:
        print(
            f"p = {p_val:.4f} <= {ALPHA}: interactions ARE statistically significant.\n"
            f"=> Parameters are NOT fully independent. Check Step 3 to see\n"
            f"   which interaction terms are driving this."
        )


import matplotlib.pyplot as plt
import seaborn as sns

def print_publication_summary(df):
    """
    Generates and displays a clean, publication-ready summary table
    focusing on the critical alpha_weight:hist_percent interaction.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    
    rows = []
    main_effects = "alpha_weight + num_samples + filter_size + hist_percent"
    interactions = " + ".join(INTERACTIONS)
    
    # 1. Grab Global Model Data
    formula_global = f"cgvqm_score ~ C(scene) + {main_effects} + {interactions}"
    model_g = smf.ols(formula_global, data=df).fit()
    anova_g = anova_lm(model_g, typ=3)
    
    term = "alpha_weight:hist_percent"
    f_stat = anova_g.loc[term, "F"]
    p_val = anova_g.loc[term, "PR(>F)"]
    eta_g = anova_g.loc[term, "sum_sq"] / (anova_g.loc[term, "sum_sq"] + anova_g.loc["Residual", "sum_sq"])
    
    rows.append({
        "Analysis Scope": "Global Model (All Scenes)",
        "F-Statistic": f"{f_stat:.2f}",
        "p-value": "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}",
        "Partial η²": f"{eta_g:.4f}",
        "Interpretation": "Significant (Large Effect)"
    })
    
    # 2. Grab Per-Scene Data
    for scene in df["scene"].unique():
        scene_df = df[df["scene"] == scene]
        formula_scene = f"cgvqm_score ~ {main_effects} + {interactions}"
        model_s = smf.ols(formula_scene, data=scene_df).fit()
        anova_s = anova_lm(model_s, typ=3)
        
        f_stat_s = anova_s.loc[term, "F"]
        p_val_s = anova_s.loc[term, "PR(>F)"]
        eta_s = anova_s.loc[term, "sum_sq"] / (anova_s.loc[term, "sum_sq"] + anova_s.loc["Residual", "sum_sq"])
        
        rows.append({
            "Analysis Scope": f"Scene: {scene}",
            "F-Statistic": f"{f_stat_s:.2f}",
            "p-value": "< 0.0001" if p_val_s < 0.0001 else f"{p_val_s:.4f}",
            "Partial η²": f"{eta_s:.4f}",
            "Interpretation": "Significant (Massive)" if eta_s > 0.4 else "Significant (Medium)"
        })
        
    summary_df = pd.DataFrame(rows)
    
    print("\n" + "="*60)
    print("SIGNIFICANT VALUES TABLE")
    print("="*60)
    # If running in a Jupyter Notebook, use display(summary_df)
    # Otherwise, to_string(index=False) keeps it beautiful in the terminal
    print(summary_df.to_string(index=False))
    print("="*60)

    # Optional: If you want to drop the F-statistic completely for brevity:
    # summary_df_clean = summary_df.drop(columns=["F-Statistic"])

def print_terminal_tables(df):
    """
    Generates and displays two comprehensive summary tables in the terminal.
    Significant interactions (p < 0.05) are automatically bolded using ANSI codes.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    import pandas as pd

    main_effects = "alpha_weight + num_samples + filter_size + hist_percent"
    interactions = " + ".join(INTERACTIONS)
    
    # Terminal formatting constants
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    # ---------------------------------------------------------
    # TABLE 1: GLOBAL MODEL (ALL INTERACTIONS)
    # ---------------------------------------------------------
    formula_global = f"cgvqm_score ~ C(scene) + {main_effects} + {interactions}"
    model_g = smf.ols(formula_global, data=df).fit()
    anova_g = anova_lm(model_g, typ=3)
    
    print("\n" + "="*75)
    print("TABLE 1: GLOBAL MODEL - ALL 2-WAY INTERACTIONS")
    print("="*75)
    print(f"{'Interaction Term':<30} {'F-Stat':<10} {'p-value':<12} {'Partial η²':<12}")
    print("-"*75)
    
    for term in INTERACTIONS:
        f_stat = anova_g.loc[term, "F"]
        p_val = anova_g.loc[term, "PR(>F)"]
        eta = anova_g.loc[term, "sum_sq"] / (anova_g.loc[term, "sum_sq"] + anova_g.loc["Residual", "sum_sq"])
        
        p_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        row_str = f"{term:<30} {f_stat:<10.2f} {p_str:<12} {eta:<12.4f}"
        
        # Bold the entire row if it is significant
        if p_val < 0.05:
            print(f"{BOLD}{row_str}{RESET}")
        else:
            print(row_str)
            
    # ---------------------------------------------------------
    # TABLE 2: PER-SCENE COMPARISON
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print("TABLE 2: PER-SCENE INTERACTION COMPARISON")
    print("="*75)
    print(f"{'Interaction Term':<30} {'Oldmine p':<11} {'Oldmine η²':<12} {'Village p':<11} {'Village η²':<12}")
    print("-"*75)
    
    for term in INTERACTIONS:
        # Scene 1: oldmine
        df_om = df[df["scene"] == "oldmine"]
        m_om = smf.ols(f"cgvqm_score ~ {main_effects} + {interactions}", data=df_om).fit()
        a_om = anova_lm(m_om, typ=3)
        p_om = a_om.loc[term, "PR(>F)"]
        eta_om = a_om.loc[term, "sum_sq"] / (a_om.loc[term, "sum_sq"] + a_om.loc["Residual", "sum_sq"])
        
        # Scene 2: fantasticvillage
        df_fv = df[df["scene"] == "fantasticvillage"]
        m_fv = smf.ols(f"cgvqm_score ~ {main_effects} + {interactions}", data=df_fv).fit()
        a_fv = anova_lm(m_fv, typ=3)
        p_fv = a_fv.loc[term, "PR(>F)"]
        eta_fv = a_fv.loc[term, "sum_sq"] / (a_fv.loc[term, "sum_sq"] + a_fv.loc["Residual", "sum_sq"])
        
        p_om_str = "< 0.0001" if p_om < 0.0001 else f"{p_om:.4f}"
        p_fv_str = "< 0.0001" if p_fv < 0.0001 else f"{p_fv:.4f}"
        
        row_str = f"{term:<30} {p_om_str:<11} {eta_om:<12.4f} {p_fv_str:<11} {eta_fv:<12.4f}"
        
        # Bold if either scene shows a significant interaction
        if p_om < 0.05 or p_fv < 0.05:
            print(f"{BOLD}{row_str}{RESET}")
        else:
            print(row_str)
            
    print("="*75 + "\n")

def plot_interactions(df):
    # Set up a 1-row, 3-column plot to show overall + per scene patterns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # 1. Overall Interaction
    sns.pointplot(
        data=df, x="hist_percent", y="cgvqm_score", hue="alpha_weight",
        ax=axes[0], capsize=0.1, markers="o", linestyles="-"
    )
    axes[0].set_title("Overall Interaction")
    axes[0].set_ylabel("CGVQM Score")
    
    # 2. Oldmine Scene
    sns.pointplot(
        data=df[df["scene"] == "oldmine"], x="hist_percent", y="cgvqm_score", hue="alpha_weight",
        ax=axes[1], capsize=0.1, markers="s", linestyles="--"
    )
    axes[1].set_title("Scene: Oldmine (eta² = 0.08)")
    axes[1].set_ylabel("") # Shared y-axis
    
    # 3. Fantastic Village Scene
    sns.pointplot(
        data=df[df["scene"] == "fantasticvillage"], x="hist_percent", y="cgvqm_score", hue="alpha_weight",
        ax=axes[2], capsize=0.1, markers="^", linestyles="-."
    )
    axes[2].set_title("Scene: Fantastic Village (eta² = 0.43)")
    axes[2].set_ylabel("") # Shared y-axis
    
    plt.tight_layout()
    plt.savefig("interaction_plots.png", dpi=300)
    plt.show()



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TAA parameter ANOVA: main effects + independence test."
    )
    parser.add_argument(
        "--scene", nargs="+", required=True,
        help="Scene name(s) matching folders in tests/"
    )
    # add this to the argparse block
    parser.add_argument(
        "--separate", action="store_true",
        help="Also run Model B per scene and compare interaction patterns."
    )

    args = parser.parse_args()

    df       = load_and_clean(args.scene)
    n_scenes = df["scene"].nunique()
    model_a  = run_model_a(df, n_scenes)
    model_b  = run_model_b(df, n_scenes)
    if args.separate and n_scenes > 1:
        run_per_scene(df)

    run_ftest(model_a, model_b)
    plot_interactions(df)
    print_publication_summary(df)
    print_terminal_tables(df)

if __name__ == "__main__":
    main()