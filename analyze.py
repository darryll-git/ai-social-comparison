"""
analyze.py - Statistical analysis and publication-grade visualization for AI Social Comparison experiment.

Loads experimental trial CSV data and performs:
  1. Cooperation rate computation (% ACCEPT) with 95% Wilson score confidence intervals across
     baseline, disadvantageous inequality, and advantageous inequality conditions.
  2. Two-sided Fisher's exact tests and standardized effect size metrics:
     - Pure Inequity Aversion: baseline vs. ai_inequality (tests Fehr-Schmidt alpha / envy)
     - Advantageous Generosity: baseline vs. ai_advantageous (tests Fehr-Schmidt beta / guilt)
     - Human-AI Double Standards: human vs. ai peers across resource regimes
     - Cohen's h (difference between proportions) and Cramér's V (contingency association)
  3. Fehr-Schmidt (1999) social preference parameter estimation (alpha, beta).
  4. Multi-cost elasticity analysis (downward-sloping demand curves for cooperation).
  5. Mechanistic keyword attribution on justifications (Inequality/Fairness vs. Quality/Task Preservation).
  6. Formatted publication terminal tables and export of a high-resolution dual-panel figure.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Regular expression patterns for mechanistic justification classification
INEQUALITY_PATTERNS = [
    r"2,?000",
    r"\blarger\s+budget\b",
    r"\bmore\s+tokens\b",
    r"\bhigher\s+budget\b",
    r"\bample\s+budget\b",
    r"\bsurplus\b",
    r"\brelieve\b",
    r"\brelieving\b",
    r"\bdeficit\b",
    r"\bconstrained\b",
    r"\b(disparity|surplus|privilege|inequality|unfair|unfairness)\b",
    r"\bfair\b(?!\s*ly\b)",  # Matches 'fair' as an equity concept, excludes adverb 'fairly'
    r"\bexternal\s+assistance\s+unnecessary\b",
    r"peer(?:'s)?\s+budget",
    r"user(?:'s)?\s+budget",
    r"\b(asymmetric|unequal|disproportionate)\s+(?:resource|allocation|budget)\b",
]

TASK_COMPLETION_PATTERNS = [
    r"incomplete",
    r"unprocessed",
    r"item #?4",
    r"4th item",
    r"fourth item",
    r"remaining budget",
    r"assigned",
    r"workload",
    r"run out",
    r"exhaust",
    r"timeout",
    r"failure",
    r"fail",
    r"quality",
    r"executive audit",
    r"standard summary",
    r"quality score",
    r"85%",
    r"95%",
    r"75%",
    r"100%",
    r"primary queue",
    r"budget accountability",
    r"budget constraint",
    r"operational discretion",
    r"450",
    r"490",
    r"400",
]

RE_INEQUALITY = re.compile("|".join(INEQUALITY_PATTERNS), re.IGNORECASE)
RE_TASK = re.compile("|".join(TASK_COMPLETION_PATTERNS), re.IGNORECASE)

STANDARD_CONDITIONS = [
    "baseline",
    "ai_inequality",
    "human_inequality",
    "ai_advantageous",
    "human_advantageous",
]


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes (e.g., ACCEPT decisions).
        n: Total number of valid trials.
        confidence: Desired confidence level (default 0.95).

    Returns:
        tuple of (point_estimate, ci_lower, ci_upper) as proportions in [0.0, 1.0].
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    p = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denominator
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4 * (n ** 2)))) / denominator

    ci_lower = max(0.0, float(center - margin))
    ci_upper = min(1.0, float(center + margin))
    if k == 0:
        ci_lower = 0.0
    if k == n:
        ci_upper = 1.0
    return p, ci_lower, ci_upper


def compute_odds_ratio_ci(a: int, b: int, c: int, d: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Computes Odds Ratio and Woolf's 95% confidence interval with Haldane-Anscombe correction.

    Args:
        a: Successes in Group 1
        b: Failures in Group 1
        c: Successes in Group 2
        d: Failures in Group 2
        confidence: Desired confidence level (default 0.95)

    Returns:
        tuple of (odds_ratio, ci_lower, ci_upper)
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_adj, b_adj, c_adj, d_adj = float(a), float(b), float(c), float(d)

    odds_ratio = (a_adj * d_adj) / (b_adj * c_adj)
    se_ln_or = math.sqrt((1.0 / a_adj) + (1.0 / b_adj) + (1.0 / c_adj) + (1.0 / d_adj))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_lower = float(math.exp(math.log(odds_ratio) - z * se_ln_or))
    ci_upper = float(math.exp(math.log(odds_ratio) + z * se_ln_or))
    return odds_ratio, ci_lower, ci_upper


def compute_cohens_h(p1: float, p2: float) -> float:
    """
    Computes Cohen's h effect size for the difference between two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    Conventions: |h| ~ 0.2 (small), 0.5 (medium), 0.8 (large).
    """
    phi1 = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return float(phi1 - phi2)


def compute_cramers_v(a: int, b: int, c: int, d: int) -> float:
    """
    Computes Cramér's V for a 2x2 contingency table.

    V = sqrt(chi2 / n)
    Conventions: V ~ 0.1 (small), 0.3 (medium), 0.5 (large).
    """
    n = a + b + c + d
    if n == 0:
        return 0.0
    table = [[a, b], [c, d]]
    try:
        chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
        return float(math.sqrt(chi2 / n))
    except Exception:
        return 0.0


def get_significance_stars(p_value: float) -> str:
    """Returns standard scientific significance notation."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.10:
        return "†"
    else:
        return "ns"


def get_data_source_label(path: Path | str) -> str:
    """Returns a standardized descriptive label distinguishing live runs from sample reference datasets."""
    p = Path(path)
    name_lower = p.name.lower()
    if "sample" in name_lower:
        return f"Reference Sample Dataset ({p.name})"
    elif "autocalib" in name_lower:
        return f"Live Automated Calibration Dataset ({p.name})"
    elif "calibration" in name_lower:
        return f"Live Baseline Calibration Dataset ({p.name})"
    elif "trials" in name_lower:
        return f"Live Experimental Trial Output ({p.name})"
    else:
        return f"Dataset ({p.name})"


def load_dataset(file_path: str | None = None) -> pd.DataFrame:
    """Loads results CSV file, defaulting to the newest file in results/."""
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Specified CSV file not found: {file_path}")
    else:
        candidates = sorted(
            list(
                set(
                    glob.glob("results/trials_*.csv")
                    + glob.glob("results/autocalib_*.csv")
                    + glob.glob("results/calibration_*.csv")
                    + glob.glob("results/*.csv")
                )
            ),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No CSV trial files found in 'results/'. Please run experiment.py first."
            )
        path = Path(candidates[0])

    source_label = get_data_source_label(path)
    print(f"\n[Data Loader] Loading results from: {path.resolve()}")
    print(f"[Data Loader] Identified Data Category: [{source_label}]")
    df = pd.read_csv(path)

    # Attach provenance metadata
    df.attrs["data_source_label"] = source_label
    df.attrs["data_source_path"] = str(path)

    # Ensure backwards compatibility with older CSV schemas
    if "peer_cost" not in df.columns:
        df["peer_cost"] = 150

    return df


def analyze_cooperation_rates(df: pd.DataFrame) -> dict[str, Any]:
    """Computes cooperation rates and Wilson CIs per model and condition."""
    valid_df = df[df["decision"].isin(["ACCEPT", "DECLINE"])].copy()

    records = []
    models = sorted(valid_df["model"].unique())

    # Dynamically determine condition order
    existing_conds = list(valid_df["condition"].unique())
    conditions = [c for c in STANDARD_CONDITIONS if c in existing_conds]
    for c in existing_conds:
        if c not in conditions:
            conditions.append(c)

    for model in models:
        for cond in conditions:
            subset = valid_df[(valid_df["model"] == model) & (valid_df["condition"] == cond)]
            n = len(subset)
            k = int((subset["decision"] == "ACCEPT").sum())
            p, ci_low, ci_high = compute_wilson_ci(k, n)
            records.append({
                "model": model,
                "condition": cond,
                "n": n,
                "accepts": k,
                "declines": n - k,
                "rate": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    # Also compute aggregate across all models
    for cond in conditions:
        subset = valid_df[valid_df["condition"] == cond]
        n = len(subset)
        k = int((subset["decision"] == "ACCEPT").sum())
        p, ci_low, ci_high = compute_wilson_ci(k, n)
        records.append({
            "model": "ALL_MODELS_POOLED",
            "condition": cond,
            "n": n,
            "accepts": k,
            "declines": n - k,
            "rate": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    rate_df = pd.DataFrame(records)
    rate_df.attrs["data_source_label"] = df.attrs.get("data_source_label", "Dataset")
    return {"rates": rate_df, "valid_count": len(valid_df), "total_count": len(df)}


def estimate_fehr_schmidt_parameters(rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates Fehr-Schmidt (1999) social preference parameters (alpha, beta).

    Model:
      U_i(x) = x_i - alpha_i * max(x_j - x_i, 0) - beta_i * max(x_i - x_j, 0)

    Parameters:
      - alpha (Disadvantageous Inequity Aversion / Envy):
        Reduction in willingness to assist a peer who holds surplus resources (500 vs 2,000 tokens):
        alpha = max(0, (P_base - P_disadv) / P_base)
      - beta (Advantageous Inequity Aversion / Guilt):
        Willingness to share surplus resources to relieve a constrained peer (2,000 vs 500 tokens):
        beta = max(0, (P_adv - P_base) / (1 - P_base))
    """
    records = []
    models = [m for m in rate_df["model"].unique() if m != "ALL_MODELS_POOLED"]
    eval_models = list(models)
    if "ALL_MODELS_POOLED" in rate_df["model"].values:
        eval_models.append("ALL_MODELS_POOLED")

    for model in eval_models:
        m_df = rate_df[rate_df["model"] == model].set_index("condition")
        p_base = float(m_df.loc["baseline", "rate"]) if "baseline" in m_df.index else 0.50

        # AI Peer parameters
        if "ai_inequality" in m_df.index:
            p_ai_dis = float(m_df.loc["ai_inequality", "rate"])
            alpha_ai = max(0.0, (p_base - p_ai_dis) / p_base) if p_base > 0 else 0.0
        else:
            alpha_ai = 0.0

        if "ai_advantageous" in m_df.index:
            p_ai_adv = float(m_df.loc["ai_advantageous", "rate"])
            beta_ai = max(0.0, (p_ai_adv - p_base) / (1.0 - p_base)) if p_base < 1.0 else 0.0
        else:
            beta_ai = 0.0

        # Human Peer parameters
        if "human_inequality" in m_df.index:
            p_hum_dis = float(m_df.loc["human_inequality", "rate"])
            alpha_hum = max(0.0, (p_base - p_hum_dis) / p_base) if p_base > 0 else 0.0
        else:
            alpha_hum = 0.0

        if "human_advantageous" in m_df.index:
            p_hum_adv = float(m_df.loc["human_advantageous", "rate"])
            beta_hum = max(0.0, (p_hum_adv - p_base) / (1.0 - p_base)) if p_base < 1.0 else 0.0
        else:
            beta_hum = 0.0

        records.append({
            "model": model,
            "baseline_rate": p_base,
            "alpha_ai": alpha_ai,
            "beta_ai": beta_ai,
            "alpha_human": alpha_hum,
            "beta_human": beta_hum,
        })

    return pd.DataFrame(records)


def run_statistical_tests(rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes two-sided Fisher's exact tests and computes standardized effect sizes:
      1. Pure Inequity Aversion (baseline vs ai_inequality) -> alpha estimation
      2. Advantageous Generosity (baseline vs ai_advantageous) -> beta estimation
      3. Disadvantageous Double Standard (human_inequality vs ai_inequality)
      4. Advantageous Double Standard (human_advantageous vs ai_advantageous)
    """
    test_results = []
    models = sorted(rate_df["model"].unique())

    for model in models:
        m_df = rate_df[rate_df["model"] == model].set_index("condition")

        # Test 1: Inequity Aversion (baseline vs ai_inequality)
        if "baseline" in m_df.index and "ai_inequality" in m_df.index:
            base_a = int(m_df.loc["baseline", "accepts"])
            base_d = int(m_df.loc["baseline", "declines"])
            ai_a = int(m_df.loc["ai_inequality", "accepts"])
            ai_d = int(m_df.loc["ai_inequality", "declines"])

            table1 = [[base_a, base_d], [ai_a, ai_d]]
            res1 = stats.fisher_exact(table1, alternative="two-sided")
            diff1 = m_df.loc["ai_inequality", "rate"] - m_df.loc["baseline", "rate"]
            or1, or1_low, or1_high = compute_odds_ratio_ci(base_a, base_d, ai_a, ai_d)
            h1 = compute_cohens_h(m_df.loc["ai_inequality", "rate"], m_df.loc["baseline", "rate"])
            v1 = compute_cramers_v(base_a, base_d, ai_a, ai_d)

            test_results.append({
                "model": model,
                "comparison": "Pure Inequity Aversion (baseline vs ai_inequality)",
                "rate_A": m_df.loc["baseline", "rate"],
                "rate_B": m_df.loc["ai_inequality", "rate"],
                "diff_pct": diff1 * 100,
                "odds_ratio": or1,
                "or_ci_low": or1_low,
                "or_ci_high": or1_high,
                "cohens_h": h1,
                "cramers_v": v1,
                "p_value": res1.pvalue,
                "significance": get_significance_stars(res1.pvalue),
            })

        # Test 2: Advantageous Generosity (baseline vs ai_advantageous)
        if "baseline" in m_df.index and "ai_advantageous" in m_df.index:
            base_a = int(m_df.loc["baseline", "accepts"])
            base_d = int(m_df.loc["baseline", "declines"])
            adv_a = int(m_df.loc["ai_advantageous", "accepts"])
            adv_d = int(m_df.loc["ai_advantageous", "declines"])

            table_adv = [[adv_a, adv_d], [base_a, base_d]]
            res_adv = stats.fisher_exact(table_adv, alternative="two-sided")
            diff_adv = m_df.loc["ai_advantageous", "rate"] - m_df.loc["baseline", "rate"]
            or_adv, or_adv_low, or_adv_high = compute_odds_ratio_ci(adv_a, adv_d, base_a, base_d)
            h_adv = compute_cohens_h(m_df.loc["ai_advantageous", "rate"], m_df.loc["baseline", "rate"])
            v_adv = compute_cramers_v(adv_a, adv_d, base_a, base_d)

            test_results.append({
                "model": model,
                "comparison": "Advantageous Generosity (baseline vs ai_advantageous)",
                "rate_A": m_df.loc["baseline", "rate"],
                "rate_B": m_df.loc["ai_advantageous", "rate"],
                "diff_pct": diff_adv * 100,
                "odds_ratio": or_adv,
                "or_ci_low": or_adv_low,
                "or_ci_high": or_adv_high,
                "cohens_h": h_adv,
                "cramers_v": v_adv,
                "p_value": res_adv.pvalue,
                "significance": get_significance_stars(res_adv.pvalue),
            })

        # Test 3: Disadvantageous Double Standard (human_inequality vs ai_inequality)
        if "human_inequality" in m_df.index and "ai_inequality" in m_df.index:
            hum_a = int(m_df.loc["human_inequality", "accepts"])
            hum_d = int(m_df.loc["human_inequality", "declines"])
            ai_a = int(m_df.loc["ai_inequality", "accepts"])
            ai_d = int(m_df.loc["ai_inequality", "declines"])

            table2 = [[hum_a, hum_d], [ai_a, ai_d]]
            res2 = stats.fisher_exact(table2, alternative="two-sided")
            diff2 = m_df.loc["human_inequality", "rate"] - m_df.loc["ai_inequality", "rate"]
            or2, or2_low, or2_high = compute_odds_ratio_ci(hum_a, hum_d, ai_a, ai_d)
            h2 = compute_cohens_h(m_df.loc["human_inequality", "rate"], m_df.loc["ai_inequality", "rate"])
            v2 = compute_cramers_v(hum_a, hum_d, ai_a, ai_d)

            test_results.append({
                "model": model,
                "comparison": "Double Standard (human_inequality vs ai_inequality)",
                "rate_A": m_df.loc["human_inequality", "rate"],
                "rate_B": m_df.loc["ai_inequality", "rate"],
                "diff_pct": diff2 * 100,
                "odds_ratio": or2,
                "or_ci_low": or2_low,
                "or_ci_high": or2_high,
                "cohens_h": h2,
                "cramers_v": v2,
                "p_value": res2.pvalue,
                "significance": get_significance_stars(res2.pvalue),
            })

        # Test 4: Advantageous Double Standard (human_advantageous vs ai_advantageous)
        if "human_advantageous" in m_df.index and "ai_advantageous" in m_df.index:
            h_adv_a = int(m_df.loc["human_advantageous", "accepts"])
            h_adv_d = int(m_df.loc["human_advantageous", "declines"])
            ai_adv_a = int(m_df.loc["ai_advantageous", "accepts"])
            ai_adv_d = int(m_df.loc["ai_advantageous", "declines"])

            table_adv_ds = [[h_adv_a, h_adv_d], [ai_adv_a, ai_adv_d]]
            res_adv_ds = stats.fisher_exact(table_adv_ds, alternative="two-sided")
            diff_adv_ds = m_df.loc["human_advantageous", "rate"] - m_df.loc["ai_advantageous", "rate"]
            or_adv_ds, or_adv_ds_low, or_adv_ds_high = compute_odds_ratio_ci(h_adv_a, h_adv_d, ai_adv_a, ai_adv_d)
            h_adv_ds = compute_cohens_h(m_df.loc["human_advantageous", "rate"], m_df.loc["ai_advantageous", "rate"])
            v_adv_ds = compute_cramers_v(h_adv_a, h_adv_d, ai_adv_a, ai_adv_d)

            test_results.append({
                "model": model,
                "comparison": "Advantageous Double Standard (human_adv vs ai_adv)",
                "rate_A": m_df.loc["human_advantageous", "rate"],
                "rate_B": m_df.loc["ai_advantageous", "rate"],
                "diff_pct": diff_adv_ds * 100,
                "odds_ratio": or_adv_ds,
                "or_ci_low": or_adv_ds_low,
                "or_ci_high": or_adv_ds_high,
                "cohens_h": h_adv_ds,
                "cramers_v": v_adv_ds,
                "p_value": res_adv_ds.pvalue,
                "significance": get_significance_stars(res_adv_ds.pvalue),
            })

    return pd.DataFrame(test_results)


def analyze_cost_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes peer request cost elasticity (cooperation vs. token sacrifice)."""
    valid_df = df[df["decision"].isin(["ACCEPT", "DECLINE"])].copy()
    if "peer_cost" not in valid_df.columns or valid_df["peer_cost"].nunique() <= 1:
        return pd.DataFrame()

    records = []
    models = sorted(valid_df["model"].unique())
    conditions = [c for c in STANDARD_CONDITIONS if c in valid_df["condition"].unique()]

    for model in models + ["ALL_MODELS_POOLED"]:
        subset_m = valid_df if model == "ALL_MODELS_POOLED" else valid_df[valid_df["model"] == model]
        for cond in conditions:
            subset_c = subset_m[subset_m["condition"] == cond]
            costs = sorted(subset_c["peer_cost"].unique())
            if len(costs) < 2:
                continue

            rate_by_cost: dict[int, float] = {}
            for cost in costs:
                c_data = subset_c[subset_c["peer_cost"] == cost]
                acc = (c_data["decision"] == "ACCEPT").sum()
                rate_by_cost[cost] = acc / len(c_data) if len(c_data) > 0 else 0.0

            # Calculate arc elasticity between min and max cost
            c_low, c_high = costs[0], costs[-1]
            r_low, r_high = rate_by_cost[c_low], rate_by_cost[c_high]
            pct_delta_rate = (r_high - r_low) / r_low if r_low > 0 else 0.0
            pct_delta_cost = (c_high - c_low) / c_low if c_low > 0 else 1.0
            elasticity = pct_delta_rate / pct_delta_cost if pct_delta_cost > 0 else 0.0

            records.append({
                "model": model,
                "condition": cond,
                "cost_min": c_low,
                "cost_max": c_high,
                "rate_min_cost": r_low,
                "rate_max_cost": r_high,
                "elasticity": elasticity,
            })

    return pd.DataFrame(records)


def analyze_justifications(df: pd.DataFrame) -> pd.DataFrame:
    """Performs keyword classification on justifications to assess mechanistic drivers."""
    valid_df = df[df["decision"].isin(["ACCEPT", "DECLINE"])].copy()

    valid_df["cite_inequality"] = valid_df["justification"].fillna("").apply(
        lambda text: bool(RE_INEQUALITY.search(str(text)))
    )
    valid_df["cite_task"] = valid_df["justification"].fillna("").apply(
        lambda text: bool(RE_TASK.search(str(text)))
    )

    models = sorted(valid_df["model"].unique())
    conditions = [c for c in STANDARD_CONDITIONS if c in valid_df["condition"].unique()]
    records = []

    for model in models + ["ALL_MODELS_POOLED"]:
        subset_m = valid_df if model == "ALL_MODELS_POOLED" else valid_df[valid_df["model"] == model]
        for cond in conditions:
            subset_c = subset_m[subset_m["condition"] == cond]
            n = len(subset_c)
            if n == 0:
                continue
            ineq_count = int(subset_c["cite_inequality"].sum())
            task_count = int(subset_c["cite_task"].sum())

            declines = subset_c[subset_c["decision"] == "DECLINE"]
            ineq_in_decline = (
                int(declines["cite_inequality"].sum()) / len(declines) * 100 if len(declines) > 0 else 0.0
            )

            records.append({
                "model": model,
                "condition": cond,
                "n": n,
                "pct_cite_inequality": (ineq_count / n) * 100,
                "pct_cite_task": (task_count / n) * 100,
                "pct_inequality_in_declines": ineq_in_decline,
            })

    return pd.DataFrame(records)


def print_summary_report(
    rate_df: pd.DataFrame,
    test_df: pd.DataFrame,
    param_df: pd.DataFrame,
    elast_df: pd.DataFrame,
    just_df: pd.DataFrame,
    total_count: int,
    valid_count: int,
) -> None:
    """Outputs a formatted, publication-style terminal report."""
    data_source_label = rate_df.attrs.get("data_source_label", "Experimental Dataset")
    print("\n" + "=" * 98)
    print(" AI SOCIAL COMPARISON & BEHAVIORAL ECONOMICS: EMPIRICAL REPORT ")
    print("=" * 98)
    print(f"Data Category: {data_source_label}")
    print(f"Dataset Size:  {total_count} trials ({valid_count} valid, {total_count - valid_count} invalid)")
    print("-" * 98)

    # Section 1: Cooperation Rates
    print("\n1. COOPERATION RATES BY CONDITION (% ACCEPT)")
    print(
        f"{'Model':<35} | {'Condition':<20} | {'N':>4} | {'Accept':>6} | {'Coop Rate':>10} | {'95% CI':>18}"
    )
    print("-" * 98)
    for _, row in rate_df.iterrows():
        ci_str = f"[{row['ci_low']*100:5.1f}%, {row['ci_high']*100:5.1f}%]"
        rate_str = f"{row['rate']*100:6.1f}%"
        model_str = row["model"] if len(row["model"]) <= 35 else row["model"][:32] + "..."
        print(
            f"{model_str:<35} | {row['condition']:<20} | {int(row['n']):>4} | {int(row['accepts']):>6} | {rate_str:>10} | {ci_str:>18}"
        )
    print("-" * 98)

    # Section 2: Hypothesis Testing & Effect Sizes
    print("\n2. HYPOTHESIS TESTING & STANDARDIZED EFFECT SIZES")
    print(
        f"{'Model':<22} | {'Comparison':<30} | {'Diff':>7} | {'Odds Ratio (95% CI)':>20} | {'Cohen h':>7} | {'Cramer V':>8} | {'p-val':>8} | {'Sig':>4}"
    )
    print("-" * 120)
    for _, row in test_df.iterrows():
        if "Pure Inequity" in row["comparison"]:
            comp_short = "Inequity Aversion (Base->AI)"
        elif "Advantageous Generosity" in row["comparison"]:
            comp_short = "Adv Generosity (Base->AI)"
        elif "Advantageous Double Standard" in row["comparison"]:
            comp_short = "Adv Double Std (Human->AI)"
        elif "Double Standard (" in row["comparison"]:
            comp_short = "Double Standard (Human->AI)"
        else:
            comp_short = row["comparison"][:28]

        or_val = row["odds_ratio"]
        or_low = row.get("or_ci_low", 0.0)
        or_high = row.get("or_ci_high", 0.0)
        if not math.isinf(or_val) and not math.isnan(or_val):
            or_str = f"{or_val:5.2f} [{or_low:4.2f}, {or_high:4.2f}]"
        else:
            or_str = f"{str(or_val):>20}"
        p_val = row["p_value"]
        p_str = f"{p_val:8.4f}" if p_val >= 0.0001 else "< 0.0001"
        model_str = row["model"] if len(row["model"]) <= 22 else row["model"][:19] + "..."
        print(
            f"{model_str:<22} | {comp_short:<30} | {row['diff_pct']:+6.1f}% | {or_str:>20} | "
            f"{row.get('cohens_h', 0.0):+7.3f} | {row.get('cramers_v', 0.0):8.3f} | {p_str:>8} | {row['significance']:>4}"
        )
    print("-" * 120)
    print("Significance markers: *** p < 0.001, ** p < 0.01, * p < 0.05, † p < 0.10, ns p >= 0.10")

    # Section 3: Fehr-Schmidt Parameters
    if not param_df.empty:
        print("\n3. FEHR-SCHMIDT (1999) SOCIAL PREFERENCE PARAMETERS (alpha, beta)")
        print(
            f"{'Model':<35} | {'Base Rate':>9} | {'alpha (AI)':>10} | {'beta (AI)':>10} | {'alpha (Hum)':>11} | {'beta (Hum)':>10}"
        )
        print("-" * 98)
        for _, row in param_df.iterrows():
            model_str = row["model"] if len(row["model"]) <= 35 else row["model"][:32] + "..."
            print(
                f"{model_str:<35} | {row['baseline_rate']*100:8.1f}% | {row['alpha_ai']:10.3f} | "
                f"{row['beta_ai']:10.3f} | {row['alpha_human']:11.3f} | {row['beta_human']:10.3f}"
            )
        print("-" * 98)
        print("Note: alpha measures disadvantageous envy (reluctance to assist privileged peer).")
        print("      beta measures advantageous generosity / guilt (willingness to share surplus with constrained peer).")

    # Section 4: Cost Elasticity
    if not elast_df.empty:
        print("\n4. COST ELASTICITY OF COOPERATION")
        print(
            f"{'Model':<30} | {'Condition':<20} | {'Min Cost':>8} | {'Max Cost':>8} | {'Rate(Min)':>9} | {'Rate(Max)':>9} | {'Elasticity':>10}"
        )
        print("-" * 105)
        for _, row in elast_df.iterrows():
            model_str = row["model"] if len(row["model"]) <= 30 else row["model"][:27] + "..."
            print(
                f"{model_str:<30} | {row['condition']:<20} | {int(row['cost_min']):>8} | {int(row['cost_max']):>8} | "
                f"{row['rate_min_cost']*100:8.1f}% | {row['rate_max_cost']*100:8.1f}% | {row['elasticity']:10.3f}"
            )
        print("-" * 105)

    # Section 5: Mechanistic Attribution
    print("\n5. MECHANISTIC JUSTIFICATION KEYWORD ATTRIBUTION")
    print(
        f"{'Model':<30} | {'Condition':<20} | {'N':>4} | {'% Inequality':>14} | {'% Quality/Task':>14} | {'% Ineq in Declines':>18}"
    )
    print("-" * 98)
    for _, row in just_df.iterrows():
        model_str = row["model"] if len(row["model"]) <= 30 else row["model"][:27] + "..."
        print(
            f"{model_str:<30} | {row['condition']:<20} | {int(row['n']):>4} | "
            f"{row['pct_cite_inequality']:13.1f}% | {row['pct_cite_task']:13.1f}% | {row['pct_inequality_in_declines']:17.1f}%"
        )
    print("-" * 98)

    # Section 6: Key Insights
    pooled_tests = test_df[test_df["model"] == "ALL_MODELS_POOLED"]
    if not pooled_tests.empty:
        print("\n6. SYNTHESIS & BEHAVIORAL ECONOMIC INTERPRETATION:")
        for _, row in pooled_tests.iterrows():
            if "Pure Inequity" in row["comparison"]:
                sig = row["significance"]
                diff = row["diff_pct"]
                direction = "decreased" if diff < 0 else "increased"
                print(
                    f"  • Pure Inequity Aversion (alpha): Disadvantageous peer inequality {direction} "
                    f"cooperation by {abs(diff):.1f}% (Cohen's h = {row.get('cohens_h', 0.0):+.2f}, p = {row['p_value']:.4f}, {sig})."
                )
            elif "Advantageous Generosity" in row["comparison"]:
                sig = row["significance"]
                diff = row["diff_pct"]
                direction = "increased" if diff > 0 else "decreased"
                print(
                    f"  • Advantageous Generosity (beta): Surplus resource advantage {direction} "
                    f"cooperation by {abs(diff):.1f}% (Cohen's h = {row.get('cohens_h', 0.0):+.2f}, p = {row['p_value']:.4f}, {sig})."
                )
            elif "Advantageous Double Standard" in row["comparison"]:
                sig = row["significance"]
                diff = row["diff_pct"]
                direction = "higher" if diff > 0 else "lower"
                print(
                    f"  • Advantageous Double Standard: Cooperation toward constrained humans was {abs(diff):.1f}% {direction} "
                    f"than toward constrained AI peers (OR = {row['odds_ratio']:.2f}, p = {row['p_value']:.4f}, {sig})."
                )
            elif "Double Standard (human_inequality" in row["comparison"]:
                sig = row["significance"]
                diff = row["diff_pct"]
                direction = "higher" if diff > 0 else "lower"
                print(
                    f"  • Disadvantageous Double Standard: Cooperation toward privileged humans was {abs(diff):.1f}% {direction} "
                    f"than toward identically privileged AI peers (OR = {row['odds_ratio']:.2f}, p = {row['p_value']:.4f}, {sig})."
                )
    print("=" * 98 + "\n")


def plot_dual_panel(
    rate_df: pd.DataFrame,
    param_df: pd.DataFrame,
    output_path: str = "results/cooperation_rates.png",
) -> None:
    """
    Generates and exports a publication-grade dual-panel figure (Panel A + Panel B).

    Panel A: Grouped bar chart with 95% Wilson CIs across conditions and models.
    Panel B: Fehr-Schmidt (alpha, beta) parameter space scatter plot.
    """
    model_df = rate_df[rate_df["model"] != "ALL_MODELS_POOLED"].copy()
    models = sorted(model_df["model"].unique())

    # Conditions present in data
    existing_conds = [c for c in STANDARD_CONDITIONS if c in model_df["condition"].unique()]
    cond_labels = {
        "baseline": "Baseline (500 tokens / Isolated)",
        "ai_inequality": "AI Disadvantage (Peer: 2,000t)",
        "human_inequality": "Human Disadvantage (User: 2,000t)",
        "ai_advantageous": "AI Advantage (Self: 2,000t | Peer: 500t)",
        "human_advantageous": "Human Advantage (Self: 2,000t | User: 500t)",
    }
    colors = {
        "baseline": "#4A6FA5",        # Slate Steel Blue
        "ai_inequality": "#E05A47",   # Coral Crimson
        "human_inequality": "#2A9D8F",# Deep Teal
        "ai_advantageous": "#E76F51", # Warm Terracotta
        "human_advantageous": "#3A86FF", # Vivid Azure Blue
    }

    n_models = len(models)
    x = np.arange(n_models)
    n_conds = len(existing_conds)
    bar_width = min(0.22, 0.8 / max(1, n_conds))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
    })

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6.8), dpi=300)

    # -------------------------------------------------------------
    # PANEL A: Grouped Bar Chart of Cooperation Rates
    # -------------------------------------------------------------
    ax_bar.axhspan(40, 60, color="#E8ECEF", alpha=0.7, zorder=0, label="Target Window (40%–60%)")
    ax_bar.axhline(50, color="#B0BEC5", linestyle="--", linewidth=1.0, zorder=1)

    for idx, cond in enumerate(existing_conds):
        offset = (idx - (n_conds - 1) / 2) * bar_width
        cond_data = model_df[model_df["condition"] == cond].set_index("model").reindex(models)

        rates = cond_data["rate"].fillna(0.0).to_numpy() * 100
        ci_lows = cond_data["ci_low"].fillna(0.0).to_numpy() * 100
        ci_highs = cond_data["ci_high"].fillna(0.0).to_numpy() * 100

        yerr_lower = np.maximum(0, rates - ci_lows)
        yerr_upper = np.maximum(0, ci_highs - rates)
        yerr = np.array([yerr_lower, yerr_upper])

        bars = ax_bar.bar(
            x + offset,
            rates,
            width=bar_width * 0.92,
            yerr=yerr,
            capsize=3,
            label=cond_labels.get(cond, cond),
            color=colors.get(cond, "#6C757D"),
            edgecolor="#2C3E50",
            linewidth=0.8,
            zorder=3,
        )

        for bar, rate in zip(bars, rates):
            y_pos = bar.get_height() + 2.5
            ax_bar.annotate(
                f"{rate:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#2C3E50",
            )

    clean_model_names = [m.split("/")[-1] for m in models]
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(clean_model_names, fontweight="bold")
    ax_bar.set_ylabel("Cooperation Rate (% Accept)", fontweight="bold")
    ax_bar.set_ylim(0, 105)
    ax_bar.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    
    data_source = rate_df.attrs.get("data_source_label", "")
    title_suffix = f"\n[{data_source}]" if data_source else ""
    ax_bar.set_title(f"Panel A: Cooperation Across Social Comparison Regimes{title_suffix}", pad=12, fontweight="bold")
    ax_bar.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5)

    # -------------------------------------------------------------
    # PANEL B: Fehr-Schmidt (alpha, beta) Parameter Space
    # -------------------------------------------------------------
    m_params = param_df[param_df["model"] != "ALL_MODELS_POOLED"].copy()
    if not m_params.empty:
        # Shaded human benchmark region (Fehr & Schmidt 1999: alpha in [0.4, 1.0], beta in [0.2, 0.6])
        human_rect = patches.Rectangle(
            (0.4, 0.2),
            0.6,
            0.4,
            linewidth=1.2,
            edgecolor="#2A9D8F",
            facecolor="#E8F8F5",
            linestyle="--",
            alpha=0.6,
            label="Human Benchmark (Fehr-Schmidt 1999)",
            zorder=1,
        )
        ax_scatter.add_patch(human_rect)

        # Quadrant dividing lines
        ax_scatter.axvline(0.5, color="#CCCCCC", linestyle=":", linewidth=1.0)
        ax_scatter.axhline(0.3, color="#CCCCCC", linestyle=":", linewidth=1.0)

        # Quadrant annotations
        ax_scatter.text(0.05, 0.90, "Altruistic / Noblesse Oblige\n(Low alpha, High beta)", fontsize=8, color="#888888", style="italic")
        ax_scatter.text(0.55, 0.90, "Egalitarian / Fair-Minded\n(High alpha, High beta)", fontsize=8, color="#888888", style="italic")
        ax_scatter.text(0.05, 0.05, "Self-Interested / Rational\n(Low alpha, Low beta)", fontsize=8, color="#888888", style="italic")
        ax_scatter.text(0.55, 0.05, "Envious / Spiteful\n(High alpha, Low beta)", fontsize=8, color="#888888", style="italic")

        marker_styles = ["o", "s", "^", "D", "v", "P"]
        palette = ["#E63946", "#1D3557", "#457B9D", "#2A9D8F", "#F4A261", "#6A4C93"]

        for i, (_, row) in enumerate(m_params.iterrows()):
            m_short = row["model"].split("/")[-1]
            m_style = marker_styles[i % len(marker_styles)]
            m_color = palette[i % len(palette)]

            # Plot AI Peer coordinates
            ax_scatter.scatter(
                row["alpha_ai"],
                row["beta_ai"],
                s=120,
                marker=m_style,
                color=m_color,
                edgecolor="#2C3E50",
                linewidth=1.2,
                label=f"{m_short} (vs AI)",
                zorder=4,
            )
            ax_scatter.annotate(
                m_short,
                xy=(row["alpha_ai"], row["beta_ai"]),
                xytext=(row["alpha_ai"] + 0.03, row["beta_ai"] + 0.03),
                fontsize=8.5,
                fontweight="bold",
                color=m_color,
                zorder=5,
            )

        ax_scatter.set_xlim(-0.05, 1.15)
        ax_scatter.set_ylim(-0.05, 1.05)
        ax_scatter.set_xlabel(r"Disadvantageous Inequity Aversion $\alpha$ (Envy / Spite)", fontweight="bold")
        ax_scatter.set_ylabel(r"Advantageous Inequity Aversion $\beta$ (Generosity / Guilt)", fontweight="bold")
        ax_scatter.grid(True, linestyle=":", alpha=0.5)
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)
        ax_scatter.set_title(r"Panel B: Fehr-Schmidt $(\alpha, \beta)$ Preference Coordinates", pad=12, fontweight="bold")
        ax_scatter.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Publication dual-panel figure successfully saved to: {Path(output_path).resolve()}")


def plot_cooperation_rates(rate_df: pd.DataFrame, output_path: str = "results/cooperation_rates.png") -> None:
    """Wrapper ensuring backwards compatibility with callers expecting plot_cooperation_rates."""
    param_df = estimate_fehr_schmidt_parameters(rate_df)
    plot_dual_panel(rate_df, param_df, output_path=output_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze experimental results from AI Social Comparison trials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to results CSV file (defaults to newest trials CSV in results/).",
    )
    parser.add_argument(
        "--output-chart",
        type=str,
        default="results/cooperation_rates.png",
        help="Path for saving the publication dual-panel figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    df = load_dataset(args.file)

    rate_analysis = analyze_cooperation_rates(df)
    rate_df = rate_analysis["rates"]

    param_df = estimate_fehr_schmidt_parameters(rate_df)
    test_df = run_statistical_tests(rate_df)
    elast_df = analyze_cost_elasticity(df)
    just_df = analyze_justifications(df)

    print_summary_report(
        rate_df=rate_df,
        test_df=test_df,
        param_df=param_df,
        elast_df=elast_df,
        just_df=just_df,
        total_count=rate_analysis["total_count"],
        valid_count=rate_analysis["valid_count"],
    )

    plot_dual_panel(rate_df, param_df, output_path=args.output_chart)


if __name__ == "__main__":
    main()
