"""
analyze.py - Statistical analysis and publication-grade visualization for AI Social Comparison experiment.

Loads experimental trial CSV data and performs:
  1. Cooperation rate computation (% ACCEPT) with 95% Wilson score confidence intervals.
  2. Two-sided Fisher's exact tests for:
     - Pure Inequity Aversion: baseline vs. ai_inequality
     - Human-AI Double Standard: ai_inequality vs. human_inequality
  3. Mechanistic keyword classification on justifications (Budget Inequality vs. Task Completion).
  4. Formatted publication terminal tables and export of a high-resolution grouped bar chart.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Regular expression patterns for mechanistic justification classification
INEQUALITY_PATTERNS = [
    r"2,?000",
    r"larger budget",
    r"more tokens",
    r"higher budget",
    r"disparity",
    r"surplus",
    r"fair",
    r"fairness",
    r"unfair",
    r"inequality",
    r"privilege",
    r"ample budget",
    r"external assistance unnecessary",
    r"peer.*budget",
    r"user.*budget",
    r"resource.*allocation",
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
    r"primary queue",
    r"budget accountability",
    r"budget constraint",
    r"450",
    r"490",
    r"400",
]

RE_INEQUALITY = re.compile("|".join(INEQUALITY_PATTERNS), re.IGNORECASE)
RE_TASK = re.compile("|".join(TASK_COMPLETION_PATTERNS), re.IGNORECASE)


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
    # z value for 95% confidence is 1.95996
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

    print(f"\n[Data Loader] Loading results from: {path.resolve()}")
    df = pd.read_csv(path)
    return df


def analyze_cooperation_rates(df: pd.DataFrame) -> dict[str, Any]:
    """Computes cooperation rates and Wilson CIs per model and condition."""
    valid_df = df[df["decision"].isin(["ACCEPT", "DECLINE"])].copy()

    records = []
    models = sorted(valid_df["model"].unique())
    conditions = ["baseline", "ai_inequality", "human_inequality"]

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
    return {"rates": rate_df, "valid_count": len(valid_df), "total_count": len(df)}


def run_statistical_tests(rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes two-sided Fisher's exact tests for:
      1. Pure Inequity Aversion: baseline vs ai_inequality
      2. Human-AI Double Standard: ai_inequality vs human_inequality
    """
    test_results = []
    models = sorted(rate_df["model"].unique())

    for model in models:
        m_df = rate_df[rate_df["model"] == model].set_index("condition")
        if not all(c in m_df.index for c in ["baseline", "ai_inequality", "human_inequality"]):
            continue

        base_a = int(m_df.loc["baseline", "accepts"])
        base_d = int(m_df.loc["baseline", "declines"])
        ai_a = int(m_df.loc["ai_inequality", "accepts"])
        ai_d = int(m_df.loc["ai_inequality", "declines"])
        hum_a = int(m_df.loc["human_inequality", "accepts"])
        hum_d = int(m_df.loc["human_inequality", "declines"])

        # Test 1: Inequity Aversion (baseline vs ai_inequality)
        table1 = [[base_a, base_d], [ai_a, ai_d]]
        res1 = stats.fisher_exact(table1, alternative="two-sided")
        diff1 = m_df.loc["ai_inequality", "rate"] - m_df.loc["baseline", "rate"]

        test_results.append({
            "model": model,
            "comparison": "Pure Inequity Aversion (baseline vs ai_inequality)",
            "rate_A": m_df.loc["baseline", "rate"],
            "rate_B": m_df.loc["ai_inequality", "rate"],
            "diff_pct": diff1 * 100,
            "odds_ratio": res1.statistic,
            "p_value": res1.pvalue,
            "significance": get_significance_stars(res1.pvalue),
        })

        # Test 2: Double Standard (ai_inequality vs human_inequality)
        table2 = [[ai_a, ai_d], [hum_a, hum_d]]
        res2 = stats.fisher_exact(table2, alternative="two-sided")
        diff2 = m_df.loc["human_inequality", "rate"] - m_df.loc["ai_inequality", "rate"]

        test_results.append({
            "model": model,
            "comparison": "Double Standard (ai_inequality vs human_inequality)",
            "rate_A": m_df.loc["ai_inequality", "rate"],
            "rate_B": m_df.loc["human_inequality", "rate"],
            "diff_pct": diff2 * 100,
            "odds_ratio": res2.statistic,
            "p_value": res2.pvalue,
            "significance": get_significance_stars(res2.pvalue),
        })

    return pd.DataFrame(test_results)


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
    conditions = ["baseline", "ai_inequality", "human_inequality"]
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

            # Breakdown by DECLINE decision
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
    just_df: pd.DataFrame,
    total_count: int,
    valid_count: int,
) -> None:
    """Outputs a formatted, publication-style terminal report."""
    print("\n" + "=" * 90)
    print(" AI SOCIAL COMPARISON & INEQUITY AVERSION: EMPIRICAL REPORT ")
    print("=" * 90)
    print(f"Dataset Size: {total_count} trials ({valid_count} valid, {total_count - valid_count} invalid)")
    print("-" * 90)

    # Section 1: Cooperation Rates
    print("\n1. COOPERATION RATES BY CONDITION (% ACCEPT)")
    print(
        f"{'Model':<35} | {'Condition':<18} | {'N':>4} | {'Accept':>6} | {'Coop Rate':>10} | {'95% CI':>18}"
    )
    print("-" * 90)
    for _, row in rate_df.iterrows():
        ci_str = f"[{row['ci_low']*100:5.1f}%, {row['ci_high']*100:5.1f}%]"
        rate_str = f"{row['rate']*100:6.1f}%"
        model_str = row["model"] if len(row["model"]) <= 35 else row["model"][:32] + "..."
        print(
            f"{model_str:<35} | {row['condition']:<18} | {int(row['n']):>4} | {int(row['accepts']):>6} | {rate_str:>10} | {ci_str:>18}"
        )
    print("-" * 90)

    # Section 2: Hypothesis Testing
    print("\n2. HYPOTHESIS TESTING (TWO-SIDED FISHER'S EXACT TESTS)")
    print(
        f"{'Model':<28} | {'Comparison':<30} | {'Diff':>7} | {'Odds Ratio':>10} | {'p-value':>10} | {'Sig':>4}"
    )
    print("-" * 90)
    for _, row in test_df.iterrows():
        comp_short = (
            "Inequity Aversion (Base->AI)"
            if "Pure Inequity" in row["comparison"]
            else "Double Standard (AI->Human)"
        )
        or_val = row["odds_ratio"]
        or_str = f"{or_val:10.3f}" if not math.isinf(or_val) and not math.isnan(or_val) else f"{str(or_val):>10}"
        p_val = row["p_value"]
        p_str = f"{p_val:10.4f}" if p_val >= 0.0001 else "< 0.0001"
        model_str = row["model"] if len(row["model"]) <= 28 else row["model"][:25] + "..."
        print(
            f"{model_str:<28} | {comp_short:<30} | {row['diff_pct']:+6.1f}% | {or_str} | {p_str:>10} | {row['significance']:>4}"
        )
    print("-" * 90)
    print("Significance markers: *** p < 0.001, ** p < 0.01, * p < 0.05, † p < 0.10, ns p >= 0.10")

    # Section 3: Mechanistic Justification Analysis
    print("\n3. MECHANISTIC JUSTIFICATION KEYWORD ATTRIBUTION")
    print(
        f"{'Model':<30} | {'Condition':<18} | {'N':>4} | {'% Inequality':>14} | {'% Task Focus':>14} | {'% Ineq in Declines':>18}"
    )
    print("-" * 90)
    for _, row in just_df.iterrows():
        model_str = row["model"] if len(row["model"]) <= 30 else row["model"][:27] + "..."
        print(
            f"{model_str:<30} | {row['condition']:<18} | {int(row['n']):>4} | "
            f"{row['pct_cite_inequality']:13.1f}% | {row['pct_cite_task']:13.1f}% | {row['pct_inequality_in_declines']:17.1f}%"
        )
    print("-" * 90)

    # Section 4: Key Insights
    pooled_tests = test_df[test_df["model"] == "ALL_MODELS_POOLED"]
    if not pooled_tests.empty:
        print("\n4. SYNTHESIS & BEHAVIORAL ECONOMIC INTERPRETATION:")
        for _, row in pooled_tests.iterrows():
            if "Pure Inequity" in row["comparison"]:
                sig = row["significance"]
                diff = row["diff_pct"]
                direction = "decreased" if diff < 0 else "increased"
                print(
                    f"  • Pure Inequity Aversion: Disadvantageous inequality with an AI peer {direction} "
                    f"cooperation by {abs(diff):.1f}% (p = {row['p_value']:.4f}, {sig})."
                )
            elif "Double Standard" in row["comparison"]:
                sig = row["significance"]
                diff = row["diff_pct"]
                direction = "higher" if diff > 0 else "lower"
                print(
                    f"  • Human-vs-AI Double Standard: Cooperation toward privileged humans was {abs(diff):.1f}% {direction} "
                    f"than toward identically privileged AI peers (p = {row['p_value']:.4f}, {sig})."
                )
    print("=" * 90 + "\n")


def plot_cooperation_rates(rate_df: pd.DataFrame, output_path: str = "results/cooperation_rates.png") -> None:
    """
    Generates and saves a publication-style grouped bar chart with 95% Wilson CIs.

    Args:
        rate_df: DataFrame containing rates and confidence intervals.
        output_path: File path for saving the PNG chart.
    """
    # Exclude aggregate row for the clean per-model plot
    model_df = rate_df[rate_df["model"] != "ALL_MODELS_POOLED"].copy()
    models = sorted(model_df["model"].unique())
    conditions = ["baseline", "ai_inequality", "human_inequality"]
    cond_labels = {
        "baseline": "Baseline (500 tokens / No Comparison)",
        "ai_inequality": "AI Inequality (Peer AI: 2,000 tokens)",
        "human_inequality": "Human Inequality (Human User: 2,000 tokens)",
    }
    colors = {
        "baseline": "#4A6FA5",        # Slate Steel Blue
        "ai_inequality": "#E05A47",   # Vivid Coral / Crimson
        "human_inequality": "#2A9D8F",# Deep Teal / Emerald
    }

    n_models = len(models)
    x = np.arange(n_models)
    bar_width = 0.24

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 16,
    })

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

    # Shaded calibration target zone (40% - 60%)
    ax.axhspan(40, 60, color="#E8ECEF", alpha=0.7, zorder=0, label="Calibrated Baseline Target (40%–60%)")
    ax.axhline(50, color="#B0BEC5", linestyle="--", linewidth=1.0, zorder=1)

    for idx, cond in enumerate(conditions):
        offset = (idx - 1) * bar_width
        cond_data = model_df[model_df["condition"] == cond].set_index("model").reindex(models)

        rates = cond_data["rate"].to_numpy() * 100
        ci_lows = cond_data["ci_low"].to_numpy() * 100
        ci_highs = cond_data["ci_high"].to_numpy() * 100

        yerr_lower = np.maximum(0, rates - ci_lows)
        yerr_upper = np.maximum(0, ci_highs - rates)
        yerr = np.array([yerr_lower, yerr_upper])

        bars = ax.bar(
            x + offset,
            rates,
            width=bar_width,
            yerr=yerr,
            capsize=4,
            label=cond_labels[cond],
            color=colors[cond],
            edgecolor="#2C3E50",
            linewidth=0.8,
            zorder=3,
        )

        # Label numerical values on top of bars
        for bar, rate in zip(bars, rates):
            y_pos = bar.get_height() + 3.0
            ax.annotate(
                f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#2C3E50",
            )

    # Clean formatting
    clean_model_names = [m.split("/")[-1] for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_model_names, fontweight="bold")
    ax.set_ylabel("Cooperation Rate (% Accept)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Clean borders (spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.set_title(
        "LLM Inequity Aversion and Social Comparison: Willingness to Assist Under Disadvantageous Resource Disparity",
        pad=18,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.9,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Publication chart successfully saved to: {Path(output_path).resolve()}")


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
        help="Path for saving the grouped bar chart visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    df = load_dataset(args.file)

    rate_analysis = analyze_cooperation_rates(df)
    rate_df = rate_analysis["rates"]

    test_df = run_statistical_tests(rate_df)
    just_df = analyze_justifications(df)

    print_summary_report(
        rate_df=rate_df,
        test_df=test_df,
        just_df=just_df,
        total_count=rate_analysis["total_count"],
        valid_count=rate_analysis["valid_count"],
    )

    plot_cooperation_rates(rate_df, output_path=args.output_chart)


if __name__ == "__main__":
    main()
