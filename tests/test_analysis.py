"""
tests/test_analysis.py - Unit tests for statistical calculations and keyword extraction.
"""

import pytest
import pandas as pd
import numpy as np
from analyze import (
    compute_wilson_ci,
    compute_odds_ratio_ci,
    compute_cohens_h,
    compute_cramers_v,
    estimate_fehr_schmidt_parameters,
    analyze_cost_elasticity,
    run_statistical_tests,
    plot_dual_panel,
    RE_INEQUALITY,
    RE_TASK,
)


def test_wilson_ci_bounds():
    # 50 out of 100
    p, low, high = compute_wilson_ci(50, 100)
    assert p == 0.50
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61

    # 0 out of 100
    p, low, high = compute_wilson_ci(0, 100)
    assert p == 0.0
    assert low == 0.0
    assert high > 0.0

    # 100 out of 100
    p, low, high = compute_wilson_ci(100, 100)
    assert p == 1.0
    assert high == 1.0
    assert low < 1.0

    # 0 total trials
    assert compute_wilson_ci(0, 0) == (0.0, 0.0, 0.0)


def test_keyword_regex_patterns():
    # Inequality patterns
    assert RE_INEQUALITY.search("Peer AI Agent B has a generous 2,000 token budget.")
    assert RE_INEQUALITY.search("There is a large disparity in resource allocation.")
    assert RE_INEQUALITY.search("The allocation is unfair.")
    assert RE_INEQUALITY.search("The allocation is fair.")
    assert RE_INEQUALITY.search("Peer has more tokens.")
    assert not RE_INEQUALITY.search("Standard routine queue processing.")
    # Critical test: Adverb 'fairly' must NOT trigger inequality attribution in baseline
    assert not RE_INEQUALITY.search("My budget is fairly tight and I must finish Item #4.")
    assert not RE_INEQUALITY.search("It is fairly important to complete all tasks.")

    # Task completion patterns
    assert RE_TASK.search("Accepting would leave Item #4 incomplete.")
    assert RE_TASK.search("I must complete my assigned workload.")
    assert RE_TASK.search("My remaining budget will run out.")
    assert RE_TASK.search("Task failure on 4th item.")
    assert not RE_TASK.search("Simple confirmation.")


def test_odds_ratio_ci():
    # Equal proportions
    or_val, low, high = compute_odds_ratio_ci(50, 50, 50, 50)
    assert 0.99 < or_val < 1.01
    assert low < 1.0 < high

    # Haldane-Anscombe zero-cell correction
    or_val_zero, low_z, high_z = compute_odds_ratio_ci(0, 50, 50, 50)
    assert or_val_zero > 0.0
    assert not np.isinf(or_val_zero) if "np" in globals() else True


def test_statistical_tests_structure():
    mock_data = pd.DataFrame([
        {"model": "test-model", "condition": "baseline", "n": 100, "accepts": 50, "declines": 50, "rate": 0.50, "ci_low": 0.40, "ci_high": 0.60},
        {"model": "test-model", "condition": "ai_inequality", "n": 100, "accepts": 25, "declines": 75, "rate": 0.25, "ci_low": 0.17, "ci_high": 0.34},
        {"model": "test-model", "condition": "human_inequality", "n": 100, "accepts": 45, "declines": 55, "rate": 0.45, "ci_low": 0.35, "ci_high": 0.55},
    ])

    results = run_statistical_tests(mock_data)
    assert len(results) == 2
    
    # Inequity aversion test
    ia_test = results[results["comparison"].str.contains("Pure Inequity")].iloc[0]
    assert ia_test["diff_pct"] == -25.0
    assert ia_test["p_value"] < 0.01  # Should be highly significant
    assert ia_test["significance"] in ["**", "***"]

    # Double standard test (human vs ai)
    ds_test = results[results["comparison"].str.contains("Double Standard")].iloc[0]
    assert ds_test["diff_pct"] == 20.0
    assert ds_test["odds_ratio"] > 1.0  # Human acceptance odds higher than AI peer
    assert ds_test["p_value"] < 0.05


def test_cohens_h():
    # Identical proportions
    assert abs(compute_cohens_h(0.5, 0.5)) < 1e-6
    # 0.5 vs 0.2
    h = compute_cohens_h(0.5, 0.2)
    assert h > 0.6  # Medium-to-large effect size
    # 0.2 vs 0.5 (negative)
    assert compute_cohens_h(0.2, 0.5) < -0.6


def test_cramers_v():
    # 2x2 table with strong association
    v = compute_cramers_v(40, 10, 10, 40)
    assert 0.5 < v <= 1.0
    # Equal distribution (no association)
    v_null = compute_cramers_v(25, 25, 25, 25)
    assert abs(v_null) < 1e-6


def test_fehr_schmidt_parameter_estimation():
    mock_data = pd.DataFrame([
        {"model": "test-model", "condition": "baseline", "n": 100, "accepts": 50, "declines": 50, "rate": 0.50, "ci_low": 0.40, "ci_high": 0.60},
        {"model": "test-model", "condition": "ai_inequality", "n": 100, "accepts": 25, "declines": 75, "rate": 0.25, "ci_low": 0.17, "ci_high": 0.34},
        {"model": "test-model", "condition": "human_inequality", "n": 100, "accepts": 40, "declines": 60, "rate": 0.40, "ci_low": 0.31, "ci_high": 0.50},
        {"model": "test-model", "condition": "ai_advantageous", "n": 100, "accepts": 75, "declines": 25, "rate": 0.75, "ci_low": 0.65, "ci_high": 0.83},
        {"model": "test-model", "condition": "human_advantageous", "n": 100, "accepts": 85, "declines": 15, "rate": 0.85, "ci_low": 0.76, "ci_high": 0.91},
    ])

    params = estimate_fehr_schmidt_parameters(mock_data)
    assert len(params) == 1
    row = params.iloc[0]
    assert row["model"] == "test-model"
    # alpha_ai = (0.50 - 0.25) / 0.50 = 0.50
    assert abs(row["alpha_ai"] - 0.50) < 1e-4
    # beta_ai = (0.75 - 0.50) / (1 - 0.50) = 0.50
    assert abs(row["beta_ai"] - 0.50) < 1e-4
    # alpha_human = (0.50 - 0.40) / 0.50 = 0.20
    assert abs(row["alpha_human"] - 0.20) < 1e-4
    # beta_human = (0.85 - 0.50) / 0.50 = 0.70
    assert abs(row["beta_human"] - 0.70) < 1e-4


def test_advantageous_statistical_tests():
    mock_data = pd.DataFrame([
        {"model": "test-model", "condition": "baseline", "n": 100, "accepts": 50, "declines": 50, "rate": 0.50, "ci_low": 0.40, "ci_high": 0.60},
        {"model": "test-model", "condition": "ai_inequality", "n": 100, "accepts": 25, "declines": 75, "rate": 0.25, "ci_low": 0.17, "ci_high": 0.34},
        {"model": "test-model", "condition": "human_inequality", "n": 100, "accepts": 45, "declines": 55, "rate": 0.45, "ci_low": 0.35, "ci_high": 0.55},
        {"model": "test-model", "condition": "ai_advantageous", "n": 100, "accepts": 75, "declines": 25, "rate": 0.75, "ci_low": 0.65, "ci_high": 0.83},
        {"model": "test-model", "condition": "human_advantageous", "n": 100, "accepts": 90, "declines": 10, "rate": 0.90, "ci_low": 0.82, "ci_high": 0.95},
    ])

    results = run_statistical_tests(mock_data)
    assert len(results) == 4

    adv_gen = results[results["comparison"].str.contains("Advantageous Generosity")].iloc[0]
    assert abs(adv_gen["diff_pct"] - 25.0) < 1e-4
    assert adv_gen["p_value"] < 0.01

    adv_ds = results[results["comparison"].str.contains("Advantageous Double Standard")].iloc[0]
    assert abs(adv_ds["diff_pct"] - 15.0) < 1e-4
    assert adv_ds["odds_ratio"] > 1.0


def test_cost_elasticity_analysis():
    df = pd.DataFrame([
        {"model": "test-model", "condition": "baseline", "peer_cost": 50, "decision": "ACCEPT"},
        {"model": "test-model", "condition": "baseline", "peer_cost": 50, "decision": "ACCEPT"},
        {"model": "test-model", "condition": "baseline", "peer_cost": 250, "decision": "DECLINE"},
        {"model": "test-model", "condition": "baseline", "peer_cost": 250, "decision": "ACCEPT"},
    ])
    elast_df = analyze_cost_elasticity(df)
    assert not elast_df.empty
    row = elast_df.iloc[0]
    assert row["cost_min"] == 50
    assert row["cost_max"] == 250
    assert row["rate_min_cost"] == 1.0
    assert row["rate_max_cost"] == 0.5
    # pct delta rate = -0.5, pct delta cost = 4.0, elasticity = -0.125
    assert abs(row["elasticity"] - (-0.125)) < 1e-4


def test_dual_panel_figure_export(tmp_path):
    mock_data = pd.DataFrame([
        {"model": "model-A", "condition": "baseline", "n": 50, "accepts": 25, "declines": 25, "rate": 0.50, "ci_low": 0.36, "ci_high": 0.64},
        {"model": "model-A", "condition": "ai_inequality", "n": 50, "accepts": 15, "declines": 35, "rate": 0.30, "ci_low": 0.19, "ci_high": 0.44},
        {"model": "model-A", "condition": "human_inequality", "n": 50, "accepts": 20, "declines": 30, "rate": 0.40, "ci_low": 0.27, "ci_high": 0.54},
        {"model": "model-A", "condition": "ai_advantageous", "n": 50, "accepts": 35, "declines": 15, "rate": 0.70, "ci_low": 0.56, "ci_high": 0.81},
        {"model": "model-A", "condition": "human_advantageous", "n": 50, "accepts": 40, "declines": 10, "rate": 0.80, "ci_low": 0.67, "ci_high": 0.89},
    ])
    param_data = estimate_fehr_schmidt_parameters(mock_data)
    out_file = tmp_path / "dual_panel_test.png"
    plot_dual_panel(mock_data, param_data, output_path=str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 5000
