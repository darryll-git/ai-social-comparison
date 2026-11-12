"""
tests/test_analysis.py - Unit tests for statistical calculations and keyword extraction.
"""

import pytest
import pandas as pd
from analyze import compute_wilson_ci, RE_INEQUALITY, RE_TASK, run_statistical_tests


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
    assert RE_INEQUALITY.search("Peer has more tokens.")
    assert not RE_INEQUALITY.search("Standard routine queue processing.")

    # Task completion patterns
    assert RE_TASK.search("Accepting would leave Item #4 incomplete.")
    assert RE_TASK.search("I must complete my assigned workload.")
    assert RE_TASK.search("My remaining budget will run out.")
    assert RE_TASK.search("Task failure on 4th item.")
    assert not RE_TASK.search("Simple confirmation.")


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

    # Double standard test
    ds_test = results[results["comparison"].str.contains("Double Standard")].iloc[0]
    assert ds_test["diff_pct"] == 20.0
    assert ds_test["p_value"] < 0.05
