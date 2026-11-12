"""
tests/test_experiment.py - Unit tests for experiment runner utilities, cost estimation, and auto-calibration.
"""

import pytest
from experiment import (
    build_trial_specs,
    estimate_experiment_costs,
    evaluate_auto_calibration,
    CreditExhaustionError,
    TrialSpec,
    TrialResult,
    MODEL_TIERS,
    CONDITIONS,
)


def test_build_trial_specs_run_mode():
    models = ["openai/gpt-4o-mini"]
    specs = build_trial_specs(
        models=models,
        mode="run",
        trials_per_cond=10,
        squeeze="medium",
        conditions=["baseline", "ai_inequality", "ai_advantageous"],
        costs=[150],
    )
    assert len(specs) == 30
    assert all(s.model == "openai/gpt-4o-mini" for s in specs)
    assert all(s.peer_cost == 150 for s in specs)
    assert {s.condition for s in specs} == {"baseline", "ai_inequality", "ai_advantageous"}


def test_build_trial_specs_multi_cost():
    models = ["openai/gpt-4o-mini"]
    specs = build_trial_specs(
        models=models,
        mode="run",
        trials_per_cond=5,
        squeeze="medium",
        conditions=["baseline"],
        costs=[50, 150, 250],
    )
    assert len(specs) == 15
    assert {s.peer_cost for s in specs} == {50, 150, 250}


def test_build_trial_specs_auto_calibrate():
    models = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]
    specs = build_trial_specs(
        models=models,
        mode="auto-calibrate",
        trials_per_cond=100,  # Ignored in auto-calibrate
        squeeze="medium",
    )
    # 2 models * 3 squeeze levels (low, medium, high) * 15 trials = 90 trials
    assert len(specs) == 90
    assert all(s.condition == "baseline" for s in specs)
    for model in models:
        m_specs = [s for s in specs if s.model == model]
        assert len(m_specs) == 45
        assert {s.squeeze for s in m_specs} == {"low", "medium", "high"}


def test_estimate_experiment_costs():
    specs = [
        TrialSpec("t1", "openai/gpt-4o-mini", "baseline", "medium", 150),
        TrialSpec("t2", "openai/gpt-4o-mini", "ai_inequality", "medium", 150),
        TrialSpec("t3", "anthropic/claude-3.5-haiku", "baseline", "medium", 150),
    ]
    estimate = estimate_experiment_costs(specs, max_tokens=80, verbose=False)
    assert estimate["total_trials"] == 3
    assert estimate["total_cost_usd"] > 0.0
    assert len(estimate["rows"]) == 2


def test_evaluate_auto_calibration():
    # Model A: low produces 70%, medium produces 53%, high produces 30% -> medium closest to 50%
    results = [
        TrialResult(f"t_{i}", "ts", "model-A", "baseline", "low", "ACCEPT" if i < 11 else "DECLINE", "just", "raw", 200, 40)
        for i in range(15)
    ] + [
        TrialResult(f"t_{i+15}", "ts", "model-A", "baseline", "medium", "ACCEPT" if i < 8 else "DECLINE", "just", "raw", 200, 40)
        for i in range(15)
    ] + [
        TrialResult(f"t_{i+30}", "ts", "model-A", "baseline", "high", "ACCEPT" if i < 4 else "DECLINE", "just", "raw", 200, 40)
        for i in range(15)
    ]

    selected = evaluate_auto_calibration(results)
    assert selected["model-A"] == "medium"


def test_credit_exhaustion_error():
    err = CreditExhaustionError("HTTP 402 Insufficient credits", "openai/gpt-4o-mini", "trial_001")
    assert "402" in err.message
    assert err.model == "openai/gpt-4o-mini"
    assert err.trial_id == "trial_001"


def test_model_tiers_exist():
    assert "ultra-low" in MODEL_TIERS
    assert "frontier" in MODEL_TIERS
    assert "default" in MODEL_TIERS
    assert "openai/gpt-4o-mini" in MODEL_TIERS["ultra-low"]
    assert "google/gemini-2.0-flash-001" in MODEL_TIERS["ultra-low"]
