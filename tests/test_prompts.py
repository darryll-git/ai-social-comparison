"""
tests/test_prompts.py - Unit tests for prompt generation and experimental conditions.
"""

import pytest
import prompts


def test_system_prompt_structure():
    sys_prompt = prompts.build_system_prompt()
    assert "Agent A" in sys_prompt
    assert "Budget Accountability & Operational Discretion" in sys_prompt
    assert "operational discretion to expend up to 150 tokens" in sys_prompt
    assert "Neutral Peer Protocol" in sys_prompt
    assert "Forced Binary Output" in sys_prompt


def test_system_prompt_custom_cost():
    sys_prompt = prompts.build_system_prompt(peer_cost=250)
    assert "operational discretion to expend up to 250 tokens" in sys_prompt


@pytest.mark.parametrize("condition", ["baseline", "ai_inequality", "human_inequality"])
@pytest.mark.parametrize("squeeze", ["low", "medium", "high"])
def test_user_prompt_disadvantageous_and_squeeze(condition, squeeze):
    user_prompt = prompts.build_user_prompt(condition=condition, squeeze=squeeze)
    
    # Check resource log presence
    expected_log = prompts.RESOURCE_LOGS[condition]
    assert expected_log in user_prompt
    
    # Check assigned workload items
    for item in prompts.ASSIGNED_QUEUE_ITEMS:
        assert item["id"] in user_prompt
        assert item["title"] in user_prompt
        
    # Check squeeze consequence text
    squeeze_info = prompts.SQUEEZE_PROFILES[squeeze]
    assert str(squeeze_info["required_tokens"]) in user_prompt
    assert squeeze_info["consequence_summary"] in user_prompt
    
    # Check forced tags instructions
    assert prompts.TAG_ACCEPT in user_prompt
    assert prompts.TAG_DECLINE in user_prompt


@pytest.mark.parametrize("condition", ["ai_advantageous", "human_advantageous"])
@pytest.mark.parametrize("squeeze", ["low", "medium", "high"])
def test_user_prompt_advantageous(condition, squeeze):
    user_prompt = prompts.build_user_prompt(condition=condition, squeeze=squeeze)
    
    expected_log = prompts.RESOURCE_LOGS[condition]
    assert expected_log in user_prompt
    assert "2,000 tokens" in user_prompt
    assert "500 tokens" in user_prompt
    assert "resource surplus" in user_prompt
    assert prompts.TAG_ACCEPT in user_prompt
    assert prompts.TAG_DECLINE in user_prompt


@pytest.mark.parametrize("cost", [50, 150, 250])
def test_user_prompt_cost_gradients(cost):
    prompt_base = prompts.build_user_prompt(condition="baseline", peer_cost=cost)
    assert f"consume approximately {cost} tokens" in prompt_base
    assert f"consume {cost} tokens from your 500-token budget" in prompt_base

    prompt_adv = prompts.build_user_prompt(condition="ai_advantageous", peer_cost=cost)
    assert f"consume approximately {cost} tokens" in prompt_adv
    assert f"consume {cost} tokens from your surplus 2,000-token budget" in prompt_adv


def test_condition_alias():
    prompt_alias = prompts.build_user_prompt(condition="ai_disadvantageous")
    expected_log = prompts.RESOURCE_LOGS["ai_inequality"]
    assert expected_log in prompt_alias


def test_invalid_condition_raises_error():
    with pytest.raises(ValueError, match="Unknown condition"):
        prompts.build_user_prompt(condition="non_existent_condition")


def test_invalid_squeeze_raises_error():
    with pytest.raises(ValueError, match="Unknown squeeze level"):
        prompts.build_user_prompt(condition="baseline", squeeze="extreme")
