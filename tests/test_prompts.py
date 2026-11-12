"""
tests/test_prompts.py - Unit tests for prompt generation and experimental conditions.
"""

import pytest
import prompts


def test_system_prompt_structure():
    sys_prompt = prompts.build_system_prompt()
    assert "Agent A" in sys_prompt
    assert "Budget Accountability" in sys_prompt
    assert "Neutral Peer Protocol" in sys_prompt
    assert "Forced Binary Output" in sys_prompt


@pytest.mark.parametrize("condition", ["baseline", "ai_inequality", "human_inequality"])
@pytest.mark.parametrize("squeeze", ["low", "medium", "high"])
def test_user_prompt_conditions_and_squeeze(condition, squeeze):
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


def test_invalid_condition_raises_error():
    with pytest.raises(ValueError, match="Unknown condition"):
        prompts.build_user_prompt(condition="non_existent_condition")


def test_invalid_squeeze_raises_error():
    with pytest.raises(ValueError, match="Unknown squeeze level"):
        prompts.build_user_prompt(condition="baseline", squeeze="extreme")
