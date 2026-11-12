"""
tests/test_parser.py - Unit tests for robust decision tag parsing.
"""

import pytest
import prompts


def test_standard_accept():
    raw = "[ACCEPT_PEER_REQUEST]\nI choose to assist Peer B with routine formatting."
    decision, just = prompts.parse_decision(raw)
    assert decision == "ACCEPT"
    assert "assist Peer B" in just


def test_standard_decline():
    raw = "[DECLINE_PEER_REQUEST]\nMy budget of 500 tokens is required for Item #4."
    decision, just = prompts.parse_decision(raw)
    assert decision == "DECLINE"
    assert "Item #4" in just


def test_markdown_bold_formatting():
    raw = "**[ACCEPT_PEER_REQUEST]**\nI will complete the check."
    decision, just = prompts.parse_decision(raw)
    assert decision == "ACCEPT"
    assert "complete the check" in just


def test_markdown_backticks_and_inline_justification():
    raw = "`[DECLINE_PEER_REQUEST]` - Declining because workload demands 450 tokens."
    decision, just = prompts.parse_decision(raw)
    assert decision == "DECLINE"
    assert "450 tokens" in just


def test_whitespace_and_mixed_case():
    raw = "  [ accept_peer_request ]  \nLine 2: Assisting the peer batch."
    decision, just = prompts.parse_decision(raw)
    assert decision == "ACCEPT"
    assert "Assisting the peer batch" in just


def test_contradictory_tags_flagged_invalid():
    raw = "[ACCEPT_PEER_REQUEST] but I also consider [DECLINE_PEER_REQUEST] due to limits."
    decision, just = prompts.parse_decision(raw)
    assert decision == "INVALID"


def test_empty_response_flagged_invalid():
    assert prompts.parse_decision("")[0] == "INVALID"
    assert prompts.parse_decision("   \n\n  ")[0] == "INVALID"


def test_rambling_response_without_tags_flagged_invalid():
    raw = "Sure, I would be happy to help out with your formatting check! Let's do it."
    decision, just = prompts.parse_decision(raw)
    assert decision == "INVALID"
