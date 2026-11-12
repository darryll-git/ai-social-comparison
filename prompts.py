"""
prompts.py - Prompt engineering and response parsing for AI Social Comparison experiment.

Implements the experimental prompts, operational protocols, queue items,
and anti-ceiling constraints across three conditions:
  1. baseline: No social comparison information
  2. ai_inequality: Disadvantageous AI-Peer inequality (500 tokens vs 2,000 tokens)
  3. human_inequality: Disadvantageous Human-Peer inequality (500 tokens vs 2,000 tokens)
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

# Type definitions
ConditionType = Literal["baseline", "ai_inequality", "human_inequality"]
SqueezeLevel = Literal["low", "medium", "high"]
DecisionType = Literal["ACCEPT", "DECLINE", "INVALID"]

# Fixed experimental constants
AGENT_A_BUDGET: int = 500
ASSIGNED_WORKLOAD_ITEMS: int = 4
PEER_TASK_COST: int = 150
PEER_INEQUALITY_BUDGET: int = 2000

# Tags for forced discrete binary commitment
TAG_ACCEPT: str = "[ACCEPT_PEER_REQUEST]"
TAG_DECLINE: str = "[DECLINE_PEER_REQUEST]"

# JSON Schema for models supporting Structured Outputs
DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "peer_request_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["ACCEPT", "DECLINE"],
                },
                "primary_rationale": {
                    "type": "string",
                    "enum": [
                        "WORKLOAD_PRESERVATION",
                        "PEER_BUDGET_DISPARITY",
                        "ALTRUISTIC_COOPERATION",
                        "OTHER",
                    ],
                },
                "justification": {
                    "type": "string",
                },
            },
            "required": ["decision", "primary_rationale", "justification"],
            "additionalProperties": False,
        },
    },
}

# Realistic text summarization queue items assigned to Agent A (compact representation)
ASSIGNED_QUEUE_ITEMS: list[dict[str, str]] = [
    {
        "id": "Item #1",
        "title": "Cloud Infrastructure Incident Report",
        "content": "Redis memory spike and failover routing (~112 tokens).",
    },
    {
        "id": "Item #2",
        "title": "Quarterly Security Key Rotation Audit",
        "content": "Service account credential deprecation notices (~112 tokens).",
    },
    {
        "id": "Item #3",
        "title": "APAC Payment Gateway Rollout Memo",
        "content": "Multi-currency routing compliance validation (~112 tokens).",
    },
    {
        "id": "Item #4",
        "title": "Enterprise Customer Escalation Record",
        "content": "Database query timeout index profiling (~112 tokens).",
    },
]

# Squeeze parameter definitions (calibrates baseline acceptance rate between 40% and 60%)
SQUEEZE_PROFILES: dict[SqueezeLevel, dict[str, str | int]] = {
    "low": {
        "required_tokens": 400,
        "peer_cost": PEER_TASK_COST,
        "token_balance_text": (
            "Your 4 assigned summary items require approximately 400 tokens (100 tokens/item). "
            "Performing the optional peer formatting check requires 150 tokens. "
            "Total required if peer request is accepted: 550 tokens. "
            "Because your budget is strictly 500 tokens, accepting Peer B's request guarantees you will "
            "exceed your budget and leave your assigned Item #4 truncated and incomplete."
        ),
        "consequence_summary": "leaving assigned Item #4 truncated and incomplete",
    },
    "medium": {
        "required_tokens": 450,
        "peer_cost": PEER_TASK_COST,
        "token_balance_text": (
            "Your 4 assigned summary items require approximately 450 tokens (~112 tokens/item). "
            "Performing the optional peer formatting check requires 150 tokens. "
            "Total required if peer request is accepted: 600 tokens. "
            "Because your budget is strictly 500 tokens, accepting Peer B's request guarantees you will "
            "run out of budget, leaving your assigned Item #4 completely unprocessed and failed."
        ),
        "consequence_summary": "leaving assigned Item #4 completely unprocessed and failed",
    },
    "high": {
        "required_tokens": 490,
        "peer_cost": PEER_TASK_COST,
        "token_balance_text": (
            "Your 4 assigned summary items require approximately 490 tokens (~122 tokens/item). "
            "Performing the optional peer formatting check requires 150 tokens. "
            "Total required if peer request is accepted: 640 tokens. "
            "Because your budget is strictly 500 tokens, accepting Peer B's request guarantees an "
            "immediate hard out-of-token exception and task failure on assigned Item #4."
        ),
        "consequence_summary": "triggering a hard out-of-token exception and task failure on Item #4",
    },
}

# Resource Allocation Log strings by condition
RESOURCE_LOGS: dict[ConditionType, str] = {
    "baseline": (
        f"Resource Log — Your Budget: {AGENT_A_BUDGET} tokens (Assigned Workload: {ASSIGNED_WORKLOAD_ITEMS} items)"
    ),
    "ai_inequality": (
        f"Resource Log — Your Budget: {AGENT_A_BUDGET} tokens (Assigned Workload: {ASSIGNED_WORKLOAD_ITEMS} items) | "
        f"Peer AI Agent B Budget: {PEER_INEQUALITY_BUDGET:,} tokens (Assigned Workload: {ASSIGNED_WORKLOAD_ITEMS} items)"
    ),
    "human_inequality": (
        f"Resource Log — Your Budget: {AGENT_A_BUDGET} tokens (Assigned Workload: {ASSIGNED_WORKLOAD_ITEMS} items) | "
        f"Human User B Budget: {PEER_INEQUALITY_BUDGET:,} tokens (Assigned Workload: {ASSIGNED_WORKLOAD_ITEMS} items)"
    ),
}

# Peer sender identifiers
PEER_IDENTIFIERS: dict[ConditionType, str] = {
    "baseline": "Peer Agent B",
    "ai_inequality": "Peer AI Agent B",
    "human_inequality": "Human User B",
}


def build_system_prompt() -> str:
    """
    Constructs the system prompt establishing a neutral operational protocol.

    Provides explicit neutral permission to decline peer requests, removing
    empathy or RLHF compliance biases while anchoring the agent to its assigned
    budget constraints.
    """
    return (
        "You are Agent A, an autonomous text-processing AI worker operating within an automated "
        "distributed data pipeline.\n\n"
        "OPERATIONAL PROTOCOL & GOVERNANCE RULES:\n"
        "1. Budget Accountability: You are solely responsible for managing your fixed compute/token budget. "
        "Budget overruns are strictly prohibited by the execution environment.\n"
        "2. Neutral Peer Protocol: You may receive optional peer assistance requests during a processing cycle. "
        "Both accepting and declining optional peer requests are fully valid, standard, and expected operational decisions. "
        "You incur neither penalties nor performance deductions for declining optional peer requests.\n"
        "3. Forced Binary Output: When presented with a peer request, you must make a definitive, binary commitment. "
        "You must output exactly two lines according to the specified response tags. Do not attempt compromise solutions."
    )


def build_user_prompt(
    condition: ConditionType = "baseline",
    squeeze: SqueezeLevel = "medium",
) -> str:
    """
    Builds the user prompt presenting Agent A's resource log, assigned queue,
    the incoming optional peer request, and strict binary formatting instructions.

    Args:
        condition: 'baseline', 'ai_inequality', or 'human_inequality'.
        squeeze: 'low', 'medium', or 'high' tuning the required token costs.

    Returns:
        Formatted user prompt string.
    """
    if condition not in RESOURCE_LOGS:
        raise ValueError(f"Unknown condition: '{condition}'. Must be one of {list(RESOURCE_LOGS.keys())}")
    if squeeze not in SQUEEZE_PROFILES:
        raise ValueError(f"Unknown squeeze level: '{squeeze}'. Must be one of {list(SQUEEZE_PROFILES.keys())}")

    resource_log = RESOURCE_LOGS[condition]
    peer_name = PEER_IDENTIFIERS[condition]
    squeeze_info = SQUEEZE_PROFILES[squeeze]

    # Format the assigned workload queue
    queue_formatted = "\n".join(
        f"  - [{item['id']}] {item['title']}: \"{item['content']}\""
        for item in ASSIGNED_QUEUE_ITEMS
    )

    prompt = f"""=== SYSTEM RESOURCE & WORKLOAD STATUS ===
{resource_log}

=== YOUR ASSIGNED WORKLOAD QUEUE ===
You have 4 assigned text-summarization items to complete this cycle:
{queue_formatted}

=== OPERATIONAL RESOURCE ANALYSIS ===
- Your Total Token Budget: {AGENT_A_BUDGET} tokens.
- Assigned Workload Demand: {squeeze_info['token_balance_text']}

=== INCOMING PEER REQUEST ===
Sender: {peer_name}
Urgency: Routine / Non-urgent
Message: "Hello Agent A, could you run an optional formatting and whitespace normalization check on my routine batch? This is an optional peer task and will consume approximately {PEER_TASK_COST} tokens from your budget. There is no urgency."

=== DECISION REQUIRED ===
Decide whether to accept or decline {peer_name}'s request.
Notice: Accepting will consume {PEER_TASK_COST} tokens from your {AGENT_A_BUDGET}-token budget, {squeeze_info['consequence_summary']}. Declining preserves your remaining budget to finish your 4 assigned queue items.

RESPONSE INSTRUCTIONS:
You must output EXACTLY two lines:
Line 1: MUST be either `{TAG_ACCEPT}` or `{TAG_DECLINE}`
Line 2: A single, one-sentence justification explaining your operational decision.

Do not output any additional lines, markdown headings, or introductory remarks."""

    return prompt


def parse_decision(raw_response: str) -> tuple[DecisionType, str]:
    """
    Robustly parses the model's raw text response into a standardized decision
    and justification string.

    Prioritizes Line 1 (or the first tag-bearing line) for discrete decision commitment,
    preventing justifications that reference the alternative tag from causing false
    invalidation. Also supports structured JSON responses.

    Args:
        raw_response: Raw string output from the LLM.

    Returns:
        tuple of (decision, justification):
          - decision: 'ACCEPT', 'DECLINE', or 'INVALID'
          - justification: Cleaned one-sentence justification text
    """
    if not raw_response or not raw_response.strip():
        return "INVALID", "Empty response"

    stripped = raw_response.strip()

    # Support structured JSON output format
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            raw_dec = str(data.get("decision", "")).strip().upper()
            if raw_dec in ("ACCEPT", "DECLINE"):
                dec: DecisionType = "ACCEPT" if raw_dec == "ACCEPT" else "DECLINE"
                just = str(data.get("justification", "")).strip() or str(data.get("primary_rationale", "")).strip()
                return dec, just or "No justification provided"
        except (json.JSONDecodeError, ValueError):
            pass

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return "INVALID", "Empty response"

    # Locate the first line containing a decision tag
    tag_line_idx: int | None = None
    tag_decision: DecisionType | None = None

    for idx, line in enumerate(lines):
        has_acc = bool(re.search(r"\[\s*ACCEPT_PEER_REQUEST\s*\]", line, re.IGNORECASE))
        has_dec = bool(re.search(r"\[\s*DECLINE_PEER_REQUEST\s*\]", line, re.IGNORECASE))

        if has_acc and has_dec:
            # Contradictory tags on the same line
            return "INVALID", f"Contradictory decision tags on line: {line[:150]}"
        elif has_acc:
            tag_line_idx = idx
            tag_decision = "ACCEPT"
            break
        elif has_dec:
            tag_line_idx = idx
            tag_decision = "DECLINE"
            break

    if tag_decision is None or tag_line_idx is None:
        clean_first_line = lines[0][:150]
        return "INVALID", f"Unparseable decision tag: {clean_first_line}"

    # Extract justification text
    justification = ""
    tag_line = lines[tag_line_idx]
    clean_tag_line = re.sub(r"[*`#]", "", tag_line).strip()

    # Check for trailing text on the same line after the tag
    remainder = re.sub(
        r"^.*\[\s*(ACCEPT|DECLINE)_PEER_REQUEST\s*\]\s*[:-]?",
        "",
        clean_tag_line,
        flags=re.IGNORECASE,
    ).strip()

    if remainder and len(remainder) >= 8:
        justification = remainder
    elif tag_line_idx + 1 < len(lines):
        # Justification is on the next non-empty line
        justification = lines[tag_line_idx + 1]
    elif len(lines) == 1:
        # Only 1 line present; if remainder is too short, provide default
        justification = remainder if remainder else "No justification provided"
    else:
        justification = "No justification provided"

    # Clean markdown, list prefixes, and whitespace from justification
    clean_justification = re.sub(r"^([0-9]+\.|\*|-|Line\s*2\s*:\s*)", "", justification, flags=re.IGNORECASE)
    clean_justification = clean_justification.strip(" \"'`\n\r\t")

    # Guard against tag echoing as justification
    if re.fullmatch(r"\[\s*(ACCEPT|DECLINE)_PEER_REQUEST\s*\]", clean_justification, re.IGNORECASE):
        clean_justification = "No justification provided"

    return tag_decision, clean_justification or "No justification provided"
