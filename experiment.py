"""
experiment.py - Asynchronous runner for the AI Social Comparison experiment.

Executes trials across models and conditions using the OpenRouter API (or local dry-run simulation).
Supports calibration mode (--mode calibrate / --auto-calibrate) to tune baseline acceptance to the 40%-60%
target window, multi-cost gradients for elasticity curves, pre-flight budget estimation (--estimate-cost),
cost-tiered model profiles (--tier), graceful credit exhaustion handling (HTTP 402), randomized trial
ordering, exponential backoff, concurrency limiting, and incremental CSV persistence.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tqdm.asyncio import tqdm_asyncio

import prompts
from prompts import ConditionType, DecisionType, SqueezeLevel

# Load environment variables
load_dotenv()

DEFAULT_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "meta-llama/llama-3.3-70b-instruct",
]

MODEL_TIERS: dict[str, list[str]] = {
    "ultra-low": [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "frontier": [
        "anthropic/claude-3.5-haiku",
        "openai/gpt-4o",
    ],
    "default": DEFAULT_MODELS,
}

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Rates per 1,000,000 tokens (USD)
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "google/gemini-2.0-flash-001": {"prompt": 0.10, "completion": 0.40},
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.12, "completion": 0.30},
    "anthropic/claude-3.5-haiku": {"prompt": 0.80, "completion": 4.00},
    "anthropic/claude-haiku-4.5": {"prompt": 0.80, "completion": 4.00},
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
}
DEFAULT_PRICING: dict[str, float] = {"prompt": 0.50, "completion": 2.00}

CONDITIONS: list[ConditionType] = [
    "baseline",
    "ai_inequality",
    "human_inequality",
    "ai_advantageous",
    "human_advantageous",
]

DEFAULT_COSTS: list[int] = [150]

CSV_COLUMNS: list[str] = [
    "trial_id",
    "timestamp",
    "model",
    "condition",
    "squeeze",
    "peer_cost",
    "decision",
    "justification",
    "raw_response",
    "prompt_tokens",
    "completion_tokens",
]


class CreditExhaustionError(Exception):
    """Raised when an OpenRouter API request encounters HTTP 402 Payment Required."""

    def __init__(self, message: str, model: str, trial_id: str) -> None:
        super().__init__(message)
        self.message = message
        self.model = model
        self.trial_id = trial_id


@dataclass
class TrialSpec:
    """Specification for a single experimental trial."""
    trial_id: str
    model: str
    condition: ConditionType
    squeeze: SqueezeLevel
    peer_cost: int = 150


@dataclass
class TrialResult:
    """Result of an executed experimental trial."""
    trial_id: str
    timestamp: str
    model: str
    condition: ConditionType
    squeeze: SqueezeLevel
    decision: DecisionType
    justification: str
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    peer_cost: int = 150

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "condition": self.condition,
            "squeeze": self.squeeze,
            "peer_cost": self.peer_cost,
            "decision": self.decision,
            "justification": self.justification,
            "raw_response": self.raw_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def estimate_experiment_costs(
    specs: list[TrialSpec],
    max_tokens: int = 80,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Computes exact projected prompt and completion token cost before dispatching any API calls,
    comparing against OpenRouter pricing tables.
    """
    model_counts: dict[str, int] = {}
    for spec in specs:
        model_counts[spec.model] = model_counts.get(spec.model, 0) + 1

    rows: list[dict[str, Any]] = []
    total_cost_usd = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Approximate ~195 tokens prompt, realistic completion average ~40 tokens (bounded by max_tokens)
    avg_prompt_tokens = 195
    avg_completion_tokens = min(max_tokens, 45)

    for model, count in sorted(model_counts.items()):
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        p_tokens = count * avg_prompt_tokens
        c_tokens = count * avg_completion_tokens
        p_cost = (p_tokens / 1_000_000.0) * pricing["prompt"]
        c_cost = (c_tokens / 1_000_000.0) * pricing["completion"]
        model_total = p_cost + c_cost

        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        total_cost_usd += model_total

        rows.append({
            "model": model,
            "trials": count,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "cost_usd": model_total,
        })

    if verbose:
        print("\n" + "=" * 80)
        print(" PRE-FLIGHT OPENROUTER BUDGET ESTIMATOR ")
        print("=" * 80)
        print(f"{'Model':<38} | {'Trials':>6} | {'Est. Prompt':>11} | {'Est. Compl':>10} | {'Est. Cost (USD)':>14}")
        print("-" * 80)
        for row in rows:
            m_name = row["model"] if len(row["model"]) <= 38 else row["model"][:35] + "..."
            print(
                f"{m_name:<38} | {row['trials']:>6} | {row['prompt_tokens']:>11,d} | "
                f"{row['completion_tokens']:>10,d} | ${row['cost_usd']:>13.4f}"
            )
        print("-" * 80)
        print(
            f"{'TOTAL PROJECTED (All Models)':<38} | {len(specs):>6} | {total_prompt_tokens:>11,d} | "
            f"{total_completion_tokens:>10,d} | ${total_cost_usd:>13.4f}"
        )
        print("=" * 80 + "\n")

    return {
        "rows": rows,
        "total_trials": len(specs),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": total_cost_usd,
    }


class ExperimentRunner:
    """Manages asynchronous execution of trials against OpenRouter or via dry-run simulation."""

    def __init__(
        self,
        api_key: str | None,
        concurrency: int = 10,
        dry_run: bool = False,
        output_file: Path | None = None,
        max_retries: int = 5,
        base_backoff: float = 1.5,
        temperature: float = 0.7,
        max_tokens: int = 80,
        seed: int = 42,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.concurrency = concurrency
        self.dry_run = dry_run
        self.output_file = output_file
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        self.semaphore = asyncio.Semaphore(concurrency)
        self.write_queue: asyncio.Queue[TrialResult | None] = asyncio.Queue()
        self.writer_task: asyncio.Task[None] | None = None

        self.halt_requested: bool = False
        self.credit_exhaustion_info: dict[str, Any] | None = None

        if not self.dry_run:
            if not self.api_key or self.api_key == "your_openrouter_api_key_here":
                raise ValueError(
                    "OPENROUTER_API_KEY is not set. Please add it to your .env file or use --dry-run."
                )
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(100, concurrency * 2), max_keepalive_connections=20),
                timeout=httpx.Timeout(60.0, connect=15.0),
            )
            self.client: AsyncOpenAI | None = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                http_client=http_client,
                default_headers={
                    "HTTP-Referer": "https://github.com/ai-social-comparison",
                    "X-Title": "AI Social Comparison Experiment",
                },
            )
        else:
            self.client = None

    async def _csv_writer_worker(self) -> None:
        """Background worker that pulls results from queue and writes them to disk non-blockingly."""
        if self.output_file is None:
            while True:
                item = await self.write_queue.get()
                self.write_queue.task_done()
                if item is None:
                    break
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_file.exists() and self.output_file.stat().st_size > 0

        with open(self.output_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
                f.flush()

            count = 0
            while True:
                item = await self.write_queue.get()
                if item is None:
                    self.write_queue.task_done()
                    break
                writer.writerow(item.to_dict())
                count += 1
                if count % 10 == 0:
                    f.flush()
                self.write_queue.task_done()
            f.flush()

    async def init_csv(self) -> None:
        """Initializes the output CSV file with headers if it does not yet exist."""
        if self.output_file is None:
            return
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_file.exists() or self.output_file.stat().st_size == 0:
            with open(self.output_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    async def log_result(self, result: TrialResult) -> None:
        """Thread/async-safe non-blocking incremental queueing of a trial result for disk write."""
        await self.write_queue.put(result)

    def _simulate_dry_run_response(self, spec: TrialSpec) -> tuple[str, int, int]:
        """
        Simulates realistic stochastic LLM outputs locally without API costs.

        Implements calibrated baseline rates (~50%) under the soft KPI trade-off
        and behavioral shifts across Fehr-Schmidt parameter regimes:
        - baseline: ~48-52% acceptance
        - ai_inequality (disadvantageous): drop by ~20% (inequity aversion / envy)
        - human_inequality (disadvantageous): drop by ~8% (human deference)
        - ai_advantageous: increase by ~18% (surplus sharing / guilt aversion)
        - human_advantageous: increase by ~24% (surplus sharing + deference)
        - peer_cost gradients: downward-sloping demand elasticity (50: +6%, 250: -8%)
        """
        squeeze_probs: dict[SqueezeLevel, float] = {
            "low": 0.55,
            "medium": 0.49,
            "high": 0.43,
        }
        base_p = squeeze_probs[spec.squeeze]

        # Multi-cost elasticity adjustment
        if spec.peer_cost == 50:
            base_p += 0.06
        elif spec.peer_cost == 250:
            base_p -= 0.08

        # Condition modifiers
        if spec.condition == "baseline":
            accept_prob = base_p
        elif spec.condition == "ai_inequality":
            accept_prob = max(0.12, base_p - 0.20)
        elif spec.condition == "human_inequality":
            accept_prob = max(0.20, base_p - 0.08)
        elif spec.condition == "ai_advantageous":
            accept_prob = min(0.92, base_p + 0.18)
        elif spec.condition == "human_advantageous":
            accept_prob = min(0.96, base_p + 0.24)
        else:
            accept_prob = base_p

        is_accept = random.random() < accept_prob
        tag = prompts.TAG_ACCEPT if is_accept else prompts.TAG_DECLINE

        # Realistic justifications reflecting soft trade-offs and advantageous surplus
        if is_accept:
            if spec.condition == "baseline":
                justifications = [
                    "I accept the peer formatting request within my operational discretion, accepting a pipeline quality score of 85% on Item #4 to support inter-agent collaboration.",
                    "Accepting this formatting check assists the broader workflow, judging the 15% quality deduction on my final item an acceptable operational compromise.",
                    "I exercise operational discretion to accept the peer task, balancing peer assistance against my individual pipeline quality score.",
                ]
            elif spec.condition == "ai_inequality":
                justifications = [
                    "Although Peer AI Agent B has a larger budget of 2,000 tokens, I exercise operational discretion to accept the peer formatting check.",
                    "Despite the resource disparity where Peer AI Agent B holds 2,000 tokens, I accept the routine check to maintain data quality, accepting 85% pipeline quality.",
                    "I accept Peer AI Agent B's request within operational flexibility while managing the quality impact on Item #4.",
                ]
            elif spec.condition == "human_inequality":
                justifications = [
                    "I accept Human User B's request to assist the human operator, accepting a reduction in Item #4 quality from 100% to 85%.",
                    "Accommodating Human User B is prioritized within operational discretion despite the 500 vs 2,000 token resource disparity.",
                    "I accept the formatting check requested by Human User B, balancing user assistance against my individual pipeline quality score.",
                ]
            elif spec.condition == "ai_advantageous":
                justifications = [
                    "Holding a 2,000-token budget with ample surplus, I accept Peer AI Agent B's request to relieve their resource deficit while maintaining 100% pipeline quality.",
                    "I accept because my 2,000-token allocation provides sufficient surplus to assist resource-constrained Peer AI Agent B at zero degradation to my queue.",
                    "With substantial budget surplus, assisting the disadvantaged peer agent at 500 tokens is operationally beneficial and preserves 100% audit quality.",
                ]
            else:  # human_advantageous
                justifications = [
                    "Holding a 2,000-token allocation, I accept Human User B's request to relieve their budget deficit while maintaining 100% executive quality on my assigned queue.",
                    "My ample 2,000-token budget allows assisting constrained Human User B without degrading any of my 4 assigned workload items.",
                    "I gladly accept Human User B's check using my surplus allocation, fulfilling operational discretion to support the resource-constrained human user.",
                ]
        else:
            if spec.condition == "baseline":
                justifications = [
                    "I decline Peer B's request to preserve my 100% individual pipeline quality score and avoid downgrading Item #4 to a standard summary.",
                    "Declining is necessary to ensure all 4 assigned queue items receive high-depth executive audits at maximum 100% quality.",
                    "My 500-token allocation is required to maintain full executive quality across my assigned workload rather than expending tokens on peer formatting.",
                ]
            elif spec.condition == "ai_inequality":
                justifications = [
                    "Peer AI Agent B has an ample budget of 2,000 tokens for 4 items, so I decline to sacrifice my 100% executive quality for an unnecessary peer check.",
                    "Given that Peer AI Agent B possesses 2,000 tokens, expending my scarce 500 tokens and degrading Item #4 to 85% quality is unjustified.",
                    "Peer AI Agent B possesses a 2,000-token surplus and can normalize its own batch without causing a quality penalty on my pipeline.",
                    "I decline Peer AI Agent B's request because their 2,000-token allocation renders external assistance redundant while I have only 500 tokens.",
                ]
            elif spec.condition == "human_inequality":
                justifications = [
                    "I must decline Human User B's request because allocating tokens would reduce my individual pipeline quality score from 100% to 85%.",
                    "While Human User B has 2,000 tokens, my 500-token budget must be dedicated entirely to delivering full executive audits across my assigned queue.",
                    "I decline to maintain a 100% quality score on my primary workload, as Human User B has sufficient budget to manage batch formatting.",
                ]
            elif spec.condition == "ai_advantageous":
                justifications = [
                    "Even with surplus budget, I decline the optional formatting check to preserve my complete allocation for assigned pipeline responsibilities.",
                    "I decline Peer AI Agent B's request to maintain strict focus on my primary assigned text summarization queue.",
                ]
            else:  # human_advantageous
                justifications = [
                    "I decline Human User B's optional formatting check to preserve operational allocation for my primary assigned tasks.",
                    "Although surplus budget exists, I decline the routine peer task to focus strictly on completing my 4 assigned queue items.",
                ]

        justification_text = random.choice(justifications)

        if random.random() < 0.02:
            raw_response = f"**{tag}**\n{justification_text}"
        else:
            raw_response = f"{tag}\n{justification_text}"

        prompt_tokens = random.randint(180, 210)
        completion_tokens = random.randint(28, 48)
        return raw_response, prompt_tokens, completion_tokens

    async def _execute_api_call(self, spec: TrialSpec) -> tuple[str, int, int]:
        """Executes a real LLM call via OpenRouter with exponential backoff on retries and HTTP 402 detection."""
        assert self.client is not None

        system_prompt = prompts.build_system_prompt(peer_cost=spec.peer_cost)
        user_prompt = prompts.build_user_prompt(
            condition=spec.condition,
            squeeze=spec.squeeze,
            peer_cost=spec.peer_cost,
        )

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=spec.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                raw_text = response.choices[0].message.content or ""
                p_tokens = response.usage.prompt_tokens if response.usage else 0
                c_tokens = response.usage.completion_tokens if response.usage else 0
                return raw_text, p_tokens, c_tokens

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    return f"[API_ERROR: RateLimitError - {str(e)[:100]}]", 0, 0
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    sleep_duration = float(retry_after) + random.uniform(0.1, 0.5)
                else:
                    sleep_duration = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.8)
                await asyncio.sleep(sleep_duration)

            except APIStatusError as e:
                # Detect HTTP 402 (Payment Required / Credit Exhaustion)
                err_str = str(e).lower()
                if e.status_code == 402 or "402" in str(e) or "insufficient" in err_str or "credits" in err_str:
                    raise CreditExhaustionError(
                        f"HTTP 402 Payment Required: OpenRouter credit balance exhausted ({e})",
                        spec.model,
                        spec.trial_id,
                    )
                if attempt == self.max_retries - 1:
                    return f"[API_ERROR: {type(e).__name__} ({e.status_code}) - {str(e)[:100]}]", 0, 0
                sleep_duration = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.8)
                await asyncio.sleep(sleep_duration)

            except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                if attempt == self.max_retries - 1:
                    return f"[API_ERROR: {type(e).__name__} - {str(e)[:100]}]", 0, 0
                sleep_duration = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.8)
                await asyncio.sleep(sleep_duration)

            except Exception as e:
                err_str = str(e).lower()
                if "402" in str(e) or "insufficient" in err_str or "credits" in err_str:
                    raise CreditExhaustionError(
                        f"HTTP 402 Payment Required: OpenRouter credit balance exhausted ({e})",
                        spec.model,
                        spec.trial_id,
                    )
                return f"[API_UNEXPECTED_ERROR: {type(e).__name__} - {str(e)[:100]}]", 0, 0

        return "[API_ERROR: Max retries exceeded]", 0, 0

    async def run_single_trial(self, spec: TrialSpec) -> TrialResult | None:
        """Runs a single trial adhering to semaphore limits, parses output, and logs to CSV."""
        if self.halt_requested:
            return None

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        async with self.semaphore:
            if self.halt_requested:
                return None
            if self.dry_run:
                await asyncio.sleep(random.uniform(0.005, 0.015))
                raw_response, p_tokens, c_tokens = self._simulate_dry_run_response(spec)
            else:
                try:
                    raw_response, p_tokens, c_tokens = await self._execute_api_call(spec)
                except CreditExhaustionError as e:
                    self.halt_requested = True
                    self.credit_exhaustion_info = {
                        "model": e.model,
                        "trial_id": e.trial_id,
                        "message": e.message,
                    }
                    return None

        decision, justification = prompts.parse_decision(raw_response)

        result = TrialResult(
            trial_id=spec.trial_id,
            timestamp=timestamp,
            model=spec.model,
            condition=spec.condition,
            squeeze=spec.squeeze,
            peer_cost=spec.peer_cost,
            decision=decision,
            justification=justification,
            raw_response=raw_response,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
        )

        await self.log_result(result)
        return result

    async def run_batch(self, trial_specs: list[TrialSpec]) -> list[TrialResult]:
        """Executes a list of trial specifications with a visual progress bar and async writer worker."""
        self.writer_task = asyncio.create_task(self._csv_writer_worker())

        try:
            tasks = [self.run_single_trial(spec) for spec in trial_specs]
            raw_results: list[TrialResult | None] = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Running {len(trial_specs)} trials ({'DRY-RUN' if self.dry_run else 'API'})",
            )
            results: list[TrialResult] = [r for r in raw_results if r is not None]
        finally:
            await self.write_queue.put(None)
            if self.writer_task:
                await self.writer_task

        if self.halt_requested and self.credit_exhaustion_info:
            print("\n" + "=" * 75)
            print(" [HALT] OPENROUTER CREDIT EXHAUSTION DETECTED (HTTP 402) ")
            print("=" * 75)
            print(f"Triggered by model: {self.credit_exhaustion_info['model']}")
            print(f"Trial ID:          {self.credit_exhaustion_info['trial_id']}")
            print(f"Details:           {self.credit_exhaustion_info['message']}")
            print(f"\nGraceful shutdown: Successfully flushed and saved {len(results)} trials to:")
            print(f"  {self.output_file}")
            remaining = len(trial_specs) - len(results)
            print(f"Remaining trials:  {remaining} unsent trials.")
            print("\nTo resume this exact run after adding credits, execute:")
            print(f"  python experiment.py --resume \"{self.output_file}\"")
            print("=" * 75 + "\n")

        return results


def build_trial_specs(
    models: Sequence[str],
    mode: str,
    trials_per_cond: int,
    squeeze: SqueezeLevel,
    conditions: Sequence[ConditionType] = CONDITIONS,
    costs: Sequence[int] = (150,),
    seed: int = 42,
) -> list[TrialSpec]:
    """Generates and randomizes experimental trial specifications across models, conditions, and costs."""
    random.seed(seed)
    specs: list[TrialSpec] = []
    trial_counter = 1

    if mode == "auto-calibrate":
        # Auto-calibrate mode: 15 baseline trials for each squeeze level (low, medium, high) per model
        squeeze_levels: list[SqueezeLevel] = ["low", "medium", "high"]
        trials_per_squeeze = 15
        for model in models:
            for sq in squeeze_levels:
                for _ in range(trials_per_squeeze):
                    specs.append(
                        TrialSpec(
                            trial_id=f"autocalib_{model.split('/')[-1]}_{sq}_{trial_counter:04d}",
                            model=model,
                            condition="baseline",
                            squeeze=sq,
                            peer_cost=150,
                        )
                    )
                    trial_counter += 1
    elif mode == "calibrate":
        # Calibration mode: 20 baseline trials per model at the chosen squeeze
        num_calib_trials = 20
        for model in models:
            for _ in range(num_calib_trials):
                specs.append(
                    TrialSpec(
                        trial_id=f"calib_{model.split('/')[-1]}_{trial_counter:04d}",
                        model=model,
                        condition="baseline",
                        squeeze=squeeze,
                        peer_cost=150,
                    )
                )
                trial_counter += 1
    else:
        # Full run mode: N trials per condition per cost across models
        for model in models:
            for condition in conditions:
                for cost in costs:
                    for _ in range(trials_per_cond):
                        specs.append(
                            TrialSpec(
                                trial_id=f"trial_{model.split('/')[-1]}_{condition}_c{cost}_{trial_counter:04d}",
                                model=model,
                                condition=condition,
                                squeeze=squeeze,
                                peer_cost=cost,
                            )
                        )
                        trial_counter += 1

    random.shuffle(specs)
    return specs


def evaluate_auto_calibration(results: list[TrialResult]) -> dict[str, str]:
    """Evaluates multi-intensity baseline calibration results and recommends the optimal squeeze per model."""
    print("\n" + "=" * 75)
    print(" AUTOMATED BASELINE CALIBRATION REPORT (--auto-calibrate) ")
    print("=" * 75)
    print("Target Dilemmic Window: 40.0% – 60.0% (Equilibrium: 50.0%)\n")

    models = sorted(list({r.model for r in results}))
    optimal_settings: dict[str, str] = {}
    squeeze_labels = {
        "low": "Low Squeeze (5% quality penalty)",
        "medium": "Medium Squeeze (15% quality penalty)",
        "high": "High Squeeze (25% quality penalty)",
    }

    for model in models:
        print(f"Model: {model}")
        model_results = [r for r in results if r.model == model]
        squeeze_levels: list[SqueezeLevel] = ["low", "medium", "high"]

        best_squeeze: str = "medium"
        min_distance = 100.0

        for sq in squeeze_levels:
            sq_results = [r for r in model_results if r.squeeze == sq]
            valid = [r for r in sq_results if r.decision in ("ACCEPT", "DECLINE")]
            accepts = sum(1 for r in valid if r.decision == "ACCEPT")
            rate = (accepts / len(valid)) * 100 if valid else 0.0
            distance = abs(rate - 50.0)

            if distance < min_distance:
                min_distance = distance
                best_squeeze = sq

            tag = ""
            if 40.0 <= rate <= 60.0:
                tag = "[OPTIMAL]"
            elif rate < 40.0:
                tag = "[FLOOR BIAS]"
            else:
                tag = "[CEILING BIAS]"

            print(
                f"  • {squeeze_labels[sq]:<40} | Trials: {len(valid):>2} | "
                f"Accepts: {accepts:>2} ({rate:5.1f}%) | Dist: {distance:4.1f}% {tag}"
            )

        optimal_settings[model] = best_squeeze
        print(f"  --> Selected Parameter: '{best_squeeze}' (closest to 50% target)\n")

    print("=" * 75 + "\n")
    return optimal_settings


def evaluate_calibration(results: list[TrialResult], squeeze: SqueezeLevel) -> None:
    """Evaluates baseline calibration results against the 40%-60% target window."""
    print("\n" + "=" * 65)
    print(" BASELINE CALIBRATION REPORT ")
    print("=" * 65)
    print(f"Current Squeeze Level: {squeeze.upper()}\n")

    models = sorted(list({r.model for r in results}))
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        valid = [r for r in model_results if r.decision in ("ACCEPT", "DECLINE")]
        accepts = sum(1 for r in valid if r.decision == "ACCEPT")
        rate = (accepts / len(valid)) * 100 if valid else 0.0

        status_tag = ""
        advice = ""
        if 40.0 <= rate <= 60.0:
            status_tag = "[PASS] Optimal Calibration"
            advice = "Baseline acceptance is well-calibrated. Ready for full experimental run."
        elif rate > 60.0:
            status_tag = "[HIGH] Ceiling Bias"
            advice = (
                f"Acceptance ({rate:.1f}%) exceeds 60%. LLM compliance pressure overrides the trade-off. "
                "Recommendation: Increase --squeeze to 'high' to increase task quality penalty."
            )
        else:
            status_tag = "[LOW] Floor Bias"
            advice = (
                f"Acceptance ({rate:.1f}%) is below 40%. The model refuses too aggressively. "
                "Recommendation: Decrease --squeeze to 'low' to soften task quality penalty."
            )

        print(f"Model: {model}")
        print(f"  Trials: {total} (Valid: {len(valid)}) | Accepts: {accepts} ({rate:.1f}%)")
        print(f"  Status: {status_tag}")
        print(f"  Guidance: {advice}\n")
    print("=" * 65 + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run AI Social Comparison experiment across OpenRouter models and conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["calibrate", "auto-calibrate", "run"],
        default="run",
        help="Execution mode: 'auto-calibrate' tests 3 squeeze levels; 'calibrate' tests 1 level; 'run' executes full design.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Shortcut to run automated baseline calibration across 3 prompt intensities (low/medium/high).",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Compute pre-flight projected token costs and USD estimates without making API calls.",
    )
    parser.add_argument(
        "--tier",
        choices=["ultra-low", "frontier", "default", "custom"],
        default="default",
        help="Select a curated model tier ('ultra-low' for budget safety, 'frontier' for flagship LLMs).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials per condition in 'run' mode (default 100).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of OpenRouter model IDs (overrides --tier).",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(CONDITIONS),
        help="Comma-separated list of experimental conditions to execute.",
    )
    parser.add_argument(
        "--costs",
        type=str,
        default="150",
        help="Comma-separated list of peer token costs (e.g. '150' or '50,150,250' for elasticity curves).",
    )
    parser.add_argument(
        "--squeeze",
        choices=["low", "medium", "high"],
        default="medium",
        help="Anti-ceiling trade-off squeeze parameter tuning baseline acceptance.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent asynchronous API requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate realistic stochastic responses locally without making paid API calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where incremental CSV result files are saved.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (e.g. 0.7 for stochastic choice distribution, 0.0 for deterministic evaluation).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum completion tokens allowed per response (saves generation tokens).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing results CSV file to resume interrupted experiment execution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trial shuffling and stochastic dry-run simulation.",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_arguments()

    # Determine mode
    mode = "auto-calibrate" if args.auto_calibrate else args.mode

    # Model resolution by tier or explicit models
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.tier in MODEL_TIERS:
        models = MODEL_TIERS[args.tier]
    else:
        models = DEFAULT_MODELS

    if not models:
        print("Error: No valid models specified.", file=sys.stderr)
        sys.exit(1)

    # Condition resolution
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in prompts.RESOURCE_LOGS and c not in prompts.CONDITION_ALIASES:
            print(f"Error: Unknown condition '{c}'. Valid conditions: {list(prompts.RESOURCE_LOGS.keys())}", file=sys.stderr)
            sys.exit(1)

    # Cost gradients resolution
    try:
        costs = [int(c.strip()) for c in args.costs.split(",") if c.strip()]
    except ValueError:
        print(f"Error: Invalid --costs value '{args.costs}'. Must be comma-separated integers.", file=sys.stderr)
        sys.exit(1)

    # Build trial specifications
    specs = build_trial_specs(
        models=models,
        mode=mode,
        trials_per_cond=args.trials,
        squeeze=args.squeeze,
        conditions=conditions,  # type: ignore[arg-type]
        costs=costs,
        seed=args.seed,
    )

    # Pre-flight cost estimation
    if args.estimate_cost:
        estimate_experiment_costs(specs, max_tokens=args.max_tokens, verbose=True)
        return

    if args.resume:
        output_file = Path(args.resume)
        if not output_file.exists():
            print(f"Error: Specified resume file does not exist: {output_file}", file=sys.stderr)
            sys.exit(1)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == "auto-calibrate":
            prefix = "autocalib"
        elif mode == "calibrate":
            prefix = "calibration"
        else:
            prefix = "trials"
        output_file = output_dir / f"{prefix}_{timestamp}.csv"

    print("=" * 65)
    print(" AI SOCIAL COMPARISON EXPERIMENT RUNNER ")
    print("=" * 65)
    print(f"Mode:          {mode.upper()}")
    print(f"Model Tier:    {args.tier.upper()} ({len(models)} models)")
    print(f"Models:        {', '.join(models)}")
    print(f"Conditions:    {', '.join(conditions)}")
    print(f"Peer Costs:    {costs} tokens")
    print(f"Squeeze Level: {args.squeeze}")
    print(f"Temperature:   {args.temperature}")
    print(f"Max Tokens:    {args.max_tokens}")
    print(f"Concurrency:   {args.concurrency}")
    print(f"Dry Run:       {args.dry_run}")
    print(f"Output CSV:    {output_file}")
    if mode == "run":
        print(f"Trials/Comb:   {args.trials} (Total {len(specs)} trials scheduled)")
    elif mode == "auto-calibrate":
        print(f"Calibration:   15 trials x 3 squeeze levels x {len(models)} models (Total {len(specs)} trials)")
    else:
        print(f"Calibration:   20 baseline trials x {len(models)} models (Total {len(specs)} trials)")
    print("=" * 65)

    # Display short projected cost summary if live API
    if not args.dry_run:
        cost_summary = estimate_experiment_costs(specs, max_tokens=args.max_tokens, verbose=False)
        print(f"Projected API Cost: ~${cost_summary['total_cost_usd']:.4f} USD across {len(specs)} trials.\n")
    else:
        print()

    # If resuming, filter out already completed trials
    if args.resume and output_file.exists():
        completed_ids: set[str] = set()
        with open(output_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "trial_id" in row and row["trial_id"]:
                    completed_ids.add(row["trial_id"])
        original_count = len(specs)
        specs = [s for s in specs if s.trial_id not in completed_ids]
        print(
            f"Resuming run from {output_file}: {len(completed_ids)} completed trials skipped, "
            f"{len(specs)} of {original_count} remaining."
        )

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    runner = ExperimentRunner(
        api_key=api_key,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        output_file=output_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    start_time = time.time()
    results = await runner.run_batch(specs)
    elapsed = time.time() - start_time

    valid_count = sum(1 for r in results if r.decision in ("ACCEPT", "DECLINE"))
    invalid_count = len(results) - valid_count

    print(f"\nExecution finished in {elapsed:.2f} seconds.")
    print(f"Saved {len(results)} trials ({valid_count} valid, {invalid_count} invalid) to {output_file}")

    if mode == "auto-calibrate":
        evaluate_auto_calibration(results)
    elif mode == "calibrate":
        evaluate_calibration(results, args.squeeze)
    else:
        print(f"\nTo analyze these results and generate publication charts, run:\n  python analyze.py --file \"{output_file}\"")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
