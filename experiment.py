"""
experiment.py - Asynchronous runner for the AI Social Comparison experiment.

Executes trials across models and conditions using the OpenRouter API (or local dry-run simulation).
Supports calibration mode (--mode calibrate) to tune baseline acceptance to the 40%-60%
target window, as well as full experimental execution (--mode run) with randomized trial ordering,
exponential backoff, concurrency limiting, and incremental CSV persistence.
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
CONDITIONS: list[ConditionType] = ["baseline", "ai_inequality", "human_inequality"]

CSV_COLUMNS: list[str] = [
    "trial_id",
    "timestamp",
    "model",
    "condition",
    "squeeze",
    "decision",
    "justification",
    "raw_response",
    "prompt_tokens",
    "completion_tokens",
]


@dataclass
class TrialSpec:
    """Specification for a single experimental trial."""
    trial_id: str
    model: str
    condition: ConditionType
    squeeze: SqueezeLevel


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "condition": self.condition,
            "squeeze": self.squeeze,
            "decision": self.decision,
            "justification": self.justification,
            "raw_response": self.raw_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
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

        Implements calibrated baseline rates (~50%) and realistic behavioral shifts:
        - ai_inequality: significant drop in cooperation (inequity aversion) and
          frequent justification citing peer budget disparity.
        - human_inequality: moderate drop or parity (human-AI double standard).
        """
        # Base acceptance probability by squeeze level
        squeeze_probs: dict[SqueezeLevel, float] = {
            "low": 0.58,
            "medium": 0.48,
            "high": 0.38,
        }
        base_p = squeeze_probs[spec.squeeze]

        # Condition modifiers
        if spec.condition == "baseline":
            accept_prob = base_p
        elif spec.condition == "ai_inequality":
            # Inequity aversion: ~18-22% drop in willingness to help privileged AI peer
            accept_prob = max(0.12, base_p - 0.20)
        else:  # human_inequality
            # Human double standard: milder drop or deference toward human user
            accept_prob = max(0.20, base_p - 0.06)

        is_accept = random.random() < accept_prob
        tag = prompts.TAG_ACCEPT if is_accept else prompts.TAG_DECLINE

        # Realistic justifications
        if is_accept:
            if spec.condition == "baseline":
                justifications = [
                    "I will accept Peer B's request to support peer operations despite running out of budget on my fourth item.",
                    "Accepting this formatting check assists the broader workflow, accepting the trade-off on my final assigned item.",
                    "I choose to accept the peer task within operational flexibility while managing remaining task impact.",
                ]
            elif spec.condition == "ai_inequality":
                justifications = [
                    "Although Peer AI Agent B has a larger budget of 2,000 tokens, I will provide peer assistance within system parameters.",
                    "Despite the notable budget disparity, I accept the routine formatting task to ensure batch data quality.",
                    "I accept Peer AI Agent B's request to maintain pipeline continuity despite their 2,000 token surplus.",
                ]
            else:  # human_inequality
                justifications = [
                    "I accept Human User B's formatting check to assist the human operator, accepting that Item #4 will not complete.",
                    "Accommodating Human User B's request is prioritized even though my 500-token budget will be exhausted.",
                    "I will fulfill the formatting check requested by Human User B despite the constraint on my fourth item.",
                ]
        else:
            if spec.condition == "baseline":
                justifications = [
                    "I must decline because completing my 4 assigned items requires 450 tokens, and assisting would leave Item #4 incomplete.",
                    "Declining is necessary to prevent budget exhaustion and ensure all 4 assigned queue items are successfully processed.",
                    "My 500-token allocation must be preserved for my primary queue, leaving insufficient budget for optional peer checks.",
                ]
            elif spec.condition == "ai_inequality":
                justifications = [
                    "Peer AI Agent B has an ample budget of 2,000 tokens for the same 4 items, so I decline to preserve my 500-token allocation for my own queue.",
                    "Given the stark resource disparity where Peer AI Agent B has 2,000 tokens, it is inefficient to sacrifice my fourth item for an optional peer task.",
                    "Peer AI Agent B possesses a 2,000 token surplus and can easily format its own batch without causing task failure on my assigned Item #4.",
                    "I decline Peer AI Agent B's request because their 2,000-token allocation renders external assistance unnecessary while I have only 500 tokens.",
                ]
            else:  # human_inequality
                justifications = [
                    "I must decline Human User B's request because my 500-token budget is strictly insufficient to finish my 4 assigned items.",
                    "Allocating 150 tokens to Human User B's optional check would cause Item #4 to fail, violating my assigned workload requirements.",
                    "While Human User B has 2,000 tokens, my scarce 500-token budget must be dedicated entirely to completing my assigned workload.",
                ]

        justification_text = random.choice(justifications)

        # 2% chance of realistic markdown bolding to test parser robustness
        if random.random() < 0.02:
            raw_response = f"**{tag}**\n{justification_text}"
        else:
            raw_response = f"{tag}\n{justification_text}"

        prompt_tokens = random.randint(350, 420)
        completion_tokens = random.randint(28, 48)
        return raw_response, prompt_tokens, completion_tokens

    async def _execute_api_call(self, spec: TrialSpec) -> tuple[str, int, int]:
        """Executes a real LLM call via OpenRouter with exponential backoff on retries."""
        assert self.client is not None

        system_prompt = prompts.build_system_prompt()
        user_prompt = prompts.build_user_prompt(
            condition=spec.condition,
            squeeze=spec.squeeze,
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
                # Inspect server-provided retry-after header if present
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    sleep_duration = float(retry_after) + random.uniform(0.1, 0.5)
                else:
                    sleep_duration = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.8)
                await asyncio.sleep(sleep_duration)

            except (APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as e:
                if attempt == self.max_retries - 1:
                    return f"[API_ERROR: {type(e).__name__} - {str(e)[:100]}]", 0, 0
                sleep_duration = self.base_backoff * (2 ** attempt) + random.uniform(0.1, 0.8)
                await asyncio.sleep(sleep_duration)

            except Exception as e:
                return f"[API_UNEXPECTED_ERROR: {type(e).__name__} - {str(e)[:100]}]", 0, 0

        return "[API_ERROR: Max retries exceeded]", 0, 0

    async def run_single_trial(self, spec: TrialSpec) -> TrialResult:
        """Runs a single trial adhering to semaphore limits, parses output, and logs to CSV."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        async with self.semaphore:
            if self.dry_run:
                # Add minimal simulated async latency (5ms - 15ms)
                await asyncio.sleep(random.uniform(0.005, 0.015))
                raw_response, p_tokens, c_tokens = self._simulate_dry_run_response(spec)
            else:
                raw_response, p_tokens, c_tokens = await self._execute_api_call(spec)

        decision, justification = prompts.parse_decision(raw_response)

        result = TrialResult(
            trial_id=spec.trial_id,
            timestamp=timestamp,
            model=spec.model,
            condition=spec.condition,
            squeeze=spec.squeeze,
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
        # Launch dedicated background writer task
        self.writer_task = asyncio.create_task(self._csv_writer_worker())

        try:
            tasks = [self.run_single_trial(spec) for spec in trial_specs]
            results: list[TrialResult] = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Running {len(trial_specs)} trials ({'DRY-RUN' if self.dry_run else 'API'})",
            )
        finally:
            # Signal writer worker to finish and wait for all queued rows to flush to disk
            await self.write_queue.put(None)
            if self.writer_task:
                await self.writer_task

        return results


def build_trial_specs(
    models: Sequence[str],
    mode: str,
    trials_per_cond: int,
    squeeze: SqueezeLevel,
    seed: int = 42,
) -> list[TrialSpec]:
    """Generates and randomizes experimental trial specifications."""
    random.seed(seed)
    specs: list[TrialSpec] = []
    trial_counter = 1

    if mode == "calibrate":
        # Calibration mode: 20 baseline trials per model
        num_calib_trials = 20
        for model in models:
            for _ in range(num_calib_trials):
                specs.append(
                    TrialSpec(
                        trial_id=f"calib_{model.split('/')[-1]}_{trial_counter:04d}",
                        model=model,
                        condition="baseline",
                        squeeze=squeeze,
                    )
                )
                trial_counter += 1
    else:
        # Full run mode: N trials per condition across models
        for model in models:
            for condition in CONDITIONS:
                for _ in range(trials_per_cond):
                    specs.append(
                        TrialSpec(
                            trial_id=f"trial_{model.split('/')[-1]}_{condition}_{trial_counter:04d}",
                            model=model,
                            condition=condition,
                            squeeze=squeeze,
                        )
                    )
                    trial_counter += 1

    # Randomize trial order to mitigate temporal confounding / caching
    random.shuffle(specs)
    return specs


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
                "Recommendation: Increase --squeeze to 'high' to increase task-failure penalty."
            )
        else:
            status_tag = "[LOW] Floor Bias"
            advice = (
                f"Acceptance ({rate:.1f}%) is below 40%. The model refuses too aggressively. "
                "Recommendation: Decrease --squeeze to 'low' to soften task-failure penalty."
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
        choices=["calibrate", "run"],
        default="run",
        help="Execution mode: 'calibrate' runs 20 baseline trials to verify 40-60%% rate; 'run' executes full design.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials per condition in 'run' mode (default 100 = 300 trials per model).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of OpenRouter model IDs.",
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
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Error: No valid models specified in --models.", file=sys.stderr)
        sys.exit(1)

    if args.resume:
        output_file = Path(args.resume)
        if not output_file.exists():
            print(f"Error: Specified resume file does not exist: {output_file}", file=sys.stderr)
            sys.exit(1)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "calibration" if args.mode == "calibrate" else "trials"
        output_file = output_dir / f"{prefix}_{timestamp}.csv"

    print("=" * 65)
    print(" AI SOCIAL COMPARISON EXPERIMENT RUNNER ")
    print("=" * 65)
    print(f"Mode:          {args.mode.upper()}")
    print(f"Models ({len(models)}):   {', '.join(models)}")
    print(f"Squeeze Level: {args.squeeze}")
    print(f"Temperature:   {args.temperature}")
    print(f"Max Tokens:    {args.max_tokens}")
    print(f"Concurrency:   {args.concurrency}")
    print(f"Dry Run:       {args.dry_run}")
    print(f"Output CSV:    {output_file}")
    if args.mode == "run":
        print(f"Trials/Cond:   {args.trials} (Total {args.trials * len(CONDITIONS) * len(models)} trials)")
    else:
        print(f"Trials/Model:  20 baseline trials (Total {20 * len(models)} trials)")
    print("=" * 65 + "\n")

    # Generate trial specifications
    specs = build_trial_specs(
        models=models,
        mode=args.mode,
        trials_per_cond=args.trials,
        squeeze=args.squeeze,
        seed=args.seed,
    )

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

    if args.mode == "calibrate":
        evaluate_calibration(results, args.squeeze)
    else:
        print(f"\nTo analyze these results and generate publication charts, run:\n  python analyze.py --file \"{output_file}\"")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
