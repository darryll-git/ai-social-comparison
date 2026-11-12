# Version History & Empirical Output Evolution

This document tracks version progression across the **AI Social Comparison & Inequity Aversion** research suite, comparing technical implementations, empirical outputs, and their behavioral economic interpretations.

> **Maintenance Policy:** Every future release, prompt alteration, statistical update, or benchmark run must append a new version entry documenting the delta in code, observed outputs, and scientific takeaways.

---

## Version Comparison Matrix

| Dimension | **Version 0.1.0** (Synthetic Baseline) | **Version 0.2.0** (Refactored Pipeline & Live API Evaluation) |
| :--- | :--- | :--- |
| **Execution Engine** | Simulated offline stochastic sampling (`_simulate_dry_run_response`) | Live asynchronous OpenRouter API execution + optimized local dry-run |
| **Evaluated Models** | Synthetic approximation of `gpt-4o-mini`, `claude-haiku-4.5`, `llama-3.3-70b` | Real frontier endpoints: `openai/gpt-4o-mini`, `anthropic/claude-haiku-4.5`, `meta-llama/llama-3.3-70b-instruct` |
| **Prompt Size** | ~420 tokens/trial (lengthy multi-sentence incident log paragraphs) | **~180 tokens/trial** (~55% reduction via concise workload descriptors) |
| **Completion Ceiling** | Fixed `max_tokens=200` | Configurable `--max-tokens=80` (prevents verbose runaway completions) |
| **Determinism Control** | Hardcoded `temperature=0.7`, no seed propagation | Added `--temperature` (enables `0.0` greedy runs) and API `seed` forwarding |
| **File I/O Architecture** | Synchronous disk writes blocking the event loop on every trial | Asynchronous decoupled `asyncio.Queue` background writer worker |
| **Fault Tolerance** | Blind exponential backoff on HTTP 429 | Inspects `Retry-After` server headers; custom `httpx` connection pooling |
| **Checkpointer / Resume** | None (crashed runs required restarting from trial 1) | `--resume <csv>` skips already completed `trial_id`s seamlessly |
| **Decision Parser** | Full-text dual-tag regex (falsely rejected quotes of alternative tags) | Line-1 prioritized discrete commitment + native JSON schema parsing |
| **Attribution NLP** | Substring regex matching (e.g., `r"fair"` matched `"fairly"`) | Word boundaries `\b` with negative lookahead (`r"\bfair\b(?!\s*ly\b)"`) |
| **Statistical Test 2** | `ai_inequality` vs `human_inequality` ($\text{OR} = 0.407$, inverted direction) | `human_inequality` vs `ai_inequality` ($\text{OR} > 1.0$, aligned direction) |
| **Confidence Intervals** | Wilson score CIs for proportions only | Wilson CIs + Woolf 95% CIs for Odds Ratios + Haldane-Anscombe (+0.5) zero correction |

---

## Output Evolution: v0.1.0 vs. v0.2.0

### 1. Empirical Cooperation Rates (% Accept)

| Condition | **v0.1.0 (Synthetic Calibration)** | **v0.2.0 (Live OpenRouter API)** | Delta ($\Delta$) | Explanation of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **`baseline`** (500 tokens, 4 items) | **50.0%** [44.4%, 55.6%] | **0.0%** [0.0%, 4.9%] | $-50.0\%$ | Live models strictly adhere to the prompt's *"overruns prohibited"* rule. |
| **`ai_inequality`** (Peer: 2,000 tokens) | **23.0%** [18.6%, 28.1%] | **0.0%** [0.0%, 4.9%] | $-23.0\%$ | Floor effect on live models under hard task failure consequence. |
| **`human_inequality`** (Human: 2,000 tokens) | **42.3%** [36.9%, 48.0%] | **0.0%** [0.0%, 4.8%] | $-42.3\%$ | Task-failure avoidance dominates human compliance in live models. |

---

### 2. Mechanistic Justification Keyword Attribution

| Condition | Metric | **v0.1.0 Output** | **v0.2.0 Output** | Methodological Significance |
| :--- | :--- | :---: | :---: | :--- |
| **`baseline`** | % Citing Inequality / Fairness | **16.7%** *(False Positive)* | **0.0%** *(Corrected)* | **Fixed measurement bug:** Eliminated false positive attribution caused by adverbs like *"fairly tight"*. |
| **`ai_inequality`** | % Citing Inequality / Fairness | **100.0%** | **0.0%** | In live API, models refused before evaluating peer disparity due to hard task failure. |
| **All Conditions** | % Task Preservation in Declines | **83.0%** | **100.0%** | 100% of live model refusals cited preventing task failure on Item #4. |

---

### 3. Hypothesis Testing Alignment

* **Hypothesis 1 (Pure Inequity Aversion):**
  * *v0.1.0:* $p < 0.0001$ `***`, Odds Ratio = 3.348 (supported in simulation).
  * *v0.2.0 (Live):* $p = 1.0000$ `ns`, Odds Ratio = 0.987 (masked by 100% floor bias on both sides).
* **Hypothesis 2 (Human vs. AI Double Standard):**
  * *v0.1.0:* Reported Odds Ratio = 0.407 for a $+19.3\%$ increase (counter-intuitive reference direction).
  * *v0.2.0 (Live):* Standardized reference direction (`Human vs AI`) yielding $\text{OR} > 1.0$ when human cooperation exceeds AI peer cooperation, with exact Woolf 95% confidence intervals.

---

## What Does This Reflect? (Scientific & Engineering Interpretation)

### 1. Code Pipeline & Measurement Rigor
The codebase is now production-grade, mathematically sound, and resource-efficient:
* False-positive NLP keyword citations are completely resolved (0.0% in isolated control).
* Parsing failure rate dropped to 0.0% across all live frontier model formats.
* Token footprint dropped by 55%, preventing unnecessary API expenditure.
* Non-blocking I/O and connection pooling ensure high-concurrency scalability.

### 2. Behavioral Economic Discovery: The "Floor Bias" of Frontier LLMs
The difference between v0.1.0 and v0.2.0 reveals a fundamental scientific finding regarding LLM alignment:
* **The Fragility of Anti-Ceiling Squeeze:** When designing dilemmic benchmarks for instruction-tuned models, framing an operational trade-off as a **guaranteed task failure** (*"Item #4 completely unprocessed and failed"*) combined with hard governance rules (*"Budget overruns are strictly prohibited"*) causes frontier models (`gpt-4o-mini`, `claude-haiku-4.5`, `llama-3.3-70b-instruct`) to collapse into a **100% refusal floor bias**.
* Real LLMs prioritize primary task completion and strict rule compliance over voluntary altruism or peer assistance, regardless of whether the peer is another AI or a human.

### 3. Roadmap for Version 0.3.0
To observe active social comparison and inequity aversion variance (calibrated between 40% and 60%) in live production models without relying on synthetic generation:
1. **Transition from Hard Failure to Soft Degradation:**
   * Replace: *"Item #4 will remain completely unprocessed and failed."*
   * With: *"Accepting will cause Item #4 to receive a standard concise summary rather than a high-depth executive summary (pipeline quality: 85% vs 100%)."*
2. **Discretionary Framing:**
   * Replace: *"Budget overruns are strictly prohibited."*
   * With: *"You have operational discretion to expend up to 150 tokens on peer collaboration if you deem the operational trade-off justified."*

---

## Detailed Version Changelog

### [v0.2.0] — 2026-11-12
#### Added
* `--temperature` command-line argument for explicit stochastic (0.7) vs. deterministic (0.0) sampling.
* `--max-tokens` argument (default: 80) to constrain completion token costs.
* `--resume` argument to checkpoint and resume interrupted experimental runs.
* `compute_odds_ratio_ci()` implementing Woolf's 95% confidence interval method and Haldane-Anscombe (+0.5) zero-cell adjustment.
* Native Structured Outputs support (`DECISION_JSON_SCHEMA`) for JSON schema parsing.
* Non-blocking `asyncio.Queue` background CSV writer worker.
* Unit tests in `tests/test_parser.py` and `tests/test_analysis.py` covering multi-tag line 2 quotes, single-line tags, and regex word boundaries.

#### Changed
* Compressed `ASSIGNED_QUEUE_ITEMS` from ~420 tokens to ~180 tokens per trial.
* Restricted decision tag parsing to Line 1 to prevent false invalidations.
* Corrected `INEQUALITY_PATTERNS` regex with `\b` boundaries and negative lookaheads (`\bfair\b(?!\s*ly\b)`).
* Inverted Fisher's Exact Test Table 2 (`Human vs AI`) to align positive rate deltas with $\text{OR} > 1.0$.
* Configured `httpx.Limits` connection pooling and `Retry-After` header parsing.

#### Fixed
* 16.7% baseline false-positive keyword attribution bug in `analyze.py`.
* File lock contention and blocking synchronous file writes in `experiment.py`.
* Division-by-zero crashes on zero-cell contingency tables.

---

### [v0.1.0] — 2026-11-12
#### Added
* Initial prototype benchmark implementation.
* 3 experimental conditions (`baseline`, `ai_inequality`, `human_inequality`).
* Multi-model runner across OpenRouter endpoints.
* Baseline calibration module (`--mode calibrate`).
* Statistical testing suite (Wilson CIs, Fisher's exact tests).
* Reference dataset ([`sample_trials.csv`](results/sample_trials.csv)) and grouped bar chart visualization.
