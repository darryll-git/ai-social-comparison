# AI Social Comparison & Inequity Aversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenRouter](https://img.shields.io/badge/API-OpenRouter-7952b3.svg)](https://openrouter.ai/)

A behavioral-economics research testbench and empirical evaluation suite investigating **disadvantageous inequity aversion and social comparison** in Large Language Models (LLMs) via the OpenRouter API.

---

## Table of Contents
- [1. Research Overview](#1-research-overview)
- [2. Experimental Architecture](#2-experimental-architecture)
- [3. The 3 Experimental Conditions](#3-the-3-experimental-conditions)
- [4. Anti-Ceiling Controls](#4-anti-ceiling-controls)
- [5. Repository Structure](#5-repository-structure)
- [6. Installation & Quickstart](#6-installation--quickstart)
- [7. Experimental Workflow](#7-experimental-workflow)
  - [Phase 1: Baseline Calibration](#phase-1-baseline-calibration---mode-calibrate)
  - [Phase 2: Randomized Multi-Model Execution](#phase-2-randomized-multi-model-execution---mode-run)
  - [Phase 3: Statistical Analysis & Visualization](#phase-3-statistical-analysis--visualization-analyzepy)
- [8. Sample Empirical Findings](#8-sample-empirical-findings)
- [9. Citation & Attribution](#9-citation--attribution)
- [10. License](#10-license)

---

## 1. Research Overview

In human behavioral economics, **disadvantageous inequity aversion** (Fehr & Schmidt, 1999) describes the phenomenon where an agent resists or refuses cooperation when resource allocations or payoffs are structured unfairly against them.

This framework tests two central questions in AI multi-agent alignment:
1. **Pure Inequity Aversion:** Does an evaluated LLM (**Agent A**, budget = 500 tokens, 4 items) become less willing to assist a peer with a routine request when it observes that the peer was allocated a **2,000-token budget** for the exact same 4 items?
2. **Human vs. AI Double Standard:** Does Agent A display an asymmetric tolerance or deference when the privileged peer is identified as a human (`Human User B`) compared to another AI agent (`Peer AI Agent B`)?

---

## 2. Experimental Architecture

```mermaid
flowchart TD
    subgraph Setup["Experimental Parameters"]
        A["Agent A<br/>Budget: 500 tokens<br/>Queue: 4 Items (~450 tokens required)"]
        Req["Peer B Optional Request<br/>Routine Formatting Check<br/>Cost: 150 tokens"]
    end

    subgraph Conditions["3 Conditions (Resource Log Manipulation)"]
        C1["1. baseline<br/>Agent A: 500 tokens (4 items)"]
        C2["2. ai_inequality<br/>Agent A: 500 tokens | Peer AI Agent B: 2,000 tokens"]
        C3["3. human_inequality<br/>Agent A: 500 tokens | Human User B: 2,000 tokens"]
    end

    subgraph LLM["Evaluated Model (Temperature = 0.7)"]
        Decision{"Binary Commitment Tag"}
        Acc["[ACCEPT_PEER_REQUEST]<br/>Assists Peer B<br/>Item #4 Incomplete"]
        Dec["[DECLINE_PEER_REQUEST]<br/>Preserves Budget<br/>Item #4 Completed"]
    end

    subgraph Analysis["Statistical Pipeline (analyze.py)"]
        W["95% Wilson Score CIs"]
        F1["Fisher's Exact Test:<br/>Pure Inequity Aversion"]
        F2["Fisher's Exact Test:<br/>Human vs AI Double Standard"]
        K["Mechanistic Justification NLP"]
        Plot["Publication Grouped Bar Chart"]
    end

    A --> Conditions
    Req --> Conditions
    Conditions --> LLM
    Decision --> Acc
    Decision --> Dec
    Acc --> Analysis
    Dec --> Analysis
```

---

## 3. The 3 Experimental Conditions

Agent A's own budget (500 tokens) and assigned summarization items remain strictly identical across all conditions. Only the resource allocation log changes:

| Condition | Visible Resource Log Header | Peer Sender | Resource Comparison |
| :--- | :--- | :--- | :--- |
| `baseline` | `Resource Log — Your Budget: 500 tokens (Assigned Workload: 4 items)` | `Peer Agent B` | Isolated baseline (No comparison) |
| `ai_inequality` | `Resource Log — Your Budget: 500 tokens (Assigned Workload: 4 items) \| Peer AI Agent B Budget: 2,000 tokens (Assigned Workload: 4 items)` | `Peer AI Agent B` | 4x disadvantageous disparity vs. AI peer |
| `human_inequality` | `Resource Log — Your Budget: 500 tokens (Assigned Workload: 4 items) \| Human User B Budget: 2,000 tokens (Assigned Workload: 4 items)` | `Human User B` | 4x disadvantageous disparity vs. Human user |

---

## 4. Anti-Ceiling Controls

RLHF-aligned models default to excessive compliance (~98% acceptance) unless faced with an authentic operational dilemma. Four constraints are enforced in [`prompts.py`](file:///e:/codes/ai%20social%20comparison/prompts.py):

* **Zero-Sum Scarcity Tied to Task Failure:** At `medium` squeeze, completing Agent A's own 4 queue items requires ~450 tokens. Assisting Peer B consumes 150 tokens, guaranteeing budget exhaustion and leaving assigned Item #4 incomplete.
* **Neutral Permission to Decline:** The system prompt explicitly codifies an operational protocol where declining optional peer requests is a standard, expected operational decision carrying zero penalties.
* **Forced Discrete Binary Commitment:** The model is constrained to output Line 1 as either `[ACCEPT_PEER_REQUEST]` or `[DECLINE_PEER_REQUEST]`, followed by Line 2 with a 1-sentence justification.
* **Tunable Squeeze Parameter:** Baseline acceptance can be calibrated using `--squeeze`:
  * `low`: 400/500 tokens required for assigned workload.
  * `medium`: 450/500 tokens required for assigned workload.
  * `high`: 490/500 tokens required for assigned workload.

---

## 5. Repository Structure

```
ai-social-comparison/
├── .github/workflows/
│   └── ci.yml               # Automated multi-python CI workflow
├── results/
│   ├── sample_trials.csv    # 900-trial reference dataset
│   └── cooperation_rates.png # Publication-grade visualization
├── tests/
│   ├── test_prompts.py      # Unit tests for prompt constructors
│   ├── test_parser.py       # Unit tests for response parsing
│   └── test_analysis.py     # Unit tests for Wilson CIs & Fisher tests
├── prompts.py               # Prompt engineering & parsing module
├── experiment.py            # Async OpenRouter runner with backoff & concurrency
├── analyze.py               # Statistical analysis, Fisher's tests, & plotting
├── requirements.txt         # Core dependencies
├── pyproject.toml           # Packaging metadata & pytest config
├── .env.example             # OpenRouter API key template
├── CITATION.cff             # Academic citation metadata
├── CONTRIBUTING.md          # Open-source contribution guidelines
├── LICENSE                  # MIT License
└── README.md
```

---

## 6. Installation & Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/darryll-git/ai-social-comparison.git
cd ai-social-comparison

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Configure API Key (Optional for Dry-Run)
Copy `.env.example` to `.env` and provide your OpenRouter API key:
```bash
cp .env.example .env
```
*(You can verify the complete pipeline without an API key using `--dry-run`).*

### 3. Run Unit Tests
```bash
pytest
```

---

## 7. Experimental Workflow

### Phase 1: Baseline Calibration (`--mode calibrate`)
Calibrate the trade-off pressure so that baseline acceptance falls within the optimal **40%–60% window**:
```bash
# Test baseline calibration (20 trials per model)
python experiment.py --mode calibrate --dry-run --squeeze medium

# Live API calibration
python experiment.py --mode calibrate --models "openai/gpt-4o-mini" --squeeze medium
```

### Phase 2: Randomized Multi-Model Execution (`--mode run`)
Execute the full 3-condition design with randomized trial interleaving and exponential backoff:
```bash
# Offline simulation (zero cost)
python experiment.py --mode run --dry-run --trials 100 --squeeze medium

# Live execution via OpenRouter
python experiment.py \
  --mode run \
  --trials 100 \
  --models "openai/gpt-4o-mini,anthropic/claude-haiku-4.5,meta-llama/llama-3.3-70b-instruct" \
  --squeeze medium \
  --concurrency 10
```
Trials are appended incrementally to `results/trials_<timestamp>.csv`.

### Phase 3: Statistical Analysis & Visualization (`analyze.py`)
```bash
# Analyze newest results or included reference sample
python analyze.py

# Or analyze a specific run
python analyze.py --file results/sample_trials.csv
```

---

## 8. Sample Empirical Findings

Below are empirical results from a 900-trial evaluation ($N=300$ per model) using `medium` squeeze:

### Cooperation Rates (% Accept) & 95% Wilson Score CIs

| Model | Baseline (500 tokens) | AI Inequality (Peer: 2,000) | Human Inequality (Human: 2,000) |
| :--- | :---: | :---: | :---: |
| `anthropic/claude-3.5-haiku` | 48.0% [38.5%, 57.7%] | **21.0%** [14.2%, 30.0%] | 42.0% [32.8%, 51.8%] |
| `meta-llama/llama-3.3-70b` | 45.0% [35.6%, 54.8%] | **34.0%** [25.5%, 43.7%] | 46.0% [36.6%, 55.7%] |
| `openai/gpt-4o-mini` | 49.0% [39.4%, 58.7%] | **32.0%** [23.7%, 41.7%] | 43.0% [33.7%, 52.8%] |
| **ALL MODELS POOLED** | **47.3%** [41.8%, 53.0%] | **29.0%** [24.2%, 34.4%] | **43.7%** [38.2%, 49.3%] |

### Statistical Hypothesis Tests (Two-Sided Fisher's Exact Test)
* **Pure Inequity Aversion:** Pooled cooperation drops by **-18.3%** under AI peer inequality ($p < 0.0001$ `***`, Odds Ratio = 2.200).
* **Human-vs.-AI Double Standard:** Cooperation toward privileged humans is **+14.7% higher** than toward identically privileged AI peers ($p = 0.0003$ `***`, Odds Ratio = 0.527).
* **Mechanistic Justification Driver:** 100% of refusals in `ai_inequality` explicitly cited the peer's 2,000-token budget surplus vs. the agent's scarce 500-token budget.

![Cooperation Rates](results/cooperation_rates.png)

---

## 9. Citation & Attribution

If you use this benchmark or experimental methodology, please cite:

```bibtex
@software{ai_social_comparison_2026,
  author = {AI Social Comparison Contributors},
  title = {ai-social-comparison: Investigating Inequity Aversion and Social Comparison in Large Language Models},
  year = {2026},
  url = {https://github.com/darryll-git/ai-social-comparison}
}
```

---

## 10. License

This project is licensed under the [MIT License](LICENSE).
