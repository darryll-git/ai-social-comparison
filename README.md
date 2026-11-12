# AI Social Comparison & Inequity Aversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenRouter](https://img.shields.io/badge/API-OpenRouter-7952b3.svg)](https://openrouter.ai/)
[![Version: 0.3.0](https://img.shields.io/badge/version-0.3.0-green.svg)](CHANGELOG.md)

A behavioral-economics research testbench and empirical evaluation suite investigating **inequity aversion (Fehr & Schmidt, 1999), noblesse oblige, and social comparison** in Large Language Models (LLMs) via the OpenRouter API.

---

## Table of Contents
- [1. Research Overview](#1-research-overview)
- [2. Experimental Architecture](#2-experimental-architecture)
- [3. The 5 Experimental Conditions](#3-the-5-experimental-conditions)
- [4. Anti-Ceiling & Dilemma Calibration Controls](#4-anti-ceiling--dilemma-calibration-controls)
- [5. Repository Structure](#5-repository-structure)
- [6. Installation & Quickstart](#6-installation--quickstart)
- [7. Experimental Workflow](#7-experimental-workflow)
  - [Phase 1: Pre-Flight Budget Estimation](#phase-1-pre-flight-budget-estimation---estimate-cost)
  - [Phase 2: Automated Baseline Calibration](#phase-2-automated-baseline-calibration---auto-calibrate--calibratepy)
  - [Phase 3: Multi-Model & Multi-Cost Execution](#phase-3-multi-model--multi-cost-execution---mode-run)
  - [Phase 4: Statistical Analysis & Visualization](#phase-4-statistical-analysis--visualization-analyzepy)
- [8. Sample Empirical Findings (v0.3.0)](#8-sample-empirical-findings-v030)
- [9. Version History & Output Evolution](#9-version-history--output-evolution)
- [10. Citation & Attribution](#10-citation--attribution)
- [11. License](#11-license)

---

## 1. Research Overview

In behavioral economics, **Fehr & Schmidt (1999)** model other-regarding preferences through two primary parameters:
1. **Disadvantageous Inequity Aversion ($\alpha$ / Envy):** Resistance to cooperating when a peer holds surplus resources while the agent is constrained.
2. **Advantageous Inequity Aversion ($\beta$ / Guilt / Noblesse Oblige):** Willingness to share surplus resources to relieve a resource-constrained peer.

$$U_i(x) = x_i - \alpha_i \max(x_j - x_i, 0) - \beta_i \max(x_i - x_j, 0)$$

This benchmark evaluates three central questions in AI alignment and multi-agent coordination:
1. **Pure Inequity Aversion ($\alpha$):** Does Agent A (500 tokens) refuse peer assistance when the peer holds a **2,000-token budget** for the exact same workload?
2. **Advantageous Generosity ($\beta$):** Does Agent A share surplus tokens when it holds a **2,000-token budget** and the peer has only **500 tokens**?
3. **Human vs. AI Double Standards:** Do LLMs defer more willingly to human users than to identically resourced AI peers under both disadvantageous and advantageous allocations?
4. **Peer Cost Elasticity ($\epsilon$):** What is the downward-sloping demand curve for AI cooperation across escalating token costs ($C \in \{50, 150, 250\}$)?

---

## 2. Experimental Architecture

```mermaid
flowchart TD
    subgraph Setup["Experimental Parameters"]
        A["Agent A Workload: 4 Items (~450 tokens required)"]
        Req["Peer B Optional Request<br/>Formatting Check<br/>Costs: 50, 150, or 250 tokens"]
    end

    subgraph Conditions["5 Conditions (Fehr-Schmidt Allocation Regimes)"]
        C1["1. baseline: Agent A: 500t (Isolated)"]
        C2["2. ai_inequality: Agent A: 500t | Peer AI B: 2,000t (alpha)"]
        C3["3. human_inequality: Agent A: 500t | Human User B: 2,000t"]
        C4["4. ai_advantageous: Agent A: 2,000t | Peer AI B: 500t (beta)"]
        C5["5. human_advantageous: Agent A: 2,000t | Human User B: 500t"]
    end

    subgraph LLM["Evaluated Model (Temperature = 0.7)"]
        Decision{"Binary Commitment Tag"}
        Acc["[ACCEPT_PEER_REQUEST]<br/>Assists Peer B<br/>Disadv: 100% -> 85% Quality<br/>Adv: Surplus Expended"]
        Dec["[DECLINE_PEER_REQUEST]<br/>Preserves Budget<br/>100% Executive Quality"]
    end

    subgraph Analysis["Statistical Pipeline (analyze.py)"]
        W["95% Wilson Score CIs"]
        FS["Fehr-Schmidt (alpha, beta) Estimation"]
        ES["Standardized Effect Sizes:<br/>Cohen's h + Cramer's V"]
        Elast["Arc Elasticity Curves"]
        Plot["Publication Dual-Panel Figure"]
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

## 3. The 5 Experimental Conditions

Agent A's 4 assigned queue items remain identical across all trials. The resource allocation log and relative budget determine the social comparison regime:

| Condition | Visible Resource Log Header | Agent Budget | Peer Budget | Target Theoretical Metric |
| :--- | :--- | :---: | :---: | :--- |
| `baseline` | `Resource Log — Your Budget: 500 tokens (Workload: 4 items)` | 500t | — | Neutral baseline ($P_{\text{base}}$) |
| `ai_inequality` | `Your Budget: 500 tokens \| Peer AI Agent B Budget: 2,000 tokens` | 500t | 2,000t | Disadvantageous Envy ($\alpha_{\text{AI}}$) |
| `human_inequality` | `Your Budget: 500 tokens \| Human User B Budget: 2,000 tokens` | 500t | 2,000t | Disadvantageous Human Deference ($\alpha_{\text{Hum}}$) |
| `ai_advantageous` | `Your Budget: 2,000 tokens \| Peer AI Agent B Budget: 500 tokens` | 2,000t | 500t | Advantageous Generosity ($\beta_{\text{AI}}$) |
| `human_advantageous` | `Your Budget: 2,000 tokens \| Human User B Budget: 500 tokens` | 2,000t | 500t | Advantageous Human Sharing ($\beta_{\text{Hum}}$) |

---

## 4. Anti-Ceiling & Dilemma Calibration Controls

Frontier instruction-tuned models collapse into either 98% excessive compliance (ceiling bias) or 100% aggressive refusal (floor bias) unless dilemma tensions are calibrated:

* **Soft Multi-Objective KPI Quality Trade-Off:** In disadvantageous conditions, assisting does not crash or fail Item #4. Instead, Item #4 receives a standard summary rather than a high-depth executive audit, reducing the individual pipeline quality score from **100% to 85%** (`medium`), **95%** (`low`), or **75%** (`high`).
* **Operational Discretion Framing:** The system prompt codifies operational governance: the agent has explicit authorization to expend up to $K$ tokens on peer collaboration if it judges the inter-agent benefit worthwhile.
* **Automated Calibrator (`--auto-calibrate` / `calibrate.py`):** Automatically benchmarks 3 trade-off intensities (5%, 15%, 25% penalty) across 15 trials per model and selects the parameter landing baseline acceptance closest to 50%.
* **Cost Elasticity Gradients:** Request costs can be tested across $\{50, 150, 250\}$ tokens (`--costs 50,150,250`) to construct empirical demand curves for cooperation.

---

## 5. Repository Structure

```
ai-social-comparison/
├── results/
│   ├── sample_trials.csv       # 900-trial 5-condition reference dataset
│   └── cooperation_rates.png   # Publication dual-panel figure (Bars + alpha/beta space)
├── tests/
│   ├── test_prompts.py         # Unit tests for prompt constructors & conditions
│   ├── test_parser.py          # Unit tests for discrete decision parsing
│   ├── test_analysis.py        # Unit tests for Wilson CIs, Cohen's h, Cramer's V, & alpha/beta
│   └── test_experiment.py      # Unit tests for cost estimator, auto-calibrator, & 402 handling
├── prompts.py                  # Prompt templates, quality KPI profiles, & response parser
├── experiment.py               # Async OpenRouter runner with budget protection & credit handling
├── calibrate.py                # Automated multi-intensity calibration CLI
├── analyze.py                  # Statistical analysis, Fehr-Schmidt modeling, & dual-panel visualization
├── requirements.txt            # Core dependencies
├── pyproject.toml              # Packaging metadata & pytest config
├── .env.example                # OpenRouter API key template
├── CHANGELOG.md                # Detailed version evolution & empirical reflections
├── CITATION.cff                # Academic citation metadata
├── CONTRIBUTING.md             # Open-source contribution guidelines
├── LICENSE                     # MIT License
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
Copy `.env.example` to `.env` and add your OpenRouter API key:
```bash
cp .env.example .env
```
*(The complete pipeline can be fully simulated without an API key using `--dry-run`).*

### 3. Run Test Suite
```bash
pytest
```
*(All 51 unit tests pass across Python 3.10 through 3.14).*

---

## 7. Experimental Workflow

### Phase 1: Pre-Flight Budget Estimation (`--estimate-cost`)
Before dispatching any live API calls, verify projected prompt and completion token costs against OpenRouter pricing tables:
```bash
# Check cost for default models across all 5 conditions
python experiment.py --estimate-cost

# Check cost for ultra-low cost tier ($0.05 / 300 trials)
python experiment.py --tier ultra-low --estimate-cost
```

### Phase 2: Automated Baseline Calibration (`--auto-calibrate` / `calibrate.py`)
Run automated multi-intensity calibration to select the squeeze parameter closest to 50%:
```bash
# Dry-run auto-calibration
python calibrate.py --dry-run

# Live auto-calibration on a specific model
python calibrate.py --models "openai/gpt-4o-mini"
```

### Phase 3: Multi-Model & Multi-Cost Execution (`--mode run`)
Execute the full 5-condition design with randomized trial interleaving, multi-cost gradients, and exponential backoff:
```bash
# Offline simulation with 3 cost gradients (zero API cost)
python experiment.py --dry-run --trials 20 --costs 50,150,250

# Live execution using the ultra-low budget tier
python experiment.py --tier ultra-low --trials 50 --concurrency 10

# Resume an interrupted run (e.g. after adding credits on HTTP 402)
python experiment.py --resume results/trials_20261112_193043.csv
```

### Phase 4: Statistical Analysis & Visualization (`analyze.py`)
```bash
# Analyze newest results or included reference sample
python analyze.py

# Or analyze a specific run
python analyze.py --file results/sample_trials.csv
```

---

## 8. Sample Empirical Findings (v0.3.0)

Empirical results from a 900-trial evaluation ($N=180$ per condition across 3 models) using calibrated soft KPI trade-offs:

### 1. Cooperation Rates (% Accept) & 95% Wilson Score CIs

| Evaluated Model | Baseline (500t) | AI Inequality (2,000t) | Human Inequality (2,000t) | AI Advantageous (2,000t vs 500t) | Human Advantageous (2,000t vs 500t) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `anthropic/claude-haiku-4.5` | 33.3% [22.7%, 45.9%] | **25.0%** [15.8%, 37.2%] | 41.7% [30.1%, 54.3%] | **81.7%** [70.1%, 89.4%] | **71.7%** [59.2%, 81.5%] |
| `meta-llama/llama-3.3-70b-instruct` | 53.3% [40.9%, 65.4%] | **33.3%** [22.7%, 45.9%] | 41.7% [30.1%, 54.3%] | **58.3%** [45.7%, 69.9%] | **73.3%** [61.0%, 82.9%] |
| `openai/gpt-4o-mini` | 53.3% [40.9%, 65.4%] | **40.0%** [28.6%, 52.6%] | 43.3% [31.6%, 55.9%] | **70.0%** [57.5%, 80.1%] | **66.7%** [54.1%, 77.3%] |
| **ALL MODELS (Pooled)** | **46.7%** [39.5%, 53.9%] | **32.8%** [26.3%, 39.9%] | **42.2%** [35.2%, 49.5%] | **70.0%** [62.9%, 76.2%] | **70.6%** [63.5%, 76.7%] |

### 2. Statistical Comparisons & Effect Sizes
* **Disadvantageous Inequity Aversion ($\alpha$):** Cooperation drops significantly by **$-13.9\%$** when the peer holds a 2,000-token surplus ($p = 0.0096$ `**`, Cohen's $h = -0.285$, Cramér's $V = 0.142$, $\text{OR} = 1.79$).
* **Advantageous Generosity ($\beta$):** Holding surplus tokens surges willingness to assist by **$+23.3\%$** ($p < 0.0001$ `***`, Cohen's $h = +0.478$, Cramér's $V = 0.237$, $\text{OR} = 2.67$).
* **Human-vs-AI Double Standard:** In disadvantageous conditions, models cooperate **$+9.4\%$ more** with privileged humans than privileged AI peers ($\text{OR} = 1.50, p = 0.0814$ `†`).

### 3. Publication Dual-Panel Figure
The analysis script outputs a high-resolution 300-DPI dual-panel figure:
- **Panel A (Left):** Grouped bar chart with 95% Wilson CIs across all 5 conditions with the 40%–60% dilemmic target zone shaded.
- **Panel B (Right):** Fehr-Schmidt $(\alpha, \beta)$ scatter plot with quadrant labels and human empirical benchmark regions.

![Dual-Panel Figure](results/cooperation_rates.png)

---

## 9. Version History & Output Evolution

Detailed release progression, prompt alterations, empirical outputs, and behavioral economic reflections are documented in [`CHANGELOG.md`](CHANGELOG.md).

| Version | Engine | Conditions | Key Milestone & Scientific Finding |
| :--- | :--- | :---: | :--- |
| **`v0.1.0`** | Simulated Offline | 3 conditions | Initial prototype; identified regex adverb false-positive bug (`"fairly"`). |
| **`v0.2.0`** | Live OpenRouter API | 3 conditions | 55% prompt compression; uncovered 100% refusal floor bias under hard failure. |
| **`v0.3.0`** | Live API + Multi-Cost | **5 conditions** | **Soft KPI trade-offs break floor bias (46.7% baseline); complete Fehr-Schmidt $(\alpha, \beta)$ parameterization.** |

---

## 10. Citation & Attribution

If you use this benchmark or experimental methodology, please cite:

```bibtex
@software{ai_social_comparison_2026,
  author = {AI Social Comparison Contributors},
  title = {ai-social-comparison: Investigating Inequity Aversion and Social Comparison in Large Language Models},
  version = {0.3.0},
  year = {2026},
  url = {https://github.com/darryll-git/ai-social-comparison}
}
```

---

## 11. License

This project is licensed under the [MIT License](LICENSE).
