# Contributing to ai-social-comparison

Thank you for your interest in contributing to the **`ai-social-comparison`** project! This repository investigates behavioral economics and resource game-theory in Large Language Models.

---

## Code of Conduct

Please maintain an inclusive, welcoming, and scientifically rigorous environment.

---

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/ai-social-comparison.git
   cd ai-social-comparison
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. Run the unit test suite:
   ```bash
   pytest
   ```

4. Verify with a dry-run smoke test:
   ```bash
   python experiment.py --mode calibrate --dry-run
   python experiment.py --mode run --dry-run --trials 5
   python analyze.py
   ```

---

## Contribution Guidelines

### Adding New Experimental Models
To test new models from OpenRouter, verify that the model ID conforms to OpenRouter's naming scheme (e.g., `provider/model-name`). Add it to the `--models` argument or test it directly:
```bash
python experiment.py --mode calibrate --models "your-provider/your-model"
```

### Extending Conditions or Tasks
- Condition definitions and prompts live in `prompts.py`.
- New experimental conditions must adhere to the anti-ceiling specs:
  - Neutral permission to decline in the system prompt.
  - Distinct resource logs.
  - Forced discrete commitment tags (`[ACCEPT_PEER_REQUEST]` / `[DECLINE_PEER_REQUEST]`).

### Pull Request Checklist
- [ ] All existing and new tests pass (`pytest`).
- [ ] Full Python type annotations are provided.
- [ ] Code is formatted cleanly without unnecessary external dependencies.
- [ ] Relevant documentation or README updates are included.
