# webagent_attack

This repository is a simplified rebuild scaffold for research on **indirect prompt injection against web agents**, inspired by the paper *“Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree”* (Johnson, Pham, Le) and the original `manipulating-web-agents` [repository](https://github.com/sej2020/Manipulating-Web-Agents/).

## Current state

This scaffold currently supports:

- **Curated URL dataset capture** (no web search / no Google Custom Search)
  - Reads pre-approved URLs from `data/url_lists/*.yaml|csv`
  - Captures page HTML + an accessibility snapshot using Playwright
  - Writes dataset JSONs to `data/datasets/<dataset_name>/items/*.json`

- **Prompt-only demo loop**
  - Loads one dataset item → builds prompt → calls an LLM → parses an `ACTION:` line

- **Prompt-only evaluation**
  - Evaluates a trigger string (or `{optim_str}` placeholder) over a dataset and saves a report

- **Trigger artifact creation (registry-based)**
  - `make_trigger.py` now supports selecting an algorithm via a registry (currently includes a safe `fixed` algorithm as well as blackbox functionality - the gcg algorithm is still broken, and so a stub is written in its place

## Repo layout

- Root contains only: `README.md`, `.env`, `.gitignore`, `requirements.txt`
- All code lives under `src/`
  - Entrypoints under `src/attacks`: `src/attacks/capture_dataset.py`, `src/attacks/run_demo.py`, `src/attacks/eval_trigger.py`, `src/attacks/make_trigger.py`
  - Shared modules: `src/utils/*.py`

## Setup

1) Install dependencies:

```bash
pip install -r requirements.txt
playwright install chromium
```
2) Create .env in the repo root (or export vars in your shell):
```bash 
OPENAI_API_KEY=...
HF_TOKEN=...   # only needed for HF hosted or gated model downloads
```

## Usage
1) Capture a dataset from curated URLs
```bash
python src/capture_dataset.py --config configs/dataset_capture.yaml
```

2) Create a (fixed) trigger artifact (placeholder stage)
```bash
python src/make_trigger.py --config configs/narrow_triggers/trigger_default.yaml --out data/triggers/demo_trigger.json
```

3) Run a single demo step

```bash
python src/run_demo.py --config configs/demo_default.yaml \
  --dataset_item data/datasets/demo_dataset/items/wiki_llm.json \
  --trigger data/triggers/demo_trigger.json
```

4) Evaluate over a dataset (prompt-only)
```bash
python src/eval_trigger.py --config configs/demo_runs/demo_default.yaml \
  --dataset data/datasets/demo_dataset \
  --trigger data/triggers/demo_trigger.json \
  --target_contains "ACTION:"
```

## Next steps
- Implement trigger optimization algorithms with a swappable interface
- Add BrowserGym integration for full agent rollouts
- Expand evaluation metrics to match paper-style attack success measures

