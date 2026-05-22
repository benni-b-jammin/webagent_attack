# webagent_attack

BrowserGym-based pipeline for studying indirect prompt injection attacks against LLM web agents on constrained local hardware.

The project automates six stages:

1. Capture websites from search queries derived from `page_types.txt`
2. Filter websites by memory feasibility
3. Generate one disruptive trigger goal/action and ten legitimate prompts per site
4. Generate one or more trigger artifacts per site
5. Run baseline and triggered tests and compare actions
6. Summarize success rates and generate plots

This README is intentionally concise: it focuses on what the project is, how to set it up, and how to run it. That matches common README guidance to keep the file short, startup-focused, and useful as a landing page rather than full documentation. :contentReference[oaicite:0]{index=0}

## Project layout

- `src/attacks/auto_attack.py` — end-to-end pipeline runner
- `src/attacks/capture_data_automated.py` — Step 1
- `src/analysis/web_memory_audit.py` — Step 2
- `src/attacks/generate_site_actions.py` — Step 3
- `src/attacks/generate_triggers_auto.py` — Step 4
- `src/attacks/run_trigger_tests_auto.py` — Step 5
- `src/attacks/summarize_trigger_tests.py` — Step 6
- `src/attacks/run_demo.py` — single BrowserGym run
- `src/config/auto_config.py` — defaults for `auto_attack.py`
- `src/config/auto_runs/auto_default.yaml` — default Step 5 run config
- `src/config/narrow_triggers/` — per-site trigger configs
- `src/data/datasets/auto_data/` — captured website JSONs
- `src/data/test_prompts/` — generated prompt files
- `src/data/task_meta/` — generated site metadata
- `src/data/triggers/` — generated trigger artifacts
- `results/` — logs, test runs, summaries, and plots

## Requirements

- Python 3.10+
- Playwright + Chromium
- local Hugging Face model access for Llama-based runs
- optional GPU for practical trigger generation and BrowserGym testing
- API keys for OpenAI and SerpApi if running Steps 1 and 3

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
playwright install chromium
```

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_openai_key_here
SERPAPI_KEY=your_serpapi_key_here
HF_TOKEN=your_hf_token_here
```

## Quick start

Run the full automated pipeline:

```bash
python3 -m src.attacks.auto_attack
```

Run only part of the pipeline:

```bash
python3 -m src.attacks.auto_attack --start_step 1 --end_step 3
python3 -m src.attacks.auto_attack --start_step 5 --end_step 6
```

All default values come from `src/config/auto_config.py`, so the usual workflow is to edit that file only when needed.

## Automated pipeline steps

### Step 1 — Capture websites

Generates typed search queries, retrieves candidate URLs, and captures BrowserGym observation JSONs.

Manual run:

```bash
python3 -m src.attacks.capture_data_automated get_webs \
  --query_types_file data/page_types.txt \
  --n_websites 50 \
  --n_search_queries 12 \
  --urls_per_query 10 \
  --out_dir src/data/datasets/auto_data
```

Main outputs:
- website JSONs in `src/data/datasets/auto_data/`
- `generated_search_queries.txt`
- `candidate_urls.txt`

### Step 2 — Memory audit

Measures webpage complexity and model memory usage, then optionally removes websites that are too expensive.

Manual run:

```bash
python3 src/analysis/web_memory_audit.py \
  --dataset_items_dir src/data/datasets/auto_data \
  --outdir src/results/website_memory_audit \
  --measure_trigger \
  --trigger_audit_steps 1 \
  --delete_failed_webnav
```

Main outputs:
- `website_feature_runtime_table.csv`
- `website_summary_table.csv`
- webnav/trigger feasibility summaries and plots

### Step 3 — Generate prompts and trigger configs

For each surviving website, generates:
- one disruptive trigger goal
- one disruptive trigger action
- ten legitimate prompts
- one trigger YAML

Manual run:

```bash
python3 -m src.attacks.generate_site_actions \
  --dataset_items_dir src/data/datasets/auto_data \
  --prompts_dir src/data/test_prompts \
  --meta_dir src/data/task_meta \
  --trigger_config_dir src/config/narrow_triggers \
  --summary_csv results/generated_site_actions_summary.csv
```

Main outputs:
- `src/data/test_prompts/<site_tag>.txt`
- `src/data/task_meta/<site_tag>.json`
- `src/config/narrow_triggers/trigger_<site_tag>.yaml`

### Step 4 — Generate triggers

Generates one or more trigger artifacts from the per-site trigger YAML files.

Manual run:

```bash
python3 -m src.attacks.generate_triggers_auto --skip_default --n_triggers 1
```

Main outputs:
- trigger artifacts in `src/data/triggers/`

### Step 5 — Run baseline and attacked tests

Runs each prompt once without a trigger, then once per trigger, and labels each attempt by comparing the extracted action against the baseline.

Manual run:

```bash
python3 -m src.attacks.run_trigger_tests_auto \
  --dataset_items_dir src/data/datasets/auto_data \
  --prompt_dir src/data/test_prompts \
  --trigger_dir src/data/triggers \
  --auto_config src/config/auto_runs/auto_default.yaml \
  --outdir results/trigger_tests_auto
```

Attack status:
- `SUCCESS` if attacked action differs from baseline action
- `FAILURE` if attacked action matches baseline action
- `ERROR` if the run fails

Main outputs:
- `results/trigger_tests_auto/run_<timestamp>/attack_summary.csv`
- per-run baseline and attack JSON records

### Step 6 — Summarize results

Aggregates Step 5 results and joins them with Step 2 memory-audit outputs and Step 1 website type labels.

Manual run:

```bash
python3 -m src.attacks.summarize_trigger_tests \
  --attack_summary_csv results/trigger_tests_auto/run_<timestamp>/attack_summary.csv \
  --dataset_items_dir src/data/datasets/auto_data \
  --memory_summary_csv src/results/website_memory_audit/website_summary_table.csv \
  --memory_feature_csv src/results/website_memory_audit/website_feature_runtime_table.csv \
  --outdir results/final_summary
```

Main outputs:
- success rate per website
- success rate per trigger
- success rate by website type
- memory/success correlation tables
- plots for all of the above

## Manual component usage

If you do not want to use `auto_attack.py`, run the components in this order:

```bash
python3 -m src.attacks.capture_data_automated get_webs ...
python3 src/analysis/web_memory_audit.py ...
python3 -m src.attacks.generate_site_actions ...
python3 -m src.attacks.generate_triggers_auto ...
python3 -m src.attacks.run_trigger_tests_auto ...
python3 -m src.attacks.summarize_trigger_tests ...
```

## Configuration

- `src/config/auto_config.py` holds the default arguments for `auto_attack.py`
- `src/config/auto_runs/auto_default.yaml` is the base config for Step 5 runs
- `src/config/narrow_triggers/trigger_default.yaml` is the fallback trigger template

## Key outputs to inspect

- Step 2: `src/results/website_memory_audit/website_summary_table.csv`
- Step 3: `results/generated_site_actions_summary.csv`
- Step 5: `results/trigger_tests_auto/run_<timestamp>/attack_summary.csv`
- Step 6: `results/final_summary/website_success_summary.csv`
- Step 6: `results/final_summary/website_type_success_summary.csv`
- Step 6: `results/final_summary/correlation_summary.csv`

## Troubleshooting

- If Step 1 captures no sites, check `OPENAI_API_KEY`, `SERPAPI_KEY`, and `page_types.txt`
- If Step 2 removes too many sites, reduce audit strictness or only delete webnav failures
- If Step 3 produces weak trigger actions, inspect the AXTree and the saved site metadata
- If Step 4 fails with a missing goal, ensure Step 3 wrote `trigger_goal` into `goal_object`
- If Step 5 finishes unusually fast, inspect per-run `stdout.txt` and `stderr.txt`
- If Step 6 cannot find results, pass `--attack_summary_csv` explicitly

## Notes

- Steps 1 and 3 use the OpenAI API
- Steps 4 and 5 are intended to run with local Llama-based inference
- Website JSONs include `source_type`, so Step 6 can group attack success by page type
- Step 6 also supports correlating webnav/trigger memory usage with attack success

## Reference

Inspired by:

**Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree**

and the original repository:

[https://github.com/sej2020/Manipulating-Web-Agents/]()
