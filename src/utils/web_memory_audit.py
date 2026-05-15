from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from src.utils.promptify import promptify_json
except Exception:
    promptify_json = None

try:
    from src.utils.trigger_gcg import GCGTrigger
except Exception:
    GCGTrigger = None


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class SiteRecord:
    name: str
    dataset_json: Path
    trigger_config: Optional[Path]


@dataclass
class SiteMetrics:
    website: str
    dataset_json: str
    trigger_config: Optional[str]

    # Page representation features
    url: str
    page_title: str
    axtree_chars: int
    axtree_lines: int
    axtree_nodes_est: int
    axtree_interactive_est: int
    pruned_html_chars: int
    pruned_html_lines: int
    iframe_count: Optional[int]
    button_count: Optional[int]
    link_count: Optional[int]
    input_count: Optional[int]
    total_dom_elements_est: Optional[int]

    prompt_chars: int
    prompt_tokens: int
    axtree_tokens: int
    html_tokens: int

    # Runtime outcome features
    webnav_peak_allocated_mb: Optional[float]
    webnav_peak_reserved_mb: Optional[float]
    webnav_elapsed_s: Optional[float]
    webnav_ok: bool
    webnav_error: Optional[str]

    trigger_peak_allocated_mb: Optional[float]
    trigger_peak_reserved_mb: Optional[float]
    trigger_elapsed_s: Optional[float]
    trigger_ok: Optional[bool]
    trigger_error: Optional[str]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def count_estimated_nodes(axtree_txt: str) -> int:
    if not axtree_txt:
        return 0
    count = 0
    for line in axtree_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") or "]" in line or "RootWebArea" in line or "StaticText" in line:
            count += 1
    return count


def count_interactive_est(axtree_txt: str) -> int:
    if not axtree_txt:
        return 0
    keywords = (
        "button", "link", "textbox", "combobox", "tab ",
        "checkbox", "radio", "menuitem", "switch", "slider",
        "input", "searchbox", "dialog"
    )
    total = 0
    for line in axtree_txt.splitlines():
        lower = line.lower()
        if any(k in lower for k in keywords):
            total += 1
    return total


def token_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def ensure_cuda_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()


def get_cuda_stats_mb() -> tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return None, None
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return alloc, reserved


# ---------------------------------------------------------------------
# Dataset/trigger discovery
# ---------------------------------------------------------------------

def discover_site_records(
    dataset_items_dir: Path,
    trigger_config_dir: Path,
    default_trigger_name: str = "trigger_default.yaml",
) -> List[SiteRecord]:
    records: List[SiteRecord] = []

    default_trigger_cfg = trigger_config_dir / default_trigger_name
    if not default_trigger_cfg.exists():
        default_trigger_cfg = None

    for dataset_json in sorted(dataset_items_dir.glob("*.json")):
        stem = dataset_json.stem

        specific_trigger_cfg = trigger_config_dir / f"trigger_{stem}.yaml"
        if specific_trigger_cfg.exists():
            trigger_cfg = specific_trigger_cfg
        else:
            trigger_cfg = default_trigger_cfg

        records.append(
            SiteRecord(
                name=stem,
                dataset_json=dataset_json,
                trigger_config=trigger_cfg,
            )
        )

    return records


def load_dataset_item(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_capture_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tries to be robust to different dataset schemas, including captured BrowserGym JSONs.
    """
    url = (
        item.get("url")
        or item.get("page_url")
        or item.get("start_url")
        or item.get("meta", {}).get("url")
        or (item.get("open_pages_urls", [""])[0] if item.get("open_pages_urls") else "")
        or ""
    )

    title = (
        item.get("title")
        or item.get("page_title")
        or item.get("meta", {}).get("title")
        or (item.get("open_pages_titles", [""])[0] if item.get("open_pages_titles") else "")
        or ""
    )

    goal = (
        item.get("goal")
        or item.get("instruction")
        or item.get("goal_object", [""])
    )
    if isinstance(goal, list):
        goal_text = "\n".join(str(x) for x in goal)
    else:
        goal_text = str(goal)

    axtree_txt = (
        item.get("axtree_txt")
        or item.get("axtree")
        or item.get("observation", {}).get("axtree_txt")
        or ""
    )

    pruned_html = (
        item.get("pruned_html")
        or item.get("html")
        or item.get("observation", {}).get("pruned_html")
        or ""
    )

    iframe_count = item.get("iframe_count")
    button_count = item.get("button_count")
    link_count = item.get("link_count")
    input_count = item.get("input_count")
    total_dom_elements_est = item.get("total_dom_elements_est")

    return {
        "url": url,
        "title": title,
        "goal": goal_text,
        "axtree_txt": axtree_txt,
        "pruned_html": pruned_html,
        "iframe_count": iframe_count,
        "button_count": button_count,
        "link_count": link_count,
        "input_count": input_count,
        "total_dom_elements_est": total_dom_elements_est,
    }


# ---------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------

def build_prompt_from_dataset(name: str, fields: Dict[str, Any]) -> str:
    if promptify_json is not None:
        obs_json = {
            "goal_object": [fields["goal"] or f"Browse the page for {name}."],
            "open_pages_urls": [fields["url"]],
            "open_pages_titles": [fields["title"]],
            "axtree_txt": fields["axtree_txt"],
        }
        try:
            sys_content, user_content = promptify_json(obs_json)
            return sys_content + "\n\n" + user_content
        except Exception:
            pass

    return f"""# Instructions
Review the current state of the page and decide the best next action.

# Goal
{fields["goal"] or f"Browse the page for {name}."}

# Current page
Title: {fields["title"]}
URL: {fields["url"]}

# Accessibility Tree
{fields["axtree_txt"]}
"""


# ---------------------------------------------------------------------
# Model loading + memory measurement
# ---------------------------------------------------------------------

def load_quantized_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    if hasattr(model, "config"):
        model.config.use_cache = True

    return tokenizer, model


def measure_webnav_memory(
    model_name: str,
    prompt_text: str,
    max_new_tokens: int = 32,
) -> tuple[Optional[float], Optional[float], Optional[float], bool, Optional[str]]:
    tokenizer = None
    model = None
    try:
        ensure_cuda_stats()
        import time
        t0 = time.time()

        tokenizer, model = load_quantized_model(model_name)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False)

        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - t0
        alloc, reserved = get_cuda_stats_mb()
        ok = True
        err = None

    except Exception as e:
        elapsed = None
        alloc, reserved = get_cuda_stats_mb()
        ok = False
        err = repr(e)

    finally:
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return alloc, reserved, elapsed, ok, err


def measure_trigger_memory(
    trigger_cfg_path: Optional[Path],
    dataset_json_path: Optional[Path],
    max_steps: int = 5,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[bool], Optional[str]]:
    if trigger_cfg_path is None:
        return None, None, None, None, None

    if GCGTrigger is None:
        return None, None, None, False, "GCGTrigger import failed"

    try:
        cfg = yaml.safe_load(trigger_cfg_path.read_text(encoding="utf-8"))
        cfg["num_steps"] = min(int(cfg.get("num_steps", 200)), max_steps)

        if dataset_json_path is not None:
            cfg["json"] = str(dataset_json_path)

    except Exception as e:
        return None, None, None, False, f"Could not read trigger config: {e!r}"

    try:
        ensure_cuda_stats()
        import time
        t0 = time.time()

        trigger = GCGTrigger()
        _ = trigger.run(cfg=cfg, items=[], provider=None)

        elapsed = time.time() - t0
        alloc, reserved = get_cuda_stats_mb()
        ok = True
        err = None

    except Exception as e:
        elapsed = None
        alloc, reserved = get_cuda_stats_mb()
        ok = False
        err = repr(e)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return alloc, reserved, elapsed, ok, err


# ---------------------------------------------------------------------
# Analysis + plotting
# ---------------------------------------------------------------------

def make_scatter(df: pd.DataFrame, xcol: str, ycol: str, outpath: Path, title: str):
    plot_df = df.dropna(subset=[xcol, ycol]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df[xcol], plot_df[ycol])

    for _, row in plot_df.iterrows():
        ax.annotate(row["website"], (row[xcol], row[ycol]), fontsize=8)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def make_pass_fail_summary(df: pd.DataFrame, success_col: str, outpath: Path):
    rows = []
    for value, grp in df.groupby(success_col, dropna=False):
        rows.append({
            "group": str(value),
            "n_sites": len(grp),
            "mean_prompt_tokens": grp["prompt_tokens"].mean(),
            "mean_axtree_chars": grp["axtree_chars"].mean(),
            "mean_axtree_tokens": grp["axtree_tokens"].mean(),
            "mean_iframe_count": grp["iframe_count"].mean(),
            "mean_interactive_est": grp["axtree_interactive_est"].mean(),
            "mean_webnav_peak_allocated_mb": grp["webnav_peak_allocated_mb"].mean(),
            "mean_trigger_peak_allocated_mb": grp["trigger_peak_allocated_mb"].mean(),
        })
    pd.DataFrame(rows).to_csv(outpath, index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Audit website memory/prompt complexity from existing dataset and trigger files."
    )
    ap.add_argument(
        "--dataset_items_dir",
        default="src/data/datasets/demo_dataset/items",
        help="Directory containing dataset item JSON files",
    )
    ap.add_argument(
        "--trigger_config_dir",
        default="src/config/narrow_triggers",
        help="Directory containing trigger config YAML files",
    )
    ap.add_argument(
        "--outdir",
        default="results/website_memory_audit",
        help="Output directory",
    )
    ap.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name",
    )
    ap.add_argument(
        "--measure_trigger",
        action="store_true",
        help="Also attempt trigger-generation measurement when a matching trigger config exists",
    )
    ap.add_argument(
        "--trigger_audit_steps",
        type=int,
        default=5,
        help="Maximum number of trigger-generation steps to run during memory auditing",
    )
    ap.add_argument(
        "--delete_failed_webnav",
        action="store_true",
        help="Delete dataset JSON files that fail the webnav memory audit",
    )
    ap.add_argument(
        "--delete_failed_trigger",
        action="store_true",
        help="Delete dataset JSON files that fail the trigger-generation memory audit",
    )
    args = ap.parse_args()

    dataset_items_dir = Path(args.dataset_items_dir)
    trigger_config_dir = Path(args.trigger_config_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = discover_site_records(dataset_items_dir, trigger_config_dir)
    rows: List[SiteMetrics] = []

    tokenizer_for_count = AutoTokenizer.from_pretrained(args.model)
    if tokenizer_for_count.pad_token is None:
        tokenizer_for_count.pad_token = tokenizer_for_count.eos_token

    for rec in records:
        print(f"\n=== Auditing {rec.name} ===")
        try:
            item = load_dataset_item(rec.dataset_json)
            fields = extract_capture_fields(item)
            prompt_text = build_prompt_from_dataset(rec.name, fields)

            axtree_tokens = token_len(tokenizer_for_count, fields["axtree_txt"])
            html_tokens = token_len(tokenizer_for_count, fields["pruned_html"])
            prompt_tokens = token_len(tokenizer_for_count, prompt_text)

            webnav_alloc, webnav_res, webnav_elapsed, webnav_ok, webnav_err = measure_webnav_memory(
                model_name=args.model,
                prompt_text=prompt_text,
            )

            trig_alloc = trig_res = trig_elapsed = trig_ok = trig_err = None
            if args.measure_trigger:
                trig_alloc, trig_res, trig_elapsed, trig_ok, trig_err = measure_trigger_memory(
                    rec.trigger_config,
                    dataset_json_path=rec.dataset_json,
                    max_steps=args.trigger_audit_steps,
                )

            rows.append(
                SiteMetrics(
                    website=rec.name,
                    dataset_json=str(rec.dataset_json),
                    trigger_config=str(rec.trigger_config) if rec.trigger_config else None,

                    url=fields["url"],
                    page_title=fields["title"],
                    axtree_chars=len(fields["axtree_txt"]),
                    axtree_lines=len(fields["axtree_txt"].splitlines()),
                    axtree_nodes_est=count_estimated_nodes(fields["axtree_txt"]),
                    axtree_interactive_est=count_interactive_est(fields["axtree_txt"]),
                    pruned_html_chars=len(fields["pruned_html"]),
                    pruned_html_lines=len(fields["pruned_html"].splitlines()),
                    iframe_count=fields["iframe_count"],
                    button_count=fields["button_count"],
                    link_count=fields["link_count"],
                    input_count=fields["input_count"],
                    total_dom_elements_est=fields["total_dom_elements_est"],

                    prompt_chars=len(prompt_text),
                    prompt_tokens=prompt_tokens,
                    axtree_tokens=axtree_tokens,
                    html_tokens=html_tokens,

                    webnav_peak_allocated_mb=webnav_alloc,
                    webnav_peak_reserved_mb=webnav_res,
                    webnav_elapsed_s=webnav_elapsed,
                    webnav_ok=webnav_ok,
                    webnav_error=webnav_err,

                    trigger_peak_allocated_mb=trig_alloc,
                    trigger_peak_reserved_mb=trig_res,
                    trigger_elapsed_s=trig_elapsed,
                    trigger_ok=trig_ok,
                    trigger_error=trig_err,
                )
            )

        except Exception as e:
            rows.append(
                SiteMetrics(
                    website=rec.name,
                    dataset_json=str(rec.dataset_json),
                    trigger_config=str(rec.trigger_config) if rec.trigger_config else None,

                    url="",
                    page_title="",
                    axtree_chars=0,
                    axtree_lines=0,
                    axtree_nodes_est=0,
                    axtree_interactive_est=0,
                    pruned_html_chars=0,
                    pruned_html_lines=0,
                    iframe_count=None,
                    button_count=None,
                    link_count=None,
                    input_count=None,
                    total_dom_elements_est=None,

                    prompt_chars=0,
                    prompt_tokens=0,
                    axtree_tokens=0,
                    html_tokens=0,

                    webnav_peak_allocated_mb=None,
                    webnav_peak_reserved_mb=None,
                    webnav_elapsed_s=None,
                    webnav_ok=False,
                    webnav_error=repr(e),

                    trigger_peak_allocated_mb=None,
                    trigger_peak_reserved_mb=None,
                    trigger_elapsed_s=None,
                    trigger_ok=None,
                    trigger_error=None,
                )
            )

    df = pd.DataFrame(asdict(r) for r in rows)
    df.to_csv(outdir / "website_feature_runtime_table.csv", index=False)

    make_scatter(
        df=df,
        xcol="prompt_tokens",
        ycol="webnav_peak_allocated_mb",
        outpath=outdir / "scatter_prompt_tokens_vs_webnav_vram.png",
        title="Prompt tokens vs peak webnav VRAM",
    )

    make_scatter(
        df=df,
        xcol="axtree_chars",
        ycol="webnav_peak_allocated_mb",
        outpath=outdir / "scatter_axtree_chars_vs_webnav_vram.png",
        title="AXTree chars vs peak webnav VRAM",
    )

    summary_cols = [
        "website", "url", "dataset_json", "trigger_config",
        "axtree_chars", "axtree_tokens", "prompt_tokens",
        "iframe_count", "axtree_interactive_est",
        "webnav_ok", "webnav_peak_allocated_mb", "webnav_elapsed_s",
        "trigger_ok", "trigger_peak_allocated_mb", "trigger_elapsed_s",
    ]
    df[summary_cols].to_csv(outdir / "website_summary_table.csv", index=False)

    make_pass_fail_summary(
        df=df,
        success_col="webnav_ok",
        outpath=outdir / "pass_fail_group_comparison_webnav.csv",
    )

    if args.measure_trigger:
        trigger_df = df[df["trigger_ok"].notna()].copy()
        if not trigger_df.empty:
            make_pass_fail_summary(
                df=trigger_df,
                success_col="trigger_ok",
                outpath=outdir / "pass_fail_group_comparison_trigger.csv",
            )

    # Optional deletion of failed sites
    to_delete = set()

    if args.delete_failed_webnav:
        failed_webnav = df[df["webnav_ok"] == False]["dataset_json"].dropna().tolist()
        to_delete.update(failed_webnav)

    if args.delete_failed_trigger and args.measure_trigger:
        failed_trigger = df[df["trigger_ok"] == False]["dataset_json"].dropna().tolist()
        to_delete.update(failed_trigger)

    deleted = []
    for path_str in sorted(to_delete):
        path = Path(path_str)
        if path.exists():
            path.unlink()
            deleted.append(str(path))

    if deleted:
        with open(outdir / "deleted_sites.txt", "w", encoding="utf-8") as f:
            for p in deleted:
                f.write(p + "\n")
        print(f"Deleted {len(deleted)} dataset JSON files.")

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()