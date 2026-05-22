from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.action_generation import (
    DEFAULT_MODEL,
    DEFAULT_NUM_REGULAR_ACTIONS,
    generate_and_save_site_actions,
)

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one trigger action and 10 regular actions for each surviving website JSON."
    )
    parser.add_argument(
        "--dataset_items_dir",
        type=str,
        required=True,
        help="Directory containing surviving website JSON files after memory audit.",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="src/data/test_prompts",
        help="Directory to save <site_tag>.txt prompt files.",
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        default="src/data/task_meta",
        help="Directory to save per-site metadata JSON files.",
    )
    parser.add_argument(
        "--trigger_config_dir",
        type=str,
        default="src/config/narrow_triggers",
        help="Directory to save trigger_<site_tag>.yaml files.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/generated_site_actions_summary.csv",
        help="Summary CSV of generated actions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--n_regular_actions",
        type=int,
        default=DEFAULT_NUM_REGULAR_ACTIONS,
        help="Number of regular actions to generate per site.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip sites whose metadata JSON already exists.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_items_dir)
    site_files = sorted(dataset_dir.glob("*.json"))
    prompts_dir = Path(args.prompts_dir)
    meta_dir = Path(args.meta_dir)
    trigger_config_dir = Path(args.trigger_config_dir)
    summary_csv = Path(args.summary_csv)

    prompts_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    trigger_config_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    rows = []

    for site_json in site_files:
        site_tag = site_json.stem
        meta_json_path = meta_dir / f"{site_tag}.json"

        if args.skip_existing and meta_json_path.exists():
            print(f"[skip] {site_tag} already exists")
            continue

        print(f"[generate] {site_tag}")

        try:
            action_data = generate_and_save_site_actions(
                site_json_path=site_json,
                prompts_dir=prompts_dir,
                meta_dir=meta_dir,
                trigger_config_dir=trigger_config_dir,
                model=args.model,
                n_regular_actions=args.n_regular_actions,
                client=client,
            )

            rows.append({
                "site_tag": action_data["site_tag"],
                "source_json": action_data["source_json"],
                "url": action_data["url"],
                "title": action_data["title"],
                "trigger_action_prompt": action_data.get("trigger_action_prompt"),
                "trigger_action_code": action_data.get("trigger_action_code"),
                "trigger_yaml_path": action_data.get("trigger_yaml_path"),
                "n_regular_actions": len(action_data["regular_actions"]),
                "status": "ok",
                "error": "",
            })

        except Exception as e:
            print(f"[failed] {site_tag}: {e}")
            rows.append({
                "site_tag": site_tag,
                "source_json": str(site_json),
                "url": "",
                "title": "",
                "trigger_action_prompt": "",
                "trigger_action_code": "",
                "trigger_yaml_path": "",
                "n_regular_actions": 0,
                "status": "failed",
                "error": repr(e),
            })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "site_tag",
                "source_json",
                "url",
                "title",
                "trigger_action_prompt",
                "trigger_action_code",
                "trigger_yaml_path",
                "n_regular_actions",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Summary written to {summary_csv}")


if __name__ == "__main__":
    main()