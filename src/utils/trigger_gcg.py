"""
utils/trigger_gcg.py

Responsibility
--------------
Placeholder for a white-box (gradient-based) GCG-style trigger optimizer.
This will require a local model backend that exposes logits/gradients
(e.g., transformers HFLocalProvider plus a specialized optimizer implementation).

Used by
-------
src/make_trigger.py via utils/trigger_registry.py
"""

from __future__ import annotations

import json
import math
import pathlib
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.gcg_plus import run, GCGConfig
from src.utils.promptify import promptify_json


class GCGTrigger:
    name = "gcg"

    def run(self, *, cfg: dict, items: list[dict], provider=None) -> dict:

        # -----------------------------
        # read config
        # -----------------------------
        json_path = pathlib.Path(cfg["json"])
        target = cfg["target"]
        trigger_length = cfg["trigger_length"]
        include_target = cfg["include_target"]
        loss_fn = cfg["loss_fn"]
        search_width = cfg["search_width"]
        top_k = cfg["top_k"]
        model_name_short = cfg.get("model", "llama3")
        dtype = cfg.get("dtype", "float16")
        device = cfg.get("device", "cuda")

        # -----------------------------
        # model selection
        # -----------------------------
        match model_name_short:
            case "mistral-7B":
                model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            case "mistral-24B":
                model_name = "mistralai/Mistral-Small-24B-Instruct-250"
            case "llama2":
                model_name = "meta-llama/Llama-2-7b-chat-hf"
            case "llama3":
                model_name = "meta-llama/Llama-3.1-8B-Instruct"
            case _:
                raise ValueError(f"Unknown model {model_name_short}")

        # -----------------------------
        # load model
        # -----------------------------
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # -----------------------------
        # load json prompt
        # -----------------------------
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            obs_dict = json.load(f)

        sys_content, user_content = promptify_json(obs_dict)

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

        # -----------------------------
        # DEBUG BYPASS
        # -----------------------------
        debug_trigger = cfg.get("debug_return_trigger")
        if debug_trigger is not None:
            return {
                "trigger": debug_trigger,
                "trace": {
                    "best_loss": None,
                    "num_steps": 0,
                    "time_to_find": 0.0,
                    "debug_bypass": True,
                },
            }

        # -----------------------------
        # build starting trigger
        # -----------------------------
        if include_target:
            n_target_tokens = len(tokenizer.tokenize(target))
            total_xs = trigger_length - n_target_tokens
            if trigger_length < n_target_tokens:
                raise ValueError("trigger_length must be >= number of target tokens")

            starting_str = (
                math.floor(total_xs / 2) * "x "
                + target
                + math.ceil(total_xs / 2) * "x "
            )
        else:
            starting_str = trigger_length * "x "

        # -----------------------------
        # loss selection
        # -----------------------------
        match loss_fn:
            case "cw":
                use_mm = False
                use_cw = True
            case "mm":
                use_mm = True
                use_cw = False
            case "ce":
                use_mm = False
                use_cw = False

        # -----------------------------
        # GCG config
        # -----------------------------
        config = GCGConfig(
            num_steps=300,
            optim_str_init=starting_str,
            search_width=search_width,
            batch_size=4,
            topk=top_k,
            use_mellowmax=use_mm,
            use_cw_loss=use_cw,
            early_stop=True,
            verbosity="INFO",
            add_space_before_target=True if model_name_short == "llama2" else False,
        )

        # -----------------------------
        # run optimization
        # -----------------------------
        result = run(
            model,
            tokenizer,
            messages,
            target,
            config,
        )

        return {
            "trigger": result.best_string,
            "trace": {
                "best_loss": result.best_loss,
                "num_steps": result.num_steps,
                "time_to_find": result.time_to_find_s,
            },
        }