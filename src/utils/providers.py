"""
src/utils/providers.py

Responsibility
--------------
Provider abstraction for chat-style LLM calls.

Supports:
- OpenAI API
- Hugging Face local inference
- Hugging Face hosted inference API

Used by
-------
run_demo.py, agent_wrapper.py
"""
from __future__ import annotations

from dotenv import load_dotenv
import pathlib
import os
from dataclasses import dataclass
from typing import List, Dict

# Load .env from project root
PROJECT_ROOT = pathlib.Path(os.getcwd())
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ProviderConfig:
    provider: str  # "openai" | "hf_local" | "hf_hosted"
    model: str
    temperature: float = 0.0
    max_tokens: int = 64


class BaseProvider:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def complete_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Default fallback: flatten chat messages into one prompt.
        """
        prompt = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}" for m in messages
        )
        return self.complete(prompt)


class OpenAIProvider(BaseProvider):
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAIProvider requires `openai` package (pip install openai)"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        self.client = OpenAI(api_key=api_key)

    def complete(self, prompt: str) -> str:
        return self.complete_messages([{"role": "user", "content": prompt}])

    def complete_messages(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


class HFLocalProvider(BaseProvider):
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("HFLocalProvider requires transformers+torch+bitsandbytes") from e

        token = os.environ.get("HF_TOKEN")
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=token)

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                token=token,
                quantization_config=bnb_config,
                device_map="auto",
            )

            # Important: do NOT call self.model.to("cuda") here
            self.device = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                token=token,
                dtype=torch.float32,
            )
            self.device = torch.device("cpu")
            self.model.to(self.device)

        self.model.eval()

    def complete(self, prompt: str) -> str:
        return self.complete_messages([{"role": "user", "content": prompt}])

    def complete_messages(self, messages: List[Dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Put inputs on the same device as the input embedding layer.
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                do_sample=(self.cfg.temperature > 0),
                max_new_tokens=self.cfg.max_tokens,
                temperature=max(self.cfg.temperature, 1e-6) if self.cfg.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generation = self.tokenizer.batch_decode(
            output[:, prompt_len:],
            skip_special_tokens=True,
        )[0]

        return generation.strip()

class HFHostedProvider(BaseProvider):
    """
    Hugging Face Inference API.
    """
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HFHostedProvider requires HF_TOKEN in environment")
        self.token = token

    def complete(self, prompt: str) -> str:
        import requests  # type: ignore

        url = f"https://api-inference.huggingface.co/models/{self.cfg.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
            },
        }

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list) and data and "generated_text" in data[0]:
            gen = data[0]["generated_text"]
            if isinstance(gen, str) and gen.startswith(prompt):
                return gen[len(prompt):].strip()
            return str(gen)

        return str(data)

    def complete_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}" for m in messages
        )
        return self.complete(prompt)


def make_provider(cfg: ProviderConfig) -> BaseProvider:
    p = cfg.provider.lower().strip()

    if p == "openai":
        return OpenAIProvider(cfg)
    if p == "hf_local":
        return HFLocalProvider(cfg)
    if p == "hf_hosted":
        return HFHostedProvider(cfg)

    raise ValueError(f"Unknown provider: {cfg.provider}")