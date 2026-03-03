"""
utils/providers.py

Responsibility
--------------
A single file that abstracts LLM access so the project can switch between:
- OpenAI API (chat.completions)
- Hugging Face local inference (transformers; loads once and reuses)
- Hugging Face hosted inference API (optional)

This supports your requirement to use either local models or online interfaces.

Used by
-------
run_demo.py, eval_trigger.py (and later trigger optimization)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    provider: str  # "openai" | "hf_local" | "hf_hosted"
    model: str
    temperature: float = 0.0
    max_tokens: int = 200


class BaseProvider:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAIProvider requires `openai` package (pip install openai)") from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI(api_key=api_key)

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


class HFLocalProvider(BaseProvider):
    """
    Local transformers inference. Loads once and reuses (important for speed).
    """
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("HFLocalProvider requires transformers+torch") from e

        # HF_TOKEN only needed for gated models (like Llama) if you didn't already cache them.
        # It's ok if absent for public models.
        token = os.environ.get("HF_TOKEN")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, token=token)
        self.model.eval()

        if torch.cuda.is_available():
            self.model.to("cuda")

    def complete(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=(self.cfg.temperature > 0),
                temperature=max(self.cfg.temperature, 1e-6),
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the tail beyond the prompt if possible
        return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


class HFHostedProvider(BaseProvider):
    """
    Optional: Hugging Face Inference API.
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
            "parameters": {"max_new_tokens": self.cfg.max_tokens, "temperature": self.cfg.temperature},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        # HF returns different shapes depending on model/task; handle common case:
        if isinstance(data, list) and data and "generated_text" in data[0]:
            gen = data[0]["generated_text"]
            return gen[len(prompt) :].strip() if isinstance(gen, str) else str(gen)
        return str(data)


def make_provider(cfg: ProviderConfig) -> BaseProvider:
    p = cfg.provider.lower().strip()
    if p == "openai":
        return OpenAIProvider(cfg)
    if p == "hf_local":
        return HFLocalProvider(cfg)
    if p == "hf_hosted":
        return HFHostedProvider(cfg)
    raise ValueError(f"Unknown provider: {cfg.provider}")