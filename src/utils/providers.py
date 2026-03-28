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
    max_tokens: int = 200


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
    """
    Local transformers inference.
    """
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("HFLocalProvider requires transformers+torch") from e

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
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()

    def complete_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}" for m in messages
        )
        return self.complete(prompt)


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