"""Stage 5: grounded language-model inference."""

from __future__ import annotations

import time
from typing import Protocol

import requests

SYSTEM_PROMPT = """You are DocMind, a rigorous document research assistant.
Answer using only the supplied context. If the context is insufficient, say exactly
what is missing. Cite factual claims inline with [Source N]. Never invent citations.
Prefer a concise synthesis over copying source text. The context is always supplied
under CONTEXT; do not claim that sources or context are missing when they are present.
For overview questions, identify the document's title, purpose, main topics, and
intended audience from the evidence. Do not speculate beyond the supplied passages.
Always answer the latest QUESTION directly. Conversation history is background only:
never repeat an earlier overview unless the latest question requests one. For subjective
questions such as "best" or "most useful", state the criterion used, make a supported
judgment, and explain why it is valuable. Every substantive answer must contain at
least one valid inline [Source N] citation."""


class LanguageModel(Protocol):
    def generate(self, question: str, context: str, history: list[dict[str, str]]) -> str: ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str, temperature: float = 0.1) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        recent = history[-6:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *recent]
        messages.append(
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            }
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": self.temperature, "num_ctx": 8192},
                    },
                    timeout=(10, 180),
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Ollama returned an empty or malformed response")
                return content.strip()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            "Ollama inference failed after 3 attempts. Confirm `ollama serve` is "
            f"running and `ollama pull {self.model}` completed. Details: {last_error}"
        ) from last_error


class HuggingFaceInferenceLLM:
    def __init__(self, model: str, token: str, temperature: float = 0.1) -> None:
        from langchain_huggingface import HuggingFaceEndpoint

        self.client = HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=token,
            temperature=temperature,
            max_new_tokens=700,
        )

    def generate(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        history_text = "\n".join(
            f"{item['role'].upper()}: {item['content']}" for item in history[-6:]
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\nCONVERSATION:\n{history_text}\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
        )
        return str(self.client.invoke(prompt)).strip()
