"""Validated application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    fetch_k: int = 20
    score_threshold: float | None = None
    reranker_model: str | None = None
    vector_backend: str = "faiss"
    persist_directory: Path = Path(".docmind")
    collection_name: str = "documents"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.retrieval_k < 1 or self.fetch_k < self.retrieval_k:
            raise ValueError("fetch_k must be >= retrieval_k >= 1")
        if self.vector_backend not in {"faiss", "chroma"}:
            raise ValueError("vector_backend must be 'faiss' or 'chroma'")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        values: dict[str, object] = {
            "llm_provider": os.getenv("DOCMIND_LLM_PROVIDER", "ollama"),
            "llm_model": os.getenv("DOCMIND_LLM_MODEL", "llama3.2:3b"),
            "ollama_url": os.getenv("DOCMIND_OLLAMA_URL", "http://localhost:11434"),
        }
        values.update(overrides)
        return cls(**values)
