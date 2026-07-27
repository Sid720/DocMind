"""Domain models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceChunk:
    text: str
    metadata: dict[str, Any]
    score: float | None = None

    @property
    def citation(self) -> str:
        source = self.metadata.get("source_name", self.metadata.get("source", "Unknown"))
        page = self.metadata.get("page")
        section = self.metadata.get("section")
        location = f"p. {int(page) + 1}" if isinstance(page, int) else section
        return f"{source}, {location}" if location else str(source)


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    query: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

