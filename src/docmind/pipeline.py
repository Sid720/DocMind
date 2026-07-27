"""Orchestration for DocMind's five-stage RAG process."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable

from docmind.chunking import chunk_documents
from docmind.config import Settings
from docmind.embeddings import get_embeddings
from docmind.llm import HuggingFaceInferenceLLM, LanguageModel, OllamaLLM
from docmind.loaders import load_documents
from docmind.models import RAGAnswer
from docmind.retrieval import (
    expand_insight_query,
    format_context,
    hybrid_retrieve,
    is_insight_query,
    is_overview_query,
    overview_chunks,
    rerank_sources,
    resolve_query,
    retrieve,
)
from docmind.vectorstores import build_vector_store


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embeddings: Any | None = None,
        llm: LanguageModel | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.embeddings = embeddings
        self.llm = llm
        self.store: Any | None = None
        self.indexed_chunks: list[Any] = []
        self.document_count = 0
        self.chunk_count = 0

    def index(self, paths: Iterable[str | Path]) -> dict[str, int | str]:
        documents = load_documents(paths)
        chunks = chunk_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("No text could be extracted from the supplied documents")
        embeddings = self.embeddings or get_embeddings(self.settings.embedding_model)
        self.store = build_vector_store(
            chunks,
            embeddings,
            self.settings.vector_backend,
            self.settings.persist_directory,
            self.settings.collection_name,
        )
        self.indexed_chunks = chunks
        self.document_count = len({d.metadata.get("document_id") for d in documents})
        self.chunk_count = len(chunks)
        return {
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "backend": self.settings.vector_backend,
        }

    def ask(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> RAGAnswer:
        if self.store is None:
            raise RuntimeError("Index documents before asking a question")
        if not question.strip():
            raise ValueError("Ask a non-empty question")
        conversation = history or []
        resolved_query = resolve_query(question, conversation)
        started = time.perf_counter()
        overview = is_overview_query(question)
        insight = is_insight_query(question)
        if overview and self.indexed_chunks:
            opening_sources = overview_chunks(
                self.indexed_chunks, min(2, self.settings.retrieval_k)
            )
            descriptive_sources = retrieve(
                self.store,
                "document title introduction purpose main topics table of contents "
                "intended audience",
                self.settings.retrieval_k,
                self.settings.fetch_k,
            )
            sources = []
            seen_chunks: set[str] = set()
            for source in [*opening_sources, *descriptive_sources]:
                identity = str(
                    source.metadata.get(
                        "chunk_id", (source.citation, source.text[:80])
                    )
                )
                if identity not in seen_chunks:
                    seen_chunks.add(identity)
                    sources.append(source)
                if len(sources) == self.settings.retrieval_k:
                    break
        else:
            retrieval_query = (
                expand_insight_query(resolved_query) if insight else resolved_query
            )
            sources = hybrid_retrieve(
                self.store,
                self.indexed_chunks,
                retrieval_query,
                self.settings.retrieval_k,
                self.settings.fetch_k,
            )
            if self.settings.score_threshold is not None:
                sources = [
                    source
                    for source in sources
                    if source.score is None or source.score >= self.settings.score_threshold
                ]
            if self.settings.reranker_model and len(sources) > 1:
                sources = rerank_sources(
                    retrieval_query, sources, self.settings.reranker_model
                )
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)
        if not sources:
            return RAGAnswer(
                answer="I could not find relevant evidence in the indexed documents.",
                query=question,
                diagnostics={"retrieval_ms": retrieval_ms, "sources": 0},
            )
        context = format_context(sources)
        model = self.llm or self._create_llm()
        effective_question = (
            "Provide a high-level overview of the indexed document: explain its title, "
            "purpose, main subjects, and intended audience."
            if overview
            else (
                f"{question}\nChoose the most broadly useful content based on practical "
                "GMAT value, explain your criterion, and support the judgment."
                if insight
                else question
            )
        )
        answer = model.generate(effective_question, context, conversation)
        cited_numbers = _citation_numbers(answer)
        if not cited_numbers:
            repair_question = (
                f"{effective_question}\n\nYour previous draft had no source citation:\n"
                f"{answer}\n\nRewrite the answer to address only the current question. "
                "Preserve supported conclusions and add valid inline [Source N] citations."
            )
            answer = model.generate(repair_question, context, [])
            cited_numbers = _citation_numbers(answer)
        invalid_citations = sorted(
            number for number in cited_numbers if number < 1 or number > len(sources)
        )
        citation_status = (
            "invalid" if invalid_citations else "verified" if cited_numbers else "missing"
        )
        if invalid_citations:
            invalid_pattern = "|".join(map(str, invalid_citations))
            answer = re.sub(
                rf"\[Source\s+({invalid_pattern})\]",
                "[unsupported citation removed]",
                answer,
                flags=re.IGNORECASE,
            )
        return RAGAnswer(
            answer=answer,
            sources=sources,
            query=question,
            diagnostics={
                "retrieval_ms": retrieval_ms,
                "sources": len(sources),
                "context_characters": len(context),
                "retrieval_mode": (
                    "overview" if overview else "insight-hybrid" if insight else "hybrid"
                ),
                "resolved_query": resolved_query,
                "citation_status": citation_status,
                "invalid_citations": invalid_citations,
                "reranked": bool(self.settings.reranker_model and not overview),
            },
        )

    def _create_llm(self) -> LanguageModel:
        if self.settings.llm_provider == "ollama":
            return OllamaLLM(
                self.settings.llm_model,
                self.settings.ollama_url,
                self.settings.temperature,
            )
        if self.settings.llm_provider == "huggingface":
            import os

            token = os.getenv("DOCMIND_HF_TOKEN")
            if not token:
                raise RuntimeError("Set DOCMIND_HF_TOKEN for Hugging Face inference")
            return HuggingFaceInferenceLLM(
                self.settings.llm_model, token, self.settings.temperature
            )
        raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")


def _citation_numbers(answer: str) -> set[int]:
    return {
        int(match)
        for match in re.findall(r"\[Source\s+(\d+)\]", answer, flags=re.IGNORECASE)
    }
