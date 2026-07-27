"""Stage 4: diversity-aware retrieval and source formatting."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from docmind.models import SourceChunk

OVERVIEW_QUERIES = {
    "what is it about",
    "what is this about",
    "what is the document about",
    "what is this document about",
    "summarize this",
    "summarize the document",
    "give me an overview",
    "overview",
}
OVERVIEW_PATTERNS = (
    "what is it about",
    "what is this about",
    "what is the pdf about",
    "what is this pdf about",
    "what does this document cover",
    "tell me about the document",
    "summarize",
    "overview",
)
FOLLOW_UP_PREFIXES = (
    "what about",
    "how about",
    "why is that",
    "explain that",
    "explain it",
    "tell me more",
    "and ",
)
INSIGHT_PATTERNS = (
    "best thing",
    "most useful",
    "most important",
    "key takeaway",
    "main takeaway",
    "best part",
    "most valuable",
)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'-]+")


def is_overview_query(query: str) -> bool:
    normalized = " ".join(query.lower().strip().rstrip("?.!").split())
    return normalized in OVERVIEW_QUERIES or any(
        phrase in normalized for phrase in OVERVIEW_PATTERNS
    )


def resolve_query(query: str, history: list[dict[str, str]]) -> str:
    """Attach the previous user turn to short or referential follow-up questions."""
    normalized = " ".join(query.strip().split())
    lower = normalized.lower()
    is_follow_up = (
        len(TOKEN_PATTERN.findall(normalized)) <= 5
        and (
            lower.startswith(FOLLOW_UP_PREFIXES)
            or any(word in TOKEN_PATTERN.findall(lower) for word in ("it", "that", "this"))
        )
    )
    if not is_follow_up:
        return normalized
    previous_questions = [
        item.get("content", "").strip()
        for item in reversed(history)
        if item.get("role") == "user" and item.get("content", "").strip()
    ]
    return (
        f"{previous_questions[0]} — follow-up: {normalized}"
        if previous_questions
        else normalized
    )


def is_insight_query(query: str) -> bool:
    normalized = " ".join(query.lower().strip().rstrip("?.!").split())
    return any(pattern in normalized for pattern in INSIGHT_PATTERNS)


def expand_insight_query(query: str) -> str:
    return (
        f"{query} practical high-value concepts core rules worked examples exercises "
        "common GMAT errors sentence correction"
    )


def overview_chunks(documents: list[Any], k: int = 5) -> list[SourceChunk]:
    """Select representative passages, prioritizing each document's opening."""
    if not documents:
        return []
    groups: dict[str, list[Any]] = {}
    for document in documents:
        key = str(document.metadata.get("document_id", document.metadata.get("source", "")))
        groups.setdefault(key, []).append(document)

    selected: list[Any] = []
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item.metadata.get("page", 0),
                item.metadata.get("start_index", 0),
            ),
        )
        # The beginning normally contains the title, abstract, preface, or contents.
        selected.extend(ordered[:2])

    remaining = max(0, k - len(selected))
    flattened = [item for group in groups.values() for item in group]
    if remaining and flattened:
        step = max(1, len(flattened) // remaining)
        selected.extend(flattened[::step][:remaining])

    unique: list[SourceChunk] = []
    seen: set[str] = set()
    for document in selected:
        chunk_id = str(document.metadata.get("chunk_id", id(document)))
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(
                SourceChunk(text=document.page_content, metadata=dict(document.metadata))
            )
        if len(unique) == k:
            break
    return unique


def retrieve(store: Any, query: str, k: int = 5, fetch_k: int = 20) -> list[SourceChunk]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.65},
    )
    documents = retriever.invoke(query)
    return [SourceChunk(text=doc.page_content, metadata=dict(doc.metadata)) for doc in documents]


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def lexical_retrieve(documents: list[Any], query: str, k: int = 10) -> list[SourceChunk]:
    """Dependency-free BM25 ranking for exact terms, names, numbers, and acronyms."""
    query_tokens = _tokens(query)
    if not query_tokens or not documents:
        return []
    tokenized = [_tokens(document.page_content) for document in documents]
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    frequencies = Counter(token for tokens in tokenized for token in set(tokens))
    scored: list[tuple[float, Any]] = []
    for document, tokens in zip(documents, tokenized, strict=True):
        counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            document_frequency = frequencies[token]
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            frequency = counts[token]
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(tokens) / max(1, average_length)
            )
            score += inverse_frequency * (frequency * 2.5 / denominator)
        if score > 0:
            scored.append((score, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        SourceChunk(text=document.page_content, metadata=dict(document.metadata), score=score)
        for score, document in scored[:k]
    ]


def hybrid_retrieve(
    store: Any,
    documents: list[Any],
    query: str,
    k: int = 5,
    fetch_k: int = 20,
) -> list[SourceChunk]:
    """Fuse dense MMR and BM25 rankings using reciprocal-rank fusion."""
    dense = retrieve(store, query, max(k, min(fetch_k, 10)), fetch_k)
    lexical = lexical_retrieve(documents, query, max(k, min(fetch_k, 10)))
    fused: dict[str, tuple[float, SourceChunk]] = {}
    for results, weight in ((dense, 1.0), (lexical, 0.9)):
        for rank, chunk in enumerate(results, start=1):
            identity = str(
                chunk.metadata.get("chunk_id", (chunk.citation, chunk.text[:100]))
            )
            contribution = weight / (60 + rank)
            previous_score = fused.get(identity, (0.0, chunk))[0]
            fused[identity] = (previous_score + contribution, chunk)
    ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)
    if not ranked:
        return []
    best = ranked[0][0]
    return [
        SourceChunk(
            text=chunk.text,
            metadata=chunk.metadata,
            score=round(score / best, 4),
        )
        for score, chunk in ranked[:k]
    ]


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_sources(
    query: str,
    chunks: list[SourceChunk],
    model_name: str,
    *,
    model: Any | None = None,
) -> list[SourceChunk]:
    """Apply a cross-encoder to jointly score each query/passage pair."""
    if len(chunks) < 2:
        return chunks
    encoder = model or _cross_encoder(model_name)
    scores = encoder.predict([(query, chunk.text) for chunk in chunks])
    ranked = sorted(
        zip(scores, chunks, strict=True),
        key=lambda item: float(item[0]),
        reverse=True,
    )
    return [
        SourceChunk(
            text=chunk.text,
            metadata=chunk.metadata,
            score=round(1 / (1 + math.exp(-float(score))), 4),
        )
        for score, chunk in ranked
    ]


def format_context(chunks: list[SourceChunk], max_characters: int = 14_000) -> str:
    sections: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        section = f"[SOURCE {index}: {chunk.citation}]\n{chunk.text.strip()}"
        if used + len(section) > max_characters:
            break
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)
