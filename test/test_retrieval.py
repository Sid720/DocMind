from dataclasses import dataclass

import pytest

from docmind.models import SourceChunk
from docmind.retrieval import (
    format_context,
    hybrid_retrieve,
    is_insight_query,
    is_overview_query,
    lexical_retrieve,
    overview_chunks,
    rerank_sources,
    resolve_query,
    retrieve,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict


class FakeRetriever:
    def invoke(self, query):
        return [FakeDocument("Grounded evidence.", {"source_name": "paper.pdf", "page": 2})]


class FakeStore:
    def __init__(self):
        self.kwargs = None

    def as_retriever(self, **kwargs):
        self.kwargs = kwargs
        return FakeRetriever()


def test_retrieve_uses_mmr_and_preserves_source_metadata():
    store = FakeStore()
    result = retrieve(store, "What is the result?", k=3, fetch_k=12)
    assert store.kwargs["search_type"] == "mmr"
    assert store.kwargs["search_kwargs"]["k"] == 3
    assert result[0].citation == "paper.pdf, p. 3"


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        retrieve(FakeStore(), "  ")


def test_context_is_numbered_and_bounded():
    chunks = [
        SourceChunk("alpha", {"source_name": "a.txt"}),
        SourceChunk("beta" * 100, {"source_name": "b.txt"}),
    ]
    context = format_context(chunks, max_characters=60)
    assert "[SOURCE 1: a.txt]" in context
    assert "b.txt" not in context


def test_overview_intent_detection():
    assert is_overview_query("What is it about?")
    assert is_overview_query("What is this PDF about?")
    assert is_overview_query("  SUMMARIZE THE DOCUMENT ")
    assert not is_overview_query("What is subject-verb agreement?")


def test_overview_prioritizes_opening_pages():
    documents = [
        FakeDocument("Late appendix", {"document_id": "a", "page": 20, "chunk_id": "3"}),
        FakeDocument("Book introduction", {"document_id": "a", "page": 0, "chunk_id": "1"}),
        FakeDocument("Table of contents", {"document_id": "a", "page": 1, "chunk_id": "2"}),
    ]
    result = overview_chunks(documents, k=2)
    assert [chunk.text for chunk in result] == ["Book introduction", "Table of contents"]


def test_follow_up_query_is_resolved_from_conversation():
    history = [{"role": "user", "content": "Explain subject-verb agreement"}]
    assert resolve_query("Why is that important?", history).startswith(
        "Explain subject-verb agreement"
    )
    assert resolve_query("Explain sentence correction rules", history) == (
        "Explain sentence correction rules"
    )


def test_subjective_value_question_has_dedicated_intent():
    assert is_insight_query("Give me the best thing that the book contains")
    assert is_insight_query("What is the most useful chapter?")
    assert not is_insight_query("Explain articles")


def test_lexical_retrieval_finds_exact_rare_term():
    documents = [
        FakeDocument("General grammar concepts", {"chunk_id": "1"}),
        FakeDocument("The GMAT uses sentence correction.", {"chunk_id": "2"}),
    ]
    results = lexical_retrieve(documents, "GMAT sentence correction", k=1)
    assert results[0].metadata["chunk_id"] == "2"
    assert results[0].score > 0


def test_hybrid_retrieval_fuses_dense_and_lexical_results():
    documents = [
        FakeDocument("Grounded evidence.", {"source_name": "paper.pdf", "page": 2}),
        FakeDocument("Exact GMAT terminology.", {"chunk_id": "exact"}),
    ]
    results = hybrid_retrieve(FakeStore(), documents, "GMAT terminology", k=2)
    assert len(results) == 2
    assert results[0].score == 1.0


def test_cross_encoder_reranker_reorders_passages():
    class FakeCrossEncoder:
        def predict(self, pairs):
            assert len(pairs) == 2
            return [-1.0, 3.0]

    chunks = [
        SourceChunk("weak passage", {"chunk_id": "weak"}),
        SourceChunk("strong passage", {"chunk_id": "strong"}),
    ]
    ranked = rerank_sources("query", chunks, "unused", model=FakeCrossEncoder())
    assert ranked[0].metadata["chunk_id"] == "strong"
    assert ranked[0].score > ranked[1].score
