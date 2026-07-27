from docmind.config import Settings
from docmind.pipeline import RAGPipeline


class FakeDocument:
    page_content = "The system uses maximal marginal relevance."
    metadata = {"source_name": "design.md", "page": 0}


class FakeRetriever:
    def invoke(self, query):
        return [FakeDocument()]


class FakeStore:
    def __init__(self):
        self.last_query = None

    def as_retriever(self, **kwargs):
        retriever = FakeRetriever()
        original_invoke = retriever.invoke

        def invoke(query):
            self.last_query = query
            return original_invoke(query)

        retriever.invoke = invoke
        return retriever


class FakeLLM:
    def generate(self, question, context, history):
        assert "[SOURCE 1: design.md, p. 1]" in context
        return "It uses MMR for diverse retrieval [Source 1]."


class InvalidCitationLLM:
    def generate(self, question, context, history):
        return "This statement cites a nonexistent passage [Source 99]."


class RepairableCitationLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, question, context, history):
        self.calls += 1
        if self.calls == 1:
            return "Sentence correction practice is the most useful content."
        return "Sentence correction practice is the most useful content [Source 1]."


def test_pipeline_returns_grounded_answer_and_diagnostics():
    pipeline = RAGPipeline(Settings(), llm=FakeLLM())
    pipeline.store = FakeStore()
    result = pipeline.ask("How is retrieval diversified?")
    assert "[Source 1]" in result.answer
    assert result.sources[0].citation == "design.md, p. 1"
    assert result.diagnostics["sources"] == 1
    assert result.diagnostics["citation_status"] == "verified"
    assert result.diagnostics["retrieval_mode"] == "hybrid"


def test_pipeline_uses_document_opening_for_overview_questions():
    pipeline = RAGPipeline(Settings(retrieval_k=2), llm=FakeLLM())
    store = FakeStore()
    pipeline.store = store
    pipeline.indexed_chunks = [
        FakeDocument(),
    ]
    result = pipeline.ask("What is it about?")
    assert result.diagnostics["retrieval_mode"] == "overview"
    assert "document title introduction purpose" in store.last_query


def test_pipeline_removes_out_of_range_citations():
    pipeline = RAGPipeline(Settings(), llm=InvalidCitationLLM())
    pipeline.store = FakeStore()
    pipeline.indexed_chunks = [FakeDocument()]
    result = pipeline.ask("What retrieval method is used?")
    assert "[Source 99]" not in result.answer
    assert result.diagnostics["citation_status"] == "invalid"
    assert result.diagnostics["invalid_citations"] == [99]


def test_pipeline_repairs_missing_citations_once():
    llm = RepairableCitationLLM()
    pipeline = RAGPipeline(Settings(), llm=llm)
    pipeline.store = FakeStore()
    pipeline.indexed_chunks = [FakeDocument()]
    result = pipeline.ask("What is the most useful thing in the book?")
    assert llm.calls == 2
    assert result.diagnostics["citation_status"] == "verified"
    assert result.diagnostics["retrieval_mode"] == "insight-hybrid"
