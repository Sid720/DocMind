# DocMind

DocMind is an advanced, source-grounded Retrieval-Augmented Generation (RAG)
system for asking questions across PDF, DOCX, Markdown, text, and CSV files.
It combines Hugging Face sentence embeddings, FAISS or Chroma vector search,
maximal marginal relevance retrieval, and local or hosted language-model inference.

## Architecture

```text
Documents → Parse + metadata → Recursive chunking → HF embeddings
          → FAISS / Chroma index → MMR retrieval → grounded LLM answer
                                                ↘ source references
```

The five stages are explicit and independently testable:

1. **Chunking** preserves file, page, document hash, and character-offset metadata.
2. **Embedding generation** uses normalized Hugging Face sentence embeddings.
3. **Vector search** supports fast in-memory FAISS and persistent Chroma.
4. **Context retrieval** fuses dense MMR and lexical BM25 rankings with
   reciprocal-rank fusion, improving both semantic questions and exact-term lookup.
5. **LLM inference** requires answers to stay within context and cite `[Source N]`.

## Quick start

Python 3.10–3.12 is recommended.

```bash
cd /Users/sidsgedam/Desktop/V_mvp/DocMind
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The default inference provider is local [Ollama](https://ollama.com/):

```bash
ollama pull llama3.2:3b
ollama serve
.venv/bin/python -m streamlit run app.py
```

Open the URL printed by Streamlit, upload documents, choose FAISS or Chroma,
click **Build knowledge base**, and ask questions.

## Hugging Face hosted inference

Copy `.env.example` to `.env`, set `DOCMIND_HF_TOKEN`, then select
`huggingface` and a compatible instruction model in the interface.

## Engineering details

- SHA-256 document identities support deterministic provenance.
- Page and chunk metadata survive every pipeline stage.
- MMR balances semantic relevance with evidence diversity.
- BM25 recovers exact names, acronyms, numbers, and domain terminology that dense
  search can miss.
- An optional MS MARCO cross-encoder jointly scores query/passage pairs after hybrid
  retrieval; it is enabled by default in the interface for higher precision.
- Query-intent routing gives document overviews dedicated title, introduction,
  purpose, contents, and audience retrieval.
- Short follow-up questions are resolved against recent conversation context.
- Optional confidence filtering suppresses weak evidence for precision-oriented tasks.
- Generated citation numbers are validated before an answer is displayed.
- Missing citations trigger one constrained answer-repair pass before display.
- Prompt injection in documents is constrained by a context-only system prompt.
- Uploaded files are placed in an isolated temporary directory for the session.
- Duplicate files are removed by SHA-256 identity and malformed or empty uploads fail
  with actionable errors.
- Ollama inference uses bounded retries, response validation, and an 8K context window.
- Rebuilding a Chroma knowledge base replaces its collection instead of silently
  accumulating duplicate chunks.
- The model and vector index are constructed once per knowledge-base build.
- Dependency injection makes the orchestration testable without model downloads.

## Tests

```bash
pytest -q
ruff check .
```

Tests cover configuration invariants, safe ingestion, deduplication, query-intent
routing, follow-up resolution, BM25 ranking, hybrid fusion, context bounds,
citation validation, source traceability, and end-to-end orchestration with
deterministic fakes.

The `docmind.evaluation` module provides reproducible retrieval recall@k, hit rate,
and mean reciprocal rank metrics for benchmark datasets. This keeps retrieval
quality measurable instead of relying only on attractive demonstrations.

## Project layout

```text
app.py                    Streamlit interface
src/docmind/
  loaders.py              Multi-format parsing and provenance
  chunking.py             Stage 1
  embeddings.py           Stage 2
  vectorstores.py         Stage 3
  retrieval.py            Stage 4
  llm.py                  Stage 5
  pipeline.py             Pipeline orchestration
tests/                    Fast unit tests
```

## Limitations and next research steps

The current baseline is deliberately inspectable. Production extensions could
add hybrid BM25+dense retrieval, cross-encoder reranking, evaluation with RAGAS,
OCR for scanned PDFs, tenant-isolated collections, and citation-faithfulness
scoring. Those are natural experiments rather than hidden complexity in the core.
