"""DocMind Streamlit application."""

from __future__ import annotations

# The src-layout bootstrap must run before local package imports.
# ruff: noqa: E402

import html
import os
import sys
import tempfile
from pathlib import Path

# Streamlit executes this file as a script. Add the src-layout package explicitly so
# the app remains runnable even if an editable-install .pth file is ignored by Python.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import streamlit as st
from dotenv import load_dotenv

from docmind.config import Settings
from docmind.loaders import supported_extensions
from docmind.pipeline import RAGPipeline

load_dotenv()
st.set_page_config(page_title="DocMind", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 4.5rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #263142;}
    .hero {padding: 1.2rem 0 .4rem;}
    .eyebrow {letter-spacing: .14em; text-transform: uppercase; color: #60a5fa;
              font-weight: 700; font-size: .78rem;}
    .hero h1 {font-size: 3rem; margin: .2rem 0; letter-spacing: -.04em;}
    .hero p {color: #94a3b8; max-width: 760px; font-size: 1.05rem;}
    .source-card {border: 1px solid #334155; border-radius: 12px; padding: .8rem 1rem;
                  margin: .45rem 0; background: rgba(30,41,59,.35);}
    </style>
    <div class="hero">
      <div class="eyebrow">Source-grounded document intelligence</div>
      <h1>DocMind</h1>
      <p>Ask difficult questions across your documents. Every answer is grounded in
      retrieved evidence and linked back to its source.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("index_stats", None)


def save_uploads(files: list) -> tuple[tempfile.TemporaryDirectory, list[Path]]:
    temp_dir = tempfile.TemporaryDirectory(prefix="docmind_")
    paths: list[Path] = []
    used_names: dict[str, int] = {}
    for uploaded in files:
        safe_name = Path(uploaded.name).name
        used_names[safe_name] = used_names.get(safe_name, 0) + 1
        occurrence = used_names[safe_name]
        if occurrence > 1:
            source_path = Path(safe_name)
            safe_name = f"{source_path.stem}_{occurrence}{source_path.suffix}"
        destination = Path(temp_dir.name) / safe_name
        destination.write_bytes(uploaded.getbuffer())
        paths.append(destination)
    return temp_dir, paths


initialize_state()

with st.sidebar:
    st.header("Knowledge base")
    uploaded_files = st.file_uploader(
        "Add documents",
        type=[ext.removeprefix(".") for ext in supported_extensions()],
        accept_multiple_files=True,
        help="PDF, DOCX, TXT, Markdown, and CSV",
    )
    backend = st.segmented_control(
        "Vector index", ["FAISS", "Chroma"], default="FAISS"
    )
    with st.expander("Retrieval settings"):
        chunk_size = st.slider("Chunk size", 300, 1800, 900, 100)
        overlap = st.slider("Chunk overlap", 0, min(400, chunk_size - 1), 150, 25)
        retrieval_k = st.slider("Evidence passages", 2, 10, 5)
        use_threshold = st.checkbox(
            "Filter low-confidence passages",
            value=False,
            help="Useful for precise fact lookup; leave off for broad summaries.",
        )
        score_threshold = (
            st.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05)
            if use_threshold
            else None
        )
        use_reranker = st.checkbox(
            "Cross-encoder reranking",
            value=True,
            help="Higher answer precision; downloads a small reranker on first use.",
        )
        embedding_model = st.text_input(
            "Embedding model", "sentence-transformers/all-MiniLM-L6-v2"
        )
    with st.expander("Language model"):
        provider = st.selectbox("Provider", ["ollama", "huggingface"])
        default_model = (
            os.getenv("DOCMIND_LLM_MODEL", "llama3.2:3b")
            if provider == "ollama"
            else "mistralai/Mistral-7B-Instruct-v0.3"
        )
        model = st.text_input("Model", default_model)

    index_clicked = st.button(
        "Build knowledge base", type="primary", use_container_width=True
    )
    if index_clicked:
        if not uploaded_files:
            st.warning("Upload at least one document.")
        else:
            temp_handle, upload_paths = save_uploads(uploaded_files)
            settings = Settings.from_env(
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                retrieval_k=retrieval_k,
                fetch_k=max(20, retrieval_k * 3),
                score_threshold=score_threshold,
                reranker_model=(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    if use_reranker
                    else None
                ),
                vector_backend=str(backend).lower(),
                llm_provider=provider,
                llm_model=model,
            )
            pipeline = RAGPipeline(settings)
            try:
                with st.spinner("Chunking, embedding, and indexing documents…"):
                    stats = pipeline.index(upload_paths)
                st.session_state.pipeline = pipeline
                st.session_state.index_stats = stats
                st.session_state.temp_handle = temp_handle
                st.session_state.messages = []
                st.success(f"Indexed {stats['chunks']} passages.")
            except Exception as exc:
                temp_handle.cleanup()
                st.error(f"Indexing failed: {exc}")

    if st.session_state.index_stats:
        stats = st.session_state.index_stats
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["documents"])
        col2.metric("Chunks", stats["chunks"])
        st.caption(f"Index: {str(stats['backend']).upper()}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if st.session_state.pipeline is None:
    left, center, right = st.columns([1, 1.5, 1])
    with center:
        st.info("Upload documents and build the knowledge base to begin.")
        st.markdown(
            """
            **Five-stage RAG pipeline**

            1. Structure-aware document chunking
            2. Normalized Hugging Face embeddings
            3. FAISS or persistent Chroma vector indexing
            4. Maximal marginal relevance retrieval
            5. Grounded LLM synthesis with traceable sources
            """
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Evidence · {len(message['sources'])} passages"):
                for number, source in enumerate(message["sources"], 1):
                    st.markdown(
                        f'<div class="source-card"><b>[Source {number}] '
                        f'{html.escape(source.citation)}</b><br>'
                        f'{html.escape(source.text[:650])}</div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input(
    "Ask a question about your documents…",
    disabled=st.session_state.pipeline is None,
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and reasoning…"):
            try:
                result = st.session_state.pipeline.ask(question, history)
                st.markdown(result.answer)
                with st.expander(f"Evidence · {len(result.sources)} passages", expanded=True):
                    for number, source in enumerate(result.sources, 1):
                        confidence = (
                            f" · confidence {source.score:.0%}"
                            if source.score is not None
                            else ""
                        )
                        st.markdown(
                            f'<div class="source-card"><b>[Source {number}] '
                            f'{html.escape(source.citation)}{confidence}</b><br>'
                            f'{html.escape(source.text[:650])}</div>',
                            unsafe_allow_html=True,
                        )
                citation_status = result.diagnostics.get("citation_status", "not checked")
                st.caption(
                    f"{str(result.diagnostics.get('retrieval_mode', 'retrieval')).title()} "
                    f"retrieval · {len(result.sources)} passages · "
                    f"{result.diagnostics.get('retrieval_ms', 0)} ms · "
                    f"citations {citation_status}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    }
                )
            except Exception as exc:
                st.error(str(exc))
