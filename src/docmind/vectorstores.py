"""Stage 3: interchangeable FAISS and persistent Chroma indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def build_vector_store(
    documents: list[Document],
    embeddings: Any,
    backend: str,
    persist_directory: Path,
    collection_name: str,
):
    persist_directory.mkdir(parents=True, exist_ok=True)
    if backend == "faiss":
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(documents, embeddings)
        faiss_dir = persist_directory / "faiss"
        store.save_local(str(faiss_dir))
        manifest = {
            "backend": backend,
            "collection": collection_name,
            "chunks": len(documents),
        }
        (persist_directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return store
    if backend == "chroma":
        import chromadb
        from langchain_chroma import Chroma

        chroma_directory = persist_directory / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_directory))
        try:
            client.delete_collection(collection_name)
        except ValueError:
            pass
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            client=client,
        )
    raise ValueError(f"Unknown vector backend: {backend}")
