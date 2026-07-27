"""Stage 1: structure-aware recursive chunking."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document], chunk_size: int = 900, chunk_overlap: int = 150
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        doc_id = chunk.metadata.get("document_id", "document")
        start = chunk.metadata.get("start_index", 0)
        chunk.metadata["chunk_id"] = f"{str(doc_id)[:12]}:{start}:{index}"
    return chunks

