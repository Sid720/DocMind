"""Secure, metadata-preserving document ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024


def supported_extensions() -> tuple[str, ...]:
    return tuple(LOADERS)


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []
    seen_hashes: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        suffix = path.suffix.lower()
        if suffix not in LOADERS:
            raise ValueError(f"Unsupported document type: {suffix}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"{path.name} exceeds the 200 MB safety limit")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        try:
            loader = (
                TextLoader(str(path), autodetect_encoding=True)
                if suffix in {".txt", ".md"}
                else LOADERS[suffix](str(path))
            )
            loaded = loader.load()
        except Exception as exc:
            raise ValueError(f"Could not parse {path.name}: {exc}") from exc
        for document in loaded:
            if not document.page_content.strip():
                continue
            document.metadata.update(
                {
                    "source": str(path),
                    "source_name": path.name,
                    "file_type": suffix.removeprefix("."),
                    "document_id": file_hash,
                }
            )
            documents.append(document)
    if not documents:
        raise ValueError("The uploaded files did not contain extractable text")
    return documents
