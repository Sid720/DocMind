from pathlib import Path

import pytest

from docmind.loaders import load_documents


def test_text_ingestion_preserves_provenance_and_deduplicates(tmp_path: Path):
    document = tmp_path / "guide.txt"
    document.write_text("GMAT grammar and sentence correction.", encoding="utf-8")
    loaded = load_documents([document, document])
    assert len(loaded) == 1
    assert loaded[0].metadata["source_name"] == "guide.txt"
    assert len(loaded[0].metadata["document_id"]) == 64


def test_empty_upload_is_rejected(tmp_path: Path):
    document = tmp_path / "empty.txt"
    document.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError, match="extractable text"):
        load_documents([document])


def test_unsupported_file_is_rejected(tmp_path: Path):
    document = tmp_path / "archive.zip"
    document.write_bytes(b"not a document")
    with pytest.raises(ValueError, match="Unsupported"):
        load_documents([document])
