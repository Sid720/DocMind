"""Stage 2: normalized Hugging Face sentence embeddings."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

