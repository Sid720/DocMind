import pytest

from docmind.config import Settings


def test_default_settings_are_valid():
    settings = Settings()
    assert settings.fetch_k >= settings.retrieval_k
    assert settings.chunk_overlap < settings.chunk_size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 50},
        {"chunk_size": 500, "chunk_overlap": 500},
        {"retrieval_k": 10, "fetch_k": 5},
        {"vector_backend": "unknown"},
        {"score_threshold": 1.1},
    ],
)
def test_invalid_settings_fail_fast(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
