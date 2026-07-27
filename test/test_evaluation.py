import pytest

from docmind.evaluation import evaluate_rankings


def test_retrieval_metrics_are_computed_correctly():
    result = evaluate_rankings(
        expected=[{"a", "b"}, {"x"}],
        retrieved=[["a", "z"], ["q", "x"]],
    )
    assert result.recall_at_k == 0.75
    assert result.mean_reciprocal_rank == 0.75
    assert result.hit_rate == 1.0
    assert result.cases == 2


def test_evaluation_rejects_invalid_cases():
    with pytest.raises(ValueError):
        evaluate_rankings([], [])
    with pytest.raises(ValueError):
        evaluate_rankings([set()], [[]])

