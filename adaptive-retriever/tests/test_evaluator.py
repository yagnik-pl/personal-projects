"""
Tests for IREvaluator class.
"""
import pytest
from src.evaluation.evaluator import IREvaluator


def test_ir_evaluator_macro_and_per_query():
    evaluator = IREvaluator(k_values=(1, 5, 10))

    run_results = {
        "q1": [("d1", 1.0), ("d2", 0.8), ("d3", 0.5)],
        "q2": [("d4", 0.9), ("d5", 0.7), ("d6", 0.3)],
    }
    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d5": 2},
    }

    metrics = evaluator.evaluate(run_results, qrels)
    assert "Recall@1" in metrics
    assert "Recall@5" in metrics
    assert "Recall@10" in metrics
    assert "Precision@1" in metrics
    assert "MRR" in metrics
    assert "nDCG@10" in metrics
    assert metrics["num_queries_evaluated"] == 2.0

    # Test per-query
    per_query = evaluator.evaluate_per_query(run_results, qrels)
    assert "q1" in per_query
    assert "q2" in per_query
    assert per_query["q1"]["Recall@1"] == 0.5  # 1 hit out of 2 relevant
    assert per_query["q2"]["Recall@1"] == 0.0  # d4 not in qrels, d5 at rank 2
    assert per_query["q2"]["Recall@5"] == 1.0  # d5 is in top 5
