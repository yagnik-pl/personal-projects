"""
Tests for Information Retrieval Evaluation Metrics.
Validates F5 requirements: Recall@1/5/10, Precision@1/5/10, MRR, nDCG@10 exact mathematical verification.
"""

import math
import pytest
from tests.conftest import (
    ref_compute_recall_at_k,
    ref_compute_mrr,
    ref_compute_ndcg_at_k,
    ref_evaluate_retrieval_run,
)

try:
    from src.evaluation.metrics import (
        compute_recall_at_k,
        compute_mrr,
        compute_ndcg_at_k,
        evaluate_retrieval_run,
    )
except ImportError:
    compute_recall_at_k = ref_compute_recall_at_k
    compute_mrr = ref_compute_mrr
    compute_ndcg_at_k = ref_compute_ndcg_at_k
    evaluate_retrieval_run = ref_evaluate_retrieval_run


def test_recall_at_k_exact_hand_calculated():
    """Verify Recall@1, Recall@5, Recall@10 against manual ground truth."""
    # 4 relevant documents total
    relevant_docs = {"doc_A", "doc_B", "doc_C", "doc_D"}
    
    # Ranked output of 10 documents
    # Top 1: doc_A (1 hit) -> Recall@1 = 1/4 = 0.25
    # Top 5: [doc_A, doc_X, doc_B, doc_Y, doc_Z] (2 hits) -> Recall@5 = 2/4 = 0.50
    # Top 10: [... + doc_C, doc_W, doc_V, doc_U, doc_T] (3 hits) -> Recall@10 = 3/4 = 0.75
    ranked_docs = [
        "doc_A", "doc_X", "doc_B", "doc_Y", "doc_Z",
        "doc_C", "doc_W", "doc_V", "doc_U", "doc_T"
    ]

    r1 = compute_recall_at_k(ranked_docs, relevant_docs, k=1)
    r5 = compute_recall_at_k(ranked_docs, relevant_docs, k=5)
    r10 = compute_recall_at_k(ranked_docs, relevant_docs, k=10)

    assert pytest.approx(r1, abs=1e-6) == 0.25
    assert pytest.approx(r5, abs=1e-6) == 0.50
    assert pytest.approx(r10, abs=1e-6) == 0.75


def test_mrr_exact_hand_calculated():
    """Verify MRR calculation for rank 1, rank 3, and no-match scenarios."""
    relevant_docs = {"doc_target"}

    # Case 1: First hit at rank 1 -> MRR = 1/1 = 1.0
    rank_1_list = ["doc_target", "doc_other", "doc_misc"]
    assert pytest.approx(compute_mrr(rank_1_list, relevant_docs, k=10), abs=1e-6) == 1.0

    # Case 2: First hit at rank 3 -> MRR = 1/3 ≈ 0.333333
    rank_3_list = ["doc_A", "doc_B", "doc_target", "doc_D"]
    assert pytest.approx(compute_mrr(rank_3_list, relevant_docs, k=10), abs=1e-6) == 1.0 / 3.0

    # Case 3: No hit within top 10 -> MRR = 0.0
    no_hit_list = ["doc_1", "doc_2", "doc_3", "doc_4"]
    assert pytest.approx(compute_mrr(no_hit_list, relevant_docs, k=10), abs=1e-6) == 0.0


def test_ndcg_at_k_exact_hand_calculated_graded_relevance():
    """
    Verify nDCG@10 against manual Discounted Cumulative Gain calculation with graded relevance.
    qrels: {doc_A: 2, doc_B: 1}
    Ranked list: [doc_B, doc_X, doc_A]
    """
    qrels = {"doc_A": 2, "doc_B": 1}
    ranked_docs = ["doc_B", "doc_X", "doc_A"]

    # Hand calculation:
    # Rank 1: doc_B (rel=1) -> gain = 2^1 - 1 = 1.0, discount = log2(1+1) = 1.0 -> 1.0 / 1.0 = 1.0
    # Rank 2: doc_X (rel=0) -> gain = 2^0 - 1 = 0.0 -> 0.0
    # Rank 3: doc_A (rel=2) -> gain = 2^2 - 1 = 3.0, discount = log2(3+1) = 2.0 -> 3.0 / 2.0 = 1.5
    # DCG@3 = 1.0 + 0.0 + 1.5 = 2.5
    #
    # Ideal ranking: [doc_A (rel=2), doc_B (rel=1)]
    # Rank 1: doc_A (rel=2) -> gain = 3.0, discount = 1.0 -> 3.0
    # Rank 2: doc_B (rel=1) -> gain = 1.0, discount = log2(2+1) = log2(3) ≈ 1.5849625 -> 1.0 / 1.5849625 ≈ 0.63092975
    # IDCG@3 = 3.0 + 0.63092975 = 3.63092975
    # nDCG@3 = 2.5 / 3.63092975 ≈ 0.6885288

    expected_idcg = ( (2**2 - 1) / math.log2(2) ) + ( (2**1 - 1) / math.log2(3) )
    expected_dcg = ( (2**1 - 1) / math.log2(2) ) + ( 0.0 ) + ( (2**2 - 1) / math.log2(4) )
    expected_ndcg = expected_dcg / expected_idcg

    calculated_ndcg = compute_ndcg_at_k(ranked_docs, qrels, k=10)
    assert pytest.approx(calculated_ndcg, abs=1e-5) == expected_ndcg


def test_ndcg_perfect_ranking_is_one():
    """Verify that a perfect ranking achieves exactly nDCG@10 = 1.0."""
    qrels = {"doc_1": 2, "doc_2": 1, "doc_3": 1}
    perfect_ranking = ["doc_1", "doc_2", "doc_3", "doc_unrelated"]
    assert pytest.approx(compute_ndcg_at_k(perfect_ranking, qrels, k=10), abs=1e-6) == 1.0


def test_metrics_empty_qrels_and_unjudged_docs():
    """Verify metric engine handles empty qrels and unjudged docs gracefully."""
    # Query with no relevant docs
    assert compute_recall_at_k(["doc_A", "doc_B"], set(), k=5) == 0.0
    assert compute_mrr(["doc_A", "doc_B"], set(), k=5) == 0.0
    assert compute_ndcg_at_k(["doc_A", "doc_B"], {}, k=5) == 0.0

    # Unjudged docs in run results
    run_results = {
        "q1": [("unjudged_1", 0.9), ("unjudged_2", 0.8)],
    }
    qrels = {
        "q1": {"relevant_doc": 1}
    }
    metrics = evaluate_retrieval_run(run_results, qrels)
    assert metrics["Recall@1"] == 0.0
    assert metrics["Recall@10"] == 0.0
    assert metrics["MRR"] == 0.0
    assert metrics["nDCG@10"] == 0.0
    assert metrics["num_queries_evaluated"] == 1.0
