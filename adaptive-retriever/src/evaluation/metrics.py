"""
Pure Python/NumPy/PyTorch evaluation metrics for Information Retrieval.
Implements exact Recall@K, Precision@K, MRR@K, and nDCG@K (with exponential gain).
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


def _extract_relevant_docs(qrels_or_relevant: Union[Dict[str, int], Set[str], Sequence[str]]) -> Set[str]:
    """Helper to extract positive relevant document IDs from either dict or set/list."""
    if isinstance(qrels_or_relevant, dict):
        return {doc_id for doc_id, score in qrels_or_relevant.items() if score > 0}
    elif isinstance(qrels_or_relevant, (set, list, tuple)):
        return set(qrels_or_relevant)
    return set()


def compute_recall_at_k(
    retrieved_doc_ids: Sequence[str],
    qrels_or_relevant: Union[Dict[str, int], Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Computes Recall@K for a single query.

    Args:
        retrieved_doc_ids: Ranked sequence of retrieved document IDs.
        qrels_or_relevant: Dict of {doc_id: rel_score} or Set/List of relevant doc IDs.
        k: Cutoff rank (e.g. 1, 5, 10).

    Returns:
        Recall@K score in [0.0, 1.0].
    """
    relevant_docs = _extract_relevant_docs(qrels_or_relevant)
    if not relevant_docs or k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return float(hits) / float(len(relevant_docs))


def compute_precision_at_k(
    retrieved_doc_ids: Sequence[str],
    qrels_or_relevant: Union[Dict[str, int], Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Computes Precision@K for a single query.

    Args:
        retrieved_doc_ids: Ranked sequence of retrieved document IDs.
        qrels_or_relevant: Dict of {doc_id: rel_score} or Set/List of relevant doc IDs.
        k: Cutoff rank (e.g. 1, 5, 10).

    Returns:
        Precision@K score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    relevant_docs = _extract_relevant_docs(qrels_or_relevant)
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return float(hits) / float(k)


def compute_mrr_at_k(
    retrieved_doc_ids: Sequence[str],
    qrels_or_relevant: Union[Dict[str, int], Set[str], Sequence[str]],
    k: int = 10,
) -> float:
    """
    Computes Reciprocal Rank at cutoff K for a single query.

    Args:
        retrieved_doc_ids: Ranked sequence of retrieved document IDs.
        qrels_or_relevant: Dict of {doc_id: rel_score} or Set/List of relevant doc IDs.
        k: Cutoff rank (default 10).

    Returns:
        Reciprocal rank in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    relevant_docs = _extract_relevant_docs(qrels_or_relevant)
    if not relevant_docs:
        return 0.0
    for rank_idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_docs:
            return 1.0 / float(rank_idx)
    return 0.0


# Alias for compatibility
compute_mrr = compute_mrr_at_k


def compute_ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    qrels_dict: Dict[str, int],
    k: int = 10,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at cutoff K using exponential gain: (2^rel - 1) / log2(rank + 1).

    Args:
        retrieved_doc_ids: Ranked sequence of retrieved document IDs.
        qrels_dict: Dict mapping doc_id -> relevance integer score.
        k: Cutoff rank (default 10).

    Returns:
        nDCG@K score in [0.0, 1.0].
    """
    if k <= 0 or not isinstance(qrels_dict, dict):
        return 0.0

    relevant_scores = [v for v in qrels_dict.values() if v > 0]
    if not relevant_scores:
        return 0.0

    # 1. DCG@K
    dcg = 0.0
    for rank_idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = qrels_dict.get(doc_id, 0)
        if rel > 0:
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(rank_idx + 1.0)
            dcg += gain / discount

    # 2. IDCG@K (Ideal DCG)
    ideal_rels = sorted(relevant_scores, reverse=True)[:k]
    idcg = 0.0
    for rank_idx, rel in enumerate(ideal_rels, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1.0)
        idcg += gain / discount

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_run(
    run_results: Dict[str, Sequence[Union[Tuple[str, float], str]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Computes macro-averaged retrieval metrics across all queries in a run.

    Args:
        run_results: Dict mapping query_id -> [(doc_id, score), ...] or [doc_id, ...]
        qrels: Dict mapping query_id -> {doc_id: relevance_score}
        k_values: Tuple of rank cutoffs for Recall and Precision.

    Returns:
        Dictionary of macro-averaged metrics: Recall@K, Precision@K, MRR, nDCG@10, num_queries_evaluated.
    """
    if not run_results:
        metrics = {f"Recall@{k}": 0.0 for k in k_values}
        metrics.update({f"Precision@{k}": 0.0 for k in k_values})
        metrics["MRR"] = 0.0
        metrics["nDCG@10"] = 0.0
        metrics["num_queries_evaluated"] = 0.0
        return metrics

    metrics = {f"Recall@{k}": 0.0 for k in k_values}
    metrics.update({f"Precision@{k}": 0.0 for k in k_values})
    metrics["MRR"] = 0.0
    metrics["nDCG@10"] = 0.0

    num_queries = len(run_results)
    for qid, ranked_items in run_results.items():
        if not ranked_items:
            ranked_doc_ids = []
        elif isinstance(ranked_items[0], tuple):
            ranked_doc_ids = [item[0] for item in ranked_items]
        else:
            ranked_doc_ids = list(ranked_items)

        query_qrels = qrels.get(qid, {})
        rel_set = {doc_id for doc_id, score in query_qrels.items() if score > 0}

        for k in k_values:
            metrics[f"Recall@{k}"] += compute_recall_at_k(ranked_doc_ids, rel_set, k)
            metrics[f"Precision@{k}"] += compute_precision_at_k(ranked_doc_ids, rel_set, k)

        metrics["MRR"] += compute_mrr(ranked_doc_ids, rel_set, k=10)
        metrics["nDCG@10"] += compute_ndcg_at_k(ranked_doc_ids, query_qrels, k=10)

    for k in list(metrics.keys()):
        metrics[k] /= float(num_queries)

    metrics["num_queries_evaluated"] = float(num_queries)
    return metrics
