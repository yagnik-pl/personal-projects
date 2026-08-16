"""
IREvaluator: Evaluates retrieval runs against ground truth qrels across standard IR metrics.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from src.evaluation.metrics import (
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_retrieval_run,
)


class IREvaluator:
    """
    Evaluates retrieval runs against ground truth qrels across standard IR metrics.
    """
    def __init__(
        self,
        k_values: Sequence[int] = (1, 5, 10),
        mrr_k: int = 10,
        ndcg_k: int = 10,
    ):
        self.k_values = tuple(sorted(k_values))
        self.mrr_k = mrr_k
        self.ndcg_k = ndcg_k

    def evaluate(
        self,
        run_results: Dict[str, Sequence[Union[Tuple[str, float], str]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """
        Computes macro-averaged retrieval metrics for the run.

        Args:
            run_results: Dict mapping query_id -> [(doc_id, score), ...] or [doc_id, ...]
            qrels: Dict mapping query_id -> {doc_id: relevance_score}

        Returns:
            Dict of macro-averaged retrieval metrics.
        """
        return evaluate_retrieval_run(run_results, qrels, k_values=self.k_values)

    def evaluate_per_query(
        self,
        run_results: Dict[str, Sequence[Union[Tuple[str, float], str]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns metric breakdown dictionary for each individual query.

        Args:
            run_results: Dict mapping query_id -> [(doc_id, score), ...] or [doc_id, ...]
            qrels: Dict mapping query_id -> {doc_id: relevance_score}

        Returns:
            Dict mapping query_id -> {metric_name: score}
        """
        per_query: Dict[str, Dict[str, float]] = {}

        for qid, ranked_items in run_results.items():
            if not ranked_items:
                ranked_doc_ids = []
            elif isinstance(ranked_items[0], tuple):
                ranked_doc_ids = [item[0] for item in ranked_items]
            else:
                ranked_doc_ids = list(ranked_items)

            query_qrels = qrels.get(qid, {})
            rel_set = {doc_id for doc_id, score in query_qrels.items() if score > 0}

            q_metrics: Dict[str, float] = {}
            for k in self.k_values:
                q_metrics[f"Recall@{k}"] = compute_recall_at_k(ranked_doc_ids, rel_set, k)
                q_metrics[f"Precision@{k}"] = compute_precision_at_k(ranked_doc_ids, rel_set, k)

            q_metrics["MRR"] = compute_mrr_at_k(ranked_doc_ids, rel_set, k=self.mrr_k)
            q_metrics[f"MRR@{self.mrr_k}"] = q_metrics["MRR"]
            q_metrics[f"nDCG@{self.ndcg_k}"] = compute_ndcg_at_k(ranked_doc_ids, query_qrels, k=self.ndcg_k)

            per_query[qid] = q_metrics

        return per_query
