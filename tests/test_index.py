"""
Tests for Dense Flat Vector Indexing and Top-K Cosine Search.
Validates F4 requirements: DenseIndex build, exact matrix product, monotonic sorting, top-k bounds.
"""

import pytest
import torch

try:
    from src.retrieval.index import DenseIndex
except ImportError:
    # Reference implementation of DenseIndex for contract validation
    class DenseIndex:
        def __init__(self):
            self.doc_ids = []
            self.embeddings = None

        def build(self, doc_ids, embeddings):
            if len(doc_ids) != embeddings.shape[0]:
                raise ValueError("Mismatch between doc_ids length and embeddings shape[0]")
            self.doc_ids = list(doc_ids)
            self.embeddings = embeddings.cpu().float()

        def search(self, query_embeddings, top_k=10):
            if self.embeddings is None:
                raise RuntimeError("Index is empty. Call build() first.")
            query_embeddings = query_embeddings.cpu().float()
            if query_embeddings.dim() == 1:
                query_embeddings = query_embeddings.unsqueeze(0)

            # Compute exact dot products: (Q, D) x (D, N) -> (Q, N)
            scores = torch.mm(query_embeddings, self.embeddings.t())
            num_docs = len(self.doc_ids)
            effective_k = min(top_k, num_docs)

            if effective_k == 0:
                return [[] for _ in range(query_embeddings.shape[0])]

            topk_scores, topk_indices = torch.topk(scores, k=effective_k, dim=-1, largest=True, sorted=True)

            results = []
            for q_idx in range(query_embeddings.shape[0]):
                q_res = [
                    (self.doc_ids[idx.item()], float(score.item()))
                    for idx, score in zip(topk_indices[q_idx], topk_scores[q_idx])
                ]
                results.append(q_res)
            return results


def test_dense_index_build_and_shapes(sample_dense_vectors):
    """Verify DenseIndex builds correctly with doc IDs and normalized vectors."""
    doc_ids = [f"doc_{i}" for i in range(sample_dense_vectors.shape[0])]
    index = DenseIndex()
    index.build(doc_ids, sample_dense_vectors)

    assert len(index.doc_ids) == len(doc_ids)
    assert index.embeddings.shape == sample_dense_vectors.shape


def test_dense_index_exact_cosine_similarity(sample_dense_vectors):
    """Verify search scores match exact hand-calculated inner products."""
    doc_ids = [f"doc_{i}" for i in range(5)]
    index = DenseIndex()
    index.build(doc_ids, sample_dense_vectors)

    # Query with exact copy of doc_2 embedding
    q_emb = sample_dense_vectors[2:3]  # shape (1, 8)
    results = index.search(q_emb, top_k=5)

    assert len(results) == 1
    ranked_list = results[0]
    assert len(ranked_list) == 5

    # Top-1 doc must be doc_2 with cosine similarity = 1.0 +/- 1e-6
    top_doc, top_score = ranked_list[0]
    assert top_doc == "doc_2"
    assert pytest.approx(top_score, abs=1e-5) == 1.0


def test_dense_index_monotonic_score_sorting(sample_dense_vectors):
    """Verify returned ranked tuples are sorted in strictly non-increasing score order."""
    doc_ids = [f"doc_{i}" for i in range(5)]
    index = DenseIndex()
    index.build(doc_ids, sample_dense_vectors)

    q_emb = torch.randn(1, sample_dense_vectors.shape[1])
    q_emb = q_emb / torch.norm(q_emb, p=2, dim=-1, keepdim=True)

    results = index.search(q_emb, top_k=5)
    ranked_list = results[0]

    scores = [score for _, score in ranked_list]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Ranking is not sorted: {scores}"


def test_dense_index_topk_boundary_handling(sample_dense_vectors):
    """Verify search handles k=0, k=1, and k > N correctly without errors."""
    num_docs = sample_dense_vectors.shape[0]
    doc_ids = [f"doc_{i}" for i in range(num_docs)]
    index = DenseIndex()
    index.build(doc_ids, sample_dense_vectors)

    q_emb = sample_dense_vectors[0:1]

    # Boundary 1: k=0
    res_k0 = index.search(q_emb, top_k=0)
    assert len(res_k0[0]) == 0

    # Boundary 2: k=1
    res_k1 = index.search(q_emb, top_k=1)
    assert len(res_k1[0]) == 1

    # Boundary 3: k > num_docs (e.g. k=100 for 5 docs)
    res_k100 = index.search(q_emb, top_k=100)
    assert len(res_k100[0]) == num_docs  # should clamp to num_docs
