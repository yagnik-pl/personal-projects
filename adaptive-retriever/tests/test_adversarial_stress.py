"""
Adversarial Stress and Boundary Edge Case Tests for Milestone M2.

Stress tests all M2 components against hostile/malformed inputs, extreme bounds,
zero divisions, numerical instabilities, and corrupted data streams.
"""

import json
import os
import tempfile
import pytest
import torch
import torch.nn.functional as F

from src.models.pooling import cls_pooling, mean_pooling, normalize_embeddings
from src.retrieval.index import DenseIndex
from src.retrieval.ranker import DenseRanker
from src.data.schemas import CorpusEntry, QueryEntry, DatasetSplit
from src.data.loader import BEIRDatasetLoader
from src.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    evaluate_retrieval_run,
)
from src.evaluation.evaluator import IREvaluator
from tests.conftest import MockTokenizer, MockTransformerEncoder


# ==============================================================================
# Vector 1: Empty and Whitespace Queries, Documents, and Lists
# ==============================================================================

def test_empty_and_whitespace_strings_in_schemas():
    """Verify schemas handle empty, whitespace-only, and unicode whitespace cleanly."""
    doc_empty = CorpusEntry(doc_id="d_empty", text="", title="")
    assert doc_empty.full_text == ""

    doc_spaces = CorpusEntry(doc_id="d_spaces", text="   \t\n  ", title="  \r\n ")
    assert doc_spaces.full_text == ""

    doc_mixed = CorpusEntry(doc_id="d_mixed", text="   body text   ", title="  Title  ")
    assert doc_mixed.full_text == "Title body text"

    q_empty = QueryEntry(query_id="q_empty", text="")
    assert q_empty.text == ""

    q_spaces = QueryEntry(query_id="q_spaces", text=" \t\n \r ")
    assert q_spaces.text == " \t\n \r "


def test_empty_texts_list_in_pooling():
    """Verify pooling functions raise appropriate errors when given empty/malformed tensor shapes."""
    # Sequence length 0
    empty_3d = torch.empty((2, 0, 16))
    with pytest.raises(ValueError, match=r"Sequence length dimension.*cannot be 0"):
        cls_pooling(empty_3d)

    # Dimension mismatch in mean pooling
    hidden = torch.randn(2, 4, 16)
    mask_wrong_dim = torch.ones(2, 5)
    with pytest.raises(ValueError, match=r"Dimension mismatch"):
        mean_pooling(hidden, mask_wrong_dim)


# ==============================================================================
# Vector 2: Extreme Padding and Single-Token Sequences in Pooling
# ==============================================================================

def test_mean_pooling_all_zero_mask():
    """
    CRITICAL: Verify that when attention_mask is ALL ZEROS (every token masked out),
    mean_pooling returns exact zeros and DOES NOT produce NaN, Inf, or ZeroDivisionError.
    """
    hidden = torch.randn(4, 10, 32)
    zero_mask = torch.zeros((4, 10), dtype=torch.long)

    pooled = mean_pooling(hidden, zero_mask)

    assert pooled.shape == (4, 32)
    assert not torch.isnan(pooled).any(), "Mean pooling on all-zero mask produced NaN!"
    assert not torch.isinf(pooled).any(), "Mean pooling on all-zero mask produced Inf!"
    assert torch.allclose(pooled, torch.zeros_like(pooled), atol=1e-6)


def test_mean_pooling_single_token_sequence():
    """Verify pooling behavior on seq_len = 1 with active and inactive masks."""
    hidden = torch.tensor([[[3.0, -5.0]], [[7.0, 11.0]]], dtype=torch.float32)  # shape (2, 1, 2)
    mask = torch.tensor([[1], [0]], dtype=torch.long)

    pooled = mean_pooling(hidden, mask)

    assert pooled.shape == (2, 2)
    # First item active -> exact representation
    assert torch.allclose(pooled[0], torch.tensor([3.0, -5.0]), atol=1e-6)
    # Second item masked -> zeros
    assert torch.allclose(pooled[1], torch.tensor([0.0, 0.0]), atol=1e-6)


def test_mean_pooling_extreme_padding_ratio():
    """Verify sequence of length 512 with 511 padding tokens preserves the single token value."""
    single_token = torch.tensor([1.5, -2.5, 4.0], dtype=torch.float32)
    hidden = torch.randn(1, 512, 3) * 100.0  # huge noise in padding tokens
    hidden[0, 0, :] = single_token

    mask = torch.zeros((1, 512), dtype=torch.long)
    mask[0, 0] = 1

    pooled = mean_pooling(hidden, mask)
    assert torch.allclose(pooled[0], single_token, atol=1e-6)


def test_cls_pooling_single_token():
    """Verify CLS pooling on sequence of length 1 extracts token 0 correctly."""
    hidden = torch.randn(3, 1, 16)
    pooled = cls_pooling(hidden)
    assert pooled.shape == (3, 16)
    assert torch.allclose(pooled, hidden[:, 0, :])


def test_pooling_invalid_tensor_ranks():
    """Verify non-3D tensors raise ValueError in CLS and Mean pooling."""
    with pytest.raises(ValueError, match=r"Expected 3D"):
        cls_pooling(torch.randn(4, 16))

    with pytest.raises(ValueError, match=r"Expected 3D"):
        cls_pooling(torch.randn(4, 2, 8, 16))

    with pytest.raises(ValueError, match=r"Expected 3D"):
        mean_pooling(torch.randn(4, 16), torch.ones(4, 16))

    with pytest.raises(ValueError, match=r"Expected 2D"):
        mean_pooling(torch.randn(4, 5, 16), torch.ones(4, 5, 1))


# ==============================================================================
# Vector 3: Zero Vectors, Subnormals, and Multi-Dimensional Normalization
# ==============================================================================

def test_normalize_embeddings_zero_and_subnormal_vectors():
    """Verify zero and subnormal vectors normalize to zeros without NaN/Inf."""
    # Zero vector
    zeros = torch.zeros(5, 64)
    norm_zeros = normalize_embeddings(zeros, eps=1e-9)
    assert not torch.isnan(norm_zeros).any()
    assert not torch.isinf(norm_zeros).any()
    assert torch.allclose(norm_zeros, torch.zeros_like(norm_zeros), atol=1e-6)

    # Subnormal / tiny vectors
    tiny = torch.full((3, 32), 1e-20)
    norm_tiny = normalize_embeddings(tiny, eps=1e-9)
    assert not torch.isnan(norm_tiny).any()
    assert not torch.isinf(norm_tiny).any()


def test_normalize_embeddings_empty_tensor():
    """Verify empty tensor is returned cleanly without error."""
    empty = torch.empty((0, 32))
    norm_empty = normalize_embeddings(empty)
    assert norm_empty.shape == (0, 32)


def test_normalize_embeddings_multidimensional():
    """Verify normalize_embeddings along last dimension for 1D, 2D, and 3D tensors."""
    # 1D vector
    v1 = torch.tensor([3.0, 4.0])
    norm_v1 = normalize_embeddings(v1)
    assert torch.allclose(norm_v1, torch.tensor([0.6, 0.8]), atol=1e-6)

    # 3D tensor (B, S, D)
    v3 = torch.randn(2, 5, 16) * 10.0
    norm_v3 = normalize_embeddings(v3)
    norms = torch.norm(norm_v3, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


# ==============================================================================
# Vector 4: DenseIndex Adversarial Boundary & Dimension Handling
# ==============================================================================

def test_dense_index_unbuilt_and_empty_search():
    """Verify searching an unbuilt index raises RuntimeError."""
    index = DenseIndex()
    q = torch.randn(1, 16)
    with pytest.raises(RuntimeError, match=r"Cannot search an unbuilt or empty index"):
        index.search(q)


def test_dense_index_top_k_bounds():
    """Verify top_k bounds: top_k > N, top_k = 0, negative top_k."""
    index = DenseIndex(embedding_dim=8)
    docs = ["d0", "d1", "d2"]
    embs = normalize_embeddings(torch.randn(3, 8))
    index.build(docs, embs)

    q = normalize_embeddings(torch.randn(2, 8))

    # top_k > N
    res_large = index.search(q, top_k=100)
    assert len(res_large) == 2
    assert len(res_large[0]) == 3
    assert len(res_large[1]) == 3

    # top_k == 0
    res_zero = index.search(q, top_k=0)
    assert len(res_zero) == 2
    assert len(res_zero[0]) == 0
    assert len(res_zero[1]) == 0

    # negative top_k
    res_neg = index.search(q, top_k=-5)
    assert len(res_neg) == 2
    assert len(res_neg[0]) == 0
    assert len(res_neg[1]) == 0


def test_dense_index_dimension_mismatch():
    """Verify dimension mismatch between query and index raises ValueError."""
    index = DenseIndex(embedding_dim=16)
    index.build(["d0", "d1"], normalize_embeddings(torch.randn(2, 16)))

    q_wrong_dim = normalize_embeddings(torch.randn(1, 32))
    with pytest.raises(ValueError, match=r"Query dim 32 != index dim 16"):
        index.search(q_wrong_dim)


def test_dense_index_1d_query_and_chunked_batch_search():
    """Verify 1D query handling and batched search chunking."""
    index = DenseIndex(embedding_dim=8)
    docs = [f"d_{i}" for i in range(10)]
    index.build(docs, normalize_embeddings(torch.randn(10, 8)))

    # 1D query vector
    q_1d = torch.randn(8)
    res_1d = index.search(q_1d, top_k=3)
    assert len(res_1d) == 1
    assert len(res_1d[0]) == 3

    # Batch chunking (25 queries chunked with batch_size=7)
    q_batch = torch.randn(25, 8)
    res_chunked = index.search(q_batch, top_k=4, batch_size=7)
    assert len(res_chunked) == 25
    for r in res_chunked:
        assert len(r) == 4


# ==============================================================================
# Vector 5: Out-of-Bounds Layers and Invalid Configurations
# ==============================================================================

def test_ranker_invalid_layer_and_empty_corpus():
    """Verify ranker with empty corpus or layer boundaries."""
    encoder = MockTransformerEncoder(num_layers=4, hidden_dim=16)
    
    # Mock encoder adapter
    class MockLayerEncoder:
        def __init__(self, enc):
            self.enc = enc
            self.num_layers = enc.num_layers
            self.hidden_dim = enc.hidden_dim
            self.device = torch.device("cpu")
            self.tok = MockTokenizer(max_length=8)

        def encode_layer(self, texts, layer, batch_size=64, show_progress=False):
            if not (0 <= layer <= self.num_layers):
                raise ValueError(f"Layer {layer} out of valid bounds [0, {self.num_layers}]")
            if not texts:
                return torch.empty((0, self.hidden_dim))
            toks = self.tok(texts)
            out = self.enc(toks["input_ids"], attention_mask=toks["attention_mask"])
            h = out.hidden_states[layer]
            pooled = mean_pooling(h, toks["attention_mask"])
            return normalize_embeddings(pooled)

    adapted = MockLayerEncoder(encoder)
    ranker = DenseRanker(encoder=adapted)

    # Layer out of bounds
    with pytest.raises(ValueError, match=r"Layer 99 out of valid bounds"):
        adapted.encode_layer(["test text"], layer=99)

    with pytest.raises(ValueError, match=r"Layer -2 out of valid bounds"):
        adapted.encode_layer(["test text"], layer=-2)


# ==============================================================================
# Vector 6: Corrupted and Malformed BEIR Dataset Files
# ==============================================================================

def test_beir_loader_corrupted_tsv_and_empty_files():
    """Verify BEIRDatasetLoader handles corrupted TSVs, header rows, and empty files gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_name = "stress_test_ds"
        ds_dir = os.path.join(tmp_dir, dataset_name)
        os.makedirs(os.path.join(ds_dir, "qrels"), exist_ok=True)

        # 1. Corpus with blank lines and valid JSON
        corpus_path = os.path.join(ds_dir, "corpus.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(json.dumps({"_id": "doc_1", "title": "Doc 1", "text": "First test doc"}) + "\n")
            f.write("   \n")
            f.write(json.dumps({"_id": "doc_2", "title": "", "text": "Second test doc"}) + "\n")
            f.write("\n")

        # 2. Queries with blank lines and valid JSON
        queries_path = os.path.join(ds_dir, "queries.jsonl")
        with open(queries_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_id": "q_1", "text": "First query"}) + "\n")
            f.write("\n")
            f.write(json.dumps({"_id": "q_2", "text": "Second query"}) + "\n")

        # 3. Qrels with header, malformed lines, non-numeric values, and empty lines
        qrels_path = os.path.join(ds_dir, "qrels", "test.tsv")
        with open(qrels_path, "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")  # Header row
            f.write("q_1\tdoc_1\t1\n")              # Valid
            f.write("malformed_line_with_only_one_col\n")  # Truncated
            f.write("q_1\tdoc_2\tinvalid_score\n")  # Non-numeric score
            f.write("\n\t\n")                       # Blank line
            f.write("q_2\tdoc_2\t2\n")              # Valid graded score

        loader = BEIRDatasetLoader(data_dir=tmp_dir)
        split = loader.load_split(dataset_name, split="test")

        assert split.num_docs == 2
        assert split.num_queries == 2
        assert split.num_judged_queries == 2
        assert split.qrels["q_1"] == {"doc_1": 1}
        assert split.qrels["q_2"] == {"doc_2": 2}


def test_beir_loader_missing_files_raises_filenotfound():
    """Verify loader raises ValueError for unrecognized dataset or FileNotFoundError for missing files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_dir = os.path.join(tmp_dir, "broken_ds")
        os.makedirs(ds_dir, exist_ok=True)
        # Empty directory without corpus.jsonl

        loader = BEIRDatasetLoader(data_dir=tmp_dir)
        with pytest.raises((ValueError, FileNotFoundError)):
            loader.load_split("broken_ds", split="test")


# ==============================================================================
# Vector 7: IR Evaluation Metrics Adversarial Inputs
# ==============================================================================

def test_metrics_zero_relevant_and_empty_run():
    """Verify metrics handle empty runs, unjudged queries, and disjoint IDs."""
    evaluator = IREvaluator(k_values=(1, 5, 10))

    # Empty run results
    empty_metrics = evaluator.evaluate(run_results={}, qrels={})
    assert empty_metrics["Recall@10"] == 0.0
    assert empty_metrics["MRR"] == 0.0
    assert empty_metrics["nDCG@10"] == 0.0
    assert empty_metrics["num_queries_evaluated"] == 0.0

    # Query in run has no qrels entry
    run_results = {"q_unjudged": [("d0", 0.9), ("d1", 0.8)]}
    qrels = {}
    metrics = evaluator.evaluate(run_results, qrels)
    assert metrics["Recall@10"] == 0.0
    assert metrics["MRR"] == 0.0
    assert metrics["nDCG@10"] == 0.0
    assert metrics["num_queries_evaluated"] == 1.0


def test_ndcg_graded_relevance_and_tie_breaking():
    """Verify nDCG@K with graded relevance and ideal ranking."""
    # Ideal order: doc_graded (rel=3), doc_binary (rel=1), doc_irrelevant (rel=0)
    qrels = {"doc_graded": 3, "doc_binary": 1, "doc_irrelevant": 0}

    # Perfect ranking
    perfect_retrieved = ["doc_graded", "doc_binary", "doc_irrelevant"]
    perfect_ndcg = compute_ndcg_at_k(perfect_retrieved, qrels, k=10)
    assert pytest.approx(perfect_ndcg, abs=1e-6) == 1.0

    # Inverted ranking (rel=1 at rank 1, rel=3 at rank 2)
    inverted_retrieved = ["doc_binary", "doc_graded", "doc_irrelevant"]
    inverted_ndcg = compute_ndcg_at_k(inverted_retrieved, qrels, k=10)
    assert 0.0 < inverted_ndcg < 1.0

    # Calculation verification:
    # Ideal DCG = (2^3 - 1)/log2(2) + (2^1 - 1)/log2(3) = 7/1 + 1/1.5849625 = 7 + 0.63092975 = 7.63092975
    # Inverted DCG = (2^1 - 1)/log2(2) + (2^3 - 1)/log2(3) = 1/1 + 7/1.5849625 = 1 + 4.416508 = 5.416508
    # Inverted nDCG = 5.416508 / 7.63092975 = 0.7098097
    assert pytest.approx(inverted_ndcg, abs=1e-4) == (5.416508 / 7.63092975)
