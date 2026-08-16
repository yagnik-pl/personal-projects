"""
Modular Pytest Fixtures and Mock Test Harness for AdaptiveRetriever.

Provides 100% offline, CPU-executable mock models, tokenizers, synthetic
embeddings, and standard BEIR-style datasets for unit and integration testing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Reference Schemas (Interface Contracts)
# ==============================================================================

@dataclass
class CorpusEntry:
    id: str
    title: str = ""
    text: str = ""


@dataclass
class QueryEntry:
    id: str
    text: str


@dataclass
class DatasetSplit:
    name: str
    corpus: Dict[str, CorpusEntry] = field(default_factory=dict)
    queries: Dict[str, QueryEntry] = field(default_factory=dict)
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ==============================================================================
# Pure-PyTorch Reference Primitives for Standalone/Contract Testing
# ==============================================================================

def ref_cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract representation corresponding to first token (CLS)."""
    return hidden_states[:, 0, :]


def ref_mean_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute masked mean pooling across token dimension."""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def ref_normalize_embeddings(
    embeddings: torch.Tensor, p: float = 2.0, eps: float = 1e-9
) -> torch.Tensor:
    """Normalize embeddings to unit L2 norm."""
    norm = torch.norm(embeddings, p=p, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return embeddings / norm


def ref_cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D or (1, D) normalized tensors."""
    v1 = e1.squeeze()
    v2 = e2.squeeze()
    return float(torch.dot(v1, v2).item())


def ref_norm_delta(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """Compute Euclidean norm difference between two normalized tensors."""
    v1 = e1.squeeze()
    v2 = e2.squeeze()
    return float(torch.norm(v1 - v2, p=2).item())


# ==============================================================================
# Metric Reference Functions (Exact Mathematical Derivations)
# ==============================================================================

def ref_compute_recall_at_k(
    ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int
) -> float:
    """Exact Recall@K."""
    if not relevant_doc_ids:
        return 0.0
    retrieved_at_k = set(ranked_doc_ids[:k])
    hits = len(retrieved_at_k.intersection(relevant_doc_ids))
    return hits / float(len(relevant_doc_ids))


def ref_compute_mrr(
    ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10
) -> float:
    """Exact MRR@K."""
    if not relevant_doc_ids:
        return 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ref_compute_ndcg_at_k(
    ranked_doc_ids: List[str], qrels_dict: Dict[str, int], k: int = 10
) -> float:
    """Exact nDCG@K with exponential gain formula."""
    relevant_scores = [v for v in qrels_dict.values() if v > 0]
    if not relevant_scores:
        return 0.0

    # Calculate DCG@k
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = qrels_dict.get(doc_id, 0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank + 1.0)
        dcg += gain / discount

    # Calculate Ideal DCG@k (IDCG@k)
    ideal_rels = sorted(relevant_scores, reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank + 1.0)
        idcg += gain / discount

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def ref_evaluate_retrieval_run(
    run_results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Calculate aggregated retrieval metrics over all queries."""
    if not run_results:
        return {f"Recall@{k}": 0.0 for k in k_values}

    metrics = {f"Recall@{k}": 0.0 for k in k_values}
    metrics.update({f"Precision@{k}": 0.0 for k in k_values})
    metrics["MRR"] = 0.0
    metrics["nDCG@10"] = 0.0

    num_queries = len(run_results)
    for qid, ranked_tuples in run_results.items():
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_tuples]
        query_qrels = qrels.get(qid, {})
        rel_set = {doc_id for doc_id, score in query_qrels.items() if score > 0}

        for k in k_values:
            metrics[f"Recall@{k}"] += ref_compute_recall_at_k(ranked_doc_ids, rel_set, k)
            # Precision@K
            retrieved_k = set(ranked_doc_ids[:k])
            hits = len(retrieved_k.intersection(rel_set))
            metrics[f"Precision@{k}"] += hits / float(k) if k > 0 else 0.0

        metrics["MRR"] += ref_compute_mrr(ranked_doc_ids, rel_set, k=10)
        metrics["nDCG@10"] += ref_compute_ndcg_at_k(ranked_doc_ids, query_qrels, k=10)

    for k in metrics:
        metrics[k] /= float(num_queries)
    metrics["num_queries_evaluated"] = float(num_queries)
    return metrics


# ==============================================================================
# Mock Neural Architecture Components (100% Offline, CPU Only)
# ==============================================================================

class MockTokenizer:
    """Mock Tokenizer converting strings to deterministic token IDs."""
    def __init__(self, vocab_size: int = 1000, max_length: int = 32):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        max_len = max_length or self.max_length
        batch_size = len(texts)

        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, text in enumerate(texts):
            words = text.strip().split()
            # CLS token is 101, SEP is 102
            tokens = [101] + [(abs(hash(w)) % (self.vocab_size - 200)) + 105 for w in words] + [102]
            tokens = tokens[:max_len]
            length = len(tokens)

            input_ids[i, :length] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MockTransformerLayer(nn.Module):
    """A lightweight deterministic single Transformer layer for testing."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # Initialize with identity-like weights for controlled drift
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim) + 0.01 * torch.randn(hidden_dim, hidden_dim))
            self.linear.bias.zero_()

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        out = F.gelu(self.linear(hidden_states))
        # Add residual connection
        out = out + hidden_states
        return (out,)


class MockTransformerEncoder(nn.Module):
    """
    Synthetic Transformer Encoder with L layers and embeddings.
    Emulates HuggingFace BertModel interface for CPU testing.
    """
    def __init__(self, num_layers: int = 6, hidden_dim: int = 64, vocab_size: int = 1000):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([MockTransformerLayer(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ):
        h_0 = self.embeddings(input_ids)
        all_hidden_states = [h_0]

        h = h_0
        for layer in self.layers:
            layer_out = layer(h, attention_mask=attention_mask)
            h = layer_out[0]
            if output_hidden_states:
                all_hidden_states.append(h)

        class ModelOutput:
            def __init__(self, last_hidden_state, hidden_states):
                self.last_hidden_state = last_hidden_state
                self.hidden_states = hidden_states

        return ModelOutput(
            last_hidden_state=h,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer(vocab_size=500, max_length=16)


@pytest.fixture
def mock_encoder():
    torch.manual_seed(42)
    return MockTransformerEncoder(num_layers=6, hidden_dim=64, vocab_size=500)


@pytest.fixture
def sample_hidden_states():
    torch.manual_seed(42)
    # Batch size 2, Sequence length 4, Hidden dim 8
    return torch.randn(2, 4, 8)


@pytest.fixture
def sample_attention_mask():
    # Batch size 2, Sequence length 4 (second sample has 2 padding tokens)
    return torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)


@pytest.fixture
def sample_corpus():
    return {
        "doc_0": CorpusEntry(id="doc_0", title="AI Retrieval", text="Dense retrieval with neural embeddings."),
        "doc_1": CorpusEntry(id="doc_1", title="Transformer Models", text="Early exit transformer architectures."),
        "doc_2": CorpusEntry(id="doc_2", title="Scientific Papers", text="Evaluation on SciFact and NFCorpus."),
        "doc_3": CorpusEntry(id="doc_3", title="Financial Search", text="Financial question answering FiQA."),
        "doc_4": CorpusEntry(id="doc_4", title="Hardware Efficiency", text="Latency optimization on consumer GPUs."),
    }


@pytest.fixture
def sample_queries():
    return {
        "q_0": QueryEntry(id="q_0", text="dense retrieval early exit"),
        "q_1": QueryEntry(id="q_1", text="evaluation on scifact dataset"),
        "q_2": QueryEntry(id="q_2", text="unknown query with no relevant documents"),
    }


@pytest.fixture
def sample_qrels():
    return {
        "q_0": {"doc_0": 1, "doc_1": 2},
        "q_1": {"doc_2": 1},
        "q_2": {},  # Empty qrels for edge case testing
    }


@pytest.fixture
def sample_dataset_split(sample_corpus, sample_queries, sample_qrels):
    return DatasetSplit(
        name="test_split",
        corpus=sample_corpus,
        queries=sample_queries,
        qrels=sample_qrels,
    )


@pytest.fixture
def sample_dense_vectors():
    torch.manual_seed(42)
    # 5 documents, 8 dimensions, L2 normalized
    raw = torch.randn(5, 8)
    return ref_normalize_embeddings(raw)
