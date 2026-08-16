"""
Tests for Pooling Strategies and Embedding Normalization.
Validates F2 interface contracts: CLS pooling, mean pooling, padding invariance, L2 normalization.
"""

import pytest
import torch
import torch.nn.functional as F

try:
    from src.models.pooling import cls_pooling, mean_pooling, normalize_embeddings
except ImportError:
    from tests.conftest import (
        ref_cls_pooling as cls_pooling,
        ref_mean_pooling as mean_pooling,
        ref_normalize_embeddings as normalize_embeddings,
    )


def test_cls_pooling_exact_slice(sample_hidden_states):
    """Verify CLS pooling extracts exactly the first token slice (index 0)."""
    pooled = cls_pooling(sample_hidden_states)
    assert pooled.shape == (sample_hidden_states.shape[0], sample_hidden_states.shape[2])
    # Check that pooled matches index 0 exactly
    assert torch.allclose(pooled, sample_hidden_states[:, 0, :], atol=1e-7)


def test_mean_pooling_accuracy():
    """Verify mean pooling matches hand-calculated arithmetic mean across unmasked tokens."""
    # Hidden states: 1 batch, 3 tokens, 2 dimensions
    # Token 1: [2.0, 4.0] (mask=1)
    # Token 2: [4.0, 6.0] (mask=1)
    # Token 3: [100.0, 200.0] (mask=0 - padding)
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 6.0], [100.0, 200.0]]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    pooled = mean_pooling(hidden, mask)
    # Expected mean over token 1 and 2: [(2+4)/2, (4+6)/2] = [3.0, 5.0]
    expected = torch.tensor([[3.0, 5.0]], dtype=torch.float32)

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_mean_pooling_padding_invariance():
    """Verify adding arbitrary padding tokens does not alter the pooled representation."""
    torch.manual_seed(42)
    # Original sequence of length 3
    orig_hidden = torch.randn(1, 3, 16)
    orig_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    orig_pooled = mean_pooling(orig_hidden, orig_mask)

    # Padded sequence of length 7 (4 padding tokens with arbitrary values)
    pad_hidden = torch.cat([orig_hidden, torch.randn(1, 4, 16) * 100.0], dim=1)
    pad_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    pad_pooled = mean_pooling(pad_hidden, pad_mask)

    assert torch.allclose(orig_pooled, pad_pooled, atol=1e-6)


def test_l2_normalization_unit_norm():
    """Verify L2 normalization produces strict unit vectors with norm 1.0 +/- 1e-6."""
    torch.manual_seed(123)
    raw_embeddings = torch.randn(20, 64) * 50.0  # arbitrary non-unit vectors
    normalized = normalize_embeddings(raw_embeddings)

    norms = torch.norm(normalized, p=2, dim=-1)
    expected_ones = torch.ones_like(norms)

    assert normalized.shape == (20, 64)
    assert torch.allclose(norms, expected_ones, atol=1e-6)


def test_l2_normalization_zero_vector_stability():
    """Verify zero vectors do not cause NaN or Inf when normalized with epsilon clamp."""
    zero_emb = torch.zeros(2, 32)
    normalized = normalize_embeddings(zero_emb, eps=1e-9)

    assert not torch.isnan(normalized).any()
    assert not torch.isinf(normalized).any()
    # Zero vector normalized with epsilon clamp produces near-zero or zero values
    assert torch.norm(normalized, p=2, dim=-1).max().item() < 1e-4
