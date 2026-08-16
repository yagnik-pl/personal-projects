"""
Tests for Layer-wise Hidden State Extraction and Representation Analysis.
Validates F2, F7 requirements: all L+1 layer embeddings, shapes, step-by-step vs batch extraction.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import (
    MockTokenizer,
    MockTransformerEncoder,
    ref_mean_pooling,
    ref_normalize_embeddings,
    ref_cosine_similarity,
)

try:
    from src.models.encoder import LayerWiseEncoder
except ImportError:
    LayerWiseEncoder = None


def test_extract_all_l_plus_1_hidden_states(mock_encoder, mock_tokenizer):
    """Verify extraction produces exactly L + 1 layer embeddings of shape (B, D)."""
    texts = ["query one", "query two longer"]
    tokens = mock_tokenizer(texts, max_length=16)

    output = mock_encoder(
        tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        output_hidden_states=True,
    )

    hidden_states = output.hidden_states
    assert len(hidden_states) == mock_encoder.num_layers + 1

    pooled_layers = []
    for l_idx, h in enumerate(hidden_states):
        pooled = ref_mean_pooling(h, tokens["attention_mask"])
        normalized = ref_normalize_embeddings(pooled)
        assert normalized.shape == (2, mock_encoder.hidden_dim)
        assert torch.allclose(torch.norm(normalized, p=2, dim=-1), torch.ones(2), atol=1e-6)
        pooled_layers.append(normalized)

    assert len(pooled_layers) == 7  # 1 embedding + 6 layers


def test_step_by_step_layer_forward_equivalence(mock_encoder, mock_tokenizer):
    """Verify sequential step-by-step layer passes yield identical outputs to full forward pass."""
    text = "testing sequential layer forward pass"
    tokens = mock_tokenizer(text, max_length=16)
    input_ids = tokens["input_ids"]
    mask = tokens["attention_mask"]

    # 1. Batched forward pass
    with torch.no_grad():
        batch_out = mock_encoder(input_ids, attention_mask=mask, output_hidden_states=True)
        batch_hidden = batch_out.hidden_states

    # 2. Step-by-step sequential manual pass
    with torch.no_grad():
        step_hidden = []
        h = mock_encoder.embeddings(input_ids)
        step_hidden.append(h)

        for layer in mock_encoder.layers:
            h = layer(h, attention_mask=mask)[0]
            step_hidden.append(h)

    assert len(batch_hidden) == len(step_hidden)
    for l_idx, (h_b, h_s) in enumerate(zip(batch_hidden, step_hidden)):
        assert torch.allclose(h_b, h_s, atol=1e-6), f"Mismatch at layer {l_idx}"


def test_consecutive_layer_cosine_stability_bounds(mock_encoder, mock_tokenizer):
    """Verify consecutive layer cosine similarities S(l) are strictly bounded in [-1.0, 1.0]."""
    text = "dense retrieval stability analysis"
    tokens = mock_tokenizer(text, max_length=16)

    with torch.no_grad():
        out = mock_encoder(tokens["input_ids"], attention_mask=tokens["attention_mask"], output_hidden_states=True)
        hidden_states = out.hidden_states

    normalized_embeddings = [
        ref_normalize_embeddings(ref_mean_pooling(h, tokens["attention_mask"]))
        for h in hidden_states
    ]

    similarities = []
    for l in range(1, len(normalized_embeddings)):
        s_l = ref_cosine_similarity(normalized_embeddings[l], normalized_embeddings[l - 1])
        assert -1.0 <= s_l <= 1.000001, f"Cosine similarity {s_l} out of bounds at layer {l}"
        similarities.append(s_l)

    assert len(similarities) == mock_encoder.num_layers
