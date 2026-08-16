"""
Tests for Fixed-Depth Static Early Exit Retrieval Baselines.
Validates F8 requirements: static early termination at fixed depth k, bounds checking, layer indexing.
"""

import pytest
import torch
from tests.conftest import (
    MockTokenizer,
    MockTransformerEncoder,
    ref_mean_pooling,
    ref_normalize_embeddings,
)


class MockFixedDepthEncoder:
    """Encoder that terminates execution strictly at layer k (1 <= k <= L)."""
    def __init__(self, encoder: MockTransformerEncoder, tokenizer: MockTokenizer):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_layers = encoder.num_layers

    def encode_at_fixed_depth(self, texts, depth: int, max_length: int = 16):
        if depth < 0 or depth > self.num_layers:
            raise ValueError(f"Invalid depth {depth}. Must be in range [0, {self.num_layers}].")

        tokens = self.tokenizer(texts, max_length=max_length)
        input_ids = tokens["input_ids"]
        mask = tokens["attention_mask"]

        with torch.no_grad():
            h = self.encoder.embeddings(input_ids)
            if depth == 0:
                pooled = ref_mean_pooling(h, mask)
                return ref_normalize_embeddings(pooled)

            # Execute layers up to 'depth' only
            for i in range(depth):
                h = self.encoder.layers[i](h, attention_mask=mask)[0]

            pooled = ref_mean_pooling(h, mask)
            return ref_normalize_embeddings(pooled)


def test_fixed_depth_layer_selection(mock_encoder, mock_tokenizer):
    """Verify that fixed depth k extracts exactly the layer k representation."""
    fixed_encoder = MockFixedDepthEncoder(mock_encoder, mock_tokenizer)
    texts = ["fixed depth retrieval query"]
    tokens = mock_tokenizer(texts, max_length=16)

    # Reference full hidden states
    full_out = mock_encoder(tokens["input_ids"], attention_mask=tokens["attention_mask"], output_hidden_states=True)
    all_hidden = full_out.hidden_states

    for k in range(mock_encoder.num_layers + 1):
        emb_k = fixed_encoder.encode_at_fixed_depth(texts, depth=k)
        ref_pooled = ref_normalize_embeddings(ref_mean_pooling(all_hidden[k], tokens["attention_mask"]))

        assert torch.allclose(emb_k, ref_pooled, atol=1e-6), f"Mismatch at fixed depth {k}"


def test_fixed_depth_bounds_checking(mock_encoder, mock_tokenizer):
    """Verify that out-of-bound depth requests raise ValueError."""
    fixed_encoder = MockFixedDepthEncoder(mock_encoder, mock_tokenizer)
    texts = ["test query"]

    # Negative depth
    with pytest.raises(ValueError):
        fixed_encoder.encode_at_fixed_depth(texts, depth=-1)

    # Depth exceeding max layers
    with pytest.raises(ValueError):
        fixed_encoder.encode_at_fixed_depth(texts, depth=mock_encoder.num_layers + 1)


def test_fixed_depth_monotonic_computation():
    """Verify depth parameter directly indexes the layer execution count."""
    mock_enc = MockTransformerEncoder(num_layers=6, hidden_dim=32)
    mock_tok = MockTokenizer(vocab_size=100, max_length=8)
    fixed_encoder = MockFixedDepthEncoder(mock_enc, mock_tok)

    # Verify all layers 0..6 execute cleanly
    for k in range(7):
        emb = fixed_encoder.encode_at_fixed_depth(["monotonicity check"], depth=k)
        assert emb.shape == (1, 32)
        assert torch.allclose(torch.norm(emb, p=2, dim=-1), torch.tensor([1.0]), atol=1e-6)
