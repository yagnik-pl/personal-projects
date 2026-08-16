"""
Tests for Model Architecture, Layer Inspection, and Tokenization.
Validates F1, F2 requirements on parameter budget, layer count, and shapes on CPU.
"""

import pytest
import torch
import torch.nn as nn

try:
    from src.models.encoder import LayerWiseEncoder
except ImportError:
    LayerWiseEncoder = None


def test_mock_model_layer_count_and_hidden_dim(mock_encoder):
    """Verify encoder has exact expected layer count and hidden dimensions."""
    assert hasattr(mock_encoder, "num_layers")
    assert mock_encoder.num_layers == 6
    assert mock_encoder.hidden_dim == 64
    assert len(mock_encoder.layers) == 6

    # Test forward pass shape
    batch_size = 3
    seq_len = 8
    dummy_input_ids = torch.randint(0, 500, (batch_size, seq_len))
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    output = mock_encoder(dummy_input_ids, attention_mask=dummy_mask, output_hidden_states=True)
    assert output.last_hidden_state.shape == (batch_size, seq_len, 64)
    # Total hidden states must be L + 1 (layer 0 embeddings + L transformer layers)
    assert len(output.hidden_states) == 7
    for h in output.hidden_states:
        assert h.shape == (batch_size, seq_len, 64)


def test_model_parameter_budget_under_hardware_constraint(mock_encoder):
    """Verify parameter count satisfies consumer hardware budget (< 300M parameters)."""
    total_params = sum(p.numel() for p in mock_encoder.parameters())
    trainable_params = sum(p.numel() for p in mock_encoder.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params == total_params
    # Constraint: Must comfortably fit in 4-8 GB VRAM (under 300M params)
    assert total_params < 300_000_000


def test_tokenizer_batch_and_padding_shapes(mock_tokenizer):
    """Verify tokenizer produces correctly shaped tensors and attention masks."""
    texts = [
        "short text",
        "a significantly longer sentence with multiple words for testing padding"
    ]
    encoded = mock_tokenizer(texts, max_length=16)

    assert "input_ids" in encoded
    assert "attention_mask" in encoded

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    assert input_ids.shape == (2, 16)
    assert attention_mask.shape == (2, 16)

    # First text is shorter, so it must have fewer active tokens (1s) than second text
    active_tokens_0 = attention_mask[0].sum().item()
    active_tokens_1 = attention_mask[1].sum().item()
    assert active_tokens_0 < active_tokens_1

    # Inactive positions must be 0 in attention_mask and 0 in input_ids
    assert input_ids[0, active_tokens_0:].sum().item() == 0


def test_model_cpu_execution_without_cuda(mock_encoder):
    """Verify model runs on CPU with zero CUDA dependency."""
    device = torch.device("cpu")
    mock_encoder.to(device)

    dummy_input = torch.tensor([[101, 205, 310, 102]], device=device)
    dummy_mask = torch.tensor([[1, 1, 1, 1]], device=device)

    out = mock_encoder(dummy_input, attention_mask=dummy_mask)
    assert out.last_hidden_state.device == device
    assert not out.last_hidden_state.is_cuda
