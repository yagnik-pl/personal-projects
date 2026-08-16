"""
Tests for Adaptive Early-Exit Mechanism, Stability Controllers, and Execution Modes.
Validates F9, F10 requirements: cosine stability, norm delta, patience, simulated vs dynamic equivalence.
"""

import math
import pytest
import torch
from tests.conftest import (
    MockTokenizer,
    MockTransformerEncoder,
    ref_mean_pooling,
    ref_normalize_embeddings,
    ref_cosine_similarity,
    ref_norm_delta,
)


class MockAdaptiveRetriever:
    """Mock implementation of dynamic adaptive early exit for verification."""
    def __init__(self, encoder: MockTransformerEncoder, tokenizer: MockTokenizer):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_layers = encoder.num_layers

    def encode_simulated(self, query: str, threshold: float, min_layer: int = 1, patience: int = 1):
        tokens = self.tokenizer(query, max_length=16)
        with torch.no_grad():
            out = self.encoder(tokens["input_ids"], attention_mask=tokens["attention_mask"], output_hidden_states=True)
            hidden_states = out.hidden_states

        embeddings = [
            ref_normalize_embeddings(ref_mean_pooling(h, tokens["attention_mask"]))
            for h in hidden_states
        ]

        similarities = []
        consecutive_stable = 0
        exit_layer = self.num_layers

        for l in range(1, self.num_layers + 1):
            sim = ref_cosine_similarity(embeddings[l], embeddings[l - 1])
            similarities.append(sim)

            if l >= min_layer:
                if sim >= threshold:
                    consecutive_stable += 1
                else:
                    consecutive_stable = 0

                if consecutive_stable >= patience:
                    exit_layer = l
                    break

        return embeddings[exit_layer], exit_layer, similarities

    def encode_step_by_step(self, query: str, threshold: float, min_layer: int = 1, patience: int = 1):
        tokens = self.tokenizer(query, max_length=16)
        input_ids = tokens["input_ids"]
        mask = tokens["attention_mask"]

        with torch.no_grad():
            h_prev = self.encoder.embeddings(input_ids)
            e_prev = ref_normalize_embeddings(ref_mean_pooling(h_prev, mask))

            similarities = []
            consecutive_stable = 0
            exit_layer = self.num_layers
            final_emb = None

            for l in range(1, self.num_layers + 1):
                layer_module = self.encoder.layers[l - 1]
                h_curr = layer_module(h_prev, attention_mask=mask)[0]
                e_curr = ref_normalize_embeddings(ref_mean_pooling(h_curr, mask))

                sim = ref_cosine_similarity(e_curr, e_prev)
                similarities.append(sim)

                if l >= min_layer:
                    if sim >= threshold:
                        consecutive_stable += 1
                    else:
                        consecutive_stable = 0

                    if consecutive_stable >= patience:
                        exit_layer = l
                        final_emb = e_curr
                        break

                h_prev = h_curr
                e_prev = e_curr

            if final_emb is None:
                final_emb = e_curr

        return final_emb, exit_layer, similarities


def test_cosine_stability_and_norm_delta_mathematical_identity():
    """Verify Delta_norm = sqrt(2 * (1 - S(l))) mathematical identity on unit vectors."""
    torch.manual_seed(42)
    for _ in range(10):
        v1 = ref_normalize_embeddings(torch.randn(1, 32))
        v2 = ref_normalize_embeddings(torch.randn(1, 32))

        cos_sim = ref_cosine_similarity(v1, v2)
        norm_delta = ref_norm_delta(v1, v2)

        # Theoretical relation: ||v1 - v2||_2 = sqrt(||v1||^2 + ||v2||^2 - 2<v1, v2>) = sqrt(2 - 2*S)
        expected_norm_delta = math.sqrt(max(0.0, 2.0 * (1.0 - cos_sim)))
        assert pytest.approx(norm_delta, abs=1e-5) == expected_norm_delta


def test_adaptive_exit_strict_policy_reaches_final_layer(mock_encoder, mock_tokenizer):
    """Verify that an ultra-strict threshold (tau=1.0) forces full-depth execution (layer L)."""
    retriever = MockAdaptiveRetriever(mock_encoder, mock_tokenizer)
    _, exit_layer, sims = retriever.encode_simulated("difficult query needing full depth", threshold=1.0)

    assert exit_layer == mock_encoder.num_layers
    assert len(sims) == mock_encoder.num_layers


def test_adaptive_exit_aggressive_policy_exits_at_min_layer(mock_encoder, mock_tokenizer):
    """Verify that an aggressive threshold (tau=0.0) forces early exit at min_layer."""
    retriever = MockAdaptiveRetriever(mock_encoder, mock_tokenizer)
    _, exit_layer, sims = retriever.encode_simulated(
        "simple query exiting early", threshold=0.0, min_layer=1, patience=1
    )

    assert exit_layer == 1
    assert len(sims) == 1


def test_adaptive_patience_windowing(mock_encoder, mock_tokenizer):
    """Verify patience window requires consecutive stable layers before triggering exit."""
    retriever = MockAdaptiveRetriever(mock_encoder, mock_tokenizer)
    query = "query for testing patience window"

    # First observe all similarities
    _, _, all_sims = retriever.encode_simulated(query, threshold=0.0, patience=mock_encoder.num_layers)
    
    # Choose a moderate threshold
    tau = min(all_sims) + 0.001

    # With patience=1 vs patience=2
    _, exit_p1, _ = retriever.encode_simulated(query, threshold=tau, patience=1)
    _, exit_p2, _ = retriever.encode_simulated(query, threshold=tau, patience=2)

    # Patience 2 cannot exit earlier than patience 1
    assert exit_p2 >= exit_p1


def test_simulated_vs_step_by_step_numerical_equivalence(mock_encoder, mock_tokenizer):
    """Verify simulated and step-by-step layer forward passes yield identical exit decisions and embeddings."""
    retriever = MockAdaptiveRetriever(mock_encoder, mock_tokenizer)
    queries = [
        "first test query",
        "second longer test query with more context",
        "third distinct query",
    ]

    for q in queries:
        for tau in [0.0, 0.90, 0.95, 1.0]:
            emb_sim, exit_sim, sims_sim = retriever.encode_simulated(q, threshold=tau, min_layer=1)
            emb_step, exit_step, sims_step = retriever.encode_step_by_step(q, threshold=tau, min_layer=1)

            assert exit_sim == exit_step, f"Exit layer mismatch for tau={tau}: sim={exit_sim}, step={exit_step}"
            assert torch.allclose(emb_sim, emb_step, atol=1e-5), f"Embedding mismatch for tau={tau}"
            assert len(sims_sim) == len(sims_step)
            for s1, s2 in zip(sims_sim, sims_step):
                assert pytest.approx(s1, abs=1e-5) == s2
