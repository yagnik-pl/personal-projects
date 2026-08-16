"""
Pure PyTorch pooling and normalization functions for dense representation models.
Guarantees numerical stability with zero external library overhead.
"""
from typing import Optional
import torch
import torch.nn.functional as F


def cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extracts the [CLS] token representation (index 0) from hidden states.

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)

    Returns:
        Tensor of shape (batch_size, hidden_dim)
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"Expected 3D hidden_states tensor of shape (batch_size, seq_len, hidden_dim), "
            f"got shape {list(hidden_states.shape)}"
        )
    if hidden_states.size(1) == 0:
        raise ValueError("Sequence length dimension (dim 1) cannot be 0.")
    return hidden_states[:, 0, :].contiguous()


def mean_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes attention-masked average token pooling across sequence length.

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len) with 1 for tokens, 0 for padding

    Returns:
        Tensor of shape (batch_size, hidden_dim)
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"Expected 3D hidden_states tensor of shape (batch_size, seq_len, hidden_dim), "
            f"got shape {list(hidden_states.shape)}"
        )
    if attention_mask.dim() != 2:
        raise ValueError(
            f"Expected 2D attention_mask tensor of shape (batch_size, seq_len), "
            f"got shape {list(attention_mask.shape)}"
        )
    if hidden_states.size(0) != attention_mask.size(0) or hidden_states.size(1) != attention_mask.size(1):
        raise ValueError(
            f"Dimension mismatch between hidden_states {list(hidden_states.shape)} "
            f"and attention_mask {list(attention_mask.shape)}"
        )

    # Expand attention mask: (B, T) -> (B, T, D)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)

    # Sum mask with clamp to prevent division by zero
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def normalize_embeddings(
    embeddings: torch.Tensor, p: float = 2.0, eps: float = 1e-9
) -> torch.Tensor:
    """
    Applies Lp normalization along the last dimension (embedding dimension).
    Ensures ||embeddings||_p == 1.0 for non-zero vectors.

    Args:
        embeddings: Tensor of shape (..., hidden_dim)
        p: Norm degree (default 2.0 for standard Euclidean/L2 norm)
        eps: Small epsilon for numerical stability

    Returns:
        Normalized Tensor of shape (..., hidden_dim)
    """
    if embeddings.numel() == 0:
        return embeddings
    norm = torch.norm(embeddings, p=p, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return embeddings / norm
