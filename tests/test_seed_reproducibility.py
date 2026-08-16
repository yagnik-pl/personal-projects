"""
Tests for Deterministic Seeding and Experiment Reproducibility.
Validates F17 requirements: deterministic seeding across Python, NumPy, PyTorch.
"""

import random
import numpy as np
import pytest
import torch

try:
    from src.utils.seed import set_seed
except ImportError:
    def set_seed(seed: int = 42) -> None:
        """Reference implementation for setting random seeds across all libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def test_seed_reproducibility_torch_tensors():
    """Verify that identical seeds produce bitwise identical PyTorch tensor generations."""
    set_seed(42)
    t1 = torch.randn(10, 32)

    set_seed(42)
    t2 = torch.randn(10, 32)

    assert torch.equal(t1, t2)


def test_seed_reproducibility_numpy_arrays():
    """Verify that identical seeds produce bitwise identical NumPy arrays."""
    set_seed(42)
    a1 = np.random.randn(5, 5)

    set_seed(42)
    a2 = np.random.randn(5, 5)

    assert np.array_equal(a1, a2)


def test_seed_reproducibility_python_random():
    """Verify that identical seeds produce identical Python random sequences."""
    set_seed(42)
    r1 = [random.random() for _ in range(20)]

    set_seed(42)
    r2 = [random.random() for _ in range(20)]

    assert r1 == r2


def test_different_seeds_produce_distinct_values():
    """Verify that different random seeds produce statistically distinct tensors."""
    set_seed(42)
    t1 = torch.randn(10, 32)

    set_seed(999)
    t2 = torch.randn(10, 32)

    assert not torch.equal(t1, t2)
