"""
Tests for Synthetic Dataset Generation and Fixtures.
"""
import os
import tempfile
import pytest
from src.data.synthetic import create_temp_beir_directory, generate_synthetic_dataset


def test_generate_synthetic_dataset():
    split = generate_synthetic_dataset(num_docs=25, num_queries=7, seed=42)

    assert split.name == "synthetic_test"
    assert split.num_docs == 25
    assert split.num_queries == 7
    assert split.num_judged_queries == 7
    assert split.num_judgments > 0

    # Verify seed determinism
    split2 = generate_synthetic_dataset(num_docs=25, num_queries=7, seed=42)
    assert list(split.corpus.keys()) == list(split2.corpus.keys())
    assert split.corpus["doc_000"].text == split2.corpus["doc_000"].text


def test_create_temp_beir_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_path = create_temp_beir_directory(tmp_dir, dataset_name="mock_beir", num_docs=10, num_queries=3)

        assert os.path.exists(os.path.join(ds_path, "corpus.jsonl"))
        assert os.path.exists(os.path.join(ds_path, "queries.jsonl"))
        assert os.path.exists(os.path.join(ds_path, "qrels", "test.tsv"))
