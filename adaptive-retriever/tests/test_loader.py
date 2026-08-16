"""
Tests for BEIRDatasetLoader using synthetic directory fixtures.
"""
import os
import tempfile
import pytest
from src.data.loader import BEIRDatasetLoader
from src.data.synthetic import create_temp_beir_directory


def test_beir_loader_from_local_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_temp_beir_directory(tmp_dir, dataset_name="scifact", num_docs=15, num_queries=4)

        loader = BEIRDatasetLoader(data_dir=tmp_dir)
        split = loader.load_split("scifact", split="test")

        assert split.name == "scifact"
        assert split.num_docs == 15
        assert split.num_queries == 4
        assert split.num_judged_queries == 4


def test_beir_loader_missing_dataset_raises_error():
    loader = BEIRDatasetLoader(data_dir="non_existent_data_dir_12345")
    with pytest.raises(ValueError):
        loader.download_dataset("invalid_dataset_name_xyz")
