"""
Tests for Visualization Plotting Engine.
"""
import os
import tempfile
import pytest
from src.visualization.plots import plot_layer_wise_quality


def test_plot_layer_wise_quality_renders_png_and_pdf():
    layer_records = [
        {"layer": 0, "nDCG@10": 0.15, "Recall@10": 0.20, "MRR": 0.10, "cosine_stability": 1.0},
        {"layer": 1, "nDCG@10": 0.35, "Recall@10": 0.45, "MRR": 0.30, "cosine_stability": 0.88},
        {"layer": 2, "nDCG@10": 0.50, "Recall@10": 0.60, "MRR": 0.45, "cosine_stability": 0.94},
        {"layer": 3, "nDCG@10": 0.65, "Recall@10": 0.75, "MRR": 0.60, "cosine_stability": 0.98},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        png_path = os.path.join(tmp_dir, "test_plot.png")
        pdf_path = os.path.join(tmp_dir, "test_plot.pdf")

        plot_layer_wise_quality(
            layer_records=layer_records,
            output_png_path=png_path,
            output_pdf_path=pdf_path,
            dataset_name="SciFact",
            model_name="BAAI/bge-small-en-v1.5",
        )

        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 0

        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
