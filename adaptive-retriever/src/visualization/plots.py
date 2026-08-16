"""
Publication-grade plotting utilities for AdaptiveRetriever.
Strictly enforces non-interactive 'Agg' backend for headless environments.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent GUI popups
import matplotlib.pyplot as plt
import seaborn as sns


def plot_layer_wise_quality(
    layer_records: List[Dict[str, Any]],
    output_png_path: str,
    output_pdf_path: Optional[str] = None,
    dataset_name: str = "SciFact",
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    """
    Renders a publication-quality dual-axis plot of retrieval metrics and cosine stability vs layer depth.

    Args:
        layer_records: List of dicts containing 'layer', 'nDCG@10', 'Recall@10', 'MRR', 'cosine_stability'.
        output_png_path: File path for 300 DPI PNG output.
        output_pdf_path: Optional file path for vector PDF output.
        dataset_name: Name of the evaluated dataset.
        model_name: Name of the encoder model.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    layers = [r["layer"] for r in layer_records]
    ndcg = [r.get("nDCG@10", 0.0) for r in layer_records]
    recall10 = [r.get("Recall@10", 0.0) for r in layer_records]
    mrr = [r.get("MRR", 0.0) for r in layer_records]

    # Stability starts from layer 1 (layer 0 has no predecessor)
    stability_layers = [r["layer"] for r in layer_records if r["layer"] >= 1]
    stability = [r.get("cosine_stability", 1.0) for r in layer_records if r["layer"] >= 1]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Left Axis: IR Metrics
    line1 = ax1.plot(layers, ndcg, color="#1f77b4", marker="o", linewidth=2.2, markersize=7, label="nDCG@10")
    line2 = ax1.plot(layers, recall10, color="#2ca02c", marker="s", linestyle="--", linewidth=2.0, markersize=6, label="Recall@10")
    line3 = ax1.plot(layers, mrr, color="#ff7f0e", marker="^", linestyle=":", linewidth=2.0, markersize=6, label="MRR")

    ax1.set_xlabel("Transformer Encoder Layer Depth ($l$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Retrieval Quality Metric Score", fontsize=12, fontweight="bold", color="#1f77b4")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(layers)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right Axis: Cosine Stability
    lines = line1 + line2 + line3
    if stability:
        ax2 = ax1.twinx()
        line4 = ax2.plot(
            stability_layers,
            stability,
            color="#d62728",
            marker="D",
            linestyle="-.",
            linewidth=2.2,
            markersize=6,
            label=r"Cosine Stability $S(l) = \langle e_l, e_{l-1} \rangle$",
        )
        ax2.set_ylabel("Consecutive Layer Cosine Stability", fontsize=12, fontweight="bold", color="#d62728")

        min_stab = min(stability) if stability else 0.8
        ax2.set_ylim(max(0.0, min_stab - 0.05), 1.02)
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.grid(False)
        lines = lines + line4

    # Combined Legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, framealpha=0.95, edgecolor="gray", fontsize=10)

    plt.title(
        f"Layer-wise Retrieval Quality & Semantic Stability Progression\nModel: {model_name} | Dataset: {dataset_name.upper()} (Asymmetric Mode)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()

    # Save raster PNG
    png_path = Path(output_png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    # Save vector PDF
    if output_pdf_path:
        pdf_path = Path(output_pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    plt.close(fig)
