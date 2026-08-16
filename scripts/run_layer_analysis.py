"""
CLI runner for layer-wise retrieval quality progression and consecutive cosine stability analysis.
"""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import torch

# Ensure root workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import BEIRDatasetLoader
from src.evaluation.metrics import evaluate_retrieval_run
from src.models.encoder import LayerWiseEncoder
from src.retrieval.index import DenseIndex
from src.utils.config import load_config, save_config
from src.utils.device import get_device_info, resolve_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.visualization.plots import plot_layer_wise_quality


def parse_args():
    parser = argparse.ArgumentParser(description="Run layer-wise retrieval quality analysis")
    parser.add_argument("--config", type=str, default="configs/layer_analysis.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset override")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.dataset:
        config.dataset.name = args.dataset
    if args.device:
        config.model.device = args.device

    log_dir = Path(config.output.get("log_dir", "results/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{config.experiment.name}.log"
    logger = setup_logger("AdaptiveRetriever.LayerAnalysis", log_file=str(log_file))

    logger.info(f"=== Starting Layer Analysis Experiment: {config.experiment.name} ===")
    set_seed(config.experiment.get("seed", 42))

    device = resolve_device(config.model.get("device", "auto"))
    device_info = get_device_info(device)
    logger.info(f"Resolved Device: {device} | {device_info.get('device_name', '')}")

    # 1. Load Dataset
    logger.info(f"Loading dataset '{config.dataset.name}' from '{config.dataset.data_dir}'...")
    loader = BEIRDatasetLoader(data_dir=config.dataset.data_dir)
    data = loader.load_split(config.dataset.name, split=config.dataset.get("split", "test"))
    logger.info(f"Corpus: {len(data.corpus):,} docs | Queries: {len(data.queries):,} | Qrels: {len(data.qrels):,} evaluated queries")

    # 2. Instantiate Encoder
    logger.info(f"Loading encoder: {config.model.name_or_path} (pooling: {config.model.get('pooling_strategy', 'auto')})...")
    encoder = LayerWiseEncoder(
        model_name_or_path=config.model.name_or_path,
        pooling_strategy=config.model.get("pooling_strategy"),
        max_length=config.model.get("max_length", 512),
        device=device,
    )
    num_layers = encoder.num_layers

    # 3. Build Full-Depth Document Index (Asymmetric Retrieval Mode)
    doc_ids, doc_texts = data.get_corpus_texts()
    logger.info(f"Encoding corpus at full-depth L={num_layers} for Asymmetric Retrieval Index...")
    doc_embs = encoder.encode_layer(
        doc_texts,
        layer=num_layers,
        batch_size=config.retrieval.get("batch_size", 64),
        show_progress=True,
    )

    index = DenseIndex(embedding_dim=encoder.hidden_dim, device=device)
    index.build(doc_ids=doc_ids, embeddings=doc_embs)
    logger.info(f"Built reference index with {len(doc_ids):,} documents.")

    # 4. Extract Query Representations across ALL layers (0..L)
    query_ids, query_texts = data.get_query_texts(only_judged=True)
    logger.info(f"Extracting all layer representations (0..{num_layers}) for {len(query_texts):,} queries...")
    all_query_embs = encoder.encode_all_layers(
        query_texts,
        batch_size=config.retrieval.get("batch_size", 64),
        show_progress=True,
    )

    # 5. Compute Consecutive Layer Cosine Stability
    cosine_similarities = [1.0]  # Layer 0 has no predecessor
    for l in range(1, num_layers + 1):
        prev_q = all_query_embs[l - 1]
        curr_q = all_query_embs[l]
        sims = torch.sum(prev_q * curr_q, dim=-1)
        mean_sim = float(sims.mean().item())
        cosine_similarities.append(mean_sim)
        logger.info(f"Layer {l-1:2d} -> Layer {l:2d} Cosine Stability: {mean_sim:.4f}")

    # 6. Evaluate Retrieval at Each Layer
    records = []
    logger.info("Evaluating retrieval performance layer-by-layer...")
    top_k = config.retrieval.get("top_k", 10)

    for l in range(num_layers + 1):
        q_l = all_query_embs[l]
        search_results = index.search(q_l, top_k=top_k)
        run_dict = {qid: search_results[i] for i, qid in enumerate(query_ids)}
        metrics = evaluate_retrieval_run(run_dict, data.qrels, k_values=(1, 5, 10))

        record = {
            "layer": l,
            "cosine_stability": round(cosine_similarities[l], 4) if l > 0 else 1.0,
            "Recall@1": round(metrics.get("Recall@1", 0.0), 4),
            "Recall@5": round(metrics.get("Recall@5", 0.0), 4),
            "Recall@10": round(metrics.get("Recall@10", 0.0), 4),
            "Precision@1": round(metrics.get("Precision@1", 0.0), 4),
            "Precision@5": round(metrics.get("Precision@5", 0.0), 4),
            "Precision@10": round(metrics.get("Precision@10", 0.0), 4),
            "MRR": round(metrics.get("MRR", 0.0), 4),
            "nDCG@10": round(metrics.get("nDCG@10", 0.0), 4),
        }
        records.append(record)
        logger.info(
            f"Layer {l:2d}/{num_layers:2d} | "
            f"nDCG@10: {record['nDCG@10']:.4f} | "
            f"Recall@10: {record['Recall@10']:.4f} | "
            f"MRR: {record['MRR']:.4f} | "
            f"Stability: {record['cosine_stability']:.4f}"
        )

    # 7. Save Tables
    tables_dir = Path(config.output.get("tables_dir", "results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    json_path = tables_dir / "layer_analysis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")

    csv_path = tables_dir / "layer_analysis_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to {csv_path}")

    md_path = tables_dir / "layer_analysis_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"Saved Markdown table to {md_path}")

    # 8. Render Publication Figures
    figures_dir = Path(config.output.get("figures_dir", "results/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    png_filename = config.output.get("figure_filename", "layer_wise_quality.png")
    pdf_filename = config.output.get("figure_pdf_filename", "layer_wise_quality.pdf")
    png_path = figures_dir / png_filename
    pdf_path = figures_dir / pdf_filename

    logger.info("Rendering publication-grade layer analysis plot...")
    plot_layer_wise_quality(
        layer_records=records,
        output_png_path=str(png_path),
        output_pdf_path=str(pdf_path),
        dataset_name=config.dataset.name,
        model_name=config.model.name_or_path,
    )
    logger.info(f"Saved publication figures to {png_path} and {pdf_path}")

    # Persist config snapshot
    config_snapshot_path = log_dir / f"{config.experiment.name}_config.yaml"
    save_config(config, config_snapshot_path)
    logger.info(f"Saved configuration snapshot to {config_snapshot_path}")


if __name__ == "__main__":
    main()
