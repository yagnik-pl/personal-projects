"""
CLI runner for fixed-depth baseline dense retrieval experiment.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-depth baseline retrieval")
    parser.add_argument("--config", type=str, default="configs/adaptive.yaml", help="Path to config YAML")
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
    log_file = log_dir / "fixed_depth.log"
    logger = setup_logger("AdaptiveRetriever.FixedDepth", log_file=str(log_file))

    logger.info("=== Starting Fixed-Depth Baseline Experiment ===")
    set_seed(config.experiment.get("seed", 42))

    device = resolve_device(config.model.get("device", "auto"))
    device_info = get_device_info(device)
    logger.info(f"Resolved Device: {device} | {device_info.get('device_name', '')}")

    # 1. Load Dataset
    loader = BEIRDatasetLoader(data_dir=config.dataset.data_dir)
    dataset = loader.load_split(config.dataset.name, split=config.dataset.get("split", "test"))
    corpus_ids, corpus_texts = dataset.get_corpus_texts()
    query_ids, query_texts = dataset.get_query_texts(only_judged=True)
    logger.info(f"Corpus: {len(corpus_ids):,} docs | Queries: {len(query_ids):,} evaluated queries")

    # 2. Instantiate Encoder
    encoder = LayerWiseEncoder(
        model_name_or_path=config.model.name_or_path,
        pooling_strategy=config.model.get("pooling_strategy"),
        max_length=config.model.get("max_length", 512),
        device=device,
    )
    total_layers = encoder.num_layers

    # 3. Build Full-Depth Corpus Index
    logger.info(f"Encoding corpus at full-depth L={total_layers}...")
    doc_embs = encoder.encode_layer(
        corpus_texts,
        layer=total_layers,
        batch_size=config.retrieval.get("batch_size", 64),
        show_progress=True,
    )
    index = DenseIndex(embedding_dim=encoder.hidden_dim, device=device)
    index.build(doc_ids=corpus_ids, embeddings=doc_embs)
    logger.info(f"Built reference index with {len(corpus_ids):,} documents.")

    top_k = config.retrieval.get("top_k", 10)
    fixed_depths = list(range(1, total_layers + 1))
    if hasattr(config, "fixed_depths") and hasattr(config.fixed_depths, "depths"):
        fixed_depths = list(config.fixed_depths.depths)

    results = []

    for depth in fixed_depths:
        logger.info(f"Evaluating Fixed Depth {depth}/{total_layers}...")

        # Benchmark per-query latency (warmup 5 queries, measure 50 queries)
        for qt in query_texts[:5]:
            encoder.encode_layer([qt], layer=depth)

        latencies = []
        for qt in query_texts[:50]:
            t0 = time.perf_counter()
            encoder.encode_layer([qt], layer=depth)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        median_latency = float(np.median(latencies))

        # Batch encode all test queries at this depth
        q_embs = encoder.encode_layer(
            query_texts,
            layer=depth,
            batch_size=config.retrieval.get("batch_size", 64),
        )

        search_results = index.search(q_embs, top_k=top_k)
        run_dict = {qid: search_results[i] for i, qid in enumerate(query_ids)}
        metrics = evaluate_retrieval_run(run_dict, dataset.qrels, k_values=(1, 5, 10))

        record = {
            "layer": depth,
            "Recall@1": round(metrics.get("Recall@1", 0.0), 4),
            "Recall@5": round(metrics.get("Recall@5", 0.0), 4),
            "Recall@10": round(metrics.get("Recall@10", 0.0), 4),
            "MRR": round(metrics.get("MRR", 0.0), 4),
            "nDCG@10": round(metrics.get("nDCG@10", 0.0), 4),
            "compute_pct": round((depth / total_layers) * 100.0, 1),
            "median_latency_ms": round(median_latency, 2),
        }
        results.append(record)
        logger.info(
            f"Depth {depth:2d}/{total_layers:2d} | "
            f"nDCG@10: {record['nDCG@10']:.4f} | "
            f"Recall@10: {record['Recall@10']:.4f} | "
            f"MRR: {record['MRR']:.4f} | "
            f"Compute: {record['compute_pct']}% | "
            f"Latency: {record['median_latency_ms']} ms"
        )

    tables_dir = Path(config.output.get("tables_dir", "results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON, CSV, MD
    json_path = tables_dir / "fixed_depth_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")

    df = pd.DataFrame(results)
    csv_path = tables_dir / "fixed_depth_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to {csv_path}")

    md_path = tables_dir / "fixed_depth_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"Saved Markdown table to {md_path}")


if __name__ == "__main__":
    main()
