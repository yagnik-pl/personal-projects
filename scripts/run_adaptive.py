"""
CLI runner for adaptive early-exit retrieval experiments and threshold sweeps.
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Run adaptive early-exit retrieval")
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
    log_file = log_dir / "adaptive_run.log"
    logger = setup_logger("AdaptiveRetriever.Adaptive", log_file=str(log_file))

    logger.info("=== Starting Adaptive Early-Exit Experiment ===")
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

    # 3. Build Full-Depth Document Index
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
    min_layer = getattr(config.adaptive, "min_layer", 1) if hasattr(config, "adaptive") else 1

    # Determine thresholds to evaluate
    threshold_names = {}
    thresholds_to_run = []

    if hasattr(config, "adaptive"):
        if hasattr(config.adaptive, "thresholds"):
            for name, val in config.adaptive.thresholds.items():
                if val not in thresholds_to_run:
                    thresholds_to_run.append(val)
                threshold_names[val] = name

        if hasattr(config.adaptive, "sweep_thresholds"):
            for t in config.adaptive.sweep_thresholds:
                if t not in thresholds_to_run:
                    thresholds_to_run.append(t)
    else:
        thresholds_to_run = [0.85, 0.92, 0.98]
        threshold_names = {0.85: "aggressive", 0.92: "medium", 0.98: "strict"}

    results = []
    per_query_data = {}
    distribution_data = {}

    for thresh in thresholds_to_run:
        name = threshold_names.get(thresh, f"tau_{thresh:.2f}")
        logger.info(f"Evaluating threshold: {name} (tau={thresh})...")

        # Warmup
        for qt in query_texts[:5]:
            encoder.encode_step_wise_adaptive(qt, stability_threshold=thresh, min_layer=min_layer)

        latencies = []
        exit_layers = []
        q_embs_list = []

        per_query_data[str(thresh)] = {}

        for qid, qt in zip(query_ids, query_texts):
            t0 = time.perf_counter()
            emb, exit_l, sims = encoder.encode_step_wise_adaptive(
                qt,
                stability_threshold=thresh,
                min_layer=min_layer,
                max_layer=total_layers,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            exit_layers.append(exit_l)
            q_embs_list.append(emb)
            per_query_data[str(thresh)][qid] = {
                "exit_layer": exit_l,
                "latency_ms": round(lat_ms, 2),
                "similarities": [round(s, 4) for s in sims],
            }

        q_embs = torch.cat(q_embs_list, dim=0)

        # Search index
        search_results = index.search(q_embs, top_k=top_k)
        run_dict = {qid: search_results[i] for i, qid in enumerate(query_ids)}
        metrics = evaluate_retrieval_run(run_dict, dataset.qrels, k_values=(1, 5, 10))

        avg_el = float(np.mean(exit_layers))
        med_el = float(np.median(exit_layers))
        std_el = float(np.std(exit_layers))
        med_lat = float(np.median(latencies))

        # Distribution counts
        counts = np.bincount(exit_layers, minlength=total_layers + 1)
        distribution_data[name] = counts.tolist()

        record = {
            "threshold": name,
            "threshold_val": thresh,
            "Recall@1": round(metrics.get("Recall@1", 0.0), 4),
            "Recall@5": round(metrics.get("Recall@5", 0.0), 4),
            "Recall@10": round(metrics.get("Recall@10", 0.0), 4),
            "MRR": round(metrics.get("MRR", 0.0), 4),
            "nDCG@10": round(metrics.get("nDCG@10", 0.0), 4),
            "avg_exit_layer": round(avg_el, 2),
            "median_exit_layer": round(med_el, 1),
            "std_exit_layer": round(std_el, 2),
            "compute_pct": round((avg_el / total_layers) * 100.0, 1),
            "median_latency_ms": round(med_lat, 2),
        }
        results.append(record)
        logger.info(
            f"Threshold {name:<12} (tau={thresh:.2f}) | "
            f"nDCG@10: {record['nDCG@10']:.4f} | "
            f"Recall@10: {record['Recall@10']:.4f} | "
            f"Avg Layer: {record['avg_exit_layer']:.2f}/{total_layers} | "
            f"Compute: {record['compute_pct']}% | "
            f"Latency: {record['median_latency_ms']} ms"
        )

    # Save Tables
    tables_dir = Path(config.output.get("tables_dir", "results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "adaptive_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")

    df = pd.DataFrame(results)
    csv_path = tables_dir / "adaptive_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to {csv_path}")

    md_path = tables_dir / "adaptive_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"Saved Markdown table to {md_path}")

    # Save raw query-level breakdowns
    with open(raw_dir / "adaptive_per_query.json", "w", encoding="utf-8") as f:
        json.dump(per_query_data, f, indent=2)
    with open(raw_dir / "exit_layer_distribution.json", "w", encoding="utf-8") as f:
        json.dump(distribution_data, f, indent=2)
    logger.info(f"Saved raw per-query and distribution outputs to {raw_dir}")


if __name__ == "__main__":
    main()
