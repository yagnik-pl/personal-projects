"""
CLI runner for full-depth baseline dense retrieval experiment.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run full-depth baseline retrieval")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
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
    logger = setup_logger("AdaptiveRetriever.Baseline", log_file=str(log_file))

    logger.info(f"=== Starting Experiment: {config.experiment.name} ===")
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

    # 3. Encode Corpus Documents at Full Depth L
    doc_ids, doc_texts = data.get_corpus_texts()
    logger.info(f"Encoding {len(doc_texts):,} corpus documents at full depth L={encoder.num_layers}...")
    doc_embs = encoder.encode_layer(
        doc_texts,
        layer=encoder.num_layers,
        batch_size=config.retrieval.get("batch_size", 64),
        show_progress=True,
    )

    # 4. Build Exact Flat Dense Index
    index = DenseIndex(embedding_dim=encoder.hidden_dim, device=device)
    index.build(doc_ids=doc_ids, embeddings=doc_embs)
    logger.info(f"Built exact dense flat index with {len(doc_ids):,} entries.")

    # 5. Encode Queries & Search
    query_ids, query_texts = data.get_query_texts(only_judged=True)
    logger.info(f"Encoding {len(query_texts):,} queries at full depth L={encoder.num_layers}...")
    query_embs = encoder.encode_layer(
        query_texts,
        layer=encoder.num_layers,
        batch_size=config.retrieval.get("batch_size", 64),
        show_progress=True,
    )

    top_k = config.retrieval.get("top_k", 10)
    logger.info(f"Searching top-{top_k} candidate documents per query...")
    search_results = index.search(query_embs, top_k=top_k)
    run_dict = {qid: search_results[i] for i, qid in enumerate(query_ids)}

    # 6. Evaluate IR Metrics
    logger.info("Computing standard IR benchmark metrics against ground-truth qrels...")
    metrics = evaluate_retrieval_run(run_dict, data.qrels, k_values=(1, 5, 10))

    logger.info("=" * 60)
    logger.info("BASELINE RETRIEVAL RESULTS:")
    eval_metrics_list = config.evaluation.get("metrics", ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@10"])
    for metric_name in eval_metrics_list:
        if metric_name in metrics:
            logger.info(f"  {metric_name:<15}: {metrics[metric_name]:.4f}")
    logger.info(f"  Evaluated Queries: {int(metrics.get('num_queries_evaluated', len(query_ids)))}")
    logger.info("=" * 60)

    # 7. Quality Assertion
    min_recall = config.evaluation.get("min_recall_at_10", 0.10)
    actual_recall = metrics.get("Recall@10", 0.0)
    if actual_recall < min_recall:
        logger.error(f"Quality Check FAILED: Recall@10 = {actual_recall:.4f} < threshold {min_recall:.4f}")
        sys.exit(1)
    else:
        logger.info(f"Quality Check PASSED: Recall@10 = {actual_recall:.4f} >= threshold {min_recall:.4f}")

    # 8. Export Structured Results
    tables_dir = Path(config.output.get("tables_dir", "results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    result_record = {
        "experiment": config.experiment.name,
        "model": config.model.name_or_path,
        "dataset": config.dataset.name,
        "split": config.dataset.get("split", "test"),
        "depth": encoder.num_layers,
        "device": device_info.get("device_name", str(device)),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }

    # JSON export
    json_path = tables_dir / "baseline_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")

    # CSV & Markdown export
    df = pd.DataFrame([result_record])
    csv_path = tables_dir / "baseline_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to {csv_path}")

    md_path = tables_dir / "baseline_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    logger.info(f"Saved Markdown table to {md_path}")

    # Persist config snapshot
    config_snapshot_path = log_dir / f"{config.experiment.name}_config.yaml"
    save_config(config, config_snapshot_path)
    logger.info(f"Saved configuration snapshot to {config_snapshot_path}")


if __name__ == "__main__":
    main()
