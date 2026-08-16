"""
CLI runner for qualitative analysis of query characteristics across early-exit depths.
Analyzes query length, token counts, lexical properties, and stability signals.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Ensure root workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import BEIRDatasetLoader
from src.evaluation.metrics import evaluate_retrieval_run
from src.models.encoder import LayerWiseEncoder
from src.retrieval.index import DenseIndex
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run qualitative query analysis")
    parser.add_argument("--config", type=str, default="configs/adaptive.yaml", help="Path to config YAML")
    parser.add_argument("--threshold", type=float, default=0.85, help="Stability threshold for analysis")
    parser.add_argument("--min_layer", type=int, default=10, help="Minimum layer for adaptive exit")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(42)

    logger = setup_logger("AdaptiveRetriever.QualitativeAnalysis")
    device = resolve_device("auto")
    logger.info(f"=== Starting Qualitative Query Analysis (Device: {device}) ===")

    loader = BEIRDatasetLoader(data_dir=config.dataset.data_dir)
    dataset = loader.load_split(config.dataset.name, split=config.dataset.get("split", "test"))
    corpus_ids, corpus_texts = dataset.get_corpus_texts()
    query_ids, query_texts = dataset.get_query_texts(only_judged=True)

    encoder = LayerWiseEncoder(
        model_name_or_path=config.model.name_or_path,
        pooling_strategy=config.model.get("pooling_strategy"),
        device=device,
    )
    total_layers = encoder.num_layers

    # Encode corpus at full depth
    logger.info("Encoding corpus at full-depth L=12...")
    doc_embs = encoder.encode_layer(corpus_texts, layer=total_layers, batch_size=64, show_progress=True)
    index = DenseIndex(embedding_dim=encoder.hidden_dim, device=device)
    index.build(doc_ids=corpus_ids, embeddings=doc_embs)

    # Step-wise encoding with detailed signal capture
    query_records = []
    q_embs_list = []

    for qid, text in zip(query_ids, query_texts):
        tokens = text.strip().split()
        word_count = len(tokens)
        char_count = len(text)

        emb, exit_l, sims = encoder.encode_step_wise_adaptive(
            text,
            stability_threshold=args.threshold,
            min_layer=args.min_layer,
            max_layer=total_layers,
        )
        q_embs_list.append(emb)

        query_records.append({
            "query_id": qid,
            "query_text": text,
            "word_count": word_count,
            "char_count": char_count,
            "exit_layer": exit_l,
            "stability_at_exit": round(sims[-1], 4) if sims else 1.0,
            "all_similarities": [round(s, 4) for s in sims],
        })

    q_embs = torch.cat(q_embs_list, dim=0)
    search_results = index.search(q_embs, top_k=10)

    # Compute per-query retrieval success
    for i, rec in enumerate(query_records):
        qid = rec["query_id"]
        rel_docs = dataset.qrels.get(qid, {})
        retrieved_ids = [doc_id for doc_id, _ in search_results[i]]
        hit_at_1 = retrieved_ids[0] in rel_docs if retrieved_ids else False
        hit_at_10 = any(d in rel_docs for d in retrieved_ids)
        rec["hit@1"] = int(hit_at_1)
        rec["hit@10"] = int(hit_at_10)

    df = pd.DataFrame(query_records)

    # Grouped statistics by exit layer
    summary = []
    for layer, grp in df.groupby("exit_layer"):
        summary.append({
            "exit_layer": int(layer),
            "num_queries": len(grp),
            "pct_queries": round((len(grp) / len(df)) * 100.0, 1),
            "avg_word_count": round(grp["word_count"].mean(), 1),
            "avg_char_count": round(grp["char_count"].mean(), 1),
            "hit@1_rate": round(grp["hit@1"].mean(), 4),
            "hit@10_rate": round(grp["hit@10"].mean(), 4),
            "avg_stability": round(grp["stability_at_exit"].mean(), 4),
        })

    summary_df = pd.DataFrame(summary)
    logger.info("\n" + summary_df.to_markdown(index=False))

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(tables_dir / "qualitative_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold": args.threshold,
            "min_layer": args.min_layer,
            "layer_summary": summary,
            "queries": query_records,
        }, f, indent=2)

    with open(tables_dir / "qualitative_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Qualitative Query Analysis by Exit Layer\n\n")
        f.write(f"Configuration: `threshold={args.threshold}`, `min_layer={args.min_layer}`\n\n")
        f.write("## Layer Group Statistics\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Representative Queries\n\n")
        for layer, grp in df.groupby("exit_layer"):
            f.write(f"### Exit Layer {layer} (Sample of 3 queries)\n\n")
            samples = grp.head(3)
            for _, row in samples.iterrows():
                f.write(f"- **Query**: \"{row['query_text']}\"\n")
                f.write(f"  - Length: {row['word_count']} words | Exit Stability: {row['stability_at_exit']} | Hit@10: {bool(row['hit@10'])}\n")
            f.write("\n")

    logger.info(f"Saved qualitative results to {tables_dir}/qualitative_analysis.json and .md")


if __name__ == "__main__":
    main()
