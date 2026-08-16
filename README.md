# Personal Projects Repository

A collection of research, machine learning, natural language processing, and engineering projects.

---

## Projects Directory

| Project | Category | Description | Primary Stack | Status |
|---|---|---|---|:---:|
| [**`adaptive-retriever`**](./adaptive-retriever/) | NLP & Dense Retrieval | Query-adaptive dynamic-depth early exit for dense Transformer retrievers. Investigates representation stability, accuracy–efficiency Pareto trade-offs, and layer-wise IR capacity. Inspired by EffiR (ACL 2026). | PyTorch, HuggingFace, NumPy, Matplotlib | Completed |

---

## Project Summaries

### 1. [AdaptiveRetriever](./adaptive-retriever/)
- **Core Problem**: Production dense bi-encoders evaluate all queries through all $L$ Transformer layers regardless of difficulty, creating unnecessary latency on the online critical path.
- **Key Findings**:
  - Uncovers the *Early-Layer Stability Illusion*: Layers 1–6 exhibit near-perfect consecutive cosine similarity ($S(l) > 0.993$) yet have $<1\%$ Recall@10.
  - Identifies a major contrastive phase transition at Layers 8–10 ($S(l)$ drops to $0.831$) where retrieval representations emerge.
  - Demonstrates that Layer 11 retains **$88.9\%$ of full-depth Recall@10** ($0.7512$ vs $0.8452$) and **$87.5\%$ of nDCG@10** ($0.6300$ vs $0.7200$) with a $8.3\%$ compute reduction.
- **Test Suite**: 71/71 passing unit and integration tests (100% offline CPU execution).
- **Details & Reproduction**: See [adaptive-retriever/README.md](./adaptive-retriever/README.md).
