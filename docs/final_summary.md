# AdaptiveRetriever: Final Research Summary

## Executive Overview
AdaptiveRetriever investigated whether dense Transformer retrievers can dynamically allocate encoder layer depth on a per-query basis at runtime via early-exit heuristics. Using `BAAI/bge-small-en-v1.5` on the SciFact benchmark (5,183 documents, 300 test queries), we mapped layer-wise retrieval capability, evaluated fixed-depth baselines, and analyzed dynamic early-exit behavior.

---

## Key Research Findings

1. **The Early-Layer Stability Illusion**:
   - Consecutive layer cosine similarities in layers 1–6 exceed $0.993$, yet retrieval performance in these layers is negligible ($<1\%$ Recall@10).
   - High inter-layer cosine similarity in early layers reflects structural collinearity, not semantic retrieval readiness.

2. **Representation Phase Transition (Layers 8–10)**:
   - Between Layers 8 and 10, inter-layer cosine similarity drops sharply to $0.83–0.85$, representing the active semantic reorientation phase where contrastive retrieval features are formed.

3. **Pre-Final Layer Quality Retention (Layer 11)**:
   - Layer 11 achieves $0.7512$ Recall@10 ($88.9\%$ of full-depth $0.8452$) and $0.6300$ nDCG@10 ($87.5\%$ of full-depth $0.7200$) using $91.7\%$ of compute ($11/12$ layers).

4. **Gated Adaptive Execution**:
   - An ungated early-exit policy exiting purely on cosine stability ($\tau \le 0.99$) fails catastrophically by stopping at Layer 2.
   - Gating early exits to the representation refinement zone (Layers 10–12) enables $8.7\%$ of queries to exit safely at Layer 11 with $84.6\%$ Hit@10, preserving computation without significant quality loss.

---

## Measured Performance Comparison

| Method | Exit Depth | Compute % | Recall@10 | MRR | nDCG@10 | Latency (ms) | Quality Retention |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fixed L10 | 10.0 | 83.3% | 0.2169 | 0.1061 | 0.1298 | 12.15 | 18.0% |
| Fixed L11 | 11.0 | 91.7% | 0.7512 | 0.6002 | 0.6300 | 13.10 | 87.5% |
| Full Baseline (L12) | 12.0 | 100.0% | 0.8452 | 0.6845 | 0.7200 | 11.90 | 100.0% |

---

## Limitations

1. **Pre-trained Architecture Bias**: Standard BERT/BGE encoders are trained with pooled embeddings derived solely from the final layer; intermediate representations are not explicitly regularized to be retrieval-ready.
2. **Sequential Loop Overhead**: Python-level sequential layer iteration introduces kernel dispatch overhead for single-query CPU/GPU calls relative to fused full-graph execution.
3. **Dataset Scope**: Experiments were evaluated on SciFact; generalizability to open-domain MS MARCO or web search corpora remains an area for future scaling.

---

## Reproduction Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit and integration tests (71 CPU tests)
python -m pytest tests/ -v

# 3. Download benchmark data
python scripts/download_data.py --datasets scifact

# 4. Run baseline retrieval
python scripts/run_baseline.py --config configs/baseline.yaml

# 5. Run layer-wise representation analysis
python scripts/run_layer_analysis.py --config configs/layer_analysis.yaml

# 6. Run fixed-depth and adaptive sweeps
python scripts/run_fixed_depth.py --config configs/adaptive.yaml
python scripts/run_adaptive.py --config configs/adaptive.yaml

# 7. Generate comparison tables and figures
python scripts/compare_results.py
python scripts/generate_figures.py
python scripts/run_qualitative_analysis.py --threshold 0.85 --min_layer 10
```
