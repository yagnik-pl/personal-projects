# AdaptiveRetriever: Dynamic Layer Allocation & Early Exit for Dense Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 71 Passed](https://img.shields.io/badge/Tests-71%20Passed%20(CPU)-success.svg)](file:///tests)

AdaptiveRetriever is a compact, empirical research framework that investigates whether dense Transformer retrievers can dynamically allocate encoder depth on a per-query basis at runtime via early-exit mechanisms.

Inspired by the structural redundancy observations of **EffiR** ([Lei et al., ACL 2026](https://aclanthology.org/2026.acl-long.587/)), this project investigates whether all retrieval queries require the same Transformer depth, maps representation stability across encoder layers, and evaluates accuracy–efficiency Pareto trade-offs.

---

## Architecture & System Pipeline

```
                       [ Raw Query q ]
                              │
                              ▼
                  [ Step-by-Step LayerWiseEncoder ]
                   Layer 0: Token Embedding (h_0)
                   Layer 1: TransformerLayer_1 ──> Pool & Norm ──> e_1
                   Layer 2: TransformerLayer_2 ──> Pool & Norm ──> e_2 ──> Cosine Sim S(2) >= tau? ──Yes──> [ Exit e_q = e_2 ]
                      ...                                                                                 │
                   Layer L: TransformerLayer_L ──> Pool & Norm ──> e_L ───────────────────────────────────┘
                                                                             │
                                                                             ▼
                                                                 [ Dense Flat Index ]
                                                                 (Offline full-depth doc
                                                                  embeddings E_doc in R^{N x D})
                                                                             │
                                                                             ▼
                                                                   [ Exact Cosine Top-K ]
                                                                  Scores = e_q · E_doc^T
                                                                             │
                                                                             ▼
                                                                   [ IR Evaluation Suite ]
                                                                 Recall@1/5/10, MRR, nDCG@10
```

---

## 1. Motivation & Research Questions

In modern dense retrieval architectures (bi-encoders), document indexing is executed **offline** over millions of passages. In contrast, **query encoding lies directly on the interactive, online critical path**. Standard production retrievers route every query through all $L$ Transformer layers regardless of query difficulty.

We formulate three core research questions:
1. **RQ1 (Layer-wise Capability)**: Does dense retrieval capability emerge uniformly across Transformer layers, or does it undergo phase transitions?
2. **RQ2 (Stability as an Exit Signal)**: Does inter-layer cosine similarity accurately reflect retrieval readiness?
3. **RQ3 (Dynamic-Depth Trade-off)**: Can a dynamic early-exit policy reduce computational cost while preserving benchmark retrieval quality?

---

## 2. Related Work & Novelty Positioning

| Framework | Method | Target Entity | Core Mechanism |
|---|---|---|---|
| **EffiR** ([Lei et al., ACL 2026](https://aclanthology.org/2026.acl-long.587/)) | Offline Pruning | Model MLP Layers | Static layer-drop / structured MLP pruning |
| **DeeBERT / FastBERT** (Xin et al., 2020) | Early Exit | Classification | Output entropy / classification confidence |
| **AdaptiveRetriever** (Ours) | Dynamic Early Exit | Query Bi-Encoder | Inter-layer cosine stability in asymmetric retrieval |

### Positioning Trajectory
$$\text{EffiR Observation (MLP Redundancy)} \longrightarrow \text{Our Hypothesis (Query-Adaptive Depth)} \longrightarrow \text{Our Modification (Inter-Layer Cosine Stability)}$$

1. **EffiR's Observation**: Lei et al. (ACL 2026) demonstrated that MLP parameters in LLM-based dense retrievers contain substantial static redundancy and can be pruned offline without severe quality collapse.
2. **Our Research Hypothesis**: If Transformer layers exhibit parameter redundancy, individual queries should not require identical computational depth at inference time.
3. **Our Modification**: We develop a step-by-step layer-wise encoder that measures consecutive representation cosine similarity $S(l) = \langle \mathbf{e}_l, \mathbf{e}_{l-1} \rangle$ during query encoding and evaluate dynamic early-exit policies against full-depth and fixed-depth baselines.

---

## 3. Experimental Setup

- **Encoder Model**: `BAAI/bge-small-en-v1.5` ($L=12$ layers, hidden dimension $D=384$, 33.4M parameters, CLS pooling, L2 normalization).
- **Benchmark Dataset**: [SciFact](file:///data/scifact) (5,183 corpus documents, 300 test queries with expert relevance judgments).
- **Indexing & Retrieval**: Exact dense flat inner-product search ($\mathbf{e}_q \mathbf{E}_{\text{doc}}^T$) with zero ANN approximation error.
- **Hardware Profile**: Evaluated on consumer GPU (NVIDIA RTX 4060 Laptop GPU, 8GB VRAM) and reproducible on CPU with deterministic seed 42.

---

## 4. Key Experimental Results

### 4.1 Layer-Wise Retrieval Quality & Inter-Layer Cosine Stability

| Layer | Inter-layer Stability $S(l)$ | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Quality Retention |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 (Embed) | 1.0000 | 0.0000 | 0.0033 | 0.0067 | 0.0007 | 0.0019 | 0.3% |
| 1 | 0.9196 | 0.0000 | 0.0000 | 0.0007 | 0.0006 | 0.0004 | 0.1% |
| 2 | 0.9970 | 0.0000 | 0.0000 | 0.0007 | 0.0005 | 0.0004 | 0.1% |
| 3 | 0.9933 | 0.0000 | 0.0040 | 0.0073 | 0.0010 | 0.0023 | 0.3% |
| 4 | 0.9947 | 0.0000 | 0.0000 | 0.0007 | 0.0004 | 0.0004 | 0.1% |
| 5 | 0.9974 | 0.0000 | 0.0040 | 0.0073 | 0.0010 | 0.0023 | 0.3% |
| 6 | 0.9989 | 0.0000 | 0.0040 | 0.0073 | 0.0010 | 0.0023 | 0.3% |
| 7 | 0.9928 | 0.0000 | 0.0000 | 0.0007 | 0.0006 | 0.0004 | 0.1% |
| 8 | 0.9560 | 0.0000 | 0.0033 | 0.0067 | 0.0011 | 0.0024 | 0.3% |
| 9 | 0.8310 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| **10** | 0.8588 | 0.0818 | 0.1618 | 0.2169 | 0.1061 | 0.1298 | **18.0%** |
| **11** | 0.8415 | 0.5002 | 0.6974 | 0.7512 | 0.6002 | 0.6300 | **87.5%** |
| **12 (Full)** | 0.8977 | 0.5787 | 0.7653 | 0.8452 | 0.6845 | 0.7200 | **100.0%** |

---

### 4.2 Overall Comparison: Full-Depth vs. Fixed-Depth vs. Adaptive

| Method | Recall@10 | MRR | nDCG@10 | Avg Layers | Compute % | Median Latency | Quality Drop |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fixed Depth (L10) | 0.2169 | 0.1061 | 0.1298 | 10.00 | 83.3% | 12.15 ms | -81.97% |
| **Fixed Depth (L11)** | **0.7512** | **0.6002** | **0.6300** | **11.00** | **91.7%** | **13.10 ms** | **-12.50%** |
| **Full Baseline (L12)** | **0.8452** | **0.6845** | **0.7200** | **12.00** | **100.0%** | **11.90 ms** | **0.00%** |
| Gated Adaptive ($\tau=0.85$) | 0.3540 | 0.2780 | 0.2980 | 10.47 | 87.3% | 12.80 ms | -58.61% |

---

## 5. Key Research Insights

1. **The Early-Layer Stability Illusion**:
   - Layers 1–6 exhibit near-unity cosine similarity ($S(l) > 0.993$), yet exhibit $<1\%$ Recall@10. In standard Transformer bi-encoders, early layers capture generic syntactic structure where consecutive token representations drift minimally, creating a false signal of semantic convergence.
2. **The Contrastive Phase Transition**:
   - A dramatic geometric reorganization occurs in Layers 8–10 (stability drops to $0.831$), indicating where generic language models transition into specialized contrastive retrieval embeddings.
3. **Pre-Final Layer Efficiency**:
   - Layer 11 retains $88.9\%$ of full-depth Recall@10 ($0.7512$ vs $0.8452$) and $87.5\%$ of nDCG@10 ($0.6300$ vs $0.7200$) with a $8.3\%$ reduction in Transformer layer evaluations.
4. **Necessity of Gated Control**:
   - Naive stability thresholding starting at Layer 1 collapses immediately at Layer 2. Effective adaptive early exit requires restricting exit decisions to the active refinement zone ($\ge 10$ layers).

---

## 6. Generated Publication Figures

All figures are generated at 300 DPI in `results/figures/`:
- `layer_wise_quality.png` / `.pdf`: Dual-axis plot of retrieval metrics and cosine stability across layers $0 \dots 12$.
- `quality_vs_depth.png` / `.pdf`: Recall@10, MRR, and nDCG@10 progression.
- `computation_vs_quality.png` / `.pdf`: Accuracy–efficiency Pareto frontier.
- `exit_layer_distribution.png` / `.pdf`: Query exit layer histogram.
- `threshold_comparison.png` / `.pdf`: Trade-off comparison across strict, medium, and aggressive policies.

---

## 7. Limitations & Honest Assessment

1. **Pre-trained Objective Bias**: Standard bi-encoders (e.g., BGE, MiniLM) are fine-tuned using contrastive loss applied exclusively to the final layer ($L=12$). Intermediate layers are not regularized during pre-training to be retrieval-ready.
2. **Sequential Loop Latency Overhead**: In Python/PyTorch, evaluating layers sequentially in a loop introduces kernel launch overhead that mitigates the wall-clock latency savings of exiting 1 layer early on single queries.
3. **Domain & Corpus Scale**: Results were verified on SciFact; cross-domain validation on larger open-domain benchmarks (MS MARCO) represents an important next step.

---

## 8. Reproducibility & Quickstart

### Installation
```bash
git clone https://github.com/yagnik-pl/employer-voice-ai-agent.git
cd nlp-project
pip install -r requirements.txt
```

### Run Test Suite (100% Offline CPU, 71 Tests)
```bash
python -m pytest tests/ -v
```

### Execute End-to-End Pipeline
```bash
# 1. Ingest & verify SciFact benchmark
python scripts/download_data.py --datasets scifact

# 2. Run full-depth baseline (Recall@10 = 0.8452)
python scripts/run_baseline.py --config configs/baseline.yaml

# 3. Run layer-wise representation analysis
python scripts/run_layer_analysis.py --config configs/layer_analysis.yaml

# 4. Run fixed-depth and adaptive sweeps
python scripts/run_fixed_depth.py --config configs/adaptive.yaml
python scripts/run_adaptive.py --config configs/adaptive.yaml

# 5. Compile comparison tables and publication plots
python scripts/compare_results.py
python scripts/generate_figures.py
python scripts/run_qualitative_analysis.py --threshold 0.85 --min_layer 10
```

---

## 9. References

1. **EffiR**: Yibin Lei, Shwai He, Ang Li, Andrew Yates. *Making Large Language Models Efficient Dense Retrievers*. Proceedings of ACL 2026. [ACL 2026.acl-long.587](https://aclanthology.org/2026.acl-long.587/).
2. **DeeBERT**: Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, Jimmy Lin. *DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference*. ACL 2020.
3. **BGE**: Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff. *C-Pack: Packaged Resources to Advance General Chinese Embedding*. arXiv:2309.07597, 2023.
4. **BEIR**: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS 2021 Datasets and Benchmarks.
5. **SciFact**: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi. *Fact or Fiction: Verifying Scientific Claims with Evidence*. EMNLP 2020.
