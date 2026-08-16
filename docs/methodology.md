# AdaptiveRetriever: Methodology & Mathematical Formulation

## 1. Mathematical Formulation of Dense Retrieval

A dense bi-encoder maps an input text string $x \in \mathcal{V}^*$ to a dense vector representation $\mathbf{e} \in \mathbb{R}^D$:

$$\mathbf{h}_0 = \text{Embed}(x) \in \mathbb{R}^{T \times D}$$

$$\mathbf{h}_l = \text{TransformerLayer}_l(\mathbf{h}_{l-1}) \in \mathbb{R}^{T \times D}, \quad l \in \{1, \dots, L\}$$

$$\mathbf{e}_l = \text{Normalize}\left(\text{Pool}(\mathbf{h}_l)\right) = \frac{\text{Pool}(\mathbf{h}_l)}{\|\text{Pool}(\mathbf{h}_l)\|_2}$$

For CLS pooling: $\text{Pool}(\mathbf{h}_l) = \mathbf{h}_l[0, :]$.

### Asymmetric Retrieval Setup
In production information retrieval systems:
1. **Document Corpus Encoding (Offline)**: The corpus $\mathcal{D} = \{d_1, \dots, d_N\}$ is encoded once at full depth $L$ and stored in an exact flat inner-product matrix $\mathbf{E}_{\text{doc}} \in \mathbb{R}^{N \times D}$.
2. **Query Encoding (Online Critical Path)**: The query $q$ is dynamically evaluated at depth $l_q \le L$ to produce $\mathbf{e}_{l_q} \in \mathbb{R}^D$.
3. **Similarity Scoring & Top-$K$ Ranking**:

$$\mathbf{s} = \mathbf{e}_{l_q} \mathbf{E}_{\text{doc}}^T \in \mathbb{R}^N, \quad \text{Top-}K = \text{argtopk}_{i \in \{1 \dots N\}}(\mathbf{s}_i)$$

---

## 2. Layer-Wise Cosine Stability Formulation

Between any two consecutive Transformer layers $l-1$ and $l$, the representation cosine stability $S(l)$ is defined as the inner product of the normalized pooled vectors:

$$S(l) = \langle \mathbf{e}_{l-1}, \mathbf{e}_l \rangle = \mathbf{e}_{l-1} \cdot \mathbf{e}_l \in [-1, 1]$$

The normalized Euclidean drift $\Delta_{\text{norm}}(l)$ satisfies the exact mathematical identity:

$$\Delta_{\text{norm}}(l) = \|\mathbf{e}_l - \mathbf{e}_{l-1}\|_2 = \sqrt{\|\mathbf{e}_l\|_2^2 + \|\mathbf{e}_{l-1}\|_2^2 - 2 \langle \mathbf{e}_l, \mathbf{e}_{l-1} \rangle} = \sqrt{2(1 - S(l))}$$

---

## 3. Early-Exit Controller Mechanisms

### 3.1 Heuristic Stability Policy
The heuristic early-exit controller evaluates $S(l)$ sequentially at each layer $l \ge \text{min\_layer}$:

$$l^* = \min \{ l \in [\text{min\_layer}, L] \mid S(l) \ge \tau \}$$

If no intermediate layer satisfies $S(l) \ge \tau$, the query defaults to full depth $L$.

### 3.2 Gated Layer-Range Control
Because early layers (layers 1–6) exhibit high cosine stability due to shared syntax rather than semantic retrieval convergence, the gated policy enforces:

$$\text{min\_layer} \in \{9, 10\}$$

This restricts early termination to the representation refinement zone (layers 10–12), where semantic discrimination is established.

### 3.3 Learned Controller (2-Layer MLP)
The lightweight learned controller takes the intermediate embedding $\mathbf{e}_l$ and scalar features (stability $S(l)$, normalized depth $l/L$):

$$\mathbf{z}_l = [\mathbf{e}_l \,\|\, S(l) \,\|\, l/L] \in \mathbb{R}^{D + 2}$$

$$p(\text{exit} \mid \mathbf{z}_l) = \sigma(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{z}_l + \mathbf{b}_1) + b_2)$$

$$\text{Decision}: \quad \text{Exit if } p(\text{exit} \mid \mathbf{z}_l) > 0.5$$

---

## 4. Evaluation Metrics

### Recall@K
$$\text{Recall@}K(q) = \frac{|\text{Top-}K(q) \cap \mathcal{R}_q|}{|\mathcal{R}_q|}$$

### Mean Reciprocal Rank (MRR)
$$\text{MRR}(q) = \frac{1}{\min \{ \text{rank}(d) \mid d \in \text{Top-}K(q) \cap \mathcal{R}_q \}}$$

### Normalized Discounted Cumulative Gain (nDCG@K)
$$\text{DCG@}K(q) = \sum_{i=1}^K \frac{2^{\text{rel}(i)} - 1}{\log_2(i + 1)}, \quad \text{nDCG@}K(q) = \frac{\text{DCG@}K(q)}{\text{IDCG@}K(q)}$$
