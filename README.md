# Personal Projects Repository

A collection of research, machine learning, computer vision, natural language processing, audio signal processing, and systems engineering projects.

---

## Projects Directory

| Project | Category | Description | Primary Stack | Status |
|---|---|---|---|:---:|
| [**`air-cursor`**](./air-cursor/) | Computer Vision & HCI | Real-time hand-gesture desktop control system using MediaPipe landmark extraction, EMA coordinate smoothing, gesture state estimation, and debounced OS actuation. | Python, OpenCV, MediaPipe, PyAutoGUI, pycaw | Completed |
| [**`adaptive-retriever`**](./adaptive-retriever/) | NLP & Dense Retrieval | Query-adaptive dynamic-depth early exit for dense Transformer retrievers. Investigates representation stability, accuracy–efficiency Pareto trade-offs, and layer-wise IR capacity. Inspired by EffiR (ACL 2026). | PyTorch, HuggingFace, NumPy, Matplotlib | Completed |
| [**`carnatic-raga-pattern-analysis`**](./carnatic-raga-pattern-analysis/) | Audio & Music Information Retrieval | Computational musicology workbench and multi-class machine learning classification for Carnatic ragas. Evaluates 12/22/24-bin pitch-class histograms, interval dynamics, and multi-method feature redundancy. | Python, scikit-learn, librosa, NumPy, Pandas, Matplotlib, Seaborn | Completed |
| [**`movie-ticket-booking-lld`**](./movie-ticket-booking-lld/) | Systems & Backend Engineering | Backend for a Movie Ticket Booking & Theatre Management System. Focuses on concurrent seat reservation, transactional booking, payment failure handling, and PostgreSQL schema design. | C++17, PostgreSQL, Crow, libpqxx, GoogleTest | Completed |

---

## Project Summaries

### 1. [Air Cursor](./air-cursor/)
- **Core Problem**: Touchless human-computer interaction requires low-latency, jitter-free hand tracking and robust gesture disambiguation without false click triggers or CPU overhead.
- **Key Implementation Highlights**:
  - **Vision Engine**: Real-time extraction of key hand landmarks (0, 4, 8, 12) via MediaPipe Hands with a dedicated TensorFlow shim to resolve environment dependency conflicts.
  - **Mathematical Processing**: $100\text{px}$ margin active-box bounding and Exponential Moving Average ($\alpha=0.4$) smoothing to eliminate high-frequency hand tremors.
  - **Gesture State Machine**: Disambiguates hover tracking, pinch-clicks ($<0.04$ threshold with $0.4\text{s}$ timestamp debouncing), and vertical volume modulation ($<0.03$ threshold).
  - **System Actuation**: Hardware-level cursor manipulation and Windows master audio endpoint integration via `pycaw`.
- **Details & Reproduction**: See [air-cursor/README.md](./air-cursor/README.md).

### 2. [AdaptiveRetriever](./adaptive-retriever/)
- **Core Problem**: Production dense bi-encoders evaluate all queries through all $L$ Transformer layers regardless of difficulty, creating unnecessary latency on the online critical path.
- **Key Findings**:
  - Uncovers the *Early-Layer Stability Illusion*: Layers 1–6 exhibit near-perfect consecutive cosine similarity ($S(l) > 0.993$) yet have $<1\%$ Recall@10.
  - Identifies a major contrastive phase transition at Layers 8–10 ($S(l)$ drops to $0.831$) where retrieval representations emerge.
  - Demonstrates that Layer 11 retains **$88.9\%$ of full-depth Recall@10** ($0.7512$ vs $0.8452$) and **$87.5\%$ of nDCG@10** ($0.6300$ vs $0.7200$) with a $8.3\%$ compute reduction.
- **Test Suite**: 71/71 passing unit and integration tests (100% offline CPU execution).
- **Details & Reproduction**: See [adaptive-retriever/README.md](./adaptive-retriever/README.md).

### 3. [Carnatic Raga Pattern Analysis](./carnatic-raga-pattern-analysis/)
- **Core Problem**: Multi-class categorization of modal structures (*ragas*) in Indian classical music characterized by non-tempered microtonal intonations (*śrutis*), continuous pitch ornamentation (*gamakas*), and strong performer-dependent acoustic bias.
- **Key Findings**:
  - Engineered a 96-dimensional descriptor suite encompassing tonic-normalized pitch distributions, 22-bin *śruti* histograms, interval jump fractions, and structural phrase markers (*nyas* / *tani*).
  - Executed multi-method feature importance & redundancy tiering (Random Forest MDI, Mutual Information, OOB Permutation, Hierarchical Collinearity Clustering).
  - Achieved **80.43% accuracy** and **91.30% Top-3 accuracy** across 40 Carnatic ragas under rigorous **Stratified Group K-Fold (grouped by artist)** validation.
- **Visual Artifacts**: 23 diagnostic plots covering tonal heatmaps, manifold projections (PCA/t-SNE), and feature redundancy dendrograms.
- **Details & Reproduction**: See [carnatic-raga-pattern-analysis/README.md](./carnatic-raga-pattern-analysis/README.md).

### 4. [Movie Ticket Booking LLD](./movie-ticket-booking-lld/)
- **Core Problem**: Designing a correct backend system that prevents concurrent double-booking, handles payment failures gracefully, and maintains consistent booking state under race conditions.
- **Key Implementation Highlights**:
  - **Concurrency**: PostgreSQL row-level locking (`SELECT ... FOR UPDATE`) inside ACID transactions guarantees exactly one seat reservation succeeds under 100 concurrent requests.
  - **Design Patterns**: Strategy (pricing), State (booking lifecycle), Factory (payment gateway), Observer (booking events), Repository (DB access).
  - **Payment Handling**: Mock payment service simulates success/failure/timeout; idempotency enforced via `UNIQUE(transaction_id)` constraint preventing duplicate payment processing.
  - **Seat Lock Expiry**: Dedicated background thread periodically releases expired temporary locks, freeing seats if payment is not completed within the time window.
  - **Test Coverage**: Unit tests (GoogleMock), integration tests against a real test DB, and a 100-thread concurrency test verifying exactly one booking succeeds.
- **Details & Reproduction**: See [movie-ticket-booking-lld/project_code/PROJECT.md](./movie-ticket-booking-lld/project_code/PROJECT.md).
