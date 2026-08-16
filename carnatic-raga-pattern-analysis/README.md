# Analysis of Raga Patterns in Carnatic Music

A computational musicology and machine learning workbench for feature extraction, statistical pattern analysis, exploratory data analysis (EDA), and multi-class classification of Carnatic ragas.

---

## Overview

Carnatic music is a classical Indian musical tradition defined by complex modal frameworks (*ragas*), continuous pitch ornamentation (*gamakas*), and tonic-relative pitch structures (*adhara sadja*). 

This project implements an end-to-end analytical and machine learning pipeline to:
1. Extract tonic-relative pitch representations, microtonal distributions (12-bin chromatic and 22-bin *śruti* resolution), melodic interval transitions, and structural phrase markers (*nyas* and *tani* segments).
2. Carry out exploratory data analysis with 23 diagnostic visualization artifacts across pitch stability, tonal distributions, and dimensionality reductions (PCA, t-SNE).
3. Systematically evaluate feature tiers and redundancy via multi-method importance metrics (Random Forest MDI, Mutual Information, OOB Permutation Importance, and Hierarchical Clustering).
4. Train and benchmark classical machine learning classifiers (Random Forest, Extra Trees, SVM-RBF, HistGradientBoosting, Gaussian Naive Bayes, KNN, Soft Voting Ensembles) evaluated under artist-grouped cross-validation to prevent acoustic and stylistic leakage.

---

## Repository Structure

```
carnatic-raga-pattern-analysis/
├── README.md                               # Project overview and technical documentation
├── README_AUDIO.md                         # Guide for raw audio & MFCC inference
├── carnatic-raga-identifier.py             # Open-set & closed-set raga identifier
├── train_raga_from_wavs.py                 # WAV-based audio feature extraction & model training
├── example_audio_usage.py                  # Audio inference utility script
│
└── indian_art_raga_workbench/              # Feature dataset workbench
    ├── README.md                           # Workbench quick-start guide
    ├── MODEL_REPORT.md                     # Model performance evaluation report
    ├── train_indian_art_raga.py            # Primary feature extraction & ML trainer
    ├── eda_plots.py                        # Generation script for 19+ EDA visual artifacts
    ├── feature_importance_analysis.py      # Multi-method feature importance & redundancy analysis
    ├── train_experiment.py                 # Microtonal binning comparison (12 vs 22 vs 24 bins)
    ├── add_pc22_from_pc24.py               # Proportional rebinning utility for śruti histograms
    │
    ├── outputs/                            # Processed datasets and metric summaries
    │   ├── indian_art_carnatic_features.csv     # Extracted 96-dimensional tabular feature dataset
    │   ├── indian_art_carnatic_metrics.json     # Full evaluation metrics & confusion matrices
    │   └── indian_art_carnatic_skipped_rows.csv # Data filtering & exclusion audit log
    │
    └── plots/                              # 23 publication-quality diagnostic visualizations
        ├── 01_raga_distribution.png
        ├── 02_duration_distribution.png
        ├── 03_tonic_distribution.png
        ├── 04_voiced_fraction.png
        ├── 05_pitch_class_histograms_top8.png
        ├── 05b_pitch_class_histograms_22bin_top8.png
        ├── 06_pitch_class_heatmap.png
        ├── 06b_pitch_class_heatmap_22bin.png
        ├── 07_pitch_statistics.png
        ├── 08_interval_features.png
        ├── 09_pitch_class_entropy.png
        ├── 10_segment_features.png
        ├── 11_artist_distribution.png
        ├── 12_feature_correlation.png
        ├── 13_pca_scatter.png
        ├── 14_tsne_scatter.png
        ├── 15_cents_range_by_raga.png
        ├── 16_pairplot_top5.png
        ├── 17_confusion_matrix.png
        ├── 18_model_comparison.png
        ├── 19_feature_importance_analysis.png
        ├── 19b_feature_redundancy_dendrogram.png
        └── 19c_feature_rank_comparison.png
```

---

## Feature Engineering Methodology

The tabular feature pipeline extracts 96 descriptive features per performance track:

### 1. Tonic-Normalized Pitch Distributions
Given fundamental frequency time series $f_0(t)$ and performer tonic $f_{\text{tonic}}$, pitch values in cents are computed as:
$$\text{cents}(t) = 1200 \cdot \log_2 \left( \frac{f_0(t)}{f_{\text{tonic}}} \right)$$

Statistical aggregations include mean, standard deviation, median, IQR, minimum, maximum, 25th/75th percentiles, and total dynamic cents range.

### 2. Multi-Resolution Pitch Class Histograms
- **12-bin Pitch Class (`pc12_00` to `pc12_11`)**: Chromatic scale pitch energy normalized across octave equivalents ($\text{cents} \pmod{1200}$).
- **22-bin Pitch Class (`pc22_00` to `pc22_21`)**: Microtonal *śruti* resolution (~54.55 cents per bin), capturing fine-grained raga intonation and gamaka centers.
- **24-bin Pitch Class (`pc24_00` to `pc24_23`)**: Quarter-tone resolution (50 cents per bin).
- **Pitch Class Entropy**: Shannon entropy of the 12-bin distribution measuring tonal concentration vs. melodic dispersion.

### 3. Melodic Movement & Interval Dynamics
- First-order interval differences $\Delta \text{cents} = \text{cents}(t+1) - \text{cents}(t)$.
- Statistical summaries of absolute interval magnitudes.
- Jump proportions: Fraction of melodic transitions exceeding $100\text{ cents}$ and $200\text{ cents}$.

### 4. Structural Phrase & Segment Markers
- **Nyas Segments**: Sustained resting notes indicating phrase termination characteristics.
- **Tani Segments**: Percussive/solo boundaries and melodic activity duration profiles.

---

## Feature Importance & Redundancy Analysis

To identify dominant vs. redundant descriptors, `feature_importance_analysis.py` evaluates feature rankings across three orthogonal methodologies:
1. **Random Forest Mean Decrease in Impurity (MDI)**
2. **Information-Theoretic Mutual Information Score** ($I(X; Y)$)
3. **Out-of-Bag (OOB) Permutation Importance**

### Composite Scoring & Tier Classification
A composite rank metric weights all three scores ($0.40 \times \text{RF} + 0.30 \times \text{MI} + 0.30 \times \text{Perm}$) alongside hierarchical clustering on Pearson distance ($d = 1 - |r|$) with a redundancy threshold of $|r| > 0.85$.

| Tier | Characteristics | Examples |
|---|---|---|
| **Dominant** | Top 15% composite score, essential discriminators | `pc12_07` (Panchamam / Pa), `pc12_04` (Antara Gandharam / G3), `cents_std` |
| **Useful** | Moderate-to-high independent explanatory capacity | `pc12_02`, `abs_interval_cents_mean`, `pc12_entropy`, `large_jump_fraction_100c` |
| **Marginal** | Low marginal discriminative impact | High-order pitch quantile duplicates, static length markers |
| **Redundant** | Strongly collinear ($|r| > 0.85$) with a higher-ranked feature | Collinear percentile/IQR pairings |

---

## Experimental Results

The benchmark models were trained on the processed 40-class Carnatic dataset under strict **Stratified Group K-Fold** partitioning by `artist` to prevent recording-environment and artist-specific performance style leakage.

### Model Performance Summary

| Model | Accuracy | Macro F1 | Weighted F1 | Top-3 Accuracy | Validation Protocol |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Classifier** | **0.8043** | **0.8076** | **0.8062** | **0.9130** | Grouped by Artist (5-Fold) |
| **Extra Trees Classifier** | 0.7826 | 0.7841 | 0.7819 | 0.8913 | Grouped by Artist (5-Fold) |
| **Soft Voting Ensemble** | 0.7826 | 0.7810 | 0.7798 | 0.8913 | Grouped by Artist (5-Fold) |
| **SVM (RBF Kernel)** | 0.7391 | 0.7412 | 0.7385 | 0.8696 | Grouped by Artist (5-Fold) |
| **HistGradientBoosting** | 0.7609 | 0.7634 | 0.7592 | 0.8913 | Grouped by Artist (5-Fold) |
| **Gaussian Naive Bayes** | 0.6522 | 0.6580 | 0.6514 | 0.8043 | Grouped by Artist (5-Fold) |
| **K-Nearest Neighbors** | 0.6304 | 0.6345 | 0.6288 | 0.7826 | Grouped by Artist (5-Fold) |

---

## Quick Start & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone git@github.com:yagnik-pl/personal-projects.git
cd personal-projects/carnatic-raga-pattern-analysis

# Install required dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy librosa
```

### 2. Feature Extraction & Model Training
```bash
cd indian_art_raga_workbench

# Train models using the preprocessed Carnatic feature dataset
python train_indian_art_raga.py
```

### 3. Generate Diagnostic Plots & Visual Artifacts
```bash
# Generate all 19+ EDA figures into plots/
python eda_plots.py

# Run multi-method feature importance & redundancy analysis
python feature_importance_analysis.py
```

### 4. Audio Inference
```bash
# Run open-set / closed-set raga identification from audio samples
python carnatic-raga-identifier.py test path/to/audio/folder
```

---

## Key Visualizations

All generated visualizations are stored under `indian_art_raga_workbench/plots/`:
- **`05_pitch_class_histograms_top8.png` & `05b_pitch_class_histograms_22bin_top8.png`**: Average 12-bin and 22-bin (*śruti*) tonal distribution signatures.
- **`06_pitch_class_heatmap.png` & `06b_pitch_class_heatmap_22bin.png`**: Global raga-to-pitch-class activation heatmaps across all classes.
- **`12_feature_correlation.png`**: Multi-feature correlation matrix highlighting collinear clusters.
- **`13_pca_scatter.png` & `14_tsne_scatter.png`**: 2D manifold embeddings of raga feature clusters.
- **`17_confusion_matrix.png`**: Classification confusion matrix across 40 Carnatic ragas.
- **`19_feature_importance_analysis.png`**: 6-panel composite importance, cumulative distribution, and tier classification.
- **`19b_feature_redundancy_dendrogram.png`**: Hierarchical feature redundancy clustering dendrogram.
- **`19c_feature_rank_comparison.png`**: Rank correlation bump chart comparing MDI vs. Mutual Info vs. Permutation importance.
