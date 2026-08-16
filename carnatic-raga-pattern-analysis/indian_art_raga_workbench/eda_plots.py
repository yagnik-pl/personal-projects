import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

WORKBENCH = Path(__file__).resolve().parent
CSV_PATH = WORKBENCH / "outputs" / "indian_art_carnatic_features.csv"
METRICS_PATH = WORKBENCH / "outputs" / "indian_art_carnatic_metrics.json"
PLOTS_DIR = WORKBENCH / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})
PALETTE = "viridis"


def save(fig, name):
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


def plot_raga_distribution(df):
    counts = df["raga"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.28)))
    colors = cm.viridis(np.linspace(0.2, 0.9, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Tracks")
    ax.set_title("Carnatic Raga Distribution")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "01_raga_distribution.png")


def plot_duration(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["duration_s"] / 60, bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Track Duration Distribution")
    axes[0].axvline(df["duration_s"].median() / 60, color="red", ls="--", label=f'Median {df["duration_s"].median()/60:.1f}m')
    axes[0].legend()
    top10 = df["raga"].value_counts().head(10).index
    sub = df[df["raga"].isin(top10)].copy()
    sub["dur_min"] = sub["duration_s"] / 60
    order = sub.groupby("raga")["dur_min"].median().sort_values().index
    sns.boxplot(data=sub, y="raga", x="dur_min", order=order, ax=axes[1], palette="coolwarm")
    axes[1].set_xlabel("Duration (minutes)")
    axes[1].set_title("Duration by Raga (Top 10)")
    fig.tight_layout()
    save(fig, "02_duration_distribution.png")


def plot_tonic(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["tonic_hz"], bins=40, color="#FF9800", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Tonic (Hz)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Tonic Frequency Distribution")
    top10 = df["raga"].value_counts().head(10).index
    sub = df[df["raga"].isin(top10)]
    order = sub.groupby("raga")["tonic_hz"].median().sort_values().index
    sns.boxplot(data=sub, y="raga", x="tonic_hz", order=order, ax=axes[1], palette="magma")
    axes[1].set_xlabel("Tonic (Hz)")
    axes[1].set_title("Tonic by Raga (Top 10)")
    fig.tight_layout()
    save(fig, "03_tonic_distribution.png")


def plot_voiced_fraction(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["voiced_fraction"], bins=40, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Voiced Fraction")
    ax.set_ylabel("Count")
    ax.set_title("Voiced Fraction Distribution")
    ax.axvline(df["voiced_fraction"].median(), color="red", ls="--",
               label=f'Median={df["voiced_fraction"].median():.3f}')
    ax.legend()
    fig.tight_layout()
    save(fig, "04_voiced_fraction.png")


def plot_pitch_class_histograms(df):
    pc12_cols = [f"pc12_{i:02d}" for i in range(12)]
    note_names = ["Sa", "r", "R", "g", "G", "M", "m", "P", "d", "D", "n", "N"]
    top8 = df["raga"].value_counts().head(8).index
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, raga in zip(axes.flat, top8):
        vals = df.loc[df["raga"] == raga, pc12_cols].mean().values
        colors = cm.viridis(vals / vals.max()) if vals.max() > 0 else ["gray"] * 12
        ax.bar(note_names, vals, color=colors, edgecolor="white")
        ax.set_title(raga, fontsize=10)
        ax.set_ylim(0, max(0.3, vals.max() * 1.15))
    fig.suptitle("Average 12-bin Pitch-Class Histogram per Raga (Top 8)", fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "05_pitch_class_histograms_top8.png")


def plot_pitch_class_histograms_22(df):
    pc22_cols = [f"pc22_{i:02d}" for i in range(22)]
    if pc22_cols[0] not in df.columns:
        return
    bin_labels = [str(i) for i in range(22)]
    top8 = df["raga"].value_counts().head(8).index
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, raga in zip(axes.flat, top8):
        vals = df.loc[df["raga"] == raga, pc22_cols].mean().values
        colors = cm.magma(vals / vals.max()) if vals.max() > 0 else ["gray"] * 22
        ax.bar(bin_labels, vals, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_title(raga, fontsize=10)
        ax.set_ylim(0, max(0.2, vals.max() * 1.15))
        ax.tick_params(axis="x", labelsize=6)
    fig.suptitle("Average 22-bin Pitch-Class Histogram per Raga (Top 8) — Śruti Resolution",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "05b_pitch_class_histograms_22bin_top8.png")


def plot_pc_heatmap(df):
    pc12_cols = [f"pc12_{i:02d}" for i in range(12)]
    note_names = ["Sa", "r", "R", "g", "G", "M", "m", "P", "d", "D", "n", "N"]
    agg = df.groupby("raga")[pc12_cols].mean()
    agg.columns = note_names
    agg = agg.loc[agg.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, max(6, len(agg) * 0.3)))
    sns.heatmap(agg, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="white")
    ax.set_title("Pitch-Class Profile Heatmap — 12-bin (All Ragas)")
    ax.set_xlabel("Svara (Pitch Class)")
    fig.tight_layout()
    save(fig, "06_pitch_class_heatmap.png")


def plot_pc_heatmap_22(df):
    pc22_cols = [f"pc22_{i:02d}" for i in range(22)]
    if pc22_cols[0] not in df.columns:
        return
    agg = df.groupby("raga")[pc22_cols].mean()
    agg.columns = [str(i) for i in range(22)]
    agg = agg.loc[agg.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(14, max(6, len(agg) * 0.3)))
    sns.heatmap(agg, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="white")
    ax.set_title("Pitch-Class Profile Heatmap — 22-bin / Śruti (All Ragas)")
    ax.set_xlabel("Śruti Bin")
    fig.tight_layout()
    save(fig, "06b_pitch_class_heatmap_22bin.png")


def plot_pitch_stats(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    feats = [("pitch_hz_mean", "Mean Pitch (Hz)"), ("pitch_hz_std", "Pitch Std (Hz)"),
             ("cents_mean", "Mean Cents"), ("cents_std", "Cents Std")]
    for ax, (col, label) in zip(axes.flat, feats):
        ax.hist(df[col], bins=40, color="#9C27B0", edgecolor="white", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
    fig.tight_layout()
    save(fig, "07_pitch_statistics.png")


def plot_interval_features(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(df["abs_interval_cents_mean"], bins=40, color="#E91E63", edgecolor="white", alpha=0.85)
    axes[0].set_title("Mean Absolute Interval (cents)")
    axes[0].set_xlabel("Cents")
    axes[1].hist(df["large_jump_fraction_100c"], bins=40, color="#00BCD4", edgecolor="white", alpha=0.85)
    axes[1].set_title("Large Jump Fraction (>100c)")
    axes[1].set_xlabel("Fraction")
    axes[2].hist(df["large_jump_fraction_200c"], bins=40, color="#795548", edgecolor="white", alpha=0.85)
    axes[2].set_title("Large Jump Fraction (>200c)")
    axes[2].set_xlabel("Fraction")
    fig.suptitle("Interval Movement Features", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "08_interval_features.png")


def plot_entropy(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["pc12_entropy"], bins=40, color="#3F51B5", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("PC12 Entropy")
    axes[0].set_title("Pitch-Class Entropy Distribution")
    top10 = df["raga"].value_counts().head(10).index
    sub = df[df["raga"].isin(top10)]
    order = sub.groupby("raga")["pc12_entropy"].median().sort_values().index
    sns.boxplot(data=sub, y="raga", x="pc12_entropy", order=order, ax=axes[1], palette="Spectral")
    axes[1].set_xlabel("PC12 Entropy")
    axes[1].set_title("Entropy by Raga (Top 10)")
    fig.tight_layout()
    save(fig, "09_pitch_class_entropy.png")


def plot_segments(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    nyas_counts = df["nyas_present"].value_counts()
    axes[0, 0].pie(nyas_counts.values, labels=["Present", "Absent"] if 1 in nyas_counts.index else ["Absent"],
                   autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
    axes[0, 0].set_title("Nyas Segment Presence")
    tani_counts = df["tani_present"].value_counts()
    axes[0, 1].pie(tani_counts.values, labels=["Present", "Absent"] if 1 in tani_counts.index else ["Absent"],
                   autopct="%1.1f%%", colors=["#2196F3", "#FF9800"])
    axes[0, 1].set_title("Tani Segment Presence")
    nyas_sub = df[df["nyas_present"] == 1]
    if len(nyas_sub) > 0:
        axes[1, 0].hist(nyas_sub["nyas_segment_count"], bins=30, color="#4CAF50", edgecolor="white")
    axes[1, 0].set_title("Nyas Segment Count (where present)")
    axes[1, 0].set_xlabel("Count")
    tani_sub = df[df["tani_present"] == 1]
    if len(tani_sub) > 0:
        axes[1, 1].hist(tani_sub["tani_segment_count"], bins=15, color="#2196F3", edgecolor="white")
    axes[1, 1].set_title("Tani Segment Count (where present)")
    axes[1, 1].set_xlabel("Count")
    fig.tight_layout()
    save(fig, "10_segment_features.png")


def plot_artist_distribution(df):
    counts = df["artist"].value_counts().head(20).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = cm.plasma(np.linspace(0.2, 0.9, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Tracks")
    ax.set_title("Top 20 Artists by Track Count")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "11_artist_distribution.png")


def plot_correlation(df):
    numeric_feats = ["duration_s", "voiced_fraction", "tonic_hz",
                     "pitch_hz_mean", "pitch_hz_std", "cents_mean", "cents_std",
                     "abs_interval_cents_mean", "large_jump_fraction_100c",
                     "pc12_entropy", "nyas_segment_count", "tani_segment_count"]
    existing = [c for c in numeric_feats if c in df.columns]
    corr = df[existing].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, ax=ax,
                annot=True, fmt=".2f", linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    save(fig, "12_feature_correlation.png")


def plot_pca(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    meta = {"system", "source_path", "feature_base", "mbid", "raga_id", "raga",
            "artist", "album", "composition", "track_name", "feature_version",
            "pitch_source", "tonic_source"}
    feat_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].fillna(0).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    top8 = df["raga"].value_counts().head(8).index.tolist()
    labels = df["raga"].apply(lambda r: r if r in top8 else "Other").values
    fig, ax = plt.subplots(figsize=(12, 8))
    unique = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, len(unique)))
    for label, color in zip(unique, colors):
        mask = labels == label
        alpha = 0.3 if label == "Other" else 0.75
        size = 10 if label == "Other" else 30
        ax.scatter(X2[mask, 0], X2[mask, 1], c=[color], label=label, alpha=alpha, s=size, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Projection of Carnatic Raga Features")
    ax.legend(fontsize=8, markerscale=1.5, loc="best")
    fig.tight_layout()
    save(fig, "13_pca_scatter.png")


def plot_tsne(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    meta = {"system", "source_path", "feature_base", "mbid", "raga_id", "raga",
            "artist", "album", "composition", "track_name", "feature_version",
            "pitch_source", "tonic_source"}
    feat_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].fillna(0).values
    X = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X2 = tsne.fit_transform(X)
    top8 = df["raga"].value_counts().head(8).index.tolist()
    labels = df["raga"].apply(lambda r: r if r in top8 else "Other").values
    fig, ax = plt.subplots(figsize=(12, 8))
    unique = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, len(unique)))
    for label, color in zip(unique, colors):
        mask = labels == label
        alpha = 0.3 if label == "Other" else 0.75
        size = 10 if label == "Other" else 30
        ax.scatter(X2[mask, 0], X2[mask, 1], c=[color], label=label, alpha=alpha, s=size, edgecolors="none")
    ax.set_title("t-SNE Projection of Carnatic Raga Features")
    ax.legend(fontsize=8, markerscale=1.5, loc="best")
    fig.tight_layout()
    save(fig, "14_tsne_scatter.png")


def plot_cents_range(df):
    if "cents_range" not in df.columns:
        return
    top10 = df["raga"].value_counts().head(10).index
    sub = df[df["raga"].isin(top10)]
    order = sub.groupby("raga")["cents_range"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=sub, y="raga", x="cents_range", order=order, ax=ax, palette="Set2")
    ax.set_xlabel("Pitch Range (cents)")
    ax.set_title("Pitch Range by Raga (Top 10)")
    fig.tight_layout()
    save(fig, "15_cents_range_by_raga.png")


def plot_pairplot(df):
    top5 = df["raga"].value_counts().head(5).index
    sub = df[df["raga"].isin(top5)][["raga", "pc12_entropy", "voiced_fraction",
                                      "pitch_hz_std", "cents_std"]].copy()
    g = sns.pairplot(sub, hue="raga", palette="Set1", diag_kind="kde",
                     plot_kws={"alpha": 0.6, "s": 20})
    g.figure.suptitle("Pair Plot of Key Features (Top 5 Ragas)", y=1.02)
    save(g.figure, "16_pairplot_top5.png")


def plot_confusion_matrix():
    if not METRICS_PATH.exists():
        return
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    best = metrics.get("best_model", "")
    if best and best in metrics.get("models", {}):
        cm_data = metrics["models"][best].get("confusion_matrix")
        if cm_data is None:
            return
        cm_arr = np.array(cm_data)
        labels = metrics.get("models", {}).get(best, {}).get("classification_report", {})
        class_names = [k for k in labels if k not in ("accuracy", "macro avg", "weighted avg")]
        if not class_names:
            class_names = [str(i) for i in range(cm_arr.shape[0])]
        fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.45),
                                         max(8, len(class_names) * 0.4)))
        sns.heatmap(cm_arr, xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", ax=ax, fmt="d", annot=True if cm_arr.shape[0] <= 20 else False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {best}")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        fig.tight_layout()
        save(fig, "17_confusion_matrix.png")


def plot_model_comparison():
    if not METRICS_PATH.exists():
        return
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    models_data = metrics.get("models", {})
    if not models_data:
        return
    names, accs, f1s = [], [], []
    for name, m in models_data.items():
        names.append(name)
        accs.append(m.get("accuracy", 0))
        f1s.append(m.get("macro_f1", 0))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.15, accs, 0.3, label="Accuracy", color="#2196F3")
    ax.bar(x + 0.15, f1s, 0.3, label="Macro F1", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i in range(len(names)):
        ax.text(x[i] - 0.15, accs[i] + 0.015, f"{accs[i]:.3f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.15, f1s[i] + 0.015, f"{f1s[i]:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "18_model_comparison.png")


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, {df['raga'].nunique()} ragas, {len(df.columns)} columns\n")

    print("Generating EDA plots → plots/")
    plot_raga_distribution(df)
    plot_duration(df)
    plot_tonic(df)
    plot_voiced_fraction(df)
    plot_pitch_class_histograms(df)
    plot_pitch_class_histograms_22(df)
    plot_pc_heatmap(df)
    plot_pc_heatmap_22(df)
    plot_pitch_stats(df)
    plot_interval_features(df)
    plot_entropy(df)
    plot_segments(df)
    plot_artist_distribution(df)
    plot_correlation(df)
    plot_pca(df)
    plot_tsne(df)
    plot_cents_range(df)
    plot_pairplot(df)
    plot_confusion_matrix()
    plot_model_comparison()

    print(f"\nDone! {len(list(PLOTS_DIR.glob('*.png')))} plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
