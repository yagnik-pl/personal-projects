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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

WORKBENCH = Path(__file__).resolve().parent
CSV_PATH = WORKBENCH / "outputs" / "indian_art_carnatic_features.csv"
METRICS_PATH = WORKBENCH / "outputs" / "indian_art_carnatic_metrics.json"
PLOTS_DIR = WORKBENCH / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
})


def save(fig, name):
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


def shorten_name(name, max_len=22):
    if len(name) <= max_len:
        return name
    return name[:max_len - 2] + ".."


def load_data():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    feat_cols = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        feat_cols = metrics.get("feature_columns", None)

    if feat_cols is None:
        meta = {"system", "source_path", "feature_base", "mbid", "raga_id", "raga",
                "artist", "album", "composition", "track_name", "feature_version",
                "pitch_source", "tonic_source"}
        feat_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feat_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df["raga"])

    groups = df["artist"].values if "artist" in df.columns else None

    print(f"  {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} ragas\n")
    return df, X, y, feat_cols, le, groups


def compute_rf_importance(X, y, feat_cols):
    print("  Computing Random Forest (MDI) importance ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    return pd.Series(importances, index=feat_cols).sort_values(ascending=False)


def compute_mutual_info(X, y, feat_cols):
    print("  Computing Mutual Information scores ...")
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    return pd.Series(mi, index=feat_cols).sort_values(ascending=False)


def compute_permutation_importance(X, y, feat_cols):
    print("  Computing Permutation Importance (on OOB) ...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                n_jobs=-1, oob_score=True)
    rf.fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=10,
                                    random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=feat_cols).sort_values(ascending=False)


def compute_redundancy_clusters(X, feat_cols, threshold=0.85):
    print(f"  Computing redundancy clusters (|r| > {threshold}) ...")
    corr = np.abs(np.corrcoef(X.T))
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0)
    np.fill_diagonal(corr, 0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=1 - threshold, criterion="distance")
    cluster_df = pd.DataFrame({"feature": feat_cols, "cluster": clusters})
    return cluster_df, Z, dist


def plot_panel_1_rf_importance(ax, rf_imp, top_n=30):
    top = rf_imp.head(top_n).iloc[::-1]
    colors = cm.viridis(np.linspace(0.25, 0.95, len(top)))
    ax.barh(range(len(top)), top.values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([shorten_name(n) for n in top.index], fontsize=7)
    ax.set_xlabel("MDI Importance")
    ax.set_title("① Random Forest Feature Importance (Top 30)", fontweight="bold")
    for i, v in enumerate(top.values):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=5.5, color="#333")


def plot_panel_2_mutual_info(ax, mi_scores, top_n=30):
    top = mi_scores.head(top_n).iloc[::-1]
    colors = cm.magma(np.linspace(0.25, 0.9, len(top)))
    ax.barh(range(len(top)), top.values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([shorten_name(n) for n in top.index], fontsize=7)
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("② Mutual Information Scores (Top 30)", fontweight="bold")
    for i, v in enumerate(top.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=5.5, color="#333")


def plot_panel_3_permutation(ax, perm_imp, top_n=30):
    top = perm_imp.head(top_n).iloc[::-1]
    colors = cm.cool(np.linspace(0.2, 0.85, len(top)))
    ax.barh(range(len(top)), top.values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([shorten_name(n) for n in top.index], fontsize=7)
    ax.set_xlabel("Mean Accuracy Decrease")
    ax.set_title("③ Permutation Importance (Top 30)", fontweight="bold")
    for i, v in enumerate(top.values):
        ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=5.5, color="#333")


def plot_panel_4_cumulative(ax, rf_imp):
    sorted_imp = rf_imp.sort_values(ascending=False)
    cumulative = sorted_imp.cumsum() / sorted_imp.sum()
    n_features = len(cumulative)
    x = np.arange(1, n_features + 1)

    ax.fill_between(x, cumulative.values, alpha=0.3, color="#2196F3")
    ax.plot(x, cumulative.values, color="#1565C0", linewidth=2)

    for pct, color, ls in [(0.80, "#4CAF50", "--"), (0.90, "#FF9800", "--"), (0.95, "#F44336", ":")]:
        idx = np.searchsorted(cumulative.values, pct)
        if idx < n_features:
            ax.axhline(pct, color=color, ls=ls, alpha=0.6, linewidth=1)
            ax.axvline(idx + 1, color=color, ls=ls, alpha=0.6, linewidth=1)
            ax.annotate(f"{int(pct*100)}% → {idx+1} features",
                        xy=(idx + 1, pct), fontsize=7, color=color,
                        xytext=(idx + 5, pct - 0.04),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xlabel("Number of Features (ranked)")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title("④ Cumulative Feature Importance", fontweight="bold")
    ax.set_xlim(1, n_features)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.15)


def plot_panel_5_redundancy_heatmap(ax, X, feat_cols, cluster_df, top_n=35):
    corr_full = np.corrcoef(X.T)
    mean_abs_corr = np.mean(np.abs(corr_full), axis=1)
    top_idx = np.argsort(mean_abs_corr)[-top_n:]
    sub_corr = corr_full[np.ix_(top_idx, top_idx)]
    sub_names = [shorten_name(feat_cols[i], 18) for i in top_idx]

    mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
    sns.heatmap(sub_corr, mask=mask, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=sub_names, yticklabels=sub_names,
                vmin=-1, vmax=1, linewidths=0.2, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})
    ax.tick_params(axis="both", labelsize=5.5)
    ax.set_title("⑤ Feature Correlation (Most Connected 35)", fontweight="bold")


def plot_panel_6_tier_classification(ax, rf_imp, mi_scores, perm_imp, cluster_df):
    all_feats = rf_imp.index.tolist()

    rf_norm = (rf_imp - rf_imp.min()) / (rf_imp.max() - rf_imp.min() + 1e-12)
    mi_norm = (mi_scores.reindex(all_feats).fillna(0))
    mi_norm = (mi_norm - mi_norm.min()) / (mi_norm.max() - mi_norm.min() + 1e-12)
    perm_norm = (perm_imp.reindex(all_feats).fillna(0))
    perm_norm = (perm_norm - perm_norm.min()) / (perm_norm.max() - perm_norm.min() + 1e-12)

    composite = (0.40 * rf_norm + 0.30 * mi_norm + 0.30 * perm_norm)
    composite = composite.sort_values(ascending=False)

    cluster_map = cluster_df.set_index("feature")["cluster"]
    seen_clusters = {}
    redundant = set()
    for feat in composite.index:
        cl = cluster_map.get(feat, -1)
        if cl in seen_clusters:
            redundant.add(feat)
        else:
            seen_clusters[cl] = feat

    tiers = {}
    for i, feat in enumerate(composite.index):
        score = composite[feat]
        if feat in redundant:
            tiers[feat] = "Redundant"
        elif score >= composite.quantile(0.85):
            tiers[feat] = "Dominant"
        elif score >= composite.quantile(0.50):
            tiers[feat] = "Useful"
        else:
            tiers[feat] = "Marginal"

    tier_series = pd.Series(tiers)
    tier_colors = {
        "Dominant": "#2E7D32",
        "Useful": "#1565C0",
        "Marginal": "#F57F17",
        "Redundant": "#C62828",
    }

    top40 = composite.head(40).iloc[::-1]
    bar_colors = [tier_colors[tiers[f]] for f in top40.index]
    ax.barh(range(len(top40)), top40.values, color=bar_colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(top40)))
    ax.set_yticklabels([shorten_name(n) for n in top40.index], fontsize=6.5)
    ax.set_xlabel("Composite Score (RF 40% + MI 30% + Perm 30%)")
    ax.set_title("⑥ Feature Tier Classification (Top 40)", fontweight="bold")

    patches = [mpatches.Patch(color=c, label=f"{t} ({(tier_series == t).sum()})")
               for t, c in tier_colors.items()]
    ax.legend(handles=patches, fontsize=7, loc="lower right", framealpha=0.9)

    summary = (f"Dominant: {(tier_series == 'Dominant').sum()}  |  "
               f"Useful: {(tier_series == 'Useful').sum()}  |  "
               f"Marginal: {(tier_series == 'Marginal').sum()}  |  "
               f"Redundant: {(tier_series == 'Redundant').sum()}")
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    return composite, tiers, tier_series


def plot_redundancy_dendrogram(Z, feat_cols, threshold=0.85):
    fig, ax = plt.subplots(figsize=(18, max(8, len(feat_cols) * 0.14)))
    short_labels = [shorten_name(n, 25) for n in feat_cols]
    dendrogram(Z, labels=short_labels, orientation="left", ax=ax,
               leaf_font_size=6, color_threshold=1 - threshold,
               above_threshold_color="#999")
    ax.axvline(1 - threshold, color="#C62828", ls="--", linewidth=1.5, alpha=0.7)
    ax.text(1 - threshold + 0.005, 0.98, f"Redundancy cutoff\n(|r| > {threshold})",
            transform=ax.get_xaxis_transform(), fontsize=8, color="#C62828", va="top")
    ax.set_xlabel("1 − |Correlation|")
    ax.set_title("Feature Redundancy Dendrogram — Hierarchical Clustering on |Correlation|",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "19b_feature_redundancy_dendrogram.png")


def plot_rank_comparison(rf_imp, mi_scores, perm_imp, top_n=25):
    feats = rf_imp.head(top_n).index.tolist()
    for s in [mi_scores, perm_imp]:
        for f in s.head(top_n).index:
            if f not in feats:
                feats.append(f)
    feats = feats[:40]

    rf_ranks = rf_imp.rank(ascending=False)
    mi_ranks = mi_scores.rank(ascending=False)
    perm_ranks = perm_imp.rank(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, len(feats) * 0.22)))
    methods = ["RF (MDI)", "Mutual Info", "Permutation"]
    x_pos = [0, 1, 2]

    colors = cm.tab20(np.linspace(0, 1, len(feats)))
    for i, feat in enumerate(feats):
        ranks = [rf_ranks.get(feat, len(rf_imp)),
                 mi_ranks.get(feat, len(mi_scores)),
                 perm_ranks.get(feat, len(perm_imp))]
        ax.plot(x_pos, ranks, marker="o", markersize=4, linewidth=1.2,
                color=colors[i], alpha=0.7)
        ax.text(2.05, ranks[2], shorten_name(feat, 20), fontsize=5.5,
                va="center", color=colors[i])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylabel("Rank (lower = more important)")
    ax.set_title("Feature Rank Comparison Across Methods", fontweight="bold", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.15)
    ax.set_xlim(-0.2, 3.2)
    fig.tight_layout()
    save(fig, "19c_feature_rank_comparison.png")


def main():
    df, X, y, feat_cols, le, groups = load_data()

    rf_imp = compute_rf_importance(X, y, feat_cols)
    mi_scores = compute_mutual_info(X, y, feat_cols)
    perm_imp = compute_permutation_importance(X, y, feat_cols)
    cluster_df, Z, dist = compute_redundancy_clusters(X, feat_cols)

    print("\n  Generating composite feature analysis figure ...")

    fig = plt.figure(figsize=(26, 32))
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    plot_panel_1_rf_importance(ax1, rf_imp)
    plot_panel_2_mutual_info(ax2, mi_scores)
    plot_panel_3_permutation(ax3, perm_imp)
    plot_panel_4_cumulative(ax4, rf_imp)
    plot_panel_5_redundancy_heatmap(ax5, X, feat_cols, cluster_df)
    composite, tiers, tier_series = plot_panel_6_tier_classification(
        ax6, rf_imp, mi_scores, perm_imp, cluster_df)

    fig.suptitle("Carnatic Raga Feature Analysis — Dominant, Redundant & Best Features",
                 fontsize=18, fontweight="bold", y=0.995)
    save(fig, "19_feature_importance_analysis.png")

    print("  Generating supplementary plots ...")
    plot_redundancy_dendrogram(Z, feat_cols)
    plot_rank_comparison(rf_imp, mi_scores, perm_imp)

    print("\n" + "=" * 60)
    print("FEATURE TIER SUMMARY")
    print("=" * 60)
    for tier in ["Dominant", "Useful", "Marginal", "Redundant"]:
        feats = tier_series[tier_series == tier].index.tolist()
        feats = sorted(feats, key=lambda f: composite.get(f, 0), reverse=True)
        print(f"\n  {tier.upper()} ({len(feats)} features):")
        for f in feats:
            print(f"    • {f:40s}  composite={composite[f]:.4f}")

    print(f"\nDone! Plots saved to {PLOTS_DIR}")
    print(f"  → 19_feature_importance_analysis.png  (main 6-panel figure)")
    print(f"  → 19b_feature_redundancy_dendrogram.png")
    print(f"  → 19c_feature_rank_comparison.png")


if __name__ == "__main__":
    main()
