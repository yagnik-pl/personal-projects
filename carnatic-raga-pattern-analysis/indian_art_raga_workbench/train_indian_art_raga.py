import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


WORKBENCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKBENCH_ROOT.parent
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "Indian Art Music Raga Recognition Dataset (features)"
    / "RagaDataset"
)
DEFAULT_OUTPUT_DIR = WORKBENCH_ROOT / "outputs"
RANDOM_STATE = 42
FEATURE_VERSION = "indian_art_pitch_tonic_segments_v1"

PITCH_EXTENSIONS = (".pitchSilIntrpPP", ".pitch")
TONIC_EXTENSIONS = (".tonicFine", ".tonic")
SEGMENT_EXTENSIONS = {
    "nyas": ".flatSegNyas",
    "tani": ".taniSegKNN",
}


def windows_safe_path(path):
    resolved = str(Path(path).resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def path_exists(path):
    return os.path.exists(windows_safe_path(path))


def open_text(path):
    return open(windows_safe_path(path), "r", encoding="utf-8", errors="replace")


def load_raga_mapping(dataset_root, system):
    mapping_path = dataset_root / system / "_info_" / "ragaId_to_ragaName_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata_rows(dataset_root, system):
    metadata_path = dataset_root / system / "_info_" / "path_mbid_ragaid.txt"
    rows = []
    raga_mapping = load_raga_mapping(dataset_root, system)
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            audio_path, mbid, raga_id = line.split("\t")
            parts = PurePosixPath(audio_path).parts
            artist = parts[4] if len(parts) > 4 else ""
            album = parts[5] if len(parts) > 5 else ""
            composition = parts[6] if system == "Carnatic" and len(parts) > 6 else ""
            track_name = parts[-1].removesuffix(".mp3")

            feature_stem = audio_path.replace("/audio/", "/features/")
            feature_stem = feature_stem.removesuffix(".mp3")
            feature_base = dataset_root.parent / feature_stem

            rows.append(
                {
                    "system": system,
                    "source_path": audio_path,
                    "feature_base": str(feature_base),
                    "mbid": mbid,
                    "raga_id": raga_id,
                    "raga": raga_mapping.get(raga_id, raga_id),
                    "artist": artist,
                    "album": album,
                    "composition": composition,
                    "track_name": track_name,
                }
            )
    return rows


def first_existing_path(feature_base, extensions):
    for extension in extensions:
        path = Path(f"{feature_base}{extension}")
        if path_exists(path):
            return path, extension
    return None, ""


def read_scalar(path):
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return float(line.split()[0])
    raise ValueError(f"No scalar value found in {path}")


def read_two_column_file(path):
    values = []
    with open_text(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                values.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.asarray(values, dtype=float)


def summarize_array(prefix, values):
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_iqr": 0.0,
        }

    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": float(np.percentile(values, 25)),
        f"{prefix}_q75": float(np.percentile(values, 75)),
        f"{prefix}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def pitch_class_histogram(cents, bins):
    if cents.size == 0:
        return np.zeros(bins, dtype=float)
    pitch_class = np.mod(cents, 1200.0)
    hist, _ = np.histogram(pitch_class, bins=bins, range=(0.0, 1200.0))
    hist = hist.astype(float)
    total = hist.sum()
    return hist / total if total else hist


def extract_pitch_features(pitch_path, tonic_hz):
    data = read_two_column_file(pitch_path)
    if data.size == 0:
        raise ValueError(f"No pitch rows found in {pitch_path}")

    times = data[:, 0]
    pitch_hz = data[:, 1]
    voiced = pitch_hz > 0
    voiced_pitch = pitch_hz[voiced]

    if voiced_pitch.size == 0:
        raise ValueError(f"No voiced pitch values found in {pitch_path}")

    cents = 1200.0 * np.log2(voiced_pitch / tonic_hz)
    pitch_class_12 = pitch_class_histogram(cents, 12)
    pitch_class_22 = pitch_class_histogram(cents, 22)
    pitch_class_24 = pitch_class_histogram(cents, 24)

    features = {
        "duration_s": float(np.max(times) if times.size else 0.0),
        "frame_count": int(len(pitch_hz)),
        "voiced_frame_count": int(np.sum(voiced)),
        "voiced_fraction": float(np.mean(voiced)),
        "tonic_hz": float(tonic_hz),
        "tonic_log2_hz": float(math.log2(tonic_hz)),
    }
    features.update(summarize_array("pitch_hz", voiced_pitch))
    features.update(summarize_array("cents", cents))
    features["cents_range"] = float(np.max(cents) - np.min(cents))

    if cents.size > 1:
        intervals = np.diff(cents)
        abs_intervals = np.abs(intervals)
        features.update(summarize_array("interval_cents", intervals))
        features.update(summarize_array("abs_interval_cents", abs_intervals))
        features["large_jump_fraction_100c"] = float(np.mean(abs_intervals > 100.0))
        features["large_jump_fraction_200c"] = float(np.mean(abs_intervals > 200.0))
    else:
        features.update(summarize_array("interval_cents", np.asarray([], dtype=float)))
        features.update(summarize_array("abs_interval_cents", np.asarray([], dtype=float)))
        features["large_jump_fraction_100c"] = 0.0
        features["large_jump_fraction_200c"] = 0.0

    for i, value in enumerate(pitch_class_12):
        features[f"pc12_{i:02d}"] = float(value)
    for i, value in enumerate(pitch_class_22):
        features[f"pc22_{i:02d}"] = float(value)
    for i, value in enumerate(pitch_class_24):
        features[f"pc24_{i:02d}"] = float(value)

    strongest_12 = int(np.argmax(pitch_class_12))
    features["pc12_peak_bin"] = strongest_12
    features["pc12_peak_value"] = float(pitch_class_12[strongest_12])
    features["pc12_entropy"] = float(
        -np.sum(pitch_class_12[pitch_class_12 > 0] * np.log2(pitch_class_12[pitch_class_12 > 0]))
    )

    return features


def extract_segment_features(feature_base, prefix, extension):
    segment_path = Path(f"{feature_base}{extension}")
    if not path_exists(segment_path):
        return {
            f"{prefix}_present": 0,
            f"{prefix}_segment_count": 0,
            f"{prefix}_total_duration_s": 0.0,
            f"{prefix}_duration_mean": 0.0,
            f"{prefix}_duration_median": 0.0,
            f"{prefix}_duration_max": 0.0,
        }

    data = read_two_column_file(segment_path)
    if data.size == 0:
        durations = np.asarray([], dtype=float)
    else:
        durations = np.maximum(0.0, data[:, 1] - data[:, 0])

    return {
        f"{prefix}_present": 1,
        f"{prefix}_segment_count": int(durations.size),
        f"{prefix}_total_duration_s": float(np.sum(durations)) if durations.size else 0.0,
        f"{prefix}_duration_mean": float(np.mean(durations)) if durations.size else 0.0,
        f"{prefix}_duration_median": float(np.median(durations)) if durations.size else 0.0,
        f"{prefix}_duration_max": float(np.max(durations)) if durations.size else 0.0,
    }


def build_feature_dataframe(dataset_root, system, min_raga_count):
    rows = []
    skipped = []
    for row in load_metadata_rows(dataset_root, system):
        feature_base = row["feature_base"]
        pitch_path, pitch_extension = first_existing_path(feature_base, PITCH_EXTENSIONS)
        tonic_path, tonic_extension = first_existing_path(feature_base, TONIC_EXTENSIONS)

        if pitch_path is None or tonic_path is None:
            skipped.append(
                {
                    **row,
                    "skip_reason": "missing_pitch_or_tonic",
                    "has_pitch": int(pitch_path is not None),
                    "has_tonic": int(tonic_path is not None),
                }
            )
            continue

        try:
            tonic_hz = read_scalar(tonic_path)
            features = extract_pitch_features(pitch_path, tonic_hz)
            for prefix, extension in SEGMENT_EXTENSIONS.items():
                features.update(extract_segment_features(feature_base, prefix, extension))
            rows.append(
                {
                    **row,
                    "feature_version": FEATURE_VERSION,
                    "pitch_source": pitch_extension,
                    "tonic_source": tonic_extension,
                    **features,
                }
            )
        except Exception as exc:
            skipped.append({**row, "skip_reason": f"feature_error: {exc}"})

    features_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    if features_df.empty:
        raise RuntimeError(f"No usable {system} feature rows were created.")

    if min_raga_count > 1:
        counts = features_df["raga"].value_counts()
        keep_ragas = counts[counts >= min_raga_count].index
        filtered = features_df.loc[features_df["raga"].isin(keep_ragas)].copy()
        dropped = features_df.loc[~features_df["raga"].isin(keep_ragas)].copy()
        if not dropped.empty:
            dropped["skip_reason"] = f"raga_count_below_{min_raga_count}"
            skipped_df = pd.concat([skipped_df, dropped], ignore_index=True, sort=False)
        features_df = filtered

    return features_df.reset_index(drop=True), skipped_df.reset_index(drop=True)


def get_feature_columns(df):
    metadata_cols = {
        "system",
        "source_path",
        "feature_base",
        "mbid",
        "raga_id",
        "raga",
        "artist",
        "album",
        "composition",
        "track_name",
        "feature_version",
        "pitch_source",
        "tonic_source",
    }
    return [
        c
        for c in df.columns
        if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])
    ]


def split_features(df, feature_cols, group_col):
    y = df["raga"].to_numpy()
    X = df[feature_cols]
    groups = df[group_col].fillna("").astype(str).to_numpy()

    class_count = df["raga"].nunique()
    min_group_count = int(df.groupby("raga")[group_col].nunique().min())
    n_splits = min(5, int(pd.Series(groups).nunique()), min_group_count)
    if n_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        best_split = None
        best_score = -1
        for train_idx, test_idx in splitter.split(X, y, groups):
            train_classes = set(y[train_idx])
            test_classes = set(y[test_idx])
            score = len(train_classes) + len(test_classes)
            if score > best_score:
                best_score = score
                best_split = (train_idx, test_idx)
            if len(train_classes) == class_count and len(test_classes) == class_count:
                return train_idx, test_idx, f"stratified_group_by_{group_col}"

        if best_split is not None:
            return best_split[0], best_split[1], f"stratified_group_by_{group_col}_partial"

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return train_idx, test_idx, "stratified_random"


def build_models(train_size):
    knn_neighbors = max(1, min(7, train_size))
    svm_rbf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    C=10,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=600,
        class_weight="balanced",
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    gaussian_nb = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]
    )

    return {
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=knn_neighbors)),
            ]
        ),
        "svm_rbf": svm_rbf,
        "gaussian_nb": gaussian_nb,
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": extra_trees,
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
        "soft_vote": VotingClassifier(
            estimators=[
                ("svm", svm_rbf),
                ("extra_trees", extra_trees),
                ("gaussian_nb", gaussian_nb),
            ],
            voting="soft",
        ),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test, label_encoder):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            pred,
            labels=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            pred,
            labels=label_encoder.classes_,
        ).tolist(),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        k = min(3, len(label_encoder.classes_))
        metrics[f"top_{k}_accuracy"] = float(
            top_k_accuracy_score(
                label_encoder.transform(y_test),
                proba,
                k=k,
                labels=np.arange(len(label_encoder.classes_)),
            )
        )

    top3 = metrics.get("top_3_accuracy")
    top3_text = f" top3={top3:.4f}" if top3 is not None else ""
    print(
        f"{name:<24} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}{top3_text}"
    )
    return model, metrics


def train_and_save(features_df, skipped_df, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = output_dir / f"indian_art_{args.system.lower()}_features.csv"
    skipped_csv = output_dir / f"indian_art_{args.system.lower()}_skipped_rows.csv"
    model_out = output_dir / f"indian_art_{args.system.lower()}_model.joblib"
    metrics_out = output_dir / f"indian_art_{args.system.lower()}_metrics.json"

    features_df.to_csv(dataset_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)
    print(f"Wrote dataset: {dataset_csv}")
    print(f"Wrote skipped rows: {skipped_csv}")

    feature_cols = get_feature_columns(features_df)
    train_idx, test_idx, split_mode = split_features(features_df, feature_cols, args.group_col)

    X_train = features_df.iloc[train_idx][feature_cols]
    X_test = features_df.iloc[test_idx][feature_cols]
    y_train = features_df.iloc[train_idx]["raga"].to_numpy()
    y_test = features_df.iloc[test_idx]["raga"].to_numpy()

    label_encoder = LabelEncoder().fit(features_df["raga"])
    trained_models = {}
    model_metrics = {}

    print(f"\nSplit: {split_mode} ({len(train_idx)} train / {len(test_idx)} test)")
    print("\nModel metrics")
    print("-" * 68)
    for name, model in build_models(len(train_idx)).items():
        trained_model, metrics = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            label_encoder,
        )
        trained_models[name] = trained_model
        model_metrics[name] = metrics

    best_model_name = max(
        model_metrics,
        key=lambda key: (
            model_metrics[key]["macro_f1"],
            model_metrics[key]["accuracy"],
            model_metrics[key].get("top_3_accuracy", 0.0),
        ),
    )
    final_model = deepcopy(trained_models[best_model_name])
    final_model.fit(features_df[feature_cols], features_df["raga"].to_numpy())

    bundle = {
        "model": final_model,
        "evaluation_model": trained_models[best_model_name],
        "models": trained_models,
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "classes": list(label_encoder.classes_),
        "label_encoder": label_encoder,
        "feature_version": FEATURE_VERSION,
        "system": args.system,
        "group_col": args.group_col,
    }
    joblib.dump(bundle, model_out, compress=3)

    metrics_report = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "system": args.system,
        "feature_version": FEATURE_VERSION,
        "dataset_csv": str(dataset_csv),
        "skipped_rows_csv": str(skipped_csv),
        "model_out": str(model_out),
        "num_rows": int(len(features_df)),
        "num_skipped_rows": int(len(skipped_df)),
        "num_classes": int(features_df["raga"].nunique()),
        "label_counts": features_df["raga"].value_counts().sort_index().to_dict(),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "split_mode": split_mode,
        "group_col": args.group_col,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "best_model": best_model_name,
        "saved_model_training": "refit_on_all_rows",
        "models": model_metrics,
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model: {model_out}")
    print(f"Saved metrics: {metrics_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Train models from the Indian Art Music Raga Recognition feature dataset."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--system", choices=["Carnatic"], default="Carnatic")
    parser.add_argument("--group-col", choices=["artist", "album", "mbid", "track_name"], default="artist")
    parser.add_argument("--min-raga-count", type=int, default=5)
    args = parser.parse_args()

    features_df, skipped_df = build_feature_dataframe(
        Path(args.dataset_root),
        args.system,
        min_raga_count=args.min_raga_count,
    )

    print(
        f"Built {args.system} flat dataset: "
        f"{len(features_df)} usable rows, {features_df['raga'].nunique()} classes, "
        f"{len(skipped_df)} skipped rows"
    )
    print(features_df["raga"].value_counts().sort_index().to_string())
    train_and_save(features_df, skipped_df, args)


if __name__ == "__main__":
    main()
