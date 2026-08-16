import argparse
import json
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_LABELS_CSV = PROJECT_ROOT / "raga_wav_labels.csv"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "raga_audio_features.csv"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "raga_audio_model.joblib"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "raga_audio_metrics.json"
DEFAULT_REMOVED_ROWS_CSV = PROJECT_ROOT / "raga_audio_removed_rows.csv"
RANDOM_STATE = 42
FEATURE_VERSION = "fast_mfcc_chroma_mel_v2"

CLEAN_8_CLASS_PRESET = [
    "Anandabhairavi",
    "Bhairavi",
    "Bhauli",
    "Bilahari",
    "Dhwijavanthi",
    "Gaanamurthi",
    "Hamsadhwani",
    "Thodi",
]

RAGA_ALIASES = [
    ("Anandabhairavi", ["anandabhairavi"]),
    ("Poorvikalyani", ["poorvikalyani"]),
    ("Hamsadhwani", ["hamsadhwani", "hamsadwani", "hamsadwaniraaga"]),
    ("Dhwijavanthi", ["dhwijavanthi"]),
    ("Gaanamurthi", ["gaanamurthi", "gaanamurthe", "gaanamurtheraaga"]),
    ("Nilambari", ["nilambari", "nilambariraaga"]),
    ("Bhairavi", ["bhairavi", "bhairaviraaga"]),
    ("Bilahari", ["bilahari"]),
    ("Bhauli", ["bhauli"]),
    ("Saveri", ["saveri"]),
    ("Thodi", ["thodi", "todi"]),
]

AUGMENTATION_SUFFIXES = [
    r"_noise_[+-]?\d+(?:\.\d+)?$",
    r"_pitch_shift_[+-]?\d+(?:\.\d+)?$",
    r"_time_stretch_[+-]?\d+(?:\.\d+)?$",
]


def compact_text(value):
    return re.sub(r"[^a-z]", "", value.lower())


def strip_augmentation_suffix(stem):
    cleaned = stem
    for pattern in AUGMENTATION_SUFFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def infer_raga_from_filename(path):
    compact = compact_text(path.stem)
    for raga, aliases in RAGA_ALIASES:
        if any(alias in compact for alias in aliases):
            return raga
    raise ValueError(f"Could not infer raga label from filename: {path.name}")


def source_id_from_filename(path):
    return strip_augmentation_suffix(path.stem).lower()


def build_label_dataframe(dataset_dir):
    wav_files = sorted(Path(dataset_dir).glob("**/*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under {dataset_dir}")

    rows = []
    errors = []
    for wav_file in wav_files:
        try:
            raga = infer_raga_from_filename(wav_file)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        rows.append(
            {
                "path": str(wav_file),
                "filename": wav_file.name,
                "source_id": source_id_from_filename(wav_file),
                "raga": raga,
            }
        )

    if errors:
        message = "\n".join(errors[:20])
        if len(errors) > 20:
            message += f"\n... and {len(errors) - 20} more"
        raise ValueError(message)

    return pd.DataFrame(rows)


def summarize_frames(prefix, values):
    if values.ndim == 1:
        values = values.reshape(1, -1)

    summary = {}
    for i, row in enumerate(values):
        summary[f"{prefix}{i}_mean"] = float(np.mean(row))
        summary[f"{prefix}{i}_std"] = float(np.std(row))
        summary[f"{prefix}{i}_median"] = float(np.median(row))
        summary[f"{prefix}{i}_min"] = float(np.min(row))
        summary[f"{prefix}{i}_max"] = float(np.max(row))
        summary[f"{prefix}{i}_q25"] = float(np.percentile(row, 25))
        summary[f"{prefix}{i}_q75"] = float(np.percentile(row, 75))
    return summary


def summarize_vector(prefix, values):
    return {f"{prefix}{i}": float(value) for i, value in enumerate(values)}


def rotate_to_tonic(chroma):
    profile = np.mean(chroma, axis=1)
    tonic_bin = int(np.argmax(profile))
    return np.roll(chroma, -tonic_bin, axis=0), tonic_bin


def extract_audio_features(path, sr=22050, max_seconds=30.0, n_mfcc=20):
    y, sr = librosa.load(
        path,
        sr=sr,
        mono=True,
        duration=max_seconds,
        res_type="kaiser_fast",
    )
    y, _ = librosa.effects.trim(y)
    if len(y) == 0:
        raise ValueError("audio is empty after trimming silence")
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonic_chroma, tonic_bin = rotate_to_tonic(chroma)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features = {}
    features["estimated_tonic_bin"] = float(tonic_bin)
    features.update(summarize_frames("mfcc", mfcc))
    features.update(summarize_frames("delta_mfcc", delta_mfcc))
    features.update(summarize_frames("delta2_mfcc", delta2_mfcc))
    features.update(summarize_frames("chroma", chroma))
    features.update(summarize_frames("tonic_chroma", tonic_chroma))
    features.update(summarize_frames("contrast", contrast))
    features.update(summarize_frames("mel", mel_db))
    features.update(summarize_vector("tonic_chroma_profile", np.mean(tonic_chroma, axis=1)))

    single_feature_sets = {
        "zcr": librosa.feature.zero_crossing_rate(y),
        "rms": librosa.feature.rms(y=y),
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "flatness": librosa.feature.spectral_flatness(y=y),
    }
    for prefix, values in single_feature_sets.items():
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_std"] = float(np.std(values))
        features[f"{prefix}_median"] = float(np.median(values))
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))

    return features


def build_feature_dataframe(labels_df, sr, max_seconds, n_mfcc):
    rows = []
    for index, row in labels_df.iterrows():
        try:
            features = extract_audio_features(
                row["path"],
                sr=sr,
                max_seconds=max_seconds,
                n_mfcc=n_mfcc,
            )
            rows.append(
                {
                    **row.to_dict(),
                    "feature_version": FEATURE_VERSION,
                    **features,
                }
            )
        except Exception as exc:
            warnings.warn(f"Skipping {row['filename']}: {exc}")

        if (index + 1) % 25 == 0:
            print(f"Extracted features for {index + 1}/{len(labels_df)} files")

    if not rows:
        raise RuntimeError("No features were extracted successfully.")
    return pd.DataFrame(rows)


def feature_cache_is_current(features_df):
    return (
        "feature_version" in features_df.columns
        and set(features_df["feature_version"]) == {FEATURE_VERSION}
    )


def get_feature_columns(features_df):
    metadata_cols = {"path", "filename", "source_id", "raga", "feature_version"}
    return [
        c
        for c in features_df.columns
        if c not in metadata_cols and pd.api.types.is_numeric_dtype(features_df[c])
    ]


def make_removed_rows(df, reason):
    if df.empty:
        return df.copy()

    removed = df.copy()
    removed.insert(0, "removal_reason", reason)
    return removed


def clean_feature_dataframe(
    features_df,
    keep_classes=None,
    deduplicate_features=False,
    drop_outliers=False,
    outlier_z_threshold=8.0,
    outlier_feature_fraction=0.20,
    outlier_iqr_multiplier=4.0,
):
    cleaned_df = features_df.copy()
    removed_frames = []
    report = {
        "input_rows": int(len(cleaned_df)),
        "input_classes": sorted(cleaned_df["raga"].dropna().unique().tolist()),
        "filters": {},
    }

    feature_cols = get_feature_columns(cleaned_df)
    finite_mask = np.isfinite(cleaned_df[feature_cols].to_numpy(dtype=float)).all(axis=1)
    if "path" in cleaned_df.columns:
        path_mask = cleaned_df["path"].map(lambda value: Path(value).exists()).to_numpy()
    else:
        path_mask = np.ones(len(cleaned_df), dtype=bool)
    valid_mask = finite_mask & path_mask
    invalid_rows = cleaned_df.loc[~valid_mask]
    if not invalid_rows.empty:
        removed_frames.append(make_removed_rows(invalid_rows, "invalid_or_missing_audio"))
        cleaned_df = cleaned_df.loc[valid_mask].copy()

    report["filters"]["invalid_or_missing_audio"] = {
        "removed_rows": int(len(invalid_rows)),
    }

    if keep_classes:
        keep_classes = list(dict.fromkeys(keep_classes))
        class_mask = cleaned_df["raga"].isin(keep_classes)
        removed_classes = cleaned_df.loc[~class_mask]
        if not removed_classes.empty:
            removed_frames.append(make_removed_rows(removed_classes, "excluded_class"))
            cleaned_df = cleaned_df.loc[class_mask].copy()

        report["filters"]["class_filter"] = {
            "keep_classes": keep_classes,
            "removed_rows": int(len(removed_classes)),
            "removed_by_class": removed_classes["raga"]
            .value_counts()
            .sort_index()
            .to_dict(),
        }

    if deduplicate_features:
        feature_hash = pd.util.hash_pandas_object(
            cleaned_df[feature_cols].round(8),
            index=False,
        )
        duplicate_mask = cleaned_df.assign(_feature_hash=feature_hash).duplicated(
            subset=["raga", "source_id", "_feature_hash"],
            keep="first",
        )
        duplicate_rows = cleaned_df.loc[duplicate_mask]
        if not duplicate_rows.empty:
            removed_frames.append(make_removed_rows(duplicate_rows, "duplicate_feature_row"))
            cleaned_df = cleaned_df.loc[~duplicate_mask].copy()

        report["filters"]["duplicate_feature_row"] = {
            "removed_rows": int(len(duplicate_rows)),
            "removed_by_class": duplicate_rows["raga"]
            .value_counts()
            .sort_index()
            .to_dict(),
        }

    if drop_outliers:
        outlier_indices = []
        outlier_stats = []
        for raga, group in cleaned_df.groupby("raga", sort=False):
            if len(group) < 8:
                continue

            values = group[feature_cols].to_numpy(dtype=float)
            median = np.nanmedian(values, axis=0)
            mad = np.nanmedian(np.abs(values - median), axis=0)
            scale = np.where(mad > 1e-9, 1.4826 * mad, np.nan)
            robust_z = np.abs((values - median) / scale)

            extreme_fraction = np.nanmean(robust_z > outlier_z_threshold, axis=1)
            median_score = np.nanmedian(
                np.where(np.isfinite(robust_z), robust_z, 0.0),
                axis=1,
            )
            q1, q3 = np.percentile(median_score, [25, 75])
            iqr = q3 - q1
            score_cutoff = q3 + outlier_iqr_multiplier * iqr if iqr > 1e-12 else np.inf

            outlier_mask = (extreme_fraction > outlier_feature_fraction) | (
                median_score > score_cutoff
            )
            outlier_indices.extend(group.index[outlier_mask].tolist())
            for index, score, fraction in zip(
                group.index[outlier_mask],
                median_score[outlier_mask],
                extreme_fraction[outlier_mask],
            ):
                outlier_stats.append(
                    {
                        "index": int(index),
                        "outlier_median_robust_z": float(score),
                        "outlier_extreme_feature_fraction": float(fraction),
                        "outlier_score_cutoff": float(score_cutoff),
                    }
                )

        outlier_rows = cleaned_df.loc[outlier_indices].copy()
        if not outlier_rows.empty:
            stats_df = pd.DataFrame(outlier_stats).set_index("index")
            outlier_rows = outlier_rows.join(stats_df, how="left")
            removed_frames.append(make_removed_rows(outlier_rows, "feature_space_outlier"))
            cleaned_df = cleaned_df.drop(index=outlier_indices).copy()

        report["filters"]["feature_space_outlier"] = {
            "removed_rows": int(len(outlier_rows)),
            "removed_by_class": outlier_rows["raga"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "z_threshold": float(outlier_z_threshold),
            "feature_fraction_threshold": float(outlier_feature_fraction),
            "iqr_multiplier": float(outlier_iqr_multiplier),
        }

    removed_rows = (
        pd.concat(removed_frames, ignore_index=True)
        if removed_frames
        else pd.DataFrame(columns=["removal_reason", *cleaned_df.columns])
    )
    report["output_rows"] = int(len(cleaned_df))
    report["output_classes"] = sorted(cleaned_df["raga"].dropna().unique().tolist())
    report["output_label_counts"] = (
        cleaned_df["raga"].value_counts().sort_index().to_dict()
    )
    report["output_source_counts"] = (
        cleaned_df.groupby("raga")["source_id"].nunique().sort_index().to_dict()
    )
    report["removed_rows"] = int(len(removed_rows))

    return cleaned_df.reset_index(drop=True), removed_rows, report


def split_features(df, feature_cols):
    y = df["raga"].to_numpy()
    groups = df["source_id"].to_numpy()
    X = df[feature_cols]

    class_count = df["raga"].nunique()
    min_group_count = int(df.groupby("raga")["source_id"].nunique().min())
    n_splits = min(5, int(df["source_id"].nunique()), min_group_count)
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
                return train_idx, test_idx, "stratified_group"

        if best_split is not None:
            return best_split[0], best_split[1], "stratified_group_partial"

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return train_idx, test_idx, "stratified_random"


def build_models(train_size):
    knn_neighbors = max(1, min(5, train_size))
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
        n_estimators=1000,
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
            n_estimators=1000,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": extra_trees,
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
        f"{name:<14} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}{top3_text}"
    )
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Create raga labels from WAV filenames and train audio classifiers."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--input-features-csv")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS_OUT))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--clean-8-class-preset", action="store_true")
    parser.add_argument("--keep-classes", nargs="+")
    parser.add_argument("--deduplicate-features", action="store_true")
    parser.add_argument("--drop-outliers", action="store_true")
    parser.add_argument("--outlier-z-threshold", type=float, default=8.0)
    parser.add_argument("--outlier-feature-fraction", type=float, default=0.20)
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=4.0)
    parser.add_argument("--removed-rows-csv", default="")
    args = parser.parse_args()

    labels_df = build_label_dataframe(args.dataset_dir)
    labels_df.to_csv(args.labels_csv, index=False)
    print(f"Wrote labels: {args.labels_csv}")
    print(labels_df["raga"].value_counts().sort_index().to_string())

    features_input_path = Path(args.input_features_csv or args.features_csv)
    features_output_path = Path(args.features_csv)
    if features_input_path.exists() and not args.rebuild_features:
        features_df = pd.read_csv(features_input_path)
        if feature_cache_is_current(features_df):
            print(f"Loaded cached features: {features_input_path}")
        else:
            print(
                f"Loaded cached features: {features_input_path} "
                "(older feature set; use --rebuild-features to refresh)"
            )
    else:
        features_df = build_feature_dataframe(
            labels_df,
            sr=args.sr,
            max_seconds=args.max_seconds,
            n_mfcc=args.n_mfcc,
        )
        features_df.to_csv(features_output_path, index=False)
        print(f"Wrote features: {features_output_path}")

    keep_classes = args.keep_classes
    if args.clean_8_class_preset:
        keep_classes = CLEAN_8_CLASS_PRESET if keep_classes is None else keep_classes

    cleanup_report = None
    removed_rows = pd.DataFrame()
    if (
        keep_classes
        or args.deduplicate_features
        or args.drop_outliers
    ):
        features_df, removed_rows, cleanup_report = clean_feature_dataframe(
            features_df,
            keep_classes=keep_classes,
            deduplicate_features=args.deduplicate_features,
            drop_outliers=args.drop_outliers,
            outlier_z_threshold=args.outlier_z_threshold,
            outlier_feature_fraction=args.outlier_feature_fraction,
            outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        )
        features_df.to_csv(features_output_path, index=False)
        features_df[["path", "filename", "source_id", "raga"]].to_csv(
            args.labels_csv,
            index=False,
        )
        print(f"Wrote cleaned features: {features_output_path}")
        print(f"Wrote cleaned labels: {args.labels_csv}")
        print(
            "Cleaned dataset: "
            f"{cleanup_report['input_rows']} -> {cleanup_report['output_rows']} rows, "
            f"{len(cleanup_report['output_classes'])} classes"
        )
        if args.removed_rows_csv:
            removed_rows.to_csv(args.removed_rows_csv, index=False)
            print(f"Wrote removed rows: {args.removed_rows_csv}")

    feature_cols = get_feature_columns(features_df)
    train_idx, test_idx, split_mode = split_features(features_df, feature_cols)

    X_train = features_df.iloc[train_idx][feature_cols]
    X_test = features_df.iloc[test_idx][feature_cols]
    y_train = features_df.iloc[train_idx]["raga"].to_numpy()
    y_test = features_df.iloc[test_idx]["raga"].to_numpy()

    label_encoder = LabelEncoder().fit(features_df["raga"])
    print(f"\nSplit: {split_mode} ({len(train_idx)} train / {len(test_idx)} test)")
    print("\nModel metrics")
    print("-" * 50)

    trained_models = {}
    model_metrics = {}
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
        "sr": args.sr,
        "max_seconds": args.max_seconds,
        "n_mfcc": args.n_mfcc,
    }
    joblib.dump(bundle, args.model_out)

    metrics_report = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "num_files": int(len(features_df)),
        "num_classes": int(features_df["raga"].nunique()),
        "label_counts": features_df["raga"].value_counts().sort_index().to_dict(),
        "feature_count": int(len(feature_cols)),
        "split_mode": split_mode,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "best_model": best_model_name,
        "saved_model_training": "refit_on_all_labeled_files",
        "models": model_metrics,
    }
    if cleanup_report is not None:
        metrics_report["cleanup"] = cleanup_report
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model: {args.model_out}")
    print(f"Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
