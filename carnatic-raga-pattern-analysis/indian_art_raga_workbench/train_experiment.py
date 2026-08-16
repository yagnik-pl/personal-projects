import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
ENABLE_PC22 = True
PC22_BINS = 22
FEATURE_VERSION = "clean_v2"


def open_text(path):
    return open(path, "r", encoding="utf-8", errors="replace")


def read_scalar(path):
    with open_text(path) as f:
        for line in f:
            if line.strip():
                return float(line.split()[0])
    raise ValueError("No scalar found")


def read_two_column_file(path):
    values = []
    with open_text(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    values.append((float(parts[0]), float(parts[1])))
                except:
                    pass
    return np.asarray(values)


def pitch_class_histogram(cents, bins):
    if len(cents) == 0:
        return np.zeros(bins)

    pc = np.mod(cents, 1200)
    hist, _ = np.histogram(pc, bins=bins, range=(0, 1200))
    hist = hist.astype(float)

    return hist / hist.sum() if hist.sum() else hist


def summarize_array(prefix, values):
    if len(values) == 0:
        return {}

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def extract_pitch_features(pitch_path, tonic_hz):
    data = read_two_column_file(pitch_path)

    pitch = data[:, 1]
    voiced = pitch > 0
    pitch = pitch[voiced]

    cents = 1200 * np.log2(pitch / tonic_hz)

    pc12 = pitch_class_histogram(cents, 12)
    pc24 = pitch_class_histogram(cents, 24)
    pc22 = pitch_class_histogram(cents, PC22_BINS) if ENABLE_PC22 else None

    features = {}

    features.update(summarize_array("cents", cents))

    for i, v in enumerate(pc12):
        features[f"pc12_{i}"] = float(v)

    for i, v in enumerate(pc24):
        features[f"pc24_{i}"] = float(v)

    if ENABLE_PC22:
        for i, v in enumerate(pc22):
            features[f"pc22_{i}"] = float(v)

    return features


def load_rows(dataset_root):
    metadata = dataset_root / "Carnatic" / "_info_" / "path_mbid_ragaid.txt"
    mapping = json.load(open(dataset_root / "Carnatic" / "_info_" / "ragaId_to_ragaName_mapping.json"))

    rows = []

    with open(metadata) as f:
        for line in f:
            path, mbid, raga_id = line.strip().split("\t")

            base = dataset_root.parent / path.replace("/audio/", "/features/").replace(".mp3", "")

            pitch_file = str(base) + ".pitch"
            tonic_file = str(base) + ".tonic"

            if not os.path.exists(pitch_file) or not os.path.exists(tonic_file):
                continue

            try:
                tonic = read_scalar(tonic_file)
                feats = extract_pitch_features(pitch_file, tonic)

                rows.append({
                    "raga": mapping.get(raga_id, raga_id),
                    **feats
                })
            except:
                continue

    return pd.DataFrame(rows)


def save_feature_importance(model, feature_cols):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nTop 15 Features:")
    print(df.head(15))


def compare_feature_groups(df):
    print("\n=== Feature Comparison ===")

    groups = {
        "pc12": [c for c in df.columns if c.startswith("pc12_")],
        "pc22": [c for c in df.columns if c.startswith("pc22_")],
        "pc24": [c for c in df.columns if c.startswith("pc24_")],
    }

    for name, cols in groups.items():
        if len(cols) == 0:
            continue

        X = df[cols]
        y = df["raga"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        print(f"{name:<10} accuracy = {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    df = load_rows(Path(args.dataset_root))

    print(f"Loaded {len(df)} samples")

    feature_cols = [c for c in df.columns if c != "raga"]

    X = df[feature_cols]
    y = df["raga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\nFinal Accuracy:", accuracy_score(y_test, pred))

    save_feature_importance(model, feature_cols)
    compare_feature_groups(df)


if __name__ == "__main__":
    main()
