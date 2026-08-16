import warnings
import joblib
import numpy as np
import pandas as pd
import librosa
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "Dataset.csv"
UNKNOWN_LABEL = "none_of_the_above"
USE_ONLY_MFCC = True
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def get_feature_columns(df):
    if USE_ONLY_MFCC:
        mfcc_cols = [c for c in df.columns if c.startswith("mfcc")]
        if len(mfcc_cols) > 0:
            return mfcc_cols
    return [c for c in df.columns if c not in {"filename", "raga"}]


def build_lgbm():
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=RANDOM_STATE
    )
    return model


def build_xgb():
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    )
    return model


def build_pipeline(base_model):
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", calibrated)
    ])
    return pipeline


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.176,
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def confidence_stats(proba):
    sorted_p = np.sort(proba, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2]
    margin = top1 - top2
    return top1, margin


def tune_thresholds(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred = np.argmax(proba, axis=1)

    top1, margin = confidence_stats(proba)
    correct = pred == y_val

    if np.any(correct):
        conf_thr = np.percentile(top1[correct], 10)
        margin_thr = np.percentile(margin[correct], 10)
    else:
        conf_thr = np.percentile(top1, 20)
        margin_thr = np.percentile(margin, 20)

    conf_thr = float(np.clip(conf_thr, 0.55, 0.95))
    margin_thr = float(np.clip(margin_thr, 0.03, 0.25))

    return conf_thr, margin_thr


def open_set_predict(model, X, label_encoder, conf_thr, margin_thr):
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    labels = label_encoder.inverse_transform(pred_idx)

    top1, margin = confidence_stats(proba)

    final = np.where(
        (top1 < conf_thr) | (margin < margin_thr),
        UNKNOWN_LABEL,
        labels
    )

    return final


def train_and_evaluate(name, model, X_train, X_val, X_test, y_train, y_val, y_test, label_encoder):
    print(f"\n===== {name} =====")

    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    conf_thr, margin_thr = tune_thresholds(pipeline, X_val, y_val)

    final_model = build_pipeline(model)
    final_model.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))

    pred = open_set_predict(final_model, X_test, label_encoder, conf_thr, margin_thr)
    y_true = label_encoder.inverse_transform(y_test)

    coverage = np.mean(pred != UNKNOWN_LABEL)
    accuracy = accuracy_score(y_true, pred)

    print("Coverage:", coverage)
    print("Accuracy:", accuracy)

    mask = pred != UNKNOWN_LABEL
    if np.any(mask):
        print(classification_report(y_true[mask], pred[mask]))

    return final_model


def extract_mfcc_from_audio(file_path, n_mfcc=19, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    y = librosa.util.fix_length(y, size=sr * 5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def mfcc_to_feature_vector(mfcc):
    return np.mean(mfcc, axis=1)


def extract_features_for_model(file_path, n_mfcc=19):
    mfcc = extract_mfcc_from_audio(file_path, n_mfcc=n_mfcc)
    features = mfcc_to_feature_vector(mfcc)
    feature_dict = {f"mfcc{i}": features[i] for i in range(len(features))}
    return feature_dict


def predict_from_audio(file_path, model_bundle):
    try:
        features = extract_features_for_model(file_path)
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=model_bundle["feature_cols"], fill_value=0)

        proba = model_bundle["model"].predict_proba(features_df)
        pred_idx = np.argmax(proba, axis=1)

        top1 = np.max(proba, axis=1)[0]
        top2_max = np.sort(proba[0])[-2]
        margin = top1 - top2_max

        conf_thr = model_bundle["conf_threshold"]
        margin_thr = model_bundle["margin_threshold"]

        if top1 < conf_thr or margin < margin_thr:
            prediction = model_bundle.get("unknown_label", "unknown")
        else:
            prediction = model_bundle["label_encoder"].inverse_transform([pred_idx[0]])[0]

        result = {
            "file": file_path,
            "prediction": prediction,
            "confidence": float(top1),
            "margin": float(margin),
            "status": "success"
        }

        return result
    except Exception as e:
        return {
            "file": file_path,
            "prediction": "error",
            "confidence": 0.0,
            "margin": 0.0,
            "status": f"error: {str(e)}"
        }


def main():
    df = load_data(DATA_PATH)

    X = df[get_feature_columns(df)]
    y_text = df["raga"]

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    lgbm_model = train_and_evaluate("LightGBM", build_lgbm(),
                                   X_train, X_val, X_test, y_train, y_val, y_test, le)

    xgb_model = train_and_evaluate("XGBoost", build_xgb(),
                                  X_train, X_val, X_test, y_train, y_val, y_test, le)

    feature_cols = get_feature_columns(df)
    conf_thr, margin_thr = tune_thresholds(lgbm_model, X_val, y_val)

    bundle = {
        "lgbm": lgbm_model,
        "xgb": xgb_model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "conf_threshold": conf_thr,
        "margin_threshold": margin_thr,
        "unknown_label": UNKNOWN_LABEL,
        "model": lgbm_model
    }

    joblib.dump(bundle, "models.joblib")
    print("\nModels saved to models.joblib")


def test_audio_files(test_folder="test-data"):
    model_bundle = joblib.load("models.joblib")

    test_path = Path(test_folder)
    if not test_path.exists():
        print(f"Test folder '{test_folder}' not found!")
        return

    wav_files = list(test_path.glob("**/*.wav"))

    if not wav_files:
        print(f"No .wav files found in '{test_folder}'")
        return

    print(f"\nTesting {len(wav_files)} audio file(s)...\n")
    print(f"{'File':<50} {'Prediction':<20} {'Confidence':<12} {'Status'}")
    print("-" * 95)

    for wav_file in wav_files:
        result = predict_from_audio(str(wav_file), model_bundle)
        filename = wav_file.name
        print(f"{filename:<50} {result['prediction']:<20} {result['confidence']:<12.4f} {result['status']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_folder = sys.argv[2] if len(sys.argv) > 2 else "test-data"
        test_audio_files(test_folder)
    else:
        main()