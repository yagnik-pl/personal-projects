import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PRESETS = {
    "100": {
        "model": PROJECT_ROOT / "carnatic_raga_model_100.joblib",
        "metrics": PROJECT_ROOT / "carnatic_raga_metrics_100.json",
    },
    "full": {
        "model": PROJECT_ROOT / "carnatic_raga_model.joblib",
        "metrics": PROJECT_ROOT / "carnatic_raga_metrics.json",
    },
}


def resolve_paths(args):
    preset = MODEL_PRESETS[args.model_preset]
    model_path = Path(args.model_path).resolve() if args.model_path else preset["model"]
    metrics_path = (
        Path(args.metrics_path).resolve() if args.metrics_path else preset["metrics"]
    )
    return model_path, metrics_path


def load_metrics(metrics_path):
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def force_single_threaded_prediction(model_bundle):
    model = model_bundle["model"]
    n_jobs_params = {
        name: 1
        for name in model.get_params(deep=True)
        if name.endswith("n_jobs")
    }

    if n_jobs_params:
        model.set_params(**n_jobs_params)

    clf = getattr(model, "named_steps", {}).get("clf")
    for calibrated in getattr(clf, "calibrated_classifiers_", []):
        estimator = getattr(calibrated, "estimator", None)
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1


def find_wav_files(test_folder):
    test_path = Path(test_folder)
    if not test_path.is_absolute():
        test_path = PROJECT_ROOT / test_path

    if not test_path.exists():
        raise FileNotFoundError(f"Test folder not found: {test_path}")

    return sorted(test_path.glob("**/*.wav"))


def extract_features_for_model(file_path, n_mfcc=19, sr=22050, seconds=5):
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    y = librosa.util.fix_length(y, size=sr * seconds)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfcc, axis=1)
    return {f"mfcc{i}": features[i] for i in range(len(features))}


def predict_from_audio(file_path, model_bundle):
    try:
        features = extract_features_for_model(file_path)
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(
            columns=model_bundle["feature_cols"],
            fill_value=0,
        )

        proba = model_bundle["model"].predict_proba(features_df)
        pred_idx = int(np.argmax(proba, axis=1)[0])

        sorted_proba = np.sort(proba[0])
        top1 = float(sorted_proba[-1])
        top2 = float(sorted_proba[-2]) if len(sorted_proba) > 1 else 0.0
        margin = top1 - top2

        if (
            top1 < model_bundle["conf_threshold"]
            or margin < model_bundle["margin_threshold"]
        ):
            prediction = model_bundle.get("unknown_label", "unknown")
        else:
            prediction = model_bundle["label_encoder"].inverse_transform([pred_idx])[0]

        return {
            "file": str(file_path),
            "prediction": prediction,
            "confidence": top1,
            "margin": margin,
            "status": "success",
        }
    except Exception as exc:
        return {
            "file": str(file_path),
            "prediction": "error",
            "confidence": 0.0,
            "margin": 0.0,
            "status": f"error: {exc}",
        }


def print_model_metrics(model_bundle, metrics, model_path, metrics_path):
    classes = list(model_bundle["label_encoder"].classes_)

    print("\nModel")
    print("-" * 72)
    print(f"Model file:        {model_path.name}")
    print(f"Metrics file:      {metrics_path.name if metrics else 'not found'}")
    print(f"Classes:           {len(classes)} ({', '.join(classes)})")
    print(f"Feature count:     {len(model_bundle['feature_cols'])}")
    print(f"Confidence thresh: {model_bundle['conf_threshold']:.4f}")
    print(f"Margin thresh:     {model_bundle['margin_threshold']:.4f}")

    if metrics:
        print(f"Coverage:          {metrics.get('coverage', 0):.4f}")
        print(f"Overall accuracy:  {metrics.get('overall_accuracy', 0):.4f}")
        print(f"Covered accuracy:  {metrics.get('covered_accuracy', 0):.4f}")


def print_predictions(results):
    print("\nAudio Predictions")
    print("-" * 104)
    print(
        f"{'File':<46} {'Prediction':<20} {'Confidence':>10} "
        f"{'Margin':>10} {'Status':<10}"
    )
    print("-" * 104)

    for result in results:
        filename = Path(result["file"]).name
        print(
            f"{filename:<46} {result['prediction']:<20} "
            f"{result['confidence']:>10.4f} {result['margin']:>10.4f} "
            f"{result['status']:<10}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict ragas for .wav files in test-data and print metrics."
    )
    parser.add_argument("--test-folder", default="test-data")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="100")
    parser.add_argument("--model-path")
    parser.add_argument("--metrics-path")
    args = parser.parse_args()

    model_path, metrics_path = resolve_paths(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_bundle = joblib.load(model_path)
    force_single_threaded_prediction(model_bundle)
    metrics = load_metrics(metrics_path)
    wav_files = find_wav_files(args.test_folder)

    print_model_metrics(model_bundle, metrics, model_path, metrics_path)

    if not wav_files:
        print(f"\nNo .wav files found in {args.test_folder}")
        return

    results = [predict_from_audio(str(wav_file), model_bundle) for wav_file in wav_files]
    print_predictions(results)


if __name__ == "__main__":
    main()
