# Indian Art Raga Workbench Report — Carnatic

Generated from:

`../Indian Art Music Raga Recognition Dataset (features)/RagaDataset`

This workbench uses the precomputed feature files, not raw audio. Each usable
track is converted into 96 numeric features from tonic-normalized pitch,
pitch-class histograms, interval movement, nyas segments, and tani segments.

## Outputs

| Dataset CSV | Model | Metrics |
|---|---|---|
| `outputs/indian_art_carnatic_features.csv` | `outputs/indian_art_carnatic_model.joblib` | `outputs/indian_art_carnatic_metrics.json` |

## Results

| Usable Rows | Classes | Best Model | Accuracy | Macro F1 | Top-3 Accuracy | Split |
|---:|---:|---|---:|---:|---:|---|
| 468 | 40 | Random Forest | 0.8043 | 0.8076 | 0.9130 | grouped by artist |

## Notes

- The Carnatic dataset is nearly balanced, complete, and suitable for serious prototyping.
- The split is grouped by artist to reduce artist-style leakage. Random splits would likely look better but be less honest.
- `xgboost` and `lightgbm` are not installed in the current venv, so this workbench uses sklearn models only.

## Re-run Commands

```powershell
cd indian_art_raga_workbench
..\.venv\Scripts\python.exe train_indian_art_raga.py
```
