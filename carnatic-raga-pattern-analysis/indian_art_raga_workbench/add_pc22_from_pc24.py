import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "outputs" / "indian_art_carnatic_features.csv"

def rebin_pc24_to_pc22(pc24_vals):
    n24, n22 = 24, 22
    width24 = 1200.0 / n24
    width22 = 1200.0 / n22

    pc22 = np.zeros(n22)
    for j in range(n22):
        lo22 = j * width22
        hi22 = (j + 1) * width22
        for i in range(n24):
            lo24 = i * width24
            hi24 = (i + 1) * width24
            overlap = max(0.0, min(hi22, hi24) - max(lo22, lo24))
            if overlap > 0:
                pc22[j] += pc24_vals[i] * (overlap / width24)
    total = pc22.sum()
    if total > 0:
        pc22 /= total
    return pc22


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    pc24_cols = [f"pc24_{i:02d}" for i in range(24)]
    if pc24_cols[0] not in df.columns:
        print("ERROR: pc24 columns not found in CSV!")
        return

    if "pc22_00" in df.columns:
        print("  pc22 columns already exist — overwriting.")
        df.drop(columns=[f"pc22_{i:02d}" for i in range(22)], inplace=True)

    print("  Computing pc22 from pc24 ...")
    pc22_data = np.zeros((len(df), 22))
    for idx in range(len(df)):
        pc24_vals = df.iloc[idx][pc24_cols].values.astype(float)
        pc22_data[idx] = rebin_pc24_to_pc22(pc24_vals)

    pc24_start = df.columns.get_loc("pc24_00")
    for i in range(22):
        df.insert(pc24_start + i, f"pc22_{i:02d}", pc22_data[:, i])

    print(f"  Saving -> {CSV_PATH}")
    df.to_csv(CSV_PATH, index=False)
    print(f"  Done! {len(df.columns)} columns now.")


if __name__ == "__main__":
    main()
