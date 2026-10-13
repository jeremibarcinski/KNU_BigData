"""
Stage 2b (ASOS): drops the columns that the cleaning report showed to be too
incomplete in 2014-2026 (UrbanAtmos).

Input:   ASOS_02_Daegu_asos_clean.csv           (output of clean_asos.py)
Output:  ASOS_022_Daegu_after_report_clean.csv

Why these columns: the gaps are structural, not random.
  - cloud_cover_tenths: NaN in 100% of 2014, 57% of 2015, ~22% of 2016-2017, 13% of 2018
  - visibility_10m:     NaN in 100% of 2014, 35% of 2015
Interpolation would only invent data, so the columns are dropped. If cloud cover is
needed later, it can be taken from ERA5 (Open-Meteo) and marked as modelled data.

Requirements:  pip install pandas
Run:           python clean_asos_after_report.py
"""

import pandas as pd

# ---------- SETTINGS ----------
INPUT_CSV = "ASOS_02_Daegu_asos_clean.csv"
OUTPUT_CSV = "ASOS_022_Daegu_after_report_clean.csv"

DROP_COLUMNS = ["cloud_cover_tenths", "visibility_10m"]


def main() -> None:
    df = pd.read_csv(INPUT_CSV, parse_dates=["time"], index_col="time")
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Range: {df.index.min()} -> {df.index.max()}")

    absent = [c for c in DROP_COLUMNS if c not in df.columns]
    if absent:
        raise KeyError(f"Columns to drop not found in the input file: {absent}")

    df = df.drop(columns=DROP_COLUMNS)
    print(f"\nDropped: {', '.join(DROP_COLUMNS)}")
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")
    print("\nRemaining NaN per column:")
    for col, n in df.isna().sum().items():
        print(f"    {col}: {int(n)}")

    df.to_csv(OUTPUT_CSV)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
