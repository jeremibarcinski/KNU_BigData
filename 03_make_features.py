"""
Stage 3: feature engineering for hourly Open-Meteo data (UrbanAtmos).

Input:   Daegu_clean.csv (from clean_open_meteo.py)
Output:  Daegu_features.csv

What this script does (and nothing else, for now):
  - wind direction (degrees) -> wind_dir_sin, wind_dir_cos
  - hour of day              -> hour_sin, hour_cos
  - day of year              -> doy_sin, doy_cos
  The raw wind_direction_10m column is dropped after the conversion.

Why sin/cos: these quantities are cyclic (359 deg is next to 1 deg,
23:00 is next to 00:00, 31 Dec is next to 1 Jan). A (sin, cos) pair puts
each value on a circle, so the model sees that start and end are close.
Both columns are always needed: sin alone is ambiguous (e.g. 0:00 and 12:00).

Not done here (later stages): lags, rolling windows, scaling,
train/test split.

Requirements:  pip install pandas numpy
Run:           python 03_make_features.py
"""

import numpy as np
import pandas as pd

# ---------- SETTINGS ----------
INPUT_CSV = "Daegu_clean.csv"
OUTPUT_CSV = "Daegu_features.csv"

WIND_DIRECTION_COL = "wind_direction_10m"
DAYS_IN_YEAR = 365.25  # average year length, keeps 31 Dec close to 1 Jan


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sin/cos columns for wind direction, hour of day and day of year.

    Expects a DataFrame indexed by a DatetimeIndex.
    Returns a new DataFrame; the input is not modified.
    """
    out = df.copy()

    # wind direction: degrees (0-360) -> radians -> sin/cos, then drop raw column
    if WIND_DIRECTION_COL in out.columns:
        angle = np.deg2rad(out[WIND_DIRECTION_COL])
        out["wind_dir_sin"] = np.sin(angle)
        out["wind_dir_cos"] = np.cos(angle)
        out = out.drop(columns=WIND_DIRECTION_COL)

    # hour of day: 0-23, period 24
    hour_angle = 2 * np.pi * out.index.hour / 24
    out["hour_sin"] = np.sin(hour_angle)
    out["hour_cos"] = np.cos(hour_angle)

    # day of year: 1-366, period ~365.25 (already contains the month information)
    doy_angle = 2 * np.pi * out.index.dayofyear / DAYS_IN_YEAR
    out["doy_sin"] = np.sin(doy_angle)
    out["doy_cos"] = np.cos(doy_angle)

    return out


# ---------- MAIN ----------
def main() -> None:
    df = pd.read_csv(INPUT_CSV, parse_dates=["time"], index_col="time")
    print(f"Loaded: {INPUT_CSV}, rows: {len(df)}, columns: {len(df.columns)}")

    features = add_cyclical_features(df)

    new_cols = [c for c in features.columns if c not in df.columns]
    removed_cols = [c for c in df.columns if c not in features.columns]
    print(f"Added columns:   {new_cols}")
    print(f"Removed columns: {removed_cols}")
    print(f"Result: rows: {len(features)}, columns: {len(features.columns)}")
    print(f"NaN in total: {int(features.isna().sum().sum())}")

    features.to_csv(OUTPUT_CSV)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
