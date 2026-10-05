"""
Stage 2: cleaning hourly Open-Meteo data (UrbanAtmos).

Input:   CSV from fetch_seoul_open_meteo.py (Daegu_open_meteo_hourly.csv)
Output:  Daegu_clean.csv + text report Daegu_clean_report.txt

Steps:
  1. load, sort, drop duplicate timestamps, set time as index
  2. full hourly grid (reindex), count missing hours
  3. physical range checks (impossible values -> NaN)
  4. outliers: report only, nothing is removed
  5. save (always overwrites the output files)

Not done here (stage 3): lags, time features, wind direction sin/cos,
scaling, train/test split.

Requirements:  pip install pandas numpy
Run:           python clean_open_meteo.py
"""

import numpy as np
import pandas as pd

# ---------- SETTINGS ----------
# Change these paths once you set up data/raw and data/processed folders
INPUT_CSV = "Daegu_open_meteo_hourly.csv"
OUTPUT_CSV = "Daegu_clean.csv"
REPORT_TXT = "Daegu_clean_report.txt"

OUTLIER_QUANTILE = 0.999   # threshold for the extreme-value report
TOP_N_OUTLIERS = 5         # how many largest values to list per column

# Physical ranges (min, max); None = no limit on that side
PHYSICAL_RANGES = {
    "temperature_2m": (-60, 60),
    "apparent_temperature": (-80, 80),
    "relative_humidity_2m": (0, 100),
    "dew_point_2m": (-80, 40),
    "surface_pressure": (800, 1100),
    "precipitation": (0, None),
    "wind_speed_10m": (0, None),
    "wind_direction_10m": (0, 360),
    "cloud_cover": (0, 100),
    "shortwave_radiation": (0, None),
    "pm2_5": (0, None),
    "pm10": (0, None),
    "nitrogen_dioxide": (0, None),
    "ozone": (0, None),
    "sulphur_dioxide": (0, None),
    "carbon_monoxide": (0, None),
}

DEW_POINT_TOLERANCE = 0.5  # °C, margin for rounding

report: list[str] = []


def log(text: str = "") -> None:
    """Prints a line to the console and keeps it for the report file."""
    print(text)
    report.append(text)


# ---------- STEP 1 ----------
def load_and_sort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    n_raw = len(df)
    df = df.sort_values("time")
    n_dupes = int(df.duplicated(subset="time").sum())
    df = df.drop_duplicates(subset="time", keep="first").set_index("time")
    log("1. LOADING")
    log(f"   Rows in file: {n_raw}, duplicate timestamps removed: {n_dupes}")
    log(f"   Range: {df.index.min()} -> {df.index.max()}")
    return df


# ---------- STEP 2 ----------
def ensure_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    missing = full_index.difference(df.index)
    log("\n2. TIME AXIS CONTINUITY")
    log(f"   Expected hours: {len(full_index)}, missing whole hours: {len(missing)}")
    if len(missing) > 0:
        log(f"   First missing: {[str(t) for t in missing[:5]]}")
    return df.reindex(full_index)


# ---------- STEP 3 ----------
def apply_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    log("\n3. PHYSICAL RANGE CHECKS (impossible values -> NaN)")
    total = 0
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col not in df.columns:
            continue
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        n_bad = int(bad.sum())
        if n_bad:
            df.loc[bad, col] = np.nan
            log(f"   {col}: {n_bad} outside range [{lo}, {hi}]")
            total += n_bad

    if {"dew_point_2m", "temperature_2m"} <= set(df.columns):
        bad = df["dew_point_2m"] > df["temperature_2m"] + DEW_POINT_TOLERANCE
        n_bad = int(bad.sum())
        if n_bad:
            df.loc[bad, "dew_point_2m"] = np.nan
            log(f"   dew_point_2m: {n_bad} above temperature (+{DEW_POINT_TOLERANCE} °C)")
            total += n_bad

    log(f"   Total values judged impossible: {total}")
    return df


# ---------- STEP 4 ----------
def report_outliers(df: pd.DataFrame) -> None:
    log(f"\n4. EXTREME VALUES (report only, above the {OUTLIER_QUANTILE * 100:g}th percentile)")
    skip = {"wind_direction_10m"}  # circular variable, "largest" has no meaning
    for col in df.columns:
        if col in skip:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        threshold = s.quantile(OUTLIER_QUANTILE)
        extreme = s[s > threshold]
        top = extreme.nlargest(TOP_N_OUTLIERS)
        log(f"   {col}: threshold {threshold:.2f}, extreme count: {len(extreme)}, max {s.max():.2f}")
        for ts, val in top.items():
            log(f"       {ts}  {val:.2f}")


# ---------- MAIN ----------
def main() -> None:
    df = load_and_sort(INPUT_CSV)
    df = ensure_hourly_grid(df)
    df = apply_physical_ranges(df)
    report_outliers(df)

    log("\n5. RESULT")
    log(f"   Rows: {len(df)}, columns: {len(df.columns)}")
    log(f"   Remaining NaN in total: {int(df.isna().sum().sum())}")

    df.to_csv(OUTPUT_CSV)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"   Saved: {OUTPUT_CSV} and {REPORT_TXT}")


if __name__ == "__main__":
    main()