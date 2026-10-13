"""
Stage 2 (ASOS): cleaning hourly KMA ASOS observations for Daegu (UrbanAtmos).

Input:   Daegu_asos_hourly.csv from fetch_daegu_asos.py (RAW, KMA missing-value codes kept)
Output:  Daegu_asos_clean.csv + text report Daegu_asos_clean_report.txt

Steps:
  1. load, sort, drop duplicate timestamps, check the station number
  2. full hourly grid (reindex), count missing hours
  3. keep the useful columns, rename them (with units), handle KMA missing codes:
       - -9 / -99 in columns where negative values are impossible -> NaN
       - temperature columns: -99 -> NaN; -9.0 is NaN only if it is a code and not a
         real temperature (checked against neighbouring hours and the season)
       - solar radiation and sunshine: code at night -> 0 (sun below the horizon),
         code during the day -> NaN
       - precipitation: "no rain" code -> 0 (April-October, and the 3-hourly slots
         from November to March), everything else -> NaN
  4. physical range checks (impossible values -> NaN)
  5. missing-data report per column and per year (BEFORE cutting the period)
  6. fill short gaps (<= MAX_GAP_HOURS) by linear interpolation, smooth variables only
  7. cut to the analysis window (the station moved in September 2013)
  8. extreme values and stuck sensors: report only, nothing is removed
  9. save (always overwrites the output files)

Not done here (stage 3): lags, time features, wind direction sin/cos, target
(Tmax / Tmin / tropical nights), scaling, train/test split.

Requirements:  pip install pandas numpy
Run:           python clean_asos.py
"""

import numpy as np
import pandas as pd

# ---------- SETTINGS ----------
INPUT_CSV = "ASOS_01_Daegu_asos_hourly.csv"
OUTPUT_CSV = "ASOS_02_Daegu_asos_clean.csv"
REPORT_TXT = "ASOS_02_Daegu_asos_clean_report.txt"

STATION = 143
ANALYSIS_START = "2014-01-01"   # station 143 moved in September 2013; None = keep all years
LATITUDE = 35.88                # approximate, used only to tell day from night
LONGITUDE = 128.65
MAX_GAP_HOURS = 3               # longest gap that is filled by interpolation (0 = off)
OUTLIER_QUANTILE = 0.999        # threshold for the extreme-value report
TOP_N_OUTLIERS = 5              # how many largest values to list per column
STUCK_HOURS = 12                # identical values for this many hours = suspicious
SUMMER_MONTHS = [5, 6, 7, 8, 9]  # a temperature of exactly -9.0 is impossible here
MISSING_CODES = [-9, -99]       # KMA codes for "no data"

# KMA name -> our name (unit). Everything else is dropped.
COLUMN_MAP = {
    "TA": "temperature",             # degC
    "TD": "dew_point",               # degC
    "HM": "relative_humidity",       # %
    "PV": "vapor_pressure",          # hPa
    "PA": "station_pressure",        # hPa
    "PS": "sea_level_pressure",      # hPa
    "WS": "wind_speed",              # m/s (Open-Meteo gives km/h by default!)
    "WD": "wind_direction_code",     # 36 directions, x10 = degrees, 0 = calm
    "CA_TOT": "cloud_cover_tenths",  # 0-10
    "VS": "visibility_10m",          # units of 10 m
    "SS": "sunshine_hours",          # hours of sunshine within the hour (0-1)
    "SI": "solar_radiation_mj",      # MJ/m2 per hour (1 MJ/m2 per hour = 277.8 W/m2)
    "TS": "ground_temperature",      # degC
    "RN": "precipitation",           # mm (hourly Apr-Oct, 3-hourly Nov-Mar)
    "IR": "precip_flag",             # 1/4 = precipitation data, 3 = no rain OR missing
}

# Temperature-like columns: (-9.0 can be real) -> tolerance in degC for the neighbour check
TEMP_LIKE = {"temperature": 4.0, "dew_point": 6.0, "ground_temperature": 6.0}
# Columns where any negative value is impossible, so -9 / -99 are always codes
NO_NEGATIVE = [
    "relative_humidity", "vapor_pressure", "station_pressure", "sea_level_pressure",
    "wind_speed", "wind_direction_code", "cloud_cover_tenths", "visibility_10m",
]
SOLAR = ["solar_radiation_mj", "sunshine_hours"]
# Smooth variables where short gaps are interpolated
INTERPOLATE = [
    "temperature", "dew_point", "relative_humidity", "vapor_pressure",
    "station_pressure", "sea_level_pressure", "wind_speed", "ground_temperature",
]

# Physical ranges (min, max); None = no limit on that side
PHYSICAL_RANGES = {
    "temperature": (-40, 50),
    "dew_point": (-50, 35),
    "relative_humidity": (0, 100),
    "vapor_pressure": (0, 100),
    "station_pressure": (900, 1100),
    "sea_level_pressure": (950, 1100),
    "wind_speed": (0, 60),
    "wind_direction_code": (0, 36),
    "cloud_cover_tenths": (0, 10),
    "visibility_10m": (0, 10000),
    "sunshine_hours": (0, 1),
    "solar_radiation_mj": (0, 4.5),
    "ground_temperature": (-40, 80),
    "precipitation": (0, 300),
}
DEW_POINT_TOLERANCE = 0.5  # degC, margin for rounding

report: list[str] = []


def log(text: str = "") -> None:
    """Prints a line to the console and keeps it for the report file."""
    print(text)
    report.append(text)


# ---------- SUN POSITION (only to tell day from night) ----------
def solar_elevation_deg(times_kst: pd.DatetimeIndex) -> np.ndarray:
    """Solar elevation in degrees for naive KST timestamps (NOAA approximation)."""
    t_utc = times_kst - pd.Timedelta(hours=9)
    doy = np.asarray(t_utc.dayofyear)
    hour_utc = np.asarray(t_utc.hour) + np.asarray(t_utc.minute) / 60
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour_utc - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    solar_minutes = hour_utc * 60 + eqtime + 4 * LONGITUDE
    hour_angle = np.deg2rad(solar_minutes / 4 - 180)
    lat = np.deg2rad(LATITUDE)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return 90 - np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))


def is_night(times_kst: pd.DatetimeIndex) -> np.ndarray:
    """True if the sun was below the horizon during the whole hour that ends at the timestamp."""
    now = solar_elevation_deg(times_kst)
    before = solar_elevation_deg(times_kst - pd.Timedelta(hours=1))
    return np.maximum(now, before) < -0.833


# ---------- STEP 1 ----------
def load_and_sort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    n_raw = len(df)
    df = df.sort_values("time")
    n_dupes = int(df.duplicated(subset="time").sum())
    df = df.drop_duplicates(subset="time", keep="last").set_index("time")
    log("1. LOADING")
    log(f"   Rows in file: {n_raw}, duplicate timestamps removed: {n_dupes}")
    log(f"   Range: {df.index.min()} -> {df.index.max()}")
    if "STN" in df.columns:
        other = int((df["STN"] != STATION).sum())
        log(f"   Rows from a station other than {STATION}: {other}")
        df = df[df["STN"] == STATION]
    return df


# ---------- STEP 2 ----------
def ensure_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    missing = full_index.difference(df.index)
    log("\n2. TIME AXIS CONTINUITY")
    log(f"   Expected hours: {len(full_index)}, missing whole hours: {len(missing)}")
    if len(missing) > 0:
        by_year = pd.Series(missing.year).value_counts().sort_index()
        log("   Missing hours per year: " + ", ".join(f"{y}: {n}" for y, n in by_year.items()))
    return df.reindex(full_index)


# ---------- STEP 3 ----------
def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    absent = [c for c in COLUMN_MAP if c not in df.columns]
    if absent:
        raise KeyError(f"Columns missing in the input file: {absent}")
    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    log("\n3. COLUMNS AND KMA MISSING-VALUE CODES (-9 / -99)")
    log("   Kept (KMA -> ours): " + ", ".join(f"{k}->{v}" for k, v in COLUMN_MAP.items()))
    return df


def handle_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    # 3a: negative values are impossible, so the codes are always "no data"
    for col in NO_NEGATIVE:
        mask = df[col].isin(MISSING_CODES)
        df.loc[mask, col] = np.nan
        log(f"   {col}: {int(mask.sum())} codes -> NaN")

    # 3b: temperatures, where -9.0 may be a real value
    summer = df.index.month.isin(SUMMER_MONTHS)
    for col, tol in TEMP_LIKE.items():
        s = df[col].mask(df[col] <= -90)                 # -99.0 is always a code
        n_99 = int(df[col].le(-90).sum())
        cand = s == -9.0
        s_clean = s.mask(cand)
        ref = pd.concat([s_clean.ffill(limit=6), s_clean.bfill(limit=6)], axis=1).mean(axis=1)
        far = (s - ref).abs() > tol                       # NaN reference -> False
        bad = cand & (far | summer)
        undecided = cand & ref.isna() & ~summer
        df[col] = s.mask(bad)
        log(
            f"   {col}: -99 codes: {n_99}; value -9.0: {int(cand.sum())} "
            f"(code -> NaN: {int(bad.sum())}, kept as real: {int((cand & ~bad).sum())}, "
            f"undecided and kept: {int(undecided.sum())})"
        )

    # 3c: solar radiation and sunshine: the code means "sun is down" at night
    night = is_night(df.index)
    for col in SOLAR:
        code = df[col].isin(MISSING_CODES)
        df.loc[code & night, col] = 0.0
        df.loc[code & ~night, col] = np.nan
        log(
            f"   {col}: night code -> 0: {int((code & night).sum())}, "
            f"daytime code -> NaN: {int((code & ~night).sum())}"
        )

    # 3d: precipitation. IR == 3 means "no precipitation OR missing" (assumed: no rain).
    # April-October RN is hourly; November-March RN exists only every 3 hours.
    months = df.index.month
    hours = df.index.hour
    hourly_season = np.isin(months, range(4, 11))
    three_hourly_slot = ~hourly_season & (hours % 3 == 0)
    code = df["precipitation"].isin(MISSING_CODES)
    dry = code & (df["precip_flag"] == 3) & (hourly_season | three_hourly_slot)
    df.loc[dry, "precipitation"] = 0.0
    df.loc[code & ~dry, "precipitation"] = np.nan
    log(
        f"   precipitation: 'no rain' code -> 0: {int(dry.sum())}, "
        f"other codes -> NaN: {int((code & ~dry).sum())}"
    )
    flags = df["precip_flag"].value_counts(dropna=False).sort_index()
    log("   precip_flag counts: " + ", ".join(f"{k}: {v}" for k, v in flags.items()))
    return df.drop(columns="precip_flag")


# ---------- STEP 4 ----------
def apply_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    log("\n4. PHYSICAL RANGE CHECKS (impossible values -> NaN)")
    total = 0
    for col, (lo, hi) in PHYSICAL_RANGES.items():
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

    bad = df["dew_point"] > df["temperature"] + DEW_POINT_TOLERANCE
    n_bad = int(bad.sum())
    if n_bad:
        df.loc[bad, "dew_point"] = np.nan
        log(f"   dew_point: {n_bad} above temperature (+{DEW_POINT_TOLERANCE} degC)")
        total += n_bad

    log(f"   Total values judged impossible: {total}")
    return df


# ---------- STEP 5 ----------
def report_missing(df: pd.DataFrame) -> None:
    log("\n5. MISSING DATA (NaN) AFTER CODES AND RANGE CHECKS, BEFORE FILLING")
    share = df.isna().mean() * 100
    log("   Share of NaN per column, whole file (%):")
    for col, val in share.items():
        log(f"       {col}: {val:.1f}")
    by_year = (df.isna().groupby(df.index.year).mean() * 100).round(1).T
    log("   Share of NaN per year (%), variables in rows, years in columns:")
    for line in by_year.to_string().splitlines():
        log("       " + line)
    log("   Note: precipitation has NaN by design from November to March (3-hourly data).")


# ---------- STEP 6 ----------
def fill_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, int]:
    """Interpolates gaps of at most max_gap hours; longer gaps stay untouched."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_gap)
    interpolated = s.interpolate(method="time", limit_area="inside")
    filled = s.where(~short, interpolated)
    return filled, int((short & interpolated.notna()).sum())


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    log(f"\n6. SHORT GAPS (<= {MAX_GAP_HOURS} h) FILLED BY LINEAR INTERPOLATION")
    if MAX_GAP_HOURS <= 0:
        log("   Interpolation is switched off.")
        return df
    for col in INTERPOLATE:
        df[col], n = fill_short_gaps(df[col], MAX_GAP_HOURS)
        log(f"   {col}: {n} values filled")
    return df


# ---------- STEP 7 ----------
def cut_period(df: pd.DataFrame) -> pd.DataFrame:
    log("\n7. ANALYSIS WINDOW")
    if ANALYSIS_START is None:
        log("   No cut, all years kept.")
        return df
    n_before = len(df)
    df = df.loc[ANALYSIS_START:]
    log(f"   Kept from {ANALYSIS_START}: {len(df)} of {n_before} rows")
    log(f"   Range: {df.index.min()} -> {df.index.max()}")
    return df


# ---------- STEP 8 ----------
def report_outliers(df: pd.DataFrame) -> None:
    log(f"\n8. EXTREME VALUES (report only, above the {OUTLIER_QUANTILE * 100:g}th percentile)")
    skip = {"wind_direction_code"}  # circular variable, "largest" has no meaning
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


def report_stuck(df: pd.DataFrame) -> None:
    log(f"\n   STUCK SENSOR CHECK (>= {STUCK_HOURS} identical hours, report only)")
    for col in ["temperature", "station_pressure"]:
        s = df[col]
        run_id = s.ne(s.shift()).cumsum()
        run_len = s.groupby(run_id).transform("size")
        stuck = (run_len >= STUCK_HOURS) & s.notna()
        starts = s[stuck].index.to_series().groupby(run_id[stuck]).first()
        log(f"   {col}: {len(starts)} runs, {int(stuck.sum())} rows")
        for ts in starts.head(TOP_N_OUTLIERS):
            log(f"       run starts {ts}")


# ---------- MAIN ----------
def main() -> None:
    df = load_and_sort(INPUT_CSV)
    df = ensure_hourly_grid(df)
    df = select_and_rename(df)
    df = handle_missing_codes(df)
    df = apply_physical_ranges(df)
    report_missing(df)
    df = fill_gaps(df)
    df = cut_period(df)
    report_outliers(df)
    report_stuck(df)

    log("\n9. RESULT")
    log(f"   Rows: {len(df)}, columns: {len(df.columns)}")
    log("   Remaining NaN per column:")
    for col, n in df.isna().sum().items():
        log(f"       {col}: {int(n)}")

    df.to_csv(OUTPUT_CSV)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"   Saved: {OUTPUT_CSV} and {REPORT_TXT}")


if __name__ == "__main__":
    main()
