"""
Stage 3 (ASOS): from hourly cleaned data to a daily feature table (UrbanAtmos).

Input:   ASOS_022_Daegu_after_report_clean.csv   (output of clean_asos_after_report.py)
Output:  ASOS_03_Daegu_daily_features.csv + text report ASOS_03_Daegu_daily_features_report.txt

Task: on day D (using data up to 23:00 of day D) predict day D+1:
  - daily maximum feels-like temperature (regression)
  - tropical night (night minimum AIR temperature >= 25 degC)
  - heat wave advisory, alarm, extended alarm (KMA, feels-like based) and
    advisory + night (night minimum air temperature >= 26 degC, the Daegu regional threshold)

Steps:
  1. load, check the hourly axis
  2. hourly feels-like temperature (KMA summer formula, Stull wet-bulb temperature)
  3. daily aggregates (min / max / mean / sums, wind u/v) with a minimum-hours rule
  4. night window 18:01-09:00 (observations 19:00 D ... 09:00 D+1), assigned to date D
  5. targets, computed on the full continuous series BEFORE cutting the season
  6. features: lags, rolling means, pressure change, streak counters, day-of-year sin/cos
  7. targets shifted to D+1, season cut, chronological split
  8. report and save (always overwrites the output files)

Not done here: scaling (fit the scaler on the train part in stage 4).

Requirements:  pip install pandas numpy
Run:           python 03_make_features.py
"""

import numpy as np
import pandas as pd

# ---------- SETTINGS ----------
INPUT_CSV = "ASOS_022_Daegu_after_report_clean.csv"
OUTPUT_CSV = "ASOS_03_Daegu_daily_features.csv"
REPORT_TXT = "ASOS_03_Daegu_daily_features_report.txt"

# KMA definitions (feels-like temperature, daily values)
FEELS_ADVISORY = 33.0        # daily max feels-like for the advisory
FEELS_ALARM = 35.0           # daily max feels-like for the alarm
RUN_ADVISORY = 2             # advisory: at least this many consecutive days
RUN_ALARM = 2                # alarm: at least this many consecutive days
RUN_ALARM_EXTENDED = 3       # extended alarm: 2 days + the next day, all >= FEELS_ALARM
# Night minimum is always the AIR temperature (not feels-like)
TROPICAL_NIGHT_THRESHOLD = 25.0   # tropical night: night minimum air temperature >= 25
ADVISORY_NIGHT_THRESHOLD = 26.0   # advisory + night: Daegu regional threshold (KMA table)

# Night window 18:01-09:00 = hourly observations 19:00 ... 09:00 (15 values)
NIGHT_START_HOUR = 19
NIGHT_END_HOUR = 9

# Minimum number of valid hourly values for a daily / nightly value
MIN_HOURS_DAY = 20           # of 24
MIN_HOURS_NIGHT = 12         # of 15

LAGS = [1, 2, 3, 7]          # days
ROLLING = [3, 7]             # days
# Minimum number of valid days inside a rolling window (allows one or two missing days)
ROLLING_MIN_PERIODS = {3: 2, 7: 5}
SEASON_MONTHS = [5, 6, 7, 8, 9]   # applied to day D and to D+1; check in the report (step 7)

# Chronological split by the date D of the row
TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"       # test = everything after

report: list[str] = []


def log(text: str = "") -> None:
    """Prints a line to the console and keeps it for the report file."""
    print(text)
    report.append(text)


# ---------- STEP 1 ----------
def load_hourly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    log("1. LOADING")
    log(f"   Rows: {len(df)}, columns: {len(df.columns)}")
    log(f"   Range: {df.index.min()} -> {df.index.max()}")
    log(f"   Missing whole hours: {len(full.difference(df.index))}")
    return df.reindex(full)


# ---------- STEP 2 ----------
def wet_bulb_stull(ta: pd.Series, rh: pd.Series) -> pd.Series:
    """Wet-bulb temperature (degC), Stull's estimation formula (atan in radians)."""
    return (
        ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(ta + rh)
        - np.arctan(rh - 1.67633)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def feels_like_kma(ta: pd.Series, rh: pd.Series) -> pd.Series:
    """KMA summer feels-like (wind chill) temperature, formula effective from 2022-06-02."""
    tw = wet_bulb_stull(ta, rh)
    return -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0


def add_hourly_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["feels_like"] = feels_like_kma(df["temperature"], df["relative_humidity"])
    # wind vector (meteorological convention: direction the wind blows FROM); code 0 = calm
    angle = np.deg2rad(df["wind_direction_code"] * 10.0)
    calm = df["wind_direction_code"] == 0
    df["wind_u"] = (-df["wind_speed"] * np.sin(angle)).where(~calm, 0.0)
    df["wind_v"] = (-df["wind_speed"] * np.cos(angle)).where(~calm, 0.0)
    log("\n2. HOURLY FEELS-LIKE TEMPERATURE (KMA formula, Stull wet-bulb)")
    d = (df["feels_like"] - df["temperature"]).dropna()
    log(f"   feels_like range: {df['feels_like'].min():.1f} .. {df['feels_like'].max():.1f} degC")
    log(f"   feels_like - temperature: mean {d.mean():.2f}, max {d.max():.2f} degC")
    return df


# ---------- STEP 3 ----------
def daily_stat(s: pd.Series, how: str, min_count: int = MIN_HOURS_DAY) -> pd.Series:
    """Daily min / max / mean / sum; NaN if fewer than min_count valid hours."""
    g = s.resample("D")
    out = getattr(g, how)()
    return out.where(g.count() >= min_count)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=pd.date_range(df.index.min().normalize(),
                                         df.index.max().normalize(), freq="D"))
    for col, name in [("temperature", "temperature"), ("feels_like", "feels_like"),
                      ("relative_humidity", "relative_humidity"), ("dew_point", "dew_point")]:
        for how in ["min", "max", "mean"]:
            d[f"{name}_{how}"] = daily_stat(df[col], how)
    d["station_pressure_mean"] = daily_stat(df["station_pressure"], "mean")
    d["sea_level_pressure_mean"] = daily_stat(df["sea_level_pressure"], "mean")
    d["vapor_pressure_mean"] = daily_stat(df["vapor_pressure"], "mean")
    d["wind_speed_mean"] = daily_stat(df["wind_speed"], "mean")
    d["wind_speed_max"] = daily_stat(df["wind_speed"], "max")
    d["wind_u"] = daily_stat(df["wind_u"], "mean")
    d["wind_v"] = daily_stat(df["wind_v"], "mean")
    d["solar_radiation_sum"] = daily_stat(df["solar_radiation_mj"], "sum")
    d["sunshine_sum"] = daily_stat(df["sunshine_hours"], "sum")
    d["ground_temperature_mean"] = daily_stat(df["ground_temperature"], "mean")
    d["ground_temperature_max"] = daily_stat(df["ground_temperature"], "max")
    d["precipitation_sum"] = daily_stat(df["precipitation"], "sum")
    d["temperature_range"] = d["temperature_max"] - d["temperature_min"]

    # hour of the daily feels-like maximum
    day_key = df.index.floor("D")
    idx = df["feels_like"].fillna(-np.inf).groupby(day_key).idxmax()
    d["feels_like_max_hour"] = pd.Series(idx.dt.hour, index=idx.index).where(d["feels_like_max"].notna())

    log("\n3. DAILY AGGREGATES")
    log(f"   Days: {len(d)}, aggregate columns: {len(d.columns)} (minimum {MIN_HOURS_DAY} valid hours per day)")
    return d


# ---------- STEP 4 ----------
def build_nights(df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Night minimum air temperature 18:01-09:00 for every date D (night starts in the evening of D)."""
    h = df.index.hour
    in_night = (h >= NIGHT_START_HOUR) | (h <= NIGHT_END_HOUR)
    # 19:00 D ... 09:00 D+1 must give date D  ->  shift by 10 h before flooring
    key = (df.index - pd.Timedelta(hours=10)).floor("D")
    sub = df.loc[in_night]
    g = sub.groupby(key[in_night])
    out = pd.DataFrame(index=days)
    mn = g["temperature"].min()
    ok = g["temperature"].count() >= MIN_HOURS_NIGHT
    out["night_min_air"] = mn.where(ok).reindex(days)
    log("\n4. NIGHT WINDOW 18:01-09:00 (hourly values 19:00 D ... 09:00 D+1, date = D)")
    log(f"   Nights with a valid minimum air temperature: {int(out['night_min_air'].notna().sum())} of {len(out)}")
    return out


# ---------- STEP 5 ----------
def run_flag(cond: pd.Series, min_len: int) -> pd.Series:
    """True for every day that belongs to a run of >= min_len consecutive True days."""
    c = cond.fillna(False).astype(bool)
    grp = (c != c.shift()).cumsum()
    run_len = c.groupby(grp).transform("sum")
    return c & (run_len >= min_len)


def streak(cond: pd.Series) -> pd.Series:
    """Number of consecutive True days up to and including each day (0 if False)."""
    c = cond.fillna(False).astype(bool)
    grp = (c != c.shift()).cumsum()
    return c.groupby(grp).cumsum().astype(float)


def as_flag(boolean: pd.Series, valid: pd.Series) -> pd.Series:
    """0/1 float flag, NaN where the underlying value is unknown."""
    return boolean.astype(float).where(valid.notna())


def build_targets(daily: pd.DataFrame, nights: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(index=daily.index)
    tmax = daily["feels_like_max"]
    t["tmax_feels"] = tmax
    night_min = nights["night_min_air"]
    t["tropical_night"] = as_flag(night_min >= TROPICAL_NIGHT_THRESHOLD, night_min)

    adv = run_flag(tmax >= FEELS_ADVISORY, RUN_ADVISORY)
    alarm = run_flag(tmax >= FEELS_ALARM, RUN_ALARM)
    alarm_ext = run_flag(tmax >= FEELS_ALARM, RUN_ALARM_EXTENDED)
    t["heatwave_advisory"] = as_flag(adv, tmax)
    t["heatwave_alarm"] = as_flag(alarm, tmax)
    t["heatwave_alarm_extended"] = as_flag(alarm_ext, tmax)

    warm_night = night_min >= ADVISORY_NIGHT_THRESHOLD
    valid = tmax.where(night_min.notna())
    t["heatwave_advisory_night"] = as_flag(adv & warm_night, valid)

    log("\n5. TARGETS (computed on the full continuous series, same-day version, before the shift)")
    log(f"   Tropical night: night minimum AIR temperature >= {TROPICAL_NIGHT_THRESHOLD:g}")
    log(f"   Advisory: feels-like Tmax >= {FEELS_ADVISORY:g} for >= {RUN_ADVISORY} consecutive days")
    log(f"   Alarm: feels-like Tmax >= {FEELS_ALARM:g} for >= {RUN_ALARM} consecutive days")
    log(f"   Extended alarm: feels-like Tmax >= {FEELS_ALARM:g} for >= {RUN_ALARM_EXTENDED} consecutive days")
    log(f"   Advisory + night: advisory and night minimum air temperature >= {ADVISORY_NIGHT_THRESHOLD:g}")
    log("   Positive days per month, whole series (use it to choose the season):")
    months = pd.DataFrame({c: t[c].groupby(t.index.month).sum() for c in t.columns if c != "tmax_feels"})
    for line in months.fillna(0).astype(int).to_string().splitlines():
        log("       " + line)
    return t


# ---------- STEP 6 ----------
def build_features(daily: pd.DataFrame, nights: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    f = daily.copy()

    # lags (day D uses D-1, D-2, ...)
    lag_cols = ["feels_like_max", "feels_like_min", "temperature_max", "relative_humidity_mean",
                "station_pressure_mean"]
    for col in lag_cols:
        for k in LAGS:
            f[f"{col}_lag{k}"] = daily[col].shift(k)

    # rolling means, the window ends at day D (no future data)
    for col in ["feels_like_max", "temperature_mean", "solar_radiation_sum"]:
        for w in ROLLING:
            f[f"{col}_roll{w}"] = daily[col].rolling(w, min_periods=ROLLING_MIN_PERIODS[w]).mean()

    # pressure change
    f["station_pressure_diff1"] = daily["station_pressure_mean"].diff(1)
    f["station_pressure_diff3"] = daily["station_pressure_mean"].diff(3)

    # streak counters up to and including day D
    f["hot33_streak"] = streak(daily["feels_like_max"] >= FEELS_ADVISORY)
    f["hot35_streak"] = streak(daily["feels_like_max"] >= FEELS_ALARM)

    # night features use the PREVIOUS night (D-1 evening ... D morning): the night that
    # starts on day D ends at 09:00 of D+1, which is not known at the end of day D
    f["night_min_air_prev"] = nights["night_min_air"].shift(1)
    f["tropical_night_streak_prev"] = streak(targets["tropical_night"] == 1.0).shift(1)

    # season (day of year as a circle)
    doy = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    log("\n6. FEATURES")
    log(f"   Lags {LAGS} for: {', '.join(lag_cols)}")
    log(f"   Rolling means {ROLLING} (min valid days {ROLLING_MIN_PERIODS}) for: "
        "feels_like_max, temperature_mean, solar_radiation_sum")
    log("   Streak counters: hot33_streak, hot35_streak (up to and including D); "
        "tropical-night streaks and night minima from the previous night (D-1)")
    log(f"   Total feature columns: {len(f.columns)}")
    return f


# ---------- STEP 7 ----------
def assemble(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    names = {
        "tmax_feels": "y_tmax_feels",
        "tropical_night": "y_tropical_night",
        "heatwave_advisory": "y_heatwave_advisory",
        "heatwave_alarm": "y_heatwave_alarm",
        "heatwave_alarm_extended": "y_heatwave_alarm_extended",
        "heatwave_advisory_night": "y_heatwave_advisory_night",
    }
    # the row of day D describes day D+1
    y = targets[list(names)].shift(-1).rename(columns=names)
    df = pd.concat([features, y], axis=1)
    target_cols = list(names.values())
    # tmax_air of the target day, for sanity checks only (not a target)
    df["check_tmax_air_next_day"] = features["temperature_max"].shift(-1)

    log("\n7. SHIFT TO D+1, SEASON CUT, SPLIT")
    log("   Note: heat-wave labels are runs of consecutive days, so the label of D+1 uses")
    log("   information from D+2 (and D+3 for the extended alarm) - this is part of the definition.")

    n_all = len(df)
    next_day = df.index + pd.Timedelta(days=1)
    in_season = df.index.month.isin(SEASON_MONTHS) & next_day.month.isin(SEASON_MONTHS)
    df = df.loc[in_season]
    log(f"   Season months {SEASON_MONTHS} (day D and day D+1): {len(df)} of {n_all} days")

    n_before = len(df)
    df = df.dropna(subset=target_cols)
    log(f"   Dropped days with an unknown target: {n_before - len(df)}")

    df["split"] = np.where(df.index <= pd.Timestamp(TRAIN_END), "train",
                           np.where(df.index <= pd.Timestamp(VAL_END), "val", "test"))
    df.index.name = "date"
    return df


# ---------- STEP 8 ----------
def report_result(df: pd.DataFrame) -> None:
    y_cols = [c for c in df.columns if c.startswith("y_") and c != "y_tmax_feels"]
    log("\n8. RESULT")
    log(f"   Rows: {len(df)}, columns: {len(df.columns)}")
    log(f"   Range: {df.index.min().date()} -> {df.index.max().date()}")

    log("   Rows per split: " + ", ".join(f"{k}: {v}" for k, v in df["split"].value_counts().items()))
    log("   Positive share per split (%):")
    share = (df.groupby("split")[y_cols].mean() * 100).round(1).reindex(["train", "val", "test"])
    for line in share.to_string().splitlines():
        log("       " + line)

    log("   Positive days per year (of days in the season):")
    per_year = df.groupby(df.index.year)[y_cols].sum().astype(int)
    per_year["days"] = df.groupby(df.index.year).size()
    for line in per_year.to_string().splitlines():
        log("       " + line)

    log("   NaN per feature column (only columns with NaN):")
    feat = df.drop(columns=[c for c in df.columns if c.startswith("y_") or c == "split"])
    nan = feat.isna().sum()
    nan = nan[nan > 0]
    if nan.empty:
        log("       none")
    for col, n in nan.items():
        log(f"       {col}: {int(n)}")

    log("   Note: hourly data give daily maxima/minima slightly below the official KMA values")
    log("   (KMA uses 10-minute data), so days close to a threshold can be classified differently.")


def main() -> None:
    hourly = load_hourly(INPUT_CSV)
    hourly = add_hourly_derived(hourly)
    daily = build_daily(hourly)
    nights = build_nights(hourly, daily.index)
    targets = build_targets(daily, nights)
    features = build_features(daily, nights, targets)
    df = assemble(features, targets)
    report_result(df)

    df.to_csv(OUTPUT_CSV)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"\n   Saved: {OUTPUT_CSV} and {REPORT_TXT}")


if __name__ == "__main__":
    main()