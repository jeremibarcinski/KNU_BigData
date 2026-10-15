## How to use the files

I grouped every file with the dataset it works on. Every important file has the dataset name at the beginning (for ASOS files I also added `A_` in front).

### Files corresponding to every dataset

| file | ASOS | ERA5 |
|---|---|---|
| python | `A_ASOS_01_fetch...` -> downloads ASOS data, <br>`A_ASOS_02_clean...` -> cleans ASOS data, <br>`A_ASOS_02_after_report...` -> removes columns that are not helpful (too many NaN = not good), <br>`A_ASOS_03_make_f...` -> adds features and converts hourly data into daily data | `01_fetch_daegu...` -> downloads ERA5 data, <br>`02_clean_open...` -> cleans ERA5 data (no NaN, because the model took care of them), <br>`03_make_features` -> adds features to the data, does not convert them into daily format |
| outputs | `A_ASOS_01...` -> `ASOS_01_Daegu...csv` `OUR DATA`, <br>`A_ASOS_02...` -> `ASOS_02_Daegu...csv` `cleaned data` + report, <br>`A_ASOS_02_after_report.py` -> `ASOS_022_...` `removes 2 columns` (too many NaN), <br>`A_ASOS_03...` -> `ASOS_03_Daegu_daily...` `features turned into daily data` + report | `01_fetch_daegu...` -> `ERA5_01...`, <br>`02_clean...` -> `ERA5_02...csv` data + `Daegu_clean_report.txt`, <br>`03_make_fea...` -> `ERA5_03_Daegu...` |
| other | `asos_raw` is the raw data used to build `ASOS_01_Daegu...csv`, which is our base dataset for ASOS and is used by all other steps | `Parameter_description.txt` is a file that describes the parameters in the ERA5 dataset |

`FOR MODELING DO NOT USE THE ERA5 DATASET !!!!` It is not ready for modeling yet. The only file needed for modeling is `ASOS_03_Daegu_daily_features.csv`.

## Stage 0: problem to solve

We want to predict whether a heatwave or a tropical night is going to occur.

### Conditions

| type | Advisory | Alarm | Major alert |
|---|---|---|---|
| heatwave | When the daily maximum perceived temperature is expected to remain at 33℃ or higher for two or more consecutive days:<br>`Tmax >= 33℃ for days >= 2` | When the daily maximum perceived temperature is expected to remain at 35℃ or higher for two or more consecutive days:<br>`Tmax >= 35℃ for days >= 2` | In areas where the daily maximum wind chill temperature is observed to be 35℃ or higher for three or more days:<br>`Tmax >= 35℃ for days >= 3` |
| tropical night | Tropical nights are defined as days when the nighttime minimum temperature is 25℃ or higher:<br>`Tmin >= 25℃` | none | none |

## Stage 1: Dataset

To satisfy the conditions we downloaded 2 datasets. First we used the ERA5 dataset from Open-Meteo, which is created by models based on satellite data. We combined two datasets: the Open-Meteo weather dataset and the Open-Meteo air quality dataset. Later in the project we decided to switch to the ASOS dataset from the KMA (Korea Meteorological Administration), which contains the official weather measurements in South Korea.

### Comparison

| compare | ASOS (KMA dataset) | ERA5 (Open-Meteo dataset) |
|---|---|---|
| data | measured by weather stations | retrieved from satellite-based models |
| number of variables available | | |
| number of variables picked | | |
| how many years available | | |
| years picked | from 2014 until now, because in 2013 the station's location was changed | from 2022 until now, because the air pollution dataset is available from 2022 |
| continuous variables | | |
| discrete variables | | |

## Stage 2: Cleaning data

This is not a very important step for modeling, but briefly:
- there were strange values in the dataset that we needed to replace, such as `-9.0` and `-99.0` (KMA codes for missing data)
- we put all of the data on a full hourly time axis
- we filled gaps shorter than 3 hours with linear interpolation:

```math
y = y_0 + \frac{y_1 - y_0}{x_1 - x_0} \cdot (x - x_0)
```

After the report:
- we had to remove two columns because they had too many missing values (NaN):
  - `cloud_cover_tenths`: NaN in 100% of 2014, 57% of 2015, ~22% of 2016-2017, 13% of 2018
  - `visibility_10m`: NaN in 100% of 2014, 35% of 2015

## Stage 3: daily features dataset (`ASOS_03_Daegu_daily_features.csv`)

Created by `src/03_make_features.py` from `ASOS_022_Daegu_after_report_clean.csv`.
A text report with counts per split and per year is saved next to it
(`ASOS_03_Daegu_daily_features_report.txt`).

### What the dataset is

- One row is one summer day **D** in Daegu (KMA station 143), May-September 2014-2026 (about 1976 rows).
- The features describe day D, using data only up to 23:00 of D.
- The targets (`y_*`) describe day **D+1**. The task is "predict tomorrow".
- The index is `date`, the date of D.

### Column groups

| group | columns | notes |
|---|---|---|
| Daily features | `temperature_*`, `feels_like_*`, `relative_humidity_*`, `dew_point_*` (min / max / mean), pressures, wind, solar, sunshine, ground temperature, precipitation, `temperature_range`, `feels_like_max_hour` | 26 columns |
| Memory features | `*_lag1/2/3/7`, `*_roll3/7`, `station_pressure_diff1/3`, `hot33_streak`, `hot35_streak`, `night_min_air_prev`, `tropical_night_streak_prev` | past only, nothing from the future |
| Season | `doy_sin`, `doy_cos` | day of the year as a circle |
| Targets | `y_tmax_feels` (regression); `y_tropical_night`, `y_heatwave_advisory`, `y_heatwave_advisory_night`, `y_heatwave_alarm`, `y_heatwave_alarm_extended` (binary 0/1) | day D+1 |
| Not a feature | `check_tmax_air_next_day`, `split` | see the rules below |

### Target definitions

- **Feels-like temperature**: KMA summer formula (effective from 2022-06-02) with Stull's wet-bulb temperature,
  computed hourly from air temperature and relative humidity, then aggregated to days.
- **Tropical night** (`y_tropical_night`): night minimum **air** temperature (18:01-09:00) >= 25 degC.
- **Heat wave advisory** (`y_heatwave_advisory`): daily max feels-like >= 33 degC for 2 or more consecutive days.
- **Advisory + night** (`y_heatwave_advisory_night`): advisory and night minimum air temperature >= 26 degC
  (regional threshold for Daegu).
- **Alarm** (`y_heatwave_alarm`): daily max feels-like >= 35 degC for 2 or more consecutive days.
- **Extended alarm** (`y_heatwave_alarm_extended`): daily max feels-like >= 35 degC for 3 or more consecutive days.
- **Regression target** (`y_tmax_feels`): daily maximum feels-like temperature of day D+1.

### Rules for using the data

1. **Features are every column except `y_*`, `check_tmax_air_next_day` and `split`.**
   `check_tmax_air_next_day` is tomorrow's air temperature, so it would be leakage as a feature.
   It exists only for sanity checks against official KMA values.
2. **Use the `split` column**: train 2014-2022, val 2023-2024, test 2025-2026.
   Never shuffle rows, because neighbouring days are strongly correlated.
3. **Tune on val and use test only once**, at the very end.
4. **Fit scalers and imputers on train only** (inside a pipeline). The file is stored unscaled.
5. **Missing values**: only a few remain (solar radiation, sunshine, precipitation).
   Tree models (XGBoost, LightGBM, `HistGradientBoosting`) handle NaN directly.
   For logistic regression or neural networks use median imputation fitted on train.
6. **Which targets to train**: `y_tropical_night`, `y_heatwave_advisory`, `y_heatwave_advisory_night`.
   The two alarm targets have almost no positive days (train: 0.4% and 0.0%; val and test: about 3 days each),
   so they are only described in the report, not trained or evaluated.

### Minimal loading example and train / val / test split

The split is chronological and stored in the `split` column (it is decided by the date D of the row,
and the cut points are the settings `TRAIN_END` and `VAL_END` in `03_make_features.py`).

| split | years (date D) | rows | role |
|---|---|---|---|
| `train` | 2014-2022 | 1368 | fit models, imputers and scalers |
| `val` | 2023-2024 | 304 | choose hyperparameters and the decision threshold |
| `test` | 2025-2026 | 304 | final evaluation, used once |

Share of positive days (%) in each split:

| target | train | val | test |
|---|---|---|---|
| `y_tropical_night` | 12.6 | 16.4 | 19.7 |
| `y_heatwave_advisory` | 9.3 | 15.1 | 21.4 |
| `y_heatwave_advisory_night` | 4.1 | 5.9 | 9.2 |

Rules:
- Rows are never shuffled, and no row of a later split is used to fit anything on an earlier one.
- The split falls between seasons (the data contain only May-September), so no summer is cut in two.
- The positive share grows from train to test (more hot days in recent years),
  so compare models on the same split and always report the positive share next to the metric.
- For the final model, after the choices are made on val, you may refit on train + val
  and then evaluate once on test.

```python
import pandas as pd

df = pd.read_csv("ASOS_03_Daegu_daily_features.csv", parse_dates=["date"], index_col="date")
y_cols = [c for c in df.columns if c.startswith("y_")]
X_cols = [c for c in df.columns if c not in y_cols + ["split", "check_tmax_air_next_day"]]

target = "y_heatwave_advisory"
train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
X_train, y_train = train[X_cols], train[target]
X_val, y_val = val[X_cols], val[target]
X_test, y_test = test[X_cols], test[target]
```

### Modelling advice

- **Class imbalance**: positives are rare (about 4-20% depending on target and split).
  Use `class_weight="balanced"` or `scale_pos_weight`.
- **Metrics**: do not use accuracy. Use PR-AUC, recall and precision (or F1) at a threshold chosen on val,
  and the Brier score for probabilities. Report the positive share per split next to every result.
- **Baselines to beat**: persistence (tomorrow equals today, e.g. `hot33_streak > 0`) and climatology (by day of year).
- **Tuning inside train**: expanding-window time-series cross-validation by year, not random K-fold.
- **Regression** (`y_tmax_feels`): report MAE and RMSE against the persistence baseline
  (`feels_like_max` of today).

### Known limitations

- Heat wave labels are runs of days, so the label of D+1 uses information from D+2
  (and D+3 for the extended alarm). This is part of the definition and should be mentioned in the report.
- Hourly data give daily maxima and minima slightly below the official KMA values (KMA uses 10-minute data),
  so days close to a threshold can be classified differently.
- Feels-like temperature is computed by us, not downloaded from KMA.
- 2026 ends on 29 September. The test set is small (304 days), so yearly positive counts are small and results are noisy.
