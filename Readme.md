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
| heatwave | When the daily maximum perceived temperature is expected to remain at 33℃ or higher for two or more consecutive days:<br>`Tmax >= 33℃ for days >= 2` | When the daily maximum perceived temperature is expected to remain at 35℃ or higher for two or more consecutive days:<br>`Tmax >= 35℃ for days >= 2` | In areas where the daily maximum perceived temperature is observed to be 35℃ or higher for three or more days:<br>`Tmax >= 35℃ for days >= 3` |
| tropical night | Tropical nights are defined as days when the nighttime minimum temperature is 25℃ or higher:<br>`Tmin >= 25℃` | none | none |

## Stage 1: Dataset

To satisfy the conditions we downloaded 2 datasets. First we used the ERA5 dataset from Open-Meteo, which is created by models based on satellite data. We combined two datasets: the Open-Meteo weather dataset and the Open-Meteo air quality dataset. Later in the project we decided to switch to the ASOS dataset from the KMA (Korea Meteorological Administration), which contains the official weather measurements in South Korea.

### Comparison

| compare | ASOS (KMA dataset) | ERA5 (Open-Meteo dataset) |
|---|---|---|
| data | measured by weather stations | retrieved from satellite-based models |
| number of variables available | 46 fields per hourly row in the KMA API (44 without time and station number) | many more than we used (the Open-Meteo weather archive and air-quality APIs offer dozens of hourly variables); not counted |
| number of variables picked | 15 read in cleaning (precipitation flag `IR` used only to clean precipitation) -> 14 in `ASOS_02` -> **12** in `ASOS_022` after removing `cloud_cover_tenths` and `visibility_10m` | **16**: 10 weather (`temperature_2m`, `apparent_temperature`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, `precipitation`, `wind_speed_10m`, `wind_direction_10m`, `cloud_cover`, `shortwave_radiation`) + 6 air quality (`pm2_5`, `pm10`, `nitrogen_dioxide`, `ozone`, `sulphur_dioxide`, `carbon_monoxide`) |
| how many years available | downloaded 2000-01-01 -> 2026-10-05 (27 years) | weather downloaded 2020-01-01 -> 2026-10-05; air quality only from 2022-08-04 |
| years picked | from 2014 until now, because in 2013 the station's location was changed | from 2023 until now (cleaned file: 2023-01-01 -> 2026-09-26), because the air pollution data start in August 2022, so 2023 is the first full year |
| continuous variables | 11: temperature, dew point, relative humidity, vapor pressure, station pressure, sea-level pressure, wind speed, sunshine, solar radiation, ground temperature, precipitation | all 16 (wind direction is in degrees 0-360, a circular continuous variable) |
| discrete variables | 1: `wind_direction_code` (16 compass directions + calm, stored as a code; turned into wind u / v components in stage 3) | none |

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

Created by `A_ASOS_03_make_features.py` from `ASOS_022_Daegu_after_report_clean.csv`.
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
and the cut points are the settings `TRAIN_END` and `VAL_END` in `A_ASOS_03_make_features.py`).

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
  Handle it with the decision threshold chosen on val; `class_weight="balanced"` / `scale_pos_weight`
  can be tried as a tuning option (in stage 4 it did not help and made the probabilities worse).
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

## Stage 4: Modeling

Use only `ASOS_03_Daegu_daily_features.csv` (not ERA5). Follow the rules from Stage 3 (chronological `split`, no shuffling, test used once).

### Goal

Predict tomorrow (day D+1) from the data of day D:

| task | target | type |
|---|---|---|
| tropical night | `y_tropical_night` | classification (0/1) |
| heatwave advisory | `y_heatwave_advisory` | classification (0/1) |
| heatwave advisory + night | `y_heatwave_advisory_night` | classification (0/1) |
| max feels-like temperature | `y_tmax_feels` | regression (optional) |

The alarm targets are not trained (almost no positive days).

### Step by step

1. **Load and split.** Use the loading example from Stage 3. Train on `train`, tune on `val`, keep `test` closed.
2. **Baselines first.** Every model must beat them:
   - persistence: tomorrow is like today (e.g. `hot33_streak > 0` for the advisory, `night_min_air_prev >= 25` for the tropical night),
   - climatology: the share of positive days for the same day of the year in train.
3. **Simple model.** Logistic regression in a pipeline: median imputer -> scaler -> model (class weighting none / balanced is tuned). Fit the pipeline on train only.
4. **Tree model.** `HistGradientBoosting`, XGBoost or LightGBM. They handle NaN, so no imputer is needed. Class weighting (`class_weight` / `scale_pos_weight`) is tuned as an option.
5. **Tuning.** Tune hyperparameters with expanding-window cross-validation by year inside train (for example: train 2014-2016 -> validate 2017, train 2014-2017 -> validate 2018, and so on). No random K-fold. Keep the search small, because there are only about 1368 training rows.
6. **Check on val.** Compare all models on `val` with the same metrics.
7. **Choose the threshold on val.** The default 0.5 is usually bad for rare events. Pick the threshold on val (for example the best F1, or a minimum recall such as 0.8) and then keep it fixed.
8. **Final model.** After all choices are made, you may refit on train + val with the chosen hyperparameters and threshold.
9. **Test once.** Evaluate on `test` a single time. Do not change anything after seeing the test result.

### Metrics

- Do not use accuracy.
- Classification: PR-AUC, precision, recall, F1 (at the threshold chosen on val), Brier score.
- Regression: MAE and RMSE, compared with persistence (`feels_like_max` of today).
- Always write the positive share of the split next to the metric.

### How to run

```bash
pip install pandas numpy scikit-learn matplotlib lightgbm xgboost
brew install libomp          # macOS only, needed by lightgbm and xgboost
python A_ASOS_04_modeling.py
```

Outputs: `ASOS_04_model_results.csv` (all metrics), `ASOS_04_modeling_report.txt` (tuning, thresholds,
feature importance, errors) and `ASOS_04_pr_curves_val.png` (precision-recall curves on val).
The setting `EVALUATE_TEST` was `False` while all choices were made on val; the final run used `True` once.
Do not change the script and re-run the test.

### Choices made on val

- **Models compared**: logistic regression, `HistGradientBoosting`, LightGBM, XGBoost, each tuned with
  expanding-window CV by year inside train (PR-AUC).
- **Class weighting is a tuned option** (none / balanced), not always on. Balanced weights did not improve
  the ranking (PR-AUC) and made the probabilities too high (logistic regression Brier on val 0.164 -> 0.080
  without weights). The imbalance is handled by the threshold instead.
- **Threshold**: the highest threshold with **recall >= 0.8 on val**. For a heat warning a missed hot day
  is worse than a false alarm. Persistence is a fixed 0/1 rule.
- **Best model per target** (highest PR-AUC on val, also best in CV): tropical night -> logistic regression,
  advisory -> LightGBM, advisory + night -> XGBoost. Refitted on train + val before the test.

### Results

Test = 2025-2026, evaluated once. Precision, recall and F1 use the threshold chosen on val.

| target | model | split | positive share (%) | PR-AUC | precision | recall | F1 | Brier |
|---|---|---|---|---|---|---|---|---|
| `y_tropical_night` | persistence | val | 16.4 | 0.447 | 0.620 | 0.620 | 0.620 | 0.125 |
| `y_tropical_night` | climatology | val | 16.4 | 0.542 | 0.377 | 0.800 | 0.513 | 0.105 |
| `y_tropical_night` | logistic regression | val | 16.4 | 0.742 | 0.519 | 0.800 | 0.630 | 0.080 |
| `y_tropical_night` | LightGBM | val | 16.4 | 0.579 | 0.435 | 0.800 | 0.563 | 0.103 |
| `y_tropical_night` | persistence | test | 19.7 | 0.439 | 0.600 | 0.600 | 0.600 | 0.158 |
| `y_tropical_night` | climatology | test | 19.7 | 0.502 | 0.456 | 0.867 | 0.598 | 0.131 |
| `y_tropical_night` | **best: logistic regression** | test | 19.7 | **0.706** | 0.567 | 0.850 | **0.680** | **0.089** |
| `y_heatwave_advisory` | persistence | val | 15.1 | 0.642 | 0.745 | 0.826 | 0.784 | 0.069 |
| `y_heatwave_advisory` | climatology | val | 15.1 | 0.710 | 0.356 | 0.804 | 0.493 | 0.090 |
| `y_heatwave_advisory` | logistic regression | val | 15.1 | 0.893 | 0.804 | 0.804 | 0.804 | 0.044 |
| `y_heatwave_advisory` | LightGBM | val | 15.1 | 0.912 | 0.860 | 0.804 | 0.831 | 0.044 |
| `y_heatwave_advisory` | persistence | test | 21.4 | 0.686 | 0.783 | 0.831 | 0.806 | 0.086 |
| `y_heatwave_advisory` | climatology | test | 21.4 | 0.629 | 0.519 | 0.862 | 0.647 | 0.136 |
| `y_heatwave_advisory` | **best: LightGBM** | test | 21.4 | **0.796** | 0.740 | 0.831 | 0.783 | **0.073** |
| `y_heatwave_advisory_night` | persistence | val | 5.9 | 0.230 | 0.444 | 0.444 | 0.444 | 0.066 |
| `y_heatwave_advisory_night` | climatology | val | 5.9 | 0.406 | 0.417 | 0.833 | 0.556 | 0.044 |
| `y_heatwave_advisory_night` | XGBoost | val | 5.9 | 0.518 | 0.405 | 0.833 | 0.545 | 0.041 |
| `y_heatwave_advisory_night` | persistence | test | 9.2 | 0.222 | 0.423 | 0.393 | 0.407 | 0.105 |
| `y_heatwave_advisory_night` | climatology | test | 9.2 | 0.334 | 0.368 | 0.500 | 0.424 | 0.075 |
| `y_heatwave_advisory_night` | **best: XGBoost** | test | 9.2 | **0.579** | 0.411 | 0.821 | **0.548** | **0.056** |

All other val rows (all four models per target) are in `ASOS_04_model_results.csv`.

Regression `y_tmax_feels` (degC):

| model | val MAE | val RMSE | test MAE | test RMSE |
|---|---|---|---|---|
| persistence (today's `feels_like_max`) | 1.63 | 2.16 | 1.64 | 2.21 |
| ridge | 1.46 | 1.87 | 1.46 | 1.91 |
| gradient boosting | 1.43 | 1.80 | 1.44 | 1.92 |

### Reading the results

- **Tropical night**: the model clearly beats both baselines on test (PR-AUC 0.71 vs 0.50, F1 0.68 vs 0.60)
  and keeps recall >= 0.8.
- **Heatwave advisory**: the model ranks days better than persistence (PR-AUC 0.80 vs 0.69) and gives better
  probabilities (Brier 0.073 vs 0.086), but at the fixed threshold its F1 (0.78) is slightly **below**
  persistence (0.81): same recall, more false alarms. "Today is hot -> tomorrow is hot" is a very strong rule
  for this target. The test score is also lower than on val (0.91), so some of the val result was optimistic.
- **Advisory + night**: the hardest target (few positive days). The model is much better than the baselines
  (PR-AUC 0.58 vs 0.33, recall 0.82 vs 0.39-0.50), but precision is about 0.4: more than half of the warnings
  are false alarms.
- **Regression**: the models are about 0.2 degC MAE better than persistence on both val and test.
- The positive share grows from val to test (more hot days in 2025-2026), so compare models only within a split.

### After modeling

- Look at feature importance (permutation importance on val, or SHAP) to see which features help most.
- Look at the errors: which days are missed (false negatives) and which are false alarms.
- Mention the limitations from Stage 3 in the final report (labels use D+2 / D+3, hourly vs 10-minute data, computed feels-like, small and noisy test set). Done: see section 7 of `FINAL_REPORT.md`.
- Do not add ERA5 features to the model until the ERA5 dataset is ready.

## Stage 5: Final report

The final report is [`FINAL_REPORT.md`](FINAL_REPORT.md) (authors: Jeremi Wojciech Barciński, Tenzin Dorjee).
It covers the problem, data, cleaning, features, method, results (test evaluated once), feature importance,
errors, discussion, limitations and future work. It shows `ASOS_04_pr_curves_val.png`, so keep that file
in the same folder.
