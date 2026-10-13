"""
Fetches hourly ground observations (ASOS) for Daegu (station 143) from the
KMA API Hub and saves them as a single CSV file (UrbanAtmos).

Requirements:  pip install requests pandas
API key:       set the environment variable KMA_API_KEY (never put the key in the code)
                 Linux/macOS:  export KMA_API_KEY="your_key"
                 Windows:      setx KMA_API_KEY "your_key"   (then reopen the terminal)
Usage:         python fetch_daegu_asos.py

How it works:
  - the date range is split into months (the period API has a limit per request)
  - every month is saved as a raw text file in ASOS_CACHE_DIR
  - months that are already downloaded are skipped, so the script can be stopped
    and started again (the API allows 2000 calls per day)
  - at the end all months are parsed and merged into one CSV

The CSV is RAW: missing-value codes (-9, -9.0, -99.0, "-") are kept exactly as
KMA delivers them. They are handled in the cleaning stage, because -9.0 can be
a real temperature in winter.
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------- SETTINGS ----------
STATION = 143                      # Daegu
START_DATE = "2000-01-01"          # first day to download (change if needed)
# End one week ago, the same as in fetch_daegu_open_meteo.py
END_DATE = (date.today() - timedelta(days=7)).isoformat()

# Check that this matches the address shown on the API Hub page
# "1.2 Time Data (Period Inquiry)" (without the key part)
API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"

OUTPUT_CSV = "ASOS_01_Daegu_asos_hourly.csv"
ASOS_CACHE_DIR = Path("asos_raw")
PAUSE_SECONDS = 0.5                # pause between requests

# Column order of one data row (46 fields, taken from the KMA help header)
COLUMNS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT",
    "PR", "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3",
    "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT",
    "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005",
    "TE_01", "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
]
TEXT_COLUMNS = ["WW", "CT"]        # text codes, "-" means empty


def parse_asos_text(text: str) -> pd.DataFrame:
    """Parses the text answer of the API into a DataFrame with a 'time' column."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        # errors come as JSON, e.g. {"result": {"status": 403, "message": ...}}
        raise RuntimeError(f"API returned an error: {stripped[:300]}")

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):   # header, #START7777, #7777END
            continue
        parts = line.split()
        if len(parts) != len(COLUMNS):
            raise ValueError(
                f"Unexpected number of fields: {len(parts)} instead of {len(COLUMNS)}"
            )
        rows.append(parts)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.insert(0, "time", pd.to_datetime(df.pop("TM"), format="%Y%m%d%H%M"))
    for col in df.columns:
        if col != "time" and col not in TEXT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Splits the range into months; returns (tm1, tm2) strings YYYYMMDDHHMM."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks = []
    for month_start in pd.date_range(start_ts.replace(day=1), end_ts, freq="MS"):
        first = max(month_start, start_ts)
        last = min(month_start + pd.offsets.MonthEnd(0), end_ts)
        chunks.append((first.strftime("%Y%m%d0000"), last.strftime("%Y%m%d2300")))
    return chunks


def download_chunk(tm1: str, tm2: str, key: str) -> str:
    """Downloads one chunk and returns the raw text."""
    params = {"tm1": tm1, "tm2": tm2, "stn": STATION, "help": 0, "authKey": key}
    try:
        response = requests.get(API_URL, params=params, timeout=120)
    except requests.RequestException as exc:
        # do not show the original message, it contains the URL with the key
        raise RuntimeError(f"Request failed ({type(exc).__name__})") from None
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.text


def main() -> None:
    if Path(OUTPUT_CSV).exists():
        print(f"File {OUTPUT_CSV} already exists - skipping download.")
        return

    key = os.environ.get("KMA_API_KEY")
    if not key:
        raise SystemExit("Set the environment variable KMA_API_KEY first.")

    ASOS_CACHE_DIR.mkdir(exist_ok=True)
    chunks = month_chunks(START_DATE, END_DATE)
    print(f"Months to process: {len(chunks)}")

    paths = []
    for tm1, tm2 in chunks:
        path = ASOS_CACHE_DIR / f"asos_{STATION}_{tm1[:8]}_{tm2[:8]}.txt"
        paths.append(path)
        if path.exists():
            continue
        print(f"  downloading {tm1[:8]} -> {tm2[:8]}")
        text = download_chunk(tm1, tm2, key)
        parse_asos_text(text)              # fail early if the answer is wrong
        path.write_text(text, encoding="utf-8")
        time.sleep(PAUSE_SECONDS)

    frames = [parse_asos_text(p.read_text(encoding="utf-8")) for p in paths]
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="time", keep="last")
        .sort_values("time")
    )

    expected = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    print(f"\nRows: {len(df)}, columns: {len(df.columns)}")
    print(f"Range: {df['time'].min()} -> {df['time'].max()}")
    print(f"Missing whole hours: {len(expected) - len(df)}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
