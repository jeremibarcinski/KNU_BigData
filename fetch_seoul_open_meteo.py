"""
Pobiera godzinowe dane pogodowe i o jakości powietrza dla Seulu z Open-Meteo
i zapisuje je jako jeden plik CSV (UrbanAtmos).

Wymagania:  pip install requests pandas
Uruchomienie:  python fetch_seoul_open_meteo.py
"""

from datetime import date, timedelta

import pandas as pd
import requests

# ---------- USTAWIENIA ----------
#LATITUDE = 37.5665       # Seul
#LONGITUDE = 126.9780

LATITUDE = 35.8714       # Daegu
LONGITUDE = 128.6014
START_DATE = "2023-01-01"
# Archiwum pogodowe ma kilkudniowe opóźnienie, więc kończymy tydzień temu
END_DATE = (date.today() - timedelta(days=7)).isoformat()
TIMEZONE = "Asia/Seoul"
OUTPUT_CSV = "Daegu_open_meteo_hourly.csv"

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
]

AIR_VARS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
]


def fetch_hourly(url: str, variables: list[str], start: str, end: str) -> pd.DataFrame:
    """Pobiera dane godzinowe z jednego endpointu Open-Meteo jako DataFrame."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": TIMEZONE,
    }
    response = requests.get(url, params=params, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Błąd {response.status_code} z {url}: {response.text[:300]}")
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_in_year_chunks(url: str, variables: list[str]) -> pd.DataFrame:
    """Dzieli zakres dat na roczne kawałki, żeby zapytania były lekkie."""
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        print(f"  {url.split('/')[2]}: {current.date()} -> {chunk_end.date()}")
        chunks.append(
            fetch_hourly(url, variables, current.date().isoformat(), chunk_end.date().isoformat())
        )
        current = chunk_end + pd.Timedelta(days=1)
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset="time")


def main() -> None:
    print("Pobieram dane pogodowe...")
    weather = fetch_in_year_chunks(WEATHER_URL, WEATHER_VARS)

    print("Pobieram dane o jakości powietrza...")
    air = fetch_in_year_chunks(AIR_URL, AIR_VARS)

    df = weather.merge(air, on="time", how="inner").sort_values("time")

    print(f"\nWiersze: {len(df)}, kolumny: {len(df.columns)}")
    print(f"Zakres: {df['time'].min()} -> {df['time'].max()}")
    print("\nBrakujące wartości (%):")
    print((df.isna().mean() * 100).round(2).to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nZapisano: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()