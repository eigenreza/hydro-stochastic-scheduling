"""
NVE HydAPI v1 client.
API key loaded from .env (NVE_API_KEY).
All queries are hard-capped at 2025-12-31 per project data cutoff.
"""
from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://hydapi.nve.no/api/v1"
DATA_CUTOFF = date(2025, 12, 31)
_DEFAULT_TIMEOUT = 30


def _get_key() -> str:
    key = os.environ.get("NVE_API_KEY", "")
    if not key:
        raise EnvironmentError("NVE_API_KEY not set. Add it to .env file.")
    return key


def _headers() -> dict[str, str]:
    return {"X-API-Key": _get_key(), "Accept": "application/json"}


def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> Any:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_headers(), params=params,
                             timeout=_DEFAULT_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


# ── Station discovery ───────────────────────────────────────────────────────

# NO2 price area counties (Norwegian names as stored by NVE).
# Telemark straddles NO1/NO2; upper Telemark (Tinn, Rjukan) feeds the NO1 grid.
# Only Agder and Rogaland are unambiguously within NO2.
NO2_COUNTIES = {
    "Rogaland", "Agder", "Aust-Agder", "Vest-Agder",
}

# Key watercourses in NO2 known for large regulated reservoirs
NO2_WATERCOURSES = {
    "Sira-Kvina", "Otra", "Suldalslågen", "Ulla-Førre",
    "Lygna", "Mandalselva", "Tovdalselva", "Nidelva",
    "Arendalsvassdraget", "Tveitevatn",
}


def get_all_stations() -> pd.DataFrame:
    data = _get("Stations")
    return pd.DataFrame(data.get("data", []))


def get_no2_stations(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Filter full station list to those in NO2 price area."""
    if df is None:
        df = get_all_stations()
    mask_county = df["countyName"].isin(NO2_COUNTIES)
    mask_river = df["riverName"].apply(
        lambda r: any(w.lower() in str(r).lower() for w in NO2_WATERCOURSES)
    )
    return df[mask_county | mask_river].copy()


# ── Series (parameter) metadata ─────────────────────────────────────────────

def get_series_for_station(station_id: str) -> pd.DataFrame:
    data = _get("Series", params={"StationId": station_id})
    rows = data.get("data", []) if data else []
    return pd.DataFrame(rows)


def get_discharge_series_no2() -> pd.DataFrame:
    """
    Fetch all daily discharge/inflow series in NO2 counties in one request.
    Returns a DataFrame of series metadata (one row per station×parameter).
    """
    all_rows = []
    for county in NO2_COUNTIES:
        for param in DISCHARGE_PARAMS:
            try:
                data = _get("Series", params={
                    "CountyName": county,
                    "Parameter": param,
                    "ResolutionTime": 1440,
                })
                rows = data.get("data", []) if data else []
                all_rows.extend(rows)
            except Exception:
                pass
    return pd.DataFrame(all_rows).drop_duplicates(subset=["stationId", "parameter"]) \
        if all_rows else pd.DataFrame()


# ── Time-series data ─────────────────────────────────────────────────────────

# Parameter codes of interest
# 1001 = Vannføring (discharge/streamflow, m³/s) — primary
# 1004 = Magasinvolum (reservoir volume) — used for initial storage proxy
DISCHARGE_PARAMS = {1001}
RESERVOIR_PARAM = 1004


def fetch_daily_series(
    station_id: str,
    parameter: int,
    resolution_time: int = 1440,  # daily = 1440 min
    start_date: str = "1980-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Fetch daily observations for a given station / parameter combination.
    Returns a DataFrame with columns [date, value, quality].
    Hard-capped at DATA_CUTOFF.
    """
    # Enforce hard cutoff
    end_date = min(end_date, DATA_CUTOFF.isoformat())
    params = {
        "StationId": station_id,
        "Parameter": parameter,
        "ResolutionTime": resolution_time,
        "ReferenceTime": f"{start_date}/{end_date}",
    }
    data = _get("Observations", params=params)
    if not data or not data.get("data"):
        return pd.DataFrame()
    rows = []
    for obs in data["data"]:
        for val in obs.get("observations", []):
            rows.append({
                "date": pd.to_datetime(val["time"]).normalize(),
                "value": val.get("value"),
                "quality": val.get("quality", 0),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    # Hard-cap again on the actual data
    df = df[df.index.date <= DATA_CUTOFF]
    return df
