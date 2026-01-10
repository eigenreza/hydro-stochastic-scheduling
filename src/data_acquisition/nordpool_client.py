"""
NO2 day-ahead electricity price retrieval.

Primary source: Energi Data Service (api.energidataservice.dk)
  — Denmark's public energy data API, covering all Nordic bidding zones
  — No registration or API key required
  — License: CC BY 4.0

Prices returned in EUR/MWh. Converted to NOK/MWh using Norges Bank
spot EUR/NOK exchange rates (also freely available via Norges Bank's
public SDMX API, no registration required).

Data cutoff: all data > 2025-12-31 is discarded.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger(__name__)

DATA_CUTOFF = date(2025, 12, 31)
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

_EDS_URL = "https://api.energidataservice.dk/dataset/Elspotprices"
_NB_URL = "https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP"
_RATE_LIMIT_SLEEP = 2.0   # seconds between EDS year-chunk requests


# ── EUR/NOK exchange rates (Norges Bank) ─────────────────────────────────────

def fetch_eur_nok(start_date: str = "2000-01-01", end_date: str = "2025-12-31") -> pd.Series:
    """
    Fetch daily EUR/NOK spot rate from Norges Bank SDMX-JSON API.
    Returns a Series indexed by date, values = NOK per EUR.
    """
    end = min(date.fromisoformat(end_date), DATA_CUTOFF).isoformat()
    r = requests.get(
        _NB_URL,
        params={
            "format": "sdmx-json",
            "startPeriod": start_date,
            "endPeriod": end,
            "detail": "dataonly",
        },
        timeout=30,
    )
    r.raise_for_status()
    nb = r.json()

    # Parse the SDMX-JSON structure
    datasets = nb["data"]["dataSets"]
    dim_obs = nb["data"]["structure"]["dimensions"]["observation"][0]["values"]
    dates = [v["id"] for v in dim_obs]

    series_data = datasets[0]["series"]
    first_key = next(iter(series_data))
    obs = series_data[first_key]["observations"]

    fx = {}
    for idx_str, vals in obs.items():
        idx = int(idx_str)
        if idx < len(dates):
            fx[pd.Timestamp(dates[idx])] = float(vals[0]) if vals[0] is not None else np.nan

    s = pd.Series(fx, name="eur_nok").sort_index()
    LOG.info("EUR/NOK loaded: %s → %s (%d days)", s.index.min().date(),
             s.index.max().date(), len(s))
    return s


# ── NO2 spot prices (Energi Data Service) ────────────────────────────────────

_BATCH_SLEEP = 90   # seconds between large EDS batch requests to avoid 429


def _fetch_eds_batch(start: str, end: str, offset: int = 0) -> pd.DataFrame:
    """
    Fetch up to 100,000 hourly NO2 records from EDS in one request.
    EDS returns data newest-first by default; we don't rely on order.
    """
    params = {
        "limit": 100000,
        "offset": offset,
        "filter": json.dumps({"PriceArea": "NO2"}),
        "columns": "HourUTC,SpotPriceEUR",
        "start": f"{start}T00:00",
        "end":   f"{end}T23:00",
    }
    for attempt in range(4):
        r = requests.get(_EDS_URL, params=params, timeout=120,
                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        if r.status_code == 200:
            records = r.json().get("records", [])
            LOG.info("  EDS batch %s→%s offset=%d: %d records", start, end, offset, len(records))
            return pd.DataFrame(records)
        elif r.status_code == 429:
            wait = 90 * (attempt + 1)
            LOG.warning("  EDS 429, waiting %ds …", wait)
            time.sleep(wait)
        else:
            LOG.warning("  EDS HTTP %d: %s", r.status_code, r.text[:100])
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_no2_prices(
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Fetch all available NO2 hourly day-ahead prices from EDS in three large
    batches (each ≤ 100,000 records), convert EUR→NOK using Norges Bank rates,
    and aggregate to daily averages.

    Returns a DataFrame with index=date and columns:
        price_avg_NOK_MWh   — daily mean price (NOK/MWh)
        price_std_NOK_MWh   — daily std dev (NOK/MWh)
    Hard-capped at DATA_CUTOFF.
    """
    end = min(date.fromisoformat(end_date), DATA_CUTOFF)
    start = date.fromisoformat(start_date)

    # ── EUR/NOK rates ────────────────────────────────────────────
    LOG.info("Fetching EUR/NOK exchange rates from Norges Bank …")
    eur_nok = fetch_eur_nok(start_date, end_date)
    eur_nok_daily = eur_nok.reindex(
        pd.date_range(start, end, freq="D")
    ).ffill().bfill()

    # ── NO2 hourly prices — three batches covering 2000–2025 ─────
    # ~227,760 hours total; three batches of ≤ 100,000 each
    # Split: 2000–2008, 2009–2016, 2017–2025
    LOG.info("Fetching NO2 hourly prices from Energi Data Service (3 batches) …")
    batch_windows = [
        (start_date, "2008-12-31"),
        ("2009-01-01", "2016-12-31"),
        ("2017-01-01", end_date),
    ]

    chunks = []
    for i, (b_start, b_end) in enumerate(batch_windows):
        if i > 0:
            LOG.info("  Sleeping %ds to respect EDS rate limit …", _BATCH_SLEEP)
            time.sleep(_BATCH_SLEEP)
        chunk = _fetch_eds_batch(b_start, b_end)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            "No NO2 price data returned from Energi Data Service.\n"
            "Source: https://api.energidataservice.dk/dataset/Elspotprices"
        )

    df = pd.concat(chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["HourUTC"], utc=True).dt.tz_convert(None)
    df["date"] = df["datetime"].dt.normalize()
    df = df[df["date"].dt.date <= DATA_CUTOFF]
    df = df.drop_duplicates(subset=["datetime"])

    # EUR → NOK conversion
    df["eur_nok"] = df["date"].map(eur_nok_daily.to_dict())
    df["price_NOK_MWh"] = df["SpotPriceEUR"] * df["eur_nok"]

    # Aggregate to daily
    daily = (
        df.groupby("date")["price_NOK_MWh"]
        .agg(price_avg_NOK_MWh="mean", price_std_NOK_MWh="std")
        .reset_index()
        .set_index("date")
        .sort_index()
    )
    daily.index = pd.to_datetime(daily.index)

    raw_path = RAW_DIR / "nordpool_no2_raw.parquet"
    daily.to_parquet(raw_path)
    LOG.info("Saved raw daily NO2 prices: %s (%d days)", raw_path, len(daily))
    return daily
