"""
Programmatic selection of the best inflow/discharge station in NO2.

Selection criteria (in order):
1. Station must have a discharge or inflow series (parameter 1000 or 1001)
   at daily (1440 min) resolution in NO2 counties.
2. Maximise overlap with the available Nord Pool price history
   (assumed available from ~2000-01-01 onward).
3. Minimise fraction of missing / bad-quality days within the overlap window.

The selected station is written to data/raw/selected_station.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_acquisition.nve_client import (
    DISCHARGE_PARAMS,
    fetch_daily_series,
    get_discharge_series_no2,
)

LOG = logging.getLogger(__name__)
OVERLAP_START = pd.Timestamp("2000-01-01")
DATA_CUTOFF = pd.Timestamp("2025-12-31")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Maximum stations to fully score (fetching full time series is slow)
MAX_TO_SCORE = 30


def score_station(station_id: str, parameter: int) -> dict | None:
    """
    Return a scoring dict for *station_id* or None if not suitable.
    """
    df = fetch_daily_series(
        station_id,
        parameter=parameter,
        start_date=OVERLAP_START.strftime("%Y-%m-%d"),
        end_date=DATA_CUTOFF.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return None

    full_range = pd.date_range(OVERLAP_START, DATA_CUTOFF, freq="D")
    n_expected = len(full_range)
    n_present = df["value"].notna().sum()
    completeness = n_present / n_expected

    return {
        "station_id": station_id,
        "parameter": int(parameter),
        "n_days": int(n_present),
        "completeness": float(completeness),
        "start": df.index.min().isoformat() if not df.empty else None,
        "end": df.index.max().isoformat() if not df.empty else None,
    }


def select_best_station(top_n: int = 5) -> dict:
    """
    Query the NVE Series endpoint for all NO2 daily discharge series,
    then score the top candidates by data availability.
    """
    LOG.info("Querying NO2 discharge series from NVE Series endpoint …")
    series_df = get_discharge_series_no2()
    if series_df.empty:
        raise RuntimeError("No discharge series found in NO2 counties via NVE Series endpoint.")

    LOG.info("Found %d candidate series in NO2", len(series_df))

    # Prioritise: prefer parameter 1000 (discharge) or 1001 (inflow)
    if "parameter" in series_df.columns:
        series_df = series_df.sort_values("parameter")

    # Deduplicate to one series per station (take first / best parameter)
    if "stationId" in series_df.columns:
        series_df = series_df.drop_duplicates(subset=["stationId"])

    # Score up to MAX_TO_SCORE stations
    candidates = series_df.head(MAX_TO_SCORE)
    results = []
    for _, row in candidates.iterrows():
        sid = row["stationId"]
        param = int(row.get("parameter", 1000))
        sname = row.get("stationName", "")
        river = row.get("riverName", "")
        county = row.get("countyName", "")
        LOG.debug("Scoring %s (%s) param=%d …", sid, sname, param)
        score = score_station(sid, param)
        if score:
            score["stationName"] = sname
            score["riverName"] = river
            score["countyName"] = county
            results.append(score)

    if not results:
        raise RuntimeError("No suitable stations scored in NO2.")

    df_scores = pd.DataFrame(results).sort_values("completeness", ascending=False)
    LOG.info("Top %d stations:\n%s", top_n,
             df_scores[["station_id", "stationName", "riverName",
                         "completeness", "n_days"]].head(top_n).to_string())

    df_scores.to_csv(RAW_DIR / "station_scores.csv", index=False)

    best = df_scores.iloc[0].to_dict()
    with open(RAW_DIR / "selected_station.json", "w") as f:
        json.dump(best, f, indent=2)
    LOG.info("Selected station: %s — %s (completeness=%.3f)",
             best["station_id"], best["stationName"], best["completeness"])
    return best
