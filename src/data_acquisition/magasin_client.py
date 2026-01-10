"""
Fetch weekly reservoir filling statistics for the NO2 bidding zone from NVE's
Magasinstatistikk API (https://biapi.nve.no/magasinstatistikk/).

The API is public and requires no authentication (NLOD licence, CC BY 3.0).
It tracks weekly fill levels for ~490 Norwegian reservoirs, aggregated by
elspot zone (NO1–NO5), vassdrag zone (VASS1–4), and whole-country (NO).

We use the NO2 elspot aggregate (omrType='EL', omrnr=2), which covers
Agder, most of Rogaland, Vestland (south), Vestfold, and Telemark — the same
geographic region as the Rygene/Arendalsvassdraget study system.

Data availability: weekly from ISO week 48/2001 (the earliest NO2 record)
through the current week. All 22 backtest years (2003–2024) are covered.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

_API_URL = (
    "https://biapi.nve.no/magasinstatistikk/api/"
    "Magasinstatistikk/HentOffentligData"
)
_RAW_DIR = Path("data/raw")
_CACHE_PATH = _RAW_DIR / "no2_magasin_weekly.csv"


def fetch_no2_magasin(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of weekly NO2 reservoir filling statistics.

    Columns: iso_year, iso_week, fyllingsgrad (0–1), kapasitet_TWh, fylling_TWh.

    Results are cached to data/raw/no2_magasin_weekly.csv. Pass
    force_refresh=True to re-download from the API.
    """
    if _CACHE_PATH.exists() and not force_refresh:
        return pd.read_csv(_CACHE_PATH)

    resp = requests.get(_API_URL, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    all_records = resp.json()

    no2 = [
        r for r in all_records
        if r.get("omrType") == "EL" and r.get("omrnr") == 2
    ]
    df = pd.DataFrame(no2)[
        ["iso_aar", "iso_uke", "fyllingsgrad", "kapasitet_TWh", "fylling_TWh"]
    ].rename(columns={"iso_aar": "iso_year", "iso_uke": "iso_week"})
    df = df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE_PATH, index=False)
    return df


def get_no2_week1_filling() -> dict[int, float]:
    """
    Return a dict mapping year → NO2 fyllingsgrad at ISO week 1.

    This is used as the start-of-year reservoir filling fraction for each
    backtest year. Coverage: 1995–present (all 22 backtest years included).
    """
    df = fetch_no2_magasin()
    week1 = df[df["iso_week"] == 1].copy()
    return dict(zip(week1["iso_year"].astype(int), week1["fyllingsgrad"]))
