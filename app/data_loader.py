"""
Cached data loading for the interactive demo.
All functions decorated with @st.cache_data so they run once per session.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent

# ── Annual mean prices (2003-2024, from weekly_panel.csv) ────────────────────
ANNUAL_PRICES = {
    2003: 286.9, 2004: 245.7, 2005: 233.4, 2006: 396.8, 2007: 209.0,
    2008: 324.6, 2009: 294.2, 2010: 408.7, 2011: 357.7, 2012: 218.7,
    2013: 290.3, 2014: 227.9, 2015: 175.5, 2016: 233.9, 2017: 268.9,
    2018: 417.0, 2019: 383.2, 2020: 96.7,  2021: 768.1, 2022: 2128.9,
    2023: 903.5, 2024: 583.8,
}

# Reference distribution: 2003-2020 mean + std
_ref_prices = [v for yr, v in ANNUAL_PRICES.items() if 2003 <= yr <= 2020]
PRICE_MEAN_REF = float(np.mean(_ref_prices))
PRICE_STD_REF  = float(np.std(_ref_prices, ddof=1))
EXTREME_THRESHOLD = PRICE_MEAN_REF + 1.5 * PRICE_STD_REF  # ~413 NOK/MWh

BACKTEST_YEARS = list(range(2003, 2025))
EXTREME_YEARS  = [yr for yr in BACKTEST_YEARS if ANNUAL_PRICES[yr] > EXTREME_THRESHOLD]
NORMAL_YEARS   = [yr for yr in BACKTEST_YEARS if ANNUAL_PRICES[yr] <= EXTREME_THRESHOLD]


# ── Table loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_phase_a() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/tables/backtest_results.csv")
    df["VoF_MNOK"] = df["value_of_flexibility_NOK"] / 1e6
    df["OL_MNOK"]  = df["open_loop_revenue_NOK"] / 1e6
    df["CL_MNOK"]  = df["closed_loop_revenue_NOK"] / 1e6
    df["PF_MNOK"]  = df["perfect_foresight_revenue_NOK"] / 1e6
    df["regime"] = df["year"].apply(
        lambda y: "crisis" if y in EXTREME_YEARS else "normal"
    )
    return df


@st.cache_data
def load_phase_b() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/tables/cascade_backtest_results.csv")
    df["regime"] = df["year"].apply(
        lambda y: "crisis" if y in EXTREME_YEARS else "normal"
    )
    return df


@st.cache_data
def load_phase_b_lag1() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/tables/cascade_lag1_sensitivity.csv")


@st.cache_data
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT / "data/processed/weekly_panel.csv",
        parse_dates=["week_start"],
    )
    df["year"] = df["week_start"].dt.year
    return df


@st.cache_data
def load_scenario_tree(year: int) -> dict:
    path = ROOT / f"results/scenarios/scenario_tree_{year}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_year_panel(year: int) -> pd.DataFrame:
    """Return the weekly rows for an ISO year."""
    panel = load_panel()
    return panel[panel["iso_year"] == year].reset_index(drop=True)


def s_init_for_year(year: int, model: str = "phase_b") -> float:
    """Initial Jørundland storage (GWh) at the start of a given backtest year."""
    if model == "phase_b":
        cb = load_phase_b()
        row = cb[cb["year"] == year]
        return float(row["s_init_J_GWh"].iloc[0]) if len(row) else 83.5
    # Phase A: use s_init_GWh from backtest_results
    pa = load_phase_a()
    row = pa[pa["year"] == year]
    return float(row["s_init_GWh"].iloc[0]) if "s_init_GWh" in pa.columns and len(row) else 650.0
