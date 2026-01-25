"""
Tests: no data beyond 2025-12-31 anywhere in the pipeline.
"""
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

DATA_CUTOFF = pd.Timestamp("2025-12-31")


def test_weekly_panel_cutoff():
    path = Path("data/processed/weekly_panel.csv")
    if not path.exists():
        pytest.skip("weekly_panel.csv not yet generated")
    df = pd.read_csv(path, index_col="week_start", parse_dates=True)
    max_date = df.index.max()
    assert max_date <= DATA_CUTOFF, (
        f"weekly_panel.csv contains data beyond 2025-12-31: max date = {max_date}"
    )


def test_raw_inflow_cutoff():
    path = Path("data/raw/inflow_daily_raw.parquet")
    if not path.exists():
        pytest.skip("inflow_daily_raw.parquet not yet generated")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    max_date = df.index.max()
    assert max_date <= DATA_CUTOFF, (
        f"Raw inflow contains data beyond 2025-12-31: max date = {max_date}"
    )


def test_raw_price_cutoff():
    path = Path("data/raw/nordpool_no2_raw.parquet")
    if not path.exists():
        pytest.skip("nordpool_no2_raw.parquet not yet generated")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    max_date = df.index.max()
    assert max_date <= DATA_CUTOFF, (
        f"Raw price data contains data beyond 2025-12-31: max date = {max_date}"
    )


def test_scenario_files_reference_valid_dates():
    scenario_dir = Path("results/scenarios")
    if not scenario_dir.exists():
        pytest.skip("No scenario files yet")
    for csv_path in scenario_dir.glob("scenario_inflow_*.csv"):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        max_date = df.index.max()
        assert max_date <= DATA_CUTOFF + pd.Timedelta(days=365), (
            f"{csv_path.name}: scenario dates too far in future: {max_date}"
        )


def test_backtest_results_years_in_panel():
    results_path = Path("results/tables/backtest_results.csv")
    panel_path = Path("data/processed/weekly_panel.csv")
    if not results_path.exists() or not panel_path.exists():
        pytest.skip("Backtest results or panel not yet generated")
    results = pd.read_csv(results_path)
    panel = pd.read_csv(panel_path, index_col="week_start", parse_dates=True)
    panel_years = set(panel["iso_year"].unique())
    for year in results["year"]:
        assert year in panel_years, (
            f"Backtest result for year {year} not in panel years {panel_years}"
        )
        assert year <= 2025, f"Backtest year {year} exceeds data cutoff 2025"
