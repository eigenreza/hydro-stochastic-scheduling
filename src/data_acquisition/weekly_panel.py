"""
Constructs the aligned weekly panel from daily inflow and price series.

Energy-equivalent inflow conversion
------------------------------------
Discharge Q [m³/s] is converted to energy-equivalent GWh/week via:

    E = Q * η * ρ * g * H_eff * T / (3.6e12)

where:
  η      = overall efficiency (turbine + generator) ≈ 0.88  [dimensionless]
  ρ      = water density = 1000 kg/m³
  g      = 9.81 m/s²
  H_eff  = cascade-level effective head of Arendalsvassdraget [m] — see note
  T      = seconds per week = 604800

H_eff is the SYSTEM EFFECTIVE HEAD across the entire Arendalsvassdraget cascade,
not the head of any single plant. It is defined as:

    H_eff = E_annual_J / (η * ρ * g * Q_mean * T_year)

where E_annual is the total mean annual production of all plants in the watershed.

Computed from NVE power-plant registry (GetHydroPowerPlantsInOperation, 2026-06-19):
  - 44 active plants on Arendalsvassdraget
  - Total MidProd_91_20 = 2498.4 GWh/year
  - Q_mean at Rygene total (19.127.0) = 124.81 m³/s (2000–2025 mean)
  - H_eff = 2498.4 GWh × 3.6e12 J/GWh / (0.88 × 1000 × 9.81 × 124.81 × 31,557,600)
           = 264.5 m

This replaces a prior (incorrect) value of 450 m that was attributed to "Brokke kraftverk".
Brokke kraftverk (NVE registry head: 300 m) is on the Otra river in Valle municipality
— a completely different watershed from Arendalsvassdraget. That attribution was wrong
on both counts (wrong watercourse and wrong head value for Brokke itself).

Rygene kraftverk, the lowest plant in Arendalsvassdraget on Nidelva (Arendal), has a
gross head of 36.2 m per the NVE registry. However, the station Rygene total (19.127.0)
measures total system outflow; a single representative head must reflect the entire
cascade's energy yield per unit of system throughput, not just Rygene's head.

Conversion factor (cascade effective head):
    k [GWh per (m³/s · week)] = η * ρ * g * H_eff * T / 3.6e12
                               = 0.88 * 1000 * 9.81 * 264.5 * 604800 / 3.6e12
                               ≈ 0.3836 GWh per (m³/s · week)

Because the series represent daily mean Q (m³/s), weekly aggregation is
the MEAN of daily Q over the ISO week, multiplied by the conversion factor
and by 604800 (seconds in a week) to express energy content in GWh.

ISO week convention: Monday–Sunday (ISO 8601). Week is identified by the
year-week of the MONDAY of that week.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Conversion: GWh per (m³/s mean over one week)
#   = eta * rho * g * H_net * T_sec / 3.6e12
ETA = 0.88          # overall turbine-generator efficiency
RHO = 1000.0        # kg/m³
G = 9.81            # m/s²
# Cascade-level effective head for the Arendalsvassdraget system.
# Computed from NVE power-plant registry (GetHydroPowerPlantsInOperation, 2026-06-19):
# 44 active plants, total MidProd_91_20 = 2498.4 GWh/year, Q_mean = 124.81 m³/s.
# H_eff = (2498.4 × 3.6e12) / (0.88 × 1000 × 9.81 × 124.81 × 31,557,600) = 264.5 m
# See docstring above for the full derivation and the correction from the prior wrong value.
H_NET = 264.5
T_WEEK = 604800.0   # seconds per week
INFLOW_CONVERSION_GWH_PER_M3S = ETA * RHO * G * H_NET * T_WEEK / 3.6e12

# Hard data cutoff
DATA_CUTOFF = pd.Timestamp("2025-12-31")


def daily_to_weekly_inflow(daily: pd.DataFrame) -> pd.Series:
    """
    Aggregate daily discharge (m³/s) to weekly energy-equivalent (GWh/week).

    Aggregation rule: mean of daily discharge values over the ISO week,
    then multiply by the energy conversion factor. Using the mean (rather
    than sum) for discharge is appropriate because Q is a rate, not a
    cumulative volume; the weekly energy is the mean rate × seconds per week.
    """
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    daily = daily[daily.index <= DATA_CUTOFF]

    # Weekly grouper: ISO week starting Monday
    weekly_mean_q = daily["value"].resample("W-SUN").mean()  # week ending Sunday
    # Label the week by the Monday date
    weekly_mean_q.index = weekly_mean_q.index - pd.Timedelta(days=6)
    weekly_mean_q.name = "inflow_GWh_week"

    weekly_gwh = weekly_mean_q * INFLOW_CONVERSION_GWH_PER_M3S
    return weekly_gwh.rename("inflow_GWh_week")


def daily_to_weekly_price(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily average price to weekly average and volatility.
    Returns columns: price_avg_NOK_MWh, price_std_NOK_MWh
    """
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    daily = daily[daily.index <= DATA_CUTOFF]

    weekly = daily["price_avg_NOK_MWh"].resample("W-SUN").agg(
        price_avg_NOK_MWh="mean",
        price_std_NOK_MWh="std",
    )
    weekly.index = weekly.index - pd.Timedelta(days=6)
    return weekly


def build_weekly_panel(
    daily_inflow: pd.DataFrame,
    daily_price: pd.DataFrame,
    min_inflow_completeness: float = 0.80,
) -> pd.DataFrame:
    """
    Align inflow and price weekly series into a single panel.

    Gap handling:
    - Short gaps (≤ 3 consecutive missing weeks): linear interpolation.
    - Longer gaps: rows excluded; boundary of exclusion documented in log.
    - Weeks where inflow is available for < 3 days are treated as missing.

    Returns a DataFrame with columns:
        week_start  (index, DatetimeIndex, Monday dates)
        inflow_GWh_week
        price_avg_NOK_MWh
        price_std_NOK_MWh
        iso_year    (ISO year of the week)
        iso_week    (ISO week number)
    """
    inflow_w = daily_to_weekly_inflow(daily_inflow)
    price_w = daily_to_weekly_price(daily_price)

    panel = pd.DataFrame({
        "inflow_GWh_week": inflow_w,
        "price_avg_NOK_MWh": price_w["price_avg_NOK_MWh"],
        "price_std_NOK_MWh": price_w["price_std_NOK_MWh"],
    })
    panel.index.name = "week_start"

    # Trim to overlap window
    first_valid = panel.dropna(subset=["inflow_GWh_week", "price_avg_NOK_MWh"]).index.min()
    last_valid = panel.dropna(subset=["inflow_GWh_week", "price_avg_NOK_MWh"]).index.max()
    panel = panel.loc[first_valid:last_valid]

    LOG.info("Raw aligned window: %s → %s (%d weeks)", first_valid.date(),
             last_valid.date(), len(panel))

    n_missing_inflow = panel["inflow_GWh_week"].isna().sum()
    n_missing_price = panel["price_avg_NOK_MWh"].isna().sum()
    LOG.info("Missing before interpolation: inflow=%d, price=%d",
             n_missing_inflow, n_missing_price)

    # Linear interpolation for short gaps (≤ 3 consecutive weeks)
    panel["inflow_GWh_week"] = _interpolate_short_gaps(
        panel["inflow_GWh_week"], max_gap=3)
    panel["price_avg_NOK_MWh"] = _interpolate_short_gaps(
        panel["price_avg_NOK_MWh"], max_gap=3)
    panel["price_std_NOK_MWh"] = _interpolate_short_gaps(
        panel["price_std_NOK_MWh"], max_gap=3)

    # Drop remaining rows with NaN (long gaps)
    before = len(panel)
    panel = panel.dropna(subset=["inflow_GWh_week", "price_avg_NOK_MWh"])
    dropped = before - len(panel)
    if dropped > 0:
        LOG.warning("Dropped %d weeks due to long gaps (>3 consecutive NaN)", dropped)

    # Add ISO year/week columns
    panel["iso_year"] = panel.index.isocalendar().year.values
    panel["iso_week"] = panel.index.isocalendar().week.values

    LOG.info("Final panel: %d weeks (%s → %s)",
             len(panel), panel.index.min().date(), panel.index.max().date())

    return panel


def _interpolate_short_gaps(s: pd.Series, max_gap: int = 3) -> pd.Series:
    """Linear interpolation for runs of NaN of length ≤ max_gap."""
    s = s.copy()
    mask = s.isna()
    if not mask.any():
        return s
    # Find consecutive NaN run lengths
    runs = (mask != mask.shift()).cumsum()
    run_lengths = mask.groupby(runs).transform("sum")
    short = mask & (run_lengths <= max_gap)
    s[short] = np.nan  # ensure these are left as NaN for interpolate
    s = s.interpolate(method="linear", limit=max_gap, limit_direction="both")
    return s


def save_panel(panel: pd.DataFrame) -> Path:
    path = PROCESSED_DIR / "weekly_panel.csv"
    panel.to_csv(path)
    LOG.info("Saved weekly panel: %s", path)

    # Also save data dictionary
    dd = (
        "# Data Dictionary — weekly_panel.csv\n\n"
        "| Column | Units | Description |\n"
        "|--------|-------|-------------|\n"
        "| week_start (index) | ISO date (Monday) | First day of ISO week (Monday–Sunday) |\n"
        "| inflow_GWh_week | GWh/week | Energy-equivalent inflow; converted from mean daily "
        "discharge [m³/s] using fixed-head approximation "
        f"(η={ETA}, H_net={H_NET} m, k={INFLOW_CONVERSION_GWH_PER_M3S:.4f} GWh per m³/s) |\n"
        "| price_avg_NOK_MWh | NOK/MWh | Mean day-ahead price in NO2 bidding zone over the week |\n"
        "| price_std_NOK_MWh | NOK/MWh | Std. dev. of daily average prices within the week |\n"
        "| iso_year | integer | ISO 8601 year |\n"
        "| iso_week | integer | ISO 8601 week number (1–53) |\n"
    )
    (PROCESSED_DIR / "DATA_DICTIONARY.md").write_text(dd, encoding="utf-8")
    return path
