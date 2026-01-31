"""
Per-plant weekly inflow panels for the three-plant Arendalsvassdraget cascade:
    Jørundland → Evenstad → Rygene

Data strategy
-------------
HydAPI provides a single continuous daily discharge series for the system:
    Station 19.127.0 (Rygene total, Q in m³/s, 3946 km² catchment area).
No sub-catchment discharge series exist in HydAPI for Arendalsvassdraget.

The total system flow is partitioned into per-plant LOCAL inflows using
mean-flow fractions derived from the NVE Vannkraftdatabase (energy balance
method). This is a documented approximation — see methodology.md §B.2.3.

Local inflow fractions (conserve total to Rygene total measured Q):
    f_J = 0.0706  (Jørundland headwater sub-catchment, ~279 km²)
    f_E = 0.6723  (main Nidelva direct to Evenstad, ~2653 km²)
    f_R = 0.2571  (local between Evenstad and Rygene, ~1014 km²)

Derivation:
    Mean turbined flow from NVE MidProd + head:
        Q_J = MidProd_J / (η ρ g H_J T_year) = 185.8×3.6e12 / (0.88×1000×9.81×278.6×31557600) = 8.81 m³/s
        Q_E = 121.4×3.6e12 / (0.88×1000×9.81×17.3×31557600) = 92.73 m³/s
    Cascade local inflows:
        i_J = Q_J (headwater only)
        i_E = Q_E - Q_J (main river direct to Evenstad, not via Jørundland)
        i_R = Q_Rygene_measured - Q_E = 124.81 - 92.73 = 32.08 m³/s
    Fractions of Q_Rygene_total = 124.81 m³/s:
        f_J = 8.81 / 124.81 = 0.0706
        f_E = 83.92 / 124.81 = 0.6723
        f_R = 32.08 / 124.81 = 0.2571

Physical parameters (NVE Vannkraftdatabase, 2026-06-19):
    Jørundland:  55.2 MW, H=278.6 m, k=0.4041 GWh/(m³/s·wk)
    Evenstad:    23.5 MW, H=17.3  m, k=0.0251 GWh/(m³/s·wk)
    Rygene:      55.0 MW, H=36.2  m, k=0.0525 GWh/(m³/s·wk)

Per-plant G_max (MW × 168 h/wk / 1000):
    G_max_J = 9.274 GWh/wk
    G_max_E = 3.948 GWh/wk
    G_max_R = 9.240 GWh/wk
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

# ── Physical constants per plant (NVE registry, 2026-06-19) ─────────────────
ETA = 0.88
RHO = 1000.0
G   = 9.81
T_WEEK = 604800.0   # s

# Plant parameters
PLANTS = ("Joerundland", "Evenstad", "Rygene")

PLANT_HEAD_M  = {"Joerundland": 278.6, "Evenstad": 17.3,  "Rygene": 36.2}
PLANT_MW      = {"Joerundland": 55.2,  "Evenstad": 23.5,  "Rygene": 55.0}
PLANT_G_MAX   = {p: PLANT_MW[p] * 168 / 1000 for p in PLANTS}    # GWh/wk
PLANT_K       = {p: ETA * RHO * G * PLANT_HEAD_M[p] * T_WEEK / 3.6e12
                 for p in PLANTS}  # GWh per (m³/s · week)

# Jørundland: real working storage from NVE HydAPI Nesvatn volume series
# (station 19.5.0, parameter 1004): working_storage = 262.96 mill m³;
# S_MAX = 262.96e6 m³ × 0.638 kWh/m³ / 1e6 = 167.77 GWh.
# Rounded conservatively to 167 GWh to exclude the lowest 0.87% tail.
PLANT_S_MAX   = {"Joerundland": 167.0, "Evenstad": 0.0, "Rygene": 0.0}  # GWh

# Local inflow fractions (conserve to Rygene total Q)
# Derived via energy-balance from NVE MidProd + head — see module docstring.
INFLOW_FRACTIONS = {
    "Joerundland": 0.0706,
    "Evenstad":    0.6723,
    "Rygene":      0.2571,
}
_FRAC_SUM = sum(INFLOW_FRACTIONS.values())
assert abs(_FRAC_SUM - 1.0) < 1e-4, f"Fractions sum to {_FRAC_SUM}, expected 1.0"

# Terminal water value [NOK/GWh]: long-run mean price × 1000 MWh/GWh
TERMINAL_VBAR = 400.0 * 1000   # NOK/GWh (same as single-reservoir model)


def split_system_inflow(
    inflow_total: pd.Series | np.ndarray,
) -> dict[str, pd.Series | np.ndarray]:
    """
    Partition total Rygene system inflow into per-plant local inflows
    using the fixed energy-balance fractions.

    Parameters
    ----------
    inflow_total:
        Total system inflow at Rygene (GWh/week). Can be a Series or ndarray.

    Returns
    -------
    dict mapping plant name → per-plant inflow array of same shape/type.
    """
    return {p: inflow_total * f for p, f in INFLOW_FRACTIONS.items()}


def build_cascade_panel(weekly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the existing weekly panel with per-plant local inflow columns.

    Parameters
    ----------
    weekly_panel:
        Existing single-reservoir panel (columns: inflow_GWh_week,
        price_avg_NOK_MWh, price_std_NOK_MWh, iso_year, iso_week).

    Returns
    -------
    Panel with three additional inflow columns:
        inflow_J_GWh_week, inflow_E_GWh_week, inflow_R_GWh_week
    """
    panel = weekly_panel.copy()
    total = panel["inflow_GWh_week"].values
    local = split_system_inflow(total)
    panel["inflow_J_GWh_week"] = local["Joerundland"]
    panel["inflow_E_GWh_week"] = local["Evenstad"]
    panel["inflow_R_GWh_week"] = local["Rygene"]
    return panel


def get_cascade_panel(panel_path: str | Path | None = None) -> pd.DataFrame:
    """Load the weekly panel and add per-plant inflow columns."""
    if panel_path is None:
        panel_path = Path("data/processed/weekly_panel.csv")
    panel = pd.read_csv(panel_path, index_col="week_start", parse_dates=True)
    return build_cascade_panel(panel)
