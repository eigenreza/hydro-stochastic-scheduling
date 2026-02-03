"""
Cascade backtesting engine: open-loop, closed-loop, and perfect-foresight
for the three-plant Jørundland→Evenstad→Rygene cascade.

Local inflow splitting
----------------------
The single measured discharge series (Rygene total, 19.127.0) is partitioned
into per-plant local inflows using fixed fractions derived from the NVE
energy-balance method:
    i_J = 0.0706 × Q_total   (Jørundland headwater, ~279 km²)
    i_E = 0.6723 × Q_total   (main Nidelva direct to Evenstad, ~2653 km²)
    i_R = 0.2571 × Q_total   (local between Evenstad and Rygene, ~1014 km²)

The fraction splits are applied to EVERY scenario path as well as to the
realised inflow used for backtest evaluation. This preserves exact cross-plant
correlation (r=1.0 by construction) as a documented simplification — see
methodology.md §B.2.3.

The scenario tree for each year was built using the single-station inflow
model (Rygene total); the cascade runner partitions that tree into three
per-plant scenario matrices.

Cascade revenue application
---------------------------
For open-loop and closed-loop, only g_J is committed (non-anticipative).
In backtest realisation:
    - Jørundland: exact same realised_revenue logic as single-reservoir,
      but applied to the local i_J fraction and plant-specific S_MAX, G_MAX.
    - Evenstad: revenue = p_t × min(G_MAX_E, i_E[t] + g_J[t] + x_J_actual[t]).
    - Rygene:   revenue = p_t × min(G_MAX_R, i_R[t] + g_E_actual[t] + x_E_actual[t]).
No committed plan at Evenstad or Rygene — their actual generation adjusts
per-period to the actual water flowing in from upstream.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_acquisition.cascade_panel import (
    INFLOW_FRACTIONS, PLANT_G_MAX, PLANT_S_MAX, split_system_inflow
)
from src.forecasting.price_model import generate_price_scenarios
from src.forecasting.scenario_tree import (
    STAGE_WEEKS,
    build_scenario_tree,
    load_scenario_tree,
)
from src.forecasting.seasonal_model import generate_inflow_scenarios
from src.optimization.cascade_model import (
    G_MAX_J, G_MAX_E, G_MAX_R, S_MAX_J,
    build_cascade_deterministic_lp,
    build_cascade_stochastic_lp,
    extract_cascade_schedule,
)
from src.optimization.reservoir_model import DELTA_WEEKLY, solve_model

LOG = logging.getLogger(__name__)

SEED        = 42
N_RAW_SCEN  = 200
SCENARIOS_DIR = Path("results/scenarios")

G_MAX_E_val = PLANT_G_MAX["Evenstad"]
G_MAX_R_val = PLANT_G_MAX["Rygene"]
S_MAX_J_val = PLANT_S_MAX["Joerundland"]


# ── Helper: realised cascade revenue ─────────────────────────────────────────

def cascade_realised_revenue(
    gen_J_committed: np.ndarray,  # planned g_J from LP
    inflow_J: np.ndarray,
    inflow_E: np.ndarray,
    inflow_R: np.ndarray,
    realized_price: np.ndarray,
    s_init_J: float,
    T: int = 52,
) -> dict:
    """
    Apply committed g_J plan to realised inflows for all three plants.

    Jørundland: storage dynamics with clamping (same as single-reservoir).
    Evenstad:   pass-through with no storage; g_E = min(G_MAX_E, inflow_to_E).
    Rygene:     pass-through with no storage; g_R = min(G_MAX_R, inflow_to_R).

    Returns a dict with revenues, trajectories, and clamping counts.
    """
    storage_J = np.zeros(T + 1)
    storage_J[0] = s_init_J
    gen_J_actual = gen_J_committed.copy()
    gen_E_actual = np.zeros(T)
    gen_R_actual = np.zeros(T)
    spill_J      = np.zeros(T)
    spill_E      = np.zeros(T)
    spill_R      = np.zeros(T)
    n_cmin = 0
    n_cmax = 0

    for t in range(T):
        s  = storage_J[t]
        iJ = inflow_J[t]
        g_plan = gen_J_actual[t]

        # Clamp Jørundland generation
        g_max_feas  = min(G_MAX_J, s + iJ)
        g_min_neces = max(0.0, s + iJ - S_MAX_J)

        if g_plan < g_min_neces:
            gen_J_actual[t] = g_min_neces
            n_cmax += 1
        elif g_plan > g_max_feas:
            gen_J_actual[t] = g_max_feas
            n_cmin += 1

        spill_J[t] = max(0.0, s + iJ - gen_J_actual[t] - S_MAX_J)
        storage_J[t + 1] = s + iJ - gen_J_actual[t] - spill_J[t]

        # Evenstad: pass through J's total release + local inflow E
        inflow_to_E = inflow_E[t] + gen_J_actual[t] + spill_J[t]
        gen_E_actual[t] = min(G_MAX_E_val, inflow_to_E)
        spill_E[t]      = max(0.0, inflow_to_E - G_MAX_E_val)

        # Rygene: pass through E's total release + local inflow R
        inflow_to_R = inflow_R[t] + gen_E_actual[t] + spill_E[t]
        gen_R_actual[t] = min(G_MAX_R_val, inflow_to_R)
        spill_R[t]      = max(0.0, inflow_to_R - G_MAX_R_val)

    # Truncate to T in case the panel year has 53 ISO weeks (scenario tree is T=52)
    dt    = np.array([DELTA_WEEKLY ** (-(t + 1)) for t in range(T)])
    rev_J = float(np.sum(dt * 1000 * realized_price[:T] * gen_J_actual[:T]))
    rev_E = float(np.sum(dt * 1000 * realized_price[:T] * gen_E_actual[:T]))
    rev_R = float(np.sum(dt * 1000 * realized_price[:T] * gen_R_actual[:T]))

    return {
        "rev_J": rev_J, "rev_E": rev_E, "rev_R": rev_R,
        "rev_total": rev_J + rev_E + rev_R,
        "gen_J": gen_J_actual, "gen_E": gen_E_actual, "gen_R": gen_R_actual,
        "spill_J": spill_J, "spill_E": spill_E, "spill_R": spill_R,
        "storage_J": storage_J,
        "n_clamp_min": n_cmin, "n_clamp_max": n_cmax,
    }


# ── Open-loop cascade ─────────────────────────────────────────────────────────

def run_cascade_open_loop(
    tree: dict,
    inflow_J: np.ndarray,
    inflow_E: np.ndarray,
    inflow_R: np.ndarray,
    realized_price: np.ndarray,
    s_init_J: float,
) -> dict:
    """Solve cascade stochastic LP with full non-anticipativity on g_J."""
    # Split the scenario tree inflows into per-plant inflows
    leaf_inflow_total = tree["leaf_inflow"]   # (T, N)
    T, N = leaf_inflow_total.shape

    local = split_system_inflow(leaf_inflow_total)
    li_J = local["Joerundland"]
    li_E = local["Evenstad"]
    li_R = local["Rygene"]
    leaf_price   = tree["leaf_price"]
    leaf_weights = tree["leaf_weights"]

    m = build_cascade_stochastic_lp(
        li_J, li_E, li_R, leaf_price, leaf_weights,
        s_init_J=s_init_J, T=T,
        non_anticipative_stage_weeks=[list(range(1, T + 1))],
    )
    m, status = solve_model(m)
    result = extract_cascade_schedule(m, scenario_idx=0, T=T)

    # T from scenario tree (52); pass first T weeks of realized arrays
    rlz = cascade_realised_revenue(
        result.generation_J,
        inflow_J[:T], inflow_E[:T], inflow_R[:T], realized_price[:T],
        s_init_J, T,
    )
    return {"policy": "open_loop", "solver_status": status, **rlz}


# ── Closed-loop cascade ───────────────────────────────────────────────────────

def run_cascade_closed_loop(
    year: int,
    panel: pd.DataFrame,
    inflow_idata,
    price_idata,
    inflow_J: np.ndarray,
    inflow_E: np.ndarray,
    inflow_R: np.ndarray,
    realized_price: np.ndarray,
    s_init_J: float,
    T: int = 52,
) -> dict:
    """Rolling-horizon closed-loop: re-optimise g_J at each stage start."""
    rng = np.random.default_rng(SEED + year + 200)
    year_start = panel[panel["iso_year"] == year].index.min()
    all_dates = pd.date_range(year_start, periods=T, freq="W-MON")

    gen_J_committed = np.zeros(T)
    storage_J = np.zeros(T + 1)
    storage_J[0] = s_init_J

    for stage_idx, stage_weeks in enumerate(STAGE_WEEKS):
        t_start = stage_weeks[0] - 1
        t_end   = stage_weeks[-1]
        remaining_T = T - t_start
        if remaining_T <= 0:
            break

        forecast_start = all_dates[t_start]
        current_s_J = storage_J[t_start]

        inflow_scen = generate_inflow_scenarios(
            inflow_idata, panel,
            forecast_start_week=forecast_start,
            n_weeks=remaining_T,
            n_scenarios=N_RAW_SCEN,
            rng=rng,
        )
        price_scen = generate_price_scenarios(
            price_idata, inflow_idata, panel,
            forecast_start_week=forecast_start,
            inflow_scenarios=inflow_scen,
            n_weeks=remaining_T,
            rng=rng,
        )
        sub_stage_weeks = [
            [w - t_start for w in sw if t_start < w <= T]
            for sw in STAGE_WEEKS[stage_idx:]
        ]
        sub_stage_weeks = [sw for sw in sub_stage_weeks if sw]

        tree_rem = build_scenario_tree(
            inflow_scen, price_scen, forecast_start,
            n_raw=N_RAW_SCEN,
            custom_stage_weeks=sub_stage_weeks if sub_stage_weeks else None,
        )
        li_total = tree_rem["leaf_inflow"]
        li_local = split_system_inflow(li_total)

        rem_na_weeks = [
            [w - t_start for w in sw if w > t_start]
            for sw in STAGE_WEEKS[stage_idx:]
            if any(w > t_start for w in sw)
        ]
        rem_na_weeks_adj = [
            [w for w in sw if 1 <= w <= remaining_T]
            for sw in rem_na_weeks
        ]
        rem_na_weeks_adj = [sw for sw in rem_na_weeks_adj if sw]

        m = build_cascade_stochastic_lp(
            li_local["Joerundland"],
            li_local["Evenstad"],
            li_local["Rygene"],
            tree_rem["leaf_price"],
            tree_rem["leaf_weights"],
            s_init_J=current_s_J,
            T=remaining_T,
            non_anticipative_stage_weeks=rem_na_weeks_adj if rem_na_weeks_adj else None,
        )
        m, status = solve_model(m)
        result = extract_cascade_schedule(m, scenario_idx=0, T=remaining_T)

        # Commit this stage's g_J decisions
        stage_len = len(stage_weeks)
        for k, t_abs in enumerate(range(t_start, t_start + stage_len)):
            if t_abs < T:
                gen_J_committed[t_abs] = result.generation_J[k]

        # Advance Jørundland storage through the stage using realised inflow
        for k, t_abs in enumerate(range(t_start, t_start + stage_len)):
            if t_abs >= T:
                break
            s   = storage_J[t_abs]
            iJ  = inflow_J[t_abs]
            g   = gen_J_committed[t_abs]
            g   = min(g, min(G_MAX_J, s + iJ))
            g   = max(g, max(0.0, s + iJ - S_MAX_J))
            gen_J_committed[t_abs] = g
            sp  = max(0.0, s + iJ - g - S_MAX_J)
            storage_J[t_abs + 1] = s + iJ - g - sp

        LOG.info("  Cascade stage %d: weeks %d–%d, s_J_end=%.1f GWh",
                 stage_idx + 1, stage_weeks[0], stage_weeks[-1],
                 storage_J[t_start + stage_len]
                 if t_start + stage_len <= T else storage_J[T])

    rlz = cascade_realised_revenue(
        gen_J_committed, inflow_J, inflow_E, inflow_R, realized_price, s_init_J, T
    )
    return {"policy": "closed_loop", "solver_status": "optimal", **rlz}


# ── Perfect-foresight cascade ─────────────────────────────────────────────────

def run_cascade_perfect_foresight(
    inflow_J: np.ndarray,
    inflow_E: np.ndarray,
    inflow_R: np.ndarray,
    realized_price: np.ndarray,
    s_init_J: float,
    T: int = 52,
) -> dict:
    """Deterministic cascade LP with actual realised inflow and price."""
    m = build_cascade_deterministic_lp(
        inflow_J, inflow_E, inflow_R, realized_price, s_init_J, T
    )
    m, status = solve_model(m)

    ts = range(1, T + 1)
    gen_J = np.array([float(m.g_J[t].value) for t in ts])
    gen_E = np.array([float(m.g_E[t].value) for t in ts])
    gen_R = np.array([float(m.g_R[t].value) for t in ts])
    sto_J = np.array([float(m.s_J[t].value) for t in range(1, T + 2)])
    dt    = np.array([DELTA_WEEKLY ** (-(t + 1)) for t in range(T)])

    rev_J = float(np.sum(dt * 1000 * realized_price[:T] * gen_J))
    rev_E = float(np.sum(dt * 1000 * realized_price[:T] * gen_E))
    rev_R = float(np.sum(dt * 1000 * realized_price[:T] * gen_R))

    return {
        "policy": "perfect_foresight",
        "solver_status": status,
        "rev_J": rev_J, "rev_E": rev_E, "rev_R": rev_R,
        "rev_total": rev_J + rev_E + rev_R,
        "gen_J": gen_J, "gen_E": gen_E, "gen_R": gen_R,
        "storage_J": sto_J,
    }
