"""
Backtesting engine.

For each historical year in the panel, computes:
  1. Open-loop policy revenue
  2. Closed-loop (rolling-horizon) policy revenue
  3. Perfect-foresight (deterministic hindsight) revenue

and derives the value-of-flexibility decomposition.

Open-loop policy
----------------
Solve the stochastic LP once at the start of the year with full
non-anticipativity (all scenarios share one decision path).
The resulting generation plan (g_t, x_t)_{t=1..52} is committed and
applied to the *actual* realised inflow and price path, computing the
realised revenue as Σ_t δ^t × p_t^realized × g_t^committed.

Note: the storage trajectory under the realised inflow/plan may differ from
the planned trajectory. If the committed release would drive s_t below 0,
it is clamped to 0 (physical constraint). If it would exceed S_max, spill
is applied. These clamps are documented and their frequency reported.

Closed-loop (rolling-horizon recourse) policy
----------------------------------------------
At the start of each quarter (stages 1–4), re-solve the remaining-horizon
stochastic LP using:
  - current realised storage s_t as initial condition
  - a fresh scenario tree generated from the posterior predictive for
    the remaining horizon

The first stage's decisions from each re-solve are committed to the
real path. This is the rolling-horizon approximation to true recourse /
SDDP: it is correctly described as a SIMPLIFIED MULTISTAGE POLICY that
captures recourse in a computationally tractable way, not full SDDP.
Key differences from true SDDP are documented in methodology.md.

Perfect-foresight benchmark
----------------------------
Deterministic LP solved with actual realised inflow and price paths.
Upper bound on achievable revenue.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.price_model import generate_price_scenarios
from src.forecasting.scenario_tree import (
    STAGE_WEEKS,
    build_scenario_tree,
    load_scenario_tree,
)
from src.forecasting.seasonal_model import generate_inflow_scenarios
from src.optimization.reservoir_model import (
    G_MAX,
    S_MAX,
    DELTA_WEEKLY,
    ScheduleResult,
    build_deterministic_lp,
    build_stochastic_lp,
    extract_schedule,
    solve_model,
)

LOG = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
SCENARIOS_DIR = Path("results/scenarios")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_RAW_SCEN = 200   # raw scenarios for rolling-horizon re-solves


# ── Revenue calculation ───────────────────────────────────────────────────────

def realised_revenue(
    generation: np.ndarray,   # committed GWh/week plan
    realized_inflow: np.ndarray,
    realized_price: np.ndarray,
    s_init: float,
    T: int = 52,
) -> tuple[float, np.ndarray, int, int]:
    """
    Apply committed generation plan to realised inflow/price path.
    Clamps generation when storage constraints would be violated.
    Returns (total_discounted_revenue_NOK, storage_trajectory, n_clamp_min, n_clamp_max).
    """
    storage = np.zeros(T + 1)
    storage[0] = s_init
    gen_actual = generation.copy()
    n_clamp_min = 0
    n_clamp_max = 0

    for t in range(T):
        s = storage[t]
        i = realized_inflow[t]
        g_planned = gen_actual[t]

        # Maximum generation before storage hits zero
        g_max_feasible = min(G_MAX, s + i)
        # Minimum generation (spill if storage would exceed S_max)
        g_min_necessary = max(0.0, s + i - S_MAX)

        if g_planned < g_min_necessary:
            gen_actual[t] = g_min_necessary
            n_clamp_min += 1
        elif g_planned > g_max_feasible:
            gen_actual[t] = g_max_feasible
            n_clamp_max += 1

        spill = max(0.0, s + i - gen_actual[t] - S_MAX)
        storage[t + 1] = s + i - gen_actual[t] - spill

    delta_t = np.array([DELTA_WEEKLY ** (-(t + 1)) for t in range(T)])
    revenue = float(np.sum(delta_t * 1000 * realized_price * gen_actual))
    return revenue, storage, n_clamp_min, n_clamp_max


# ── Open-loop policy ──────────────────────────────────────────────────────────

def run_open_loop(
    tree: dict,
    realized_inflow: np.ndarray,
    realized_price: np.ndarray,
    s_init: float,
) -> dict:
    """
    Open-loop: solve once with full non-anticipativity (single decision path).
    """
    leaf_inflow = tree["leaf_inflow"]
    leaf_price = tree["leaf_price"]
    leaf_weights = tree["leaf_weights"]
    T = leaf_inflow.shape[0]

    # One big stage — all scenarios share same decisions
    m = build_stochastic_lp(
        leaf_inflow, leaf_price, leaf_weights, s_init=s_init, T=T,
        non_anticipative_stage_weeks=[list(range(1, T + 1))],
    )
    m, status = solve_model(m)
    result = extract_schedule(m, scenario_idx=0, T=T)

    rev, storage, n_cmin, n_cmax = realised_revenue(
        result.generation, realized_inflow, realized_price, s_init
    )
    return {
        "policy": "open_loop",
        "planned_generation": result.generation,
        "realized_revenue_NOK": rev,
        "storage_trajectory": storage,
        "n_clamp_min": n_cmin,
        "n_clamp_max": n_cmax,
        "solver_status": status,
    }


# ── Closed-loop (rolling-horizon) policy ──────────────────────────────────────

def run_closed_loop(
    year: int,
    panel: pd.DataFrame,
    inflow_idata,
    price_idata,
    realized_inflow: np.ndarray,
    realized_price: np.ndarray,
    s_init: float,
    T: int = 52,
) -> dict:
    """
    Closed-loop rolling-horizon policy: re-optimise at each stage start.
    """
    rng = np.random.default_rng(SEED + year + 100)
    year_start = panel[panel["iso_year"] == year].index.min()
    all_dates = pd.date_range(year_start, periods=T, freq="W-MON")

    generation_committed = np.zeros(T)
    storage = np.zeros(T + 1)
    storage[0] = s_init

    for stage_idx, stage_weeks in enumerate(STAGE_WEEKS):
        t_start = stage_weeks[0] - 1    # 0-indexed
        t_end = stage_weeks[-1]         # exclusive end (0-indexed)
        remaining_T = T - t_start
        if remaining_T <= 0:
            break

        forecast_start = all_dates[t_start]
        current_s = storage[t_start]

        # Generate fresh scenario tree for remaining horizon
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
        remaining_stage_weeks = [
            [w - t_start for w in sw if w > t_start]
            for sw in STAGE_WEEKS[stage_idx:]
            if any(w > t_start for w in sw)
        ]
        # Ensure week indices are 1-based relative to remaining horizon
        remaining_stage_weeks_adj = []
        for sw in remaining_stage_weeks:
            adj = [w for w in sw if 1 <= w <= remaining_T]
            if adj:
                remaining_stage_weeks_adj.append(adj)

        # Compute stage weeks relative to this sub-horizon (1-indexed)
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
        leaf_inflow = tree_rem["leaf_inflow"]
        leaf_price = tree_rem["leaf_price"]
        leaf_weights = tree_rem["leaf_weights"]

        m = build_stochastic_lp(
            leaf_inflow, leaf_price, leaf_weights,
            s_init=current_s,
            T=remaining_T,
            non_anticipative_stage_weeks=remaining_stage_weeks_adj if remaining_stage_weeks_adj else None,
        )
        m, status = solve_model(m)
        result = extract_schedule(m, scenario_idx=0, T=remaining_T)

        # Commit this stage's decisions
        stage_len = len(stage_weeks)
        for k, t_abs in enumerate(range(t_start, t_start + stage_len)):
            if t_abs < T:
                generation_committed[t_abs] = result.generation[k]

        # Advance storage through this stage using realised inflow
        for k, t_abs in enumerate(range(t_start, t_start + stage_len)):
            if t_abs >= T:
                break
            s = storage[t_abs]
            i_real = realized_inflow[t_abs]
            g = generation_committed[t_abs]
            g = min(g, min(G_MAX, s + i_real))
            g = max(g, max(0.0, s + i_real - S_MAX))
            generation_committed[t_abs] = g
            spill = max(0.0, s + i_real - g - S_MAX)
            storage[t_abs + 1] = s + i_real - g - spill

        LOG.info("  Stage %d complete: weeks %d–%d, s_end=%.1f GWh",
                 stage_idx + 1, stage_weeks[0], stage_weeks[-1],
                 storage[t_start + stage_len] if t_start + stage_len <= T else storage[T])

    delta_t = np.array([DELTA_WEEKLY ** (-(t + 1)) for t in range(T)])
    revenue = float(np.sum(delta_t * 1000 * realized_price * generation_committed))

    return {
        "policy": "closed_loop",
        "realized_generation": generation_committed,
        "realized_revenue_NOK": revenue,
        "storage_trajectory": storage,
        "solver_status": "optimal",
    }


# ── Perfect-foresight benchmark ───────────────────────────────────────────────

def run_perfect_foresight(
    realized_inflow: np.ndarray,
    realized_price: np.ndarray,
    s_init: float,
    T: int = 52,
) -> dict:
    """
    Deterministic LP using actual realised inflow and price.
    """
    m = build_deterministic_lp(realized_inflow, realized_price, s_init, T)
    m, status = solve_model(m)

    gen = np.array([float(m.g[t].value) for t in range(1, T + 1)])
    spl = np.array([float(m.x[t].value) for t in range(1, T + 1)])
    sto = np.array([float(m.s[t].value) for t in range(1, T + 2)])
    delta_t = np.array([DELTA_WEEKLY ** (-(t + 1)) for t in range(T)])
    revenue = float(np.sum(delta_t * 1000 * realized_price * gen))

    return {
        "policy": "perfect_foresight",
        "generation": gen,
        "spill": spl,
        "realized_revenue_NOK": revenue,
        "storage_trajectory": sto,
        "solver_status": status,
    }
