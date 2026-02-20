"""
Live LP recompute for the capacity-slider demo.

Runs OL and PF cascade LPs for a chosen year with a user-supplied S_MAX_J.
Returns revenue and weekly trajectory data for plotting.

Only g_J (Jørundland release) is decided; Evenstad and Rygene are always
at capacity given local inflow, so their revenue is effectively constant
and the interesting dynamic is entirely at Jørundland.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.optimization.cascade_model as _cm
from src.data_acquisition.cascade_panel import split_system_inflow
from src.optimization.cascade_model import (
    G_MAX_E, G_MAX_R,
    build_cascade_deterministic_lp,
    build_cascade_stochastic_lp,
    extract_cascade_schedule,
)
from src.optimization.reservoir_model import DELTA_WEEKLY, solve_model


class LiveSolveError(Exception):
    """
    Raised when the live LP solve cannot produce a usable result.

    `infeasible` is True when the solver determined no physically valid
    schedule exists for the chosen settings (most commonly: the chosen
    reservoir capacity is smaller than the storage already required to
    hold the initial water level for that year). It is False for other
    solver-related failures (e.g. solver unavailable, numerical issues).
    """

    def __init__(self, message: str, infeasible: bool = False):
        super().__init__(message)
        self.infeasible = infeasible


def _solve_or_raise(m):
    """
    Solve a model, translating solver failures into LiveSolveError.

    Pyomo's HiGHS interface can fail in two different ways for an
    infeasible model: solve_model's own RuntimeError (termination
    condition not optimal), or a lower-level exception raised while
    attempting to load a solution that was never found (e.g.
    NoFeasibleSolutionError). Both are treated as solver failures here.
    """
    try:
        return solve_model(m)
    except Exception as exc:
        status = str(exc).lower()
        # Catches both "infeasible" (HiGHS termination condition) and
        # "a feasible solution was not found" (Pyomo's solution-loading error).
        infeasible = "feasible" in status
        raise LiveSolveError(str(exc), infeasible=infeasible) from exc


@contextlib.contextmanager
def _override_smax(s_max_j: float):
    """Temporarily override S_MAX_J and S_MID_J in the cascade_model module."""
    old_smax = _cm.S_MAX_J
    old_smid = _cm.S_MID_J
    _cm.S_MAX_J = s_max_j
    _cm.S_MID_J = 0.5 * s_max_j
    try:
        yield
    finally:
        _cm.S_MAX_J = old_smax
        _cm.S_MID_J = old_smid


def run_live_solve(
    year: int,
    s_max_j: float,
    tree: dict,
    panel_year: pd.DataFrame,
    s_init_j: float,
) -> dict:
    """
    Solve OL stochastic LP and PF deterministic LP with custom S_MAX_J.

    Returns a dict with:
      - ol_rev, pf_rev: total system revenues (MNOK)
      - ol_rev_j, pf_rev_j: Jørundland-only revenues (MNOK)
      - ol_storage, pf_storage: storage trajectory (T+1,) GWh
      - ol_gen_j, pf_gen_j: Jørundland generation schedule (T,) GWh/wk
      - weeks: list of ISO week numbers
      - price: realized weekly price (NOK/MWh)
      - inflow_total: realized total system inflow (GWh/wk)
    """
    T = 52
    price = panel_year["price_avg_NOK_MWh"].values[:T]
    inflow_total = panel_year["inflow_GWh_week"].values[:T]

    fracs = split_system_inflow(inflow_total.reshape(T, 1))
    iJ = fracs["Joerundland"].reshape(T)
    iE = fracs["Evenstad"].reshape(T)
    iR = fracs["Rygene"].reshape(T)

    # Scenario-tree inflows (subsample to keep the demo solve fast)
    li_total = tree["leaf_inflow"]          # (T_sc, N)
    T_sc = li_total.shape[0]
    N_full = li_total.shape[1]
    N_DEMO = 16
    idx = np.linspace(0, N_full - 1, min(N_DEMO, N_full), dtype=int)
    li_total = li_total[:, idx]
    li_local = split_system_inflow(li_total)
    li_J = li_local["Joerundland"]
    li_E = li_local["Evenstad"]
    li_R = li_local["Rygene"]
    leaf_price   = tree["leaf_price"][:, idx]
    leaf_weights = tree["leaf_weights"][idx]
    leaf_weights = leaf_weights / leaf_weights.sum()

    dt = np.array([DELTA_WEEKLY ** (-(t + 1)) for t in range(T)])

    with _override_smax(s_max_j):
        # ── OL: stochastic LP with full non-anticipativity on g_J ──────────
        m_ol = build_cascade_stochastic_lp(
            li_J, li_E, li_R, leaf_price, leaf_weights,
            s_init_J=s_init_j, T=T_sc,
            non_anticipative_stage_weeks=[list(range(1, T_sc + 1))],
        )
        m_ol, _ = _solve_or_raise(m_ol)
        sched_ol = extract_cascade_schedule(m_ol, scenario_idx=0, T=T_sc)

        # Apply committed g_J to realized inflows
        ol = _apply_cascade_realized(
            sched_ol.generation_J[:T], iJ, iE, iR, price, s_init_j, s_max_j, T, dt
        )

        # ── PF: deterministic LP with realized inflow/price ─────────────────
        m_pf = build_cascade_deterministic_lp(iJ, iE, iR, price, s_init_j, T)
        m_pf, _ = _solve_or_raise(m_pf)
        import pyomo.environ as _pyo
        ts = range(1, T + 1)
        gen_j_pf = np.array([_pyo.value(m_pf.g_J[t]) for t in ts])
        pf = _apply_cascade_realized(
            gen_j_pf, iJ, iE, iR, price, s_init_j, s_max_j, T, dt,
            pf_mode=True,
        )

    weeks = panel_year["iso_week"].values[:T].tolist()

    return {
        "ol_rev":      ol["rev_total"] / 1e6,
        "pf_rev":      pf["rev_total"] / 1e6,
        "ol_rev_j":    ol["rev_j"] / 1e6,
        "pf_rev_j":    pf["rev_j"] / 1e6,
        "ol_storage":  ol["storage_j"],
        "pf_storage":  pf["storage_j"],
        "ol_gen_j":    ol["gen_j"],
        "pf_gen_j":    pf["gen_j"],
        "weeks":       weeks,
        "price":       price,
        "inflow_total": inflow_total[:T],
    }


def _apply_cascade_realized(
    gen_j_plan: np.ndarray,
    iJ: np.ndarray,
    iE: np.ndarray,
    iR: np.ndarray,
    price: np.ndarray,
    s_init_j: float,
    s_max_j: float,
    T: int,
    dt: np.ndarray,
    pf_mode: bool = False,
) -> dict:
    """Apply a committed g_J schedule to realized inflows and compute revenues."""
    storage_j = np.zeros(T + 1)
    storage_j[0] = s_init_j
    gen_j = np.array(gen_j_plan[:T], dtype=float)
    gen_e = np.zeros(T)
    gen_r = np.zeros(T)
    spill_j = np.zeros(T)

    for t in range(T):
        s   = storage_j[t]
        g   = gen_j[t]
        g_max_feas = min(_cm.G_MAX_J, s + iJ[t])
        g_min_nec  = max(0.0, s + iJ[t] - s_max_j)
        gen_j[t]   = np.clip(g, g_min_nec, g_max_feas)
        spill_j[t] = max(0.0, s + iJ[t] - gen_j[t] - s_max_j)
        storage_j[t + 1] = s + iJ[t] - gen_j[t] - spill_j[t]

        in_e = iE[t] + gen_j[t] + spill_j[t]
        sp_e = max(0.0, in_e - G_MAX_E)
        gen_e[t] = min(G_MAX_E, in_e)

        in_r = iR[t] + gen_e[t] + sp_e
        gen_r[t] = min(G_MAX_R, in_r)

    rev_j = float(np.sum(dt * 1000 * price * gen_j))
    rev_e = float(np.sum(dt * 1000 * price * gen_e))
    rev_r = float(np.sum(dt * 1000 * price * gen_r))

    return {
        "rev_j": rev_j,
        "rev_e": rev_e,
        "rev_r": rev_r,
        "rev_total": rev_j + rev_e + rev_r,
        "storage_j": storage_j,
        "gen_j": gen_j,
    }
