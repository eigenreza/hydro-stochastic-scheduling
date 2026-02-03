"""
Three-plant Arendalsvassdraget cascade LP.

Physical structure
------------------
Cascade (upstream → downstream): Jørundland → Evenstad → Rygene.

Water released by plant n at period t flows into plant n+1 within the
SAME weekly time step (zero routing lag). This is the standard simplification
for weekly-resolution cascade scheduling when inter-plant travel time is well
below one week.

Travel-time justification:
    Jørundland → Evenstad: ~17 km along Nidelva.
    Evenstad  → Rygene:    ~26 km (Rygene is ~9 km from river mouth;
                                    Evenstad ~35 km from mouth).
    At typical low-flow velocity ~0.5 m/s: travel time ~9–15 h.
    At high-flow velocity ~2 m/s: ~2–4 h.
    Both are well below the weekly resolution → same-week routing
    assumption is conservative and defensible. Documented as Limitation B.

Reservoir structure
-------------------
Jørundland:  TRUE RESERVOIR. Working storage S_MAX_J = 167 GWh.
             Source: NVE HydAPI station 19.5.0 (Nesvatn volume series,
             parameter 1004, daily 1990–2025); EnEkv=0.638 kWh/m³
             from NVE Vannkraftdatabase. Working range: 2.87–265.83
             million m³ → 167.8 GWh usable.
Evenstad:    RUN-OF-RIVER. Head=17.3 m, no meaningful reservoir storage.
             S_MAX_E = 0 (zero storage constraint = pass-through equality).
Rygene:      RUN-OF-RIVER. Head=36.2 m, no meaningful reservoir storage.
             S_MAX_R = 0.

Cascade dynamics (same-week routing, no storage at E and R):
    s_J[t+1] = s_J[t] + i_J[t] − g_J[t] − x_J[t]
    0        = i_E[t] + (g_J[t] + x_J[t]) − g_E[t] − x_E[t]   (E: no storage)
    0        = i_R[t] + (g_E[t] + x_E[t]) − g_R[t] − x_R[t]   (R: no storage)

    Bounds:   0 ≤ s_J[t] ≤ S_MAX_J
              0 ≤ g_n[t] ≤ G_MAX_n    for n ∈ {J, E, R}
              x_n[t] ≥ 0               (spill is physical overflow)

Objective (system revenue, discounted):
    max Σ_ω π_ω Σ_t δ^t × (p_t^ω / 1000) × (g_J[t] + g_E[ω,t] + g_R[ω,t])

Note: g_J is subject to non-anticipativity (it is the controllable decision);
      g_E and g_R are ALWAYS scenario-specific because they respond to the
      realized inflow even when g_J is fixed, since i_E^ω and i_R^ω vary
      across scenarios. Spill at every node is always scenario-specific.

Non-anticipativity
------------------
Applied to g_J only (the only storage decision variable):
    Open-loop:    g_J[t] same across all scenarios at every t.
    Closed-loop:  g_J[t] same within each rolling-horizon stage.
g_E and g_R are NOT subject to non-anticipativity; they adjust per-scenario.

Relationship to single-reservoir model
---------------------------------------
The single-reservoir model (Phase A) aggregated all 44 Arendalsvassdraget
plants into one virtual reservoir with system-effective head H_eff=264.5 m
and S_MAX=1300 GWh. The cascade model here represents only these three plants
(23% of the system by MidProd: 580.6/2498.4 GWh). The two models represent
different levels of abstraction of the same physical system and are not
directly comparable in revenue scale, but the VoF/flexibility ratio is
meaningful to compare — see methodology.md §B.7.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from src.data_acquisition.cascade_panel import (
    PLANT_G_MAX,
    PLANT_S_MAX,
    TERMINAL_VBAR,
)
from src.optimization.reservoir_model import (
    R_ANNUAL,
    DELTA_WEEKLY,
    solve_model,
)

LOG = logging.getLogger(__name__)

PLANTS     = ("Joerundland", "Evenstad", "Rygene")
PLANT_IDX  = {p: i for i, p in enumerate(PLANTS)}

G_MAX_J = PLANT_G_MAX["Joerundland"]    # 9.274 GWh/wk
G_MAX_E = PLANT_G_MAX["Evenstad"]       # 3.948 GWh/wk
G_MAX_R = PLANT_G_MAX["Rygene"]         # 9.240 GWh/wk
S_MAX_J = PLANT_S_MAX["Joerundland"]    # 167.0 GWh
S_MID_J = 0.5 * S_MAX_J


@dataclass
class CascadeScheduleResult:
    """Solved cascade schedule for all plants under one scenario."""
    generation_J: np.ndarray   # GWh/wk, shape (T,)
    generation_E: np.ndarray
    generation_R: np.ndarray
    spill_J:      np.ndarray
    spill_E:      np.ndarray
    spill_R:      np.ndarray
    storage_J:    np.ndarray   # GWh, shape (T+1,)
    total_revenue: float       # discounted NOK
    solver_status: str


def build_cascade_deterministic_lp(
    inflow_J: np.ndarray,   # (T,) GWh/wk
    inflow_E: np.ndarray,
    inflow_R: np.ndarray,
    price: np.ndarray,      # (T,) NOK/MWh
    s_init_J: float,
    T: int = 52,
) -> pyo.ConcreteModel:
    """
    Deterministic cascade LP for a single realised inflow/price path.
    Used for perfect-foresight benchmark and closed-loop re-solves.
    """
    assert len(inflow_J) == len(inflow_E) == len(inflow_R) == len(price) == T

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)

    # Decision variables
    m.g_J = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, G_MAX_J))
    m.g_E = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, G_MAX_E))
    m.g_R = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, G_MAX_R))
    m.x_J = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.x_E = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.x_R = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.s_J = pyo.Var(pyo.RangeSet(1, T + 1), within=pyo.NonNegativeReals,
                    bounds=(0, S_MAX_J))
    m.tv  = pyo.Var(within=pyo.Reals)   # terminal value for Jørundland storage

    # Data dictionaries
    iJ = {t: float(inflow_J[t - 1]) for t in range(1, T + 1)}
    iE = {t: float(inflow_E[t - 1]) for t in range(1, T + 1)}
    iR = {t: float(inflow_R[t - 1]) for t in range(1, T + 1)}
    pp = {t: float(price[t - 1])    for t in range(1, T + 1)}
    dt = {t: DELTA_WEEKLY ** (-t)   for t in range(1, T + 1)}

    # Initial storage
    m.init_J = pyo.Constraint(expr=m.s_J[1] == s_init_J)

    # Jørundland dynamics
    m.dyn_J = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.s_J[t + 1] == m.s_J[t] + iJ[t] - m.g_J[t] - m.x_J[t]
    )

    # Evenstad: no storage (pass-through equality)
    m.pass_E = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.g_E[t] + m.x_E[t] == iE[t] + m.g_J[t] + m.x_J[t]
    )

    # Rygene: no storage (pass-through equality)
    m.pass_R = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.g_R[t] + m.x_R[t] == iR[t] + m.g_E[t] + m.x_E[t]
    )

    # Terminal value for Jørundland (piecewise linear)
    m.tv_c1 = pyo.Constraint(expr=m.tv <= TERMINAL_VBAR * m.s_J[T + 1])
    m.tv_c2 = pyo.Constraint(
        expr=m.tv <= TERMINAL_VBAR * S_MID_J
             + 0.5 * TERMINAL_VBAR * (m.s_J[T + 1] - S_MID_J)
    )

    # Objective: total system revenue
    def obj_rule(m):
        rev = sum(
            dt[t] * 1000 * pp[t] * (m.g_J[t] + m.g_E[t] + m.g_R[t])
            for t in range(1, T + 1)
        )
        return rev + m.tv
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m


def build_cascade_stochastic_lp(
    leaf_inflow_J: np.ndarray,   # (T, N_scen)
    leaf_inflow_E: np.ndarray,
    leaf_inflow_R: np.ndarray,
    leaf_price: np.ndarray,      # (T, N_scen)
    leaf_weights: np.ndarray,    # (N_scen,)
    s_init_J: float,
    T: int = 52,
    non_anticipative_stage_weeks: list[list[int]] | None = None,
) -> pyo.ConcreteModel:
    """
    Extensive-form cascade stochastic LP.

    Non-anticipativity applies to g_J only (the storage decision).
    g_E and g_R are always scenario-specific (they pass through whatever
    flows in, which depends on per-scenario local inflows).

    Parameters
    ----------
    non_anticipative_stage_weeks:
        Stage groupings for g_J non-anticipativity constraints.
        None → full single-stage (open-loop g_J).
        Provide STAGE_WEEKS → closed-loop g_J.
    """
    N = leaf_inflow_J.shape[1]
    assert leaf_inflow_J.shape == leaf_inflow_E.shape == leaf_inflow_R.shape \
           == leaf_price.shape == (T, N)
    assert len(leaf_weights) == N

    if non_anticipative_stage_weeks is None:
        non_anticipative_stage_weeks = [list(range(1, T + 1))]

    m = pyo.ConcreteModel()
    m.T_set = pyo.RangeSet(1, T)
    m.S_set = pyo.RangeSet(0, N - 1)

    dt = {t: DELTA_WEEKLY ** (-t) for t in range(1, T + 1)}

    # ── Jørundland (storage, non-anticipative g_J) ──────────────────────────
    m.g_J = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals, bounds=(0, G_MAX_J))
    m.x_J = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals)
    m.s_J = pyo.Var(m.S_set, pyo.RangeSet(1, T + 1),
                    within=pyo.NonNegativeReals, bounds=(0, S_MAX_J))
    m.tv  = pyo.Var(m.S_set, within=pyo.Reals)

    # ── Evenstad and Rygene (run-of-river, SCENARIO-SPECIFIC) ───────────────
    m.g_E = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals, bounds=(0, G_MAX_E))
    m.x_E = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals)
    m.g_R = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals, bounds=(0, G_MAX_R))
    m.x_R = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals)

    # Initial Jørundland storage (same for all scenarios)
    m.init_J = pyo.Constraint(
        m.S_set,
        rule=lambda m, scen: m.s_J[scen, 1] == s_init_J
    )

    # Jørundland dynamics
    def dyn_J_rule(m, scen, t):
        return m.s_J[scen, t + 1] == (m.s_J[scen, t]
                                        + float(leaf_inflow_J[t - 1, scen])
                                        - m.g_J[scen, t]
                                        - m.x_J[scen, t])
    m.dyn_J = pyo.Constraint(m.S_set, m.T_set, rule=dyn_J_rule)

    # Evenstad pass-through (no storage)
    def pass_E_rule(m, scen, t):
        return (m.g_E[scen, t] + m.x_E[scen, t]
                == float(leaf_inflow_E[t - 1, scen])
                   + m.g_J[scen, t] + m.x_J[scen, t])
    m.pass_E = pyo.Constraint(m.S_set, m.T_set, rule=pass_E_rule)

    # Rygene pass-through (no storage)
    def pass_R_rule(m, scen, t):
        return (m.g_R[scen, t] + m.x_R[scen, t]
                == float(leaf_inflow_R[t - 1, scen])
                   + m.g_E[scen, t] + m.x_E[scen, t])
    m.pass_R = pyo.Constraint(m.S_set, m.T_set, rule=pass_R_rule)

    # Terminal value for Jørundland
    m.tv_c1 = pyo.Constraint(
        m.S_set,
        rule=lambda m, scen: m.tv[scen] <= TERMINAL_VBAR * m.s_J[scen, T + 1]
    )
    m.tv_c2 = pyo.Constraint(
        m.S_set,
        rule=lambda m, scen: m.tv[scen] <= (
            TERMINAL_VBAR * S_MID_J
            + 0.5 * TERMINAL_VBAR * (m.s_J[scen, T + 1] - S_MID_J)
        )
    )

    # Non-anticipativity on g_J only
    # g_E and g_R are intentionally NOT constrained — they vary with inflow.
    na_constraints = {}
    for stage_idx, stage_weeks in enumerate(non_anticipative_stage_weeks):
        for t in stage_weeks:
            for scen in range(1, N):
                na_constraints[(stage_idx, t, scen)] = (m.g_J[scen, t] == m.g_J[0, t])
    m.non_anticip = pyo.ConstraintList()
    for expr in na_constraints.values():
        m.non_anticip.add(expr)

    # Objective
    def obj_rule(m):
        return sum(
            float(leaf_weights[scen]) * (
                sum(
                    dt[t] * 1000 * float(leaf_price[t - 1, scen])
                    * (m.g_J[scen, t] + m.g_E[scen, t] + m.g_R[scen, t])
                    for t in range(1, T + 1)
                ) + m.tv[scen]
            )
            for scen in range(N)
        )
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m


def extract_cascade_schedule(
    m: pyo.ConcreteModel,
    scenario_idx: int = 0,
    T: int = 52,
) -> CascadeScheduleResult:
    """Extract generation/spill/storage for all three plants."""
    ts = range(1, T + 1)
    g_J = np.array([pyo.value(m.g_J[scenario_idx, t]) for t in ts])
    g_E = np.array([pyo.value(m.g_E[scenario_idx, t]) for t in ts])
    g_R = np.array([pyo.value(m.g_R[scenario_idx, t]) for t in ts])
    x_J = np.array([pyo.value(m.x_J[scenario_idx, t]) for t in ts])
    x_E = np.array([pyo.value(m.x_E[scenario_idx, t]) for t in ts])
    x_R = np.array([pyo.value(m.x_R[scenario_idx, t]) for t in ts])
    s_J = np.array([pyo.value(m.s_J[scenario_idx, t]) for t in range(1, T + 2)])
    total_rev = float(pyo.value(m.obj))
    return CascadeScheduleResult(
        generation_J=g_J, generation_E=g_E, generation_R=g_R,
        spill_J=x_J, spill_E=x_E, spill_R=x_R,
        storage_J=s_J, total_revenue=total_rev, solver_status="optimal",
    )
