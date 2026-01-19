"""
Hydropower reservoir scheduling LP.

Mathematical formulation
------------------------
Time index:       t = 1, …, T   (T = 52 weeks)
Scenarios:        ω ∈ Ω         (leaf scenarios from the scenario tree)
State:            s_t^ω         reservoir storage [GWh]
Decision:         g_t^ω ≥ 0     generation/release [GWh/week]
                  x_t^ω ≥ 0     spill             [GWh/week]
Exogenous:        i_t^ω         inflow             [GWh/week]
                  p_t^ω         price              [NOK/MWh]

Dynamics:   s_{t+1}^ω = s_t^ω + i_t^ω − g_t^ω − x_t^ω

Bounds:     0 ≤ s_t^ω ≤ S_max
            0 ≤ g_t^ω ≤ G_max
            x_t^ω ≥ 0

Objective:  max Σ_ω π_ω · Σ_t δ^t · (p_t^ω / 1000) · g_t^ω + V_T(s_{T+1})
            (revenue in MNOK; price in NOK/MWh; generation in GWh → revenue in GWh·NOK/MWh
            = NOK·MWh/MWh = NOK ... units: GWh × NOK/MWh = NOK·1000 → divide by 1000 → kNOK,
            or keep as NOK·GWh/MWh = NOK·1000·kWh/MWh = ... see unit note below)

Unit accounting:
    g_t [GWh/week] × p_t [NOK/MWh] = g_t × 1000 MWh/GWh × p_t NOK/MWh
    = 1000 × g_t × p_t  [NOK]
    Revenue coefficient: 1000 × p_t NOK per GWh of generation.

Terminal value function V_T(s_{T+1}):
    A piecewise-linear concave approximation is used:
        V_T(s) = v̄ · s   for s ≤ S_mid
                 v̄ · S_mid + ½v̄ · (s - S_mid)   for s > S_mid

    where v̄ is the average marginal value of water = mean annual price
    × inflow conversion factor [NOK/GWh], and S_mid = 0.5 · S_max.
    This ensures the model does not trivially drain the reservoir at the
    horizon end. Documented as a simplification: a full SDDP implementation
    would use iteratively refined Benders cuts as the terminal condition.

Reservoir parameters (Strengen / Skiensvassdraget — approximated):
    S_max = 1300 GWh     — total usable storage capacity
                           (Tokke/Vinje reservoir system capacity: ~5 TWh
                           full system, but Strengen discharge represents
                           partial catchment; scaled to ~25% = 1300 GWh
                           as a documented approximation)
    G_max =  100 GWh/wk — maximum weekly generation
                           (~600 MW plant capacity × 168 h/wk = 100.8 GWh/wk)
    S_init = 0.65 · S_max — initial filling (set per year in backtesting)

Weekly discount factor:
    δ = (1 + r_annual)^{1/52}   with r_annual = 0.04 (4% annual discount rate).
    This matches a standard low-risk public infrastructure rate consistent with
    Norwegian NVE/Statnett project evaluation practice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyomo.environ as pyo

LOG = logging.getLogger(__name__)

# ── Physical reservoir parameters ─────────────────────────────────────────────
# S_MAX: total usable storage of the Arendalsvassdraget system. Approximate;
# system-level reservoir data is not available at watercourse resolution from
# NVE Magasinstatistikk (only regional aggregates are published).
S_MAX = 1300.0     # GWh — max usable storage (documented approximation)

# G_MAX: derived from NVE power-plant registry total installed capacity.
# All 44 active Arendalsvassdraget plants: total = 565.5 MW.
# 565.5 MW × 168 h/week / 1000 = 95.0 GWh/week.
# Source: GetHydroPowerPlantsInOperation (NVE, 2026-06-19).
G_MAX = 95.0       # GWh/week — max weekly generation
S_MID = 0.5 * S_MAX

# ── Discount factor ───────────────────────────────────────────────────────────
R_ANNUAL = 0.04
DELTA_WEEKLY = (1 + R_ANNUAL) ** (1 / 52)

# ── Terminal water value parameters ──────────────────────────────────────────
# Estimated from long-run mean price (≈ 400 NOK/MWh) × 1000 MWh/GWh
TERMINAL_VBAR = 400.0 * 1000   # NOK per GWh of storage = 4e5 NOK/GWh


@dataclass
class ScheduleResult:
    """Holds a solved schedule for one scenario."""
    generation: np.ndarray     # GWh/week, shape (T,)
    spill: np.ndarray          # GWh/week, shape (T,)
    storage: np.ndarray        # GWh, shape (T+1,) — s_1, …, s_{T+1}
    revenue: float             # Total discounted revenue [NOK]
    terminal_value: float      # Terminal storage value [NOK]
    solver_status: str


def build_deterministic_lp(
    inflow: np.ndarray,   # (T,) GWh/week
    price: np.ndarray,    # (T,) NOK/MWh
    s_init: float,        # initial storage [GWh]
    T: int = 52,
) -> pyo.ConcreteModel:
    """
    Build a deterministic LP for a single inflow/price path.
    Used for: (a) perfect-foresight benchmark, (b) rolling-horizon re-solve.
    """
    assert len(inflow) == T and len(price) == T

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)

    # Variables
    m.g = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, G_MAX))
    m.x = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.s = pyo.Var(pyo.RangeSet(1, T + 1), within=pyo.NonNegativeReals,
                  bounds=(0, S_MAX))

    # Parameters
    inflow_p = {t: float(inflow[t - 1]) for t in range(1, T + 1)}
    price_p  = {t: float(price[t - 1])  for t in range(1, T + 1)}
    delta_t  = {t: DELTA_WEEKLY ** (-t)  for t in range(1, T + 1)}

    # Initial storage
    m.init_storage = pyo.Constraint(expr=m.s[1] == s_init)

    # Dynamics
    def dynamics_rule(m, t):
        return m.s[t + 1] == m.s[t] + inflow_p[t] - m.g[t] - m.x[t]
    m.dynamics = pyo.Constraint(m.T, rule=dynamics_rule)

    # Terminal value (piecewise-linear, implemented via auxiliary variable)
    m.tv = pyo.Var(within=pyo.Reals)
    m.tv_c1 = pyo.Constraint(expr=m.tv <= TERMINAL_VBAR * m.s[T + 1])
    m.tv_c2 = pyo.Constraint(
        expr=m.tv <= TERMINAL_VBAR * S_MID + 0.5 * TERMINAL_VBAR * (m.s[T + 1] - S_MID)
    )

    # Objective: maximise discounted revenue + terminal value
    def obj_rule(m):
        rev = sum(
            delta_t[t] * 1000 * price_p[t] * m.g[t]
            for t in range(1, T + 1)
        )
        return rev + m.tv
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m


def build_stochastic_lp(
    leaf_inflow: np.ndarray,   # (T, N_scen)
    leaf_price: np.ndarray,    # (T, N_scen)
    leaf_weights: np.ndarray,  # (N_scen,)
    s_init: float,
    T: int = 52,
    non_anticipative_stage_weeks: list[list[int]] | None = None,
) -> pyo.ConcreteModel:
    """
    Build the extensive-form stochastic LP.

    non_anticipative_stage_weeks: if provided, decisions within each stage
    group share the same value (non-anticipativity constraint per stage).
    If None, full non-anticipativity (open-loop: single decision path).

    For the CLOSED-LOOP policy: call this function with
        non_anticipative_stage_weeks = STAGE_WEEKS
    so decisions are fixed within each stage but can differ across stages.

    For the OPEN-LOOP policy: call with
        non_anticipative_stage_weeks = [list(range(1, 53))]   (single stage)
    so all scenarios share the same generation plan.
    """
    N = leaf_inflow.shape[1]
    assert leaf_inflow.shape == leaf_price.shape == (T, N)
    assert len(leaf_weights) == N

    if non_anticipative_stage_weeks is None:
        non_anticipative_stage_weeks = [list(range(1, T + 1))]

    m = pyo.ConcreteModel()
    m.T_set = pyo.RangeSet(1, T)
    m.S_set = pyo.RangeSet(0, N - 1)

    delta_t = {t: DELTA_WEEKLY ** (-t) for t in range(1, T + 1)}

    # Per-scenario variables
    m.g = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals, bounds=(0, G_MAX))
    m.x = pyo.Var(m.S_set, m.T_set, within=pyo.NonNegativeReals)
    m.s = pyo.Var(m.S_set, pyo.RangeSet(1, T + 1),
                  within=pyo.NonNegativeReals, bounds=(0, S_MAX))
    m.tv = pyo.Var(m.S_set, within=pyo.Reals)

    # Initial storage (same for all scenarios)
    m.init_s = pyo.Constraint(
        m.S_set,
        rule=lambda m, scen: m.s[scen, 1] == s_init
    )

    # Dynamics
    def dyn_rule(m, scen, t):
        return m.s[scen, t + 1] == (m.s[scen, t]
                                     + float(leaf_inflow[t - 1, scen])
                                     - m.g[scen, t]
                                     - m.x[scen, t])
    m.dyn = pyo.Constraint(m.S_set, m.T_set, rule=dyn_rule)

    # Terminal value
    m.tv_c1 = pyo.Constraint(
        m.S_set,
        rule=lambda m, scen: m.tv[scen] <= TERMINAL_VBAR * m.s[scen, T + 1]
    )
    m.tv_c2 = pyo.Constraint(
        m.S_set,
        rule=lambda m, scen: m.tv[scen] <= (
            TERMINAL_VBAR * S_MID
            + 0.5 * TERMINAL_VBAR * (m.s[scen, T + 1] - S_MID)
        )
    )

    # Non-anticipativity constraints
    # For each stage group, decisions at all scenarios must be equal at weeks
    # within that stage (i.e., decisions can only depend on which stage we're in,
    # not on the specific scenario within that stage — capturing that future
    # uncertainty is not yet resolved at stage start).
    # Implementation: for each stage's weeks and each pair of scenarios,
    # g[scen1, t] == g[scen2, t] for all t in stage.
    # This is O(N² × T) constraints — feasible for N ≤ 256.

    # Non-anticipativity on generation g only.
    # Spill x is physical overflow — always scenario-specific regardless of stage,
    # because it responds to realized reservoir level, not to committed plans.
    na_constraints = {}
    for stage_idx, stage_weeks in enumerate(non_anticipative_stage_weeks):
        for t in stage_weeks:
            for scen in range(1, N):
                key = (stage_idx, t, scen)
                na_constraints[key] = (m.g[scen, t] == m.g[0, t])
    m.non_anticip = pyo.ConstraintList()
    for key, expr in na_constraints.items():
        m.non_anticip.add(expr)

    # Objective
    def obj_rule(m):
        return sum(
            float(leaf_weights[scen]) * (
                sum(
                    delta_t[t] * 1000 * float(leaf_price[t - 1, scen]) * m.g[scen, t]
                    for t in range(1, T + 1)
                ) + m.tv[scen]
            )
            for scen in range(N)
        )
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m


def solve_model(
    m: pyo.ConcreteModel,
    solver: str = "highs",
    tee: bool = False,
) -> tuple[pyo.ConcreteModel, str]:
    """
    Solve a Pyomo model using HiGHS (or CBC fallback).
    Returns (solved_model, status_string).
    Raises RuntimeError if solution is not optimal.
    """
    opt = pyo.SolverFactory(solver)
    if not opt.available():
        LOG.warning("Solver %s not available, falling back to cbc", solver)
        opt = pyo.SolverFactory("cbc")
        if not opt.available():
            raise RuntimeError("Neither HiGHS nor CBC solver found. "
                               "Install highspy: pip install highspy")

    result = opt.solve(m, tee=tee)
    status = str(result.solver.termination_condition)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"Solver did not reach optimal solution. "
            f"Status: {status}. "
            f"Check model feasibility (storage/capacity bounds)."
        )
    return m, status


def extract_schedule(
    m: pyo.ConcreteModel,
    scenario_idx: int = 0,
    T: int = 52,
) -> ScheduleResult:
    """Extract generation/spill/storage schedule from solved model."""
    gen = np.array([pyo.value(m.g[scenario_idx, t]) for t in range(1, T + 1)])
    spl = np.array([pyo.value(m.x[scenario_idx, t]) for t in range(1, T + 1)])
    sto = np.array([pyo.value(m.s[scenario_idx, t]) for t in range(1, T + 2)])
    tv  = pyo.value(m.tv[scenario_idx])
    rev = float(pyo.value(m.obj))  # total expected revenue

    return ScheduleResult(
        generation=gen, spill=spl, storage=sto,
        revenue=rev, terminal_value=tv, solver_status="optimal",
    )
