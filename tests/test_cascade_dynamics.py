"""
Tests for the three-plant Arendalsvassdraget cascade LP.

Covers:
 1. Deterministic cascade LP: mass balance at each node, bounds, solver status
 2. Stochastic cascade LP: mass balance per scenario, non-anticipativity on g_J,
    g_E and g_R are scenario-specific (NOT subject to non-anticipativity)
 3. Routing: Evenstad/Rygene receive correct upstream inflow
 4. Data cutoff: no data beyond 2025-12-31
 5. Inflow fraction: fractions sum to 1 and split conserves total
"""
import numpy as np
import pytest
import pyomo.environ as pyo

from src.data_acquisition.cascade_panel import (
    INFLOW_FRACTIONS, PLANT_G_MAX, PLANT_S_MAX, split_system_inflow
)
from src.optimization.cascade_model import (
    G_MAX_J, G_MAX_E, G_MAX_R, S_MAX_J,
    build_cascade_deterministic_lp,
    build_cascade_stochastic_lp,
    extract_cascade_schedule,
)
from src.optimization.reservoir_model import solve_model
from src.backtest.cascade_runner import cascade_realised_revenue


# ── Fixtures ─────────────────────────────────────────────────────────────────

T_TEST = 10
N_TEST = 3


@pytest.fixture
def constant_inflows():
    """Constant local inflows across all time steps."""
    T = T_TEST
    # Total = 10 GWh/wk, split by fractions
    total = np.full(T, 10.0)
    local = split_system_inflow(total)
    iJ = local["Joerundland"]
    iE = local["Evenstad"]
    iR = local["Rygene"]
    price = np.full(T, 400.0)
    return iJ, iE, iR, price, T


@pytest.fixture
def solved_det_cascade(constant_inflows):
    iJ, iE, iR, price, T = constant_inflows
    s_init = 0.5 * S_MAX_J
    m = build_cascade_deterministic_lp(iJ, iE, iR, price, s_init, T)
    m, status = solve_model(m)
    return m, status, s_init, iJ, iE, iR, T


# ── 1. Deterministic LP ───────────────────────────────────────────────────────

class TestCascadeDeterministicLP:

    def test_solves_optimally(self, solved_det_cascade):
        _, status, *_ = solved_det_cascade
        assert status == "optimal"

    def test_jørundland_mass_balance(self, solved_det_cascade):
        m, _, s_init, iJ, _, _, T = solved_det_cascade
        for t in range(1, T + 1):
            s_t  = pyo.value(m.s_J[t])
            s_t1 = pyo.value(m.s_J[t + 1])
            gJ   = pyo.value(m.g_J[t])
            xJ   = pyo.value(m.x_J[t])
            err  = abs(s_t1 - (s_t + float(iJ[t - 1]) - gJ - xJ))
            assert err < 1e-4, f"Jørundland mass balance error at t={t}: {err:.6f}"

    def test_evenstad_passthrough(self, solved_det_cascade):
        """Evenstad has no storage: inflow_to_E = g_E + x_E."""
        m, _, s_init, iJ, iE, _, T = solved_det_cascade
        for t in range(1, T + 1):
            inflow_to_E = float(iE[t - 1]) + pyo.value(m.g_J[t]) + pyo.value(m.x_J[t])
            out_E       = pyo.value(m.g_E[t]) + pyo.value(m.x_E[t])
            err = abs(inflow_to_E - out_E)
            assert err < 1e-4, f"Evenstad pass-through error at t={t}: {err:.6f}"

    def test_rygene_passthrough(self, solved_det_cascade):
        """Rygene has no storage: inflow_to_R = g_R + x_R."""
        m, _, _, _, iE, iR, T = solved_det_cascade
        for t in range(1, T + 1):
            inflow_to_R = (float(iR[t - 1])
                           + pyo.value(m.g_E[t]) + pyo.value(m.x_E[t]))
            out_R = pyo.value(m.g_R[t]) + pyo.value(m.x_R[t])
            err = abs(inflow_to_R - out_R)
            assert err < 1e-4, f"Rygene pass-through error at t={t}: {err:.6f}"

    def test_storage_bounds(self, solved_det_cascade):
        m, _, _, _, _, _, T = solved_det_cascade
        for t in range(1, T + 2):
            s = pyo.value(m.s_J[t])
            assert s >= -1e-5, f"Storage_J below 0 at t={t}: {s:.4f}"
            assert s <= S_MAX_J + 1e-5, f"Storage_J above S_max_J at t={t}: {s:.4f}"

    def test_generation_bounds(self, solved_det_cascade):
        m, _, _, _, _, _, T = solved_det_cascade
        for t in range(1, T + 1):
            gJ = pyo.value(m.g_J[t])
            gE = pyo.value(m.g_E[t])
            gR = pyo.value(m.g_R[t])
            assert 0 <= gJ <= G_MAX_J + 1e-5
            assert 0 <= gE <= G_MAX_E + 1e-5
            assert 0 <= gR <= G_MAX_R + 1e-5

    def test_spill_nonnegative(self, solved_det_cascade):
        m, _, _, _, _, _, T = solved_det_cascade
        for t in range(1, T + 1):
            assert pyo.value(m.x_J[t]) >= -1e-6
            assert pyo.value(m.x_E[t]) >= -1e-6
            assert pyo.value(m.x_R[t]) >= -1e-6

    def test_initial_storage(self, solved_det_cascade):
        m, _, s_init, _, _, _, _ = solved_det_cascade
        assert abs(pyo.value(m.s_J[1]) - s_init) < 1e-4


# ── 2. Stochastic cascade LP ─────────────────────────────────────────────────

class TestCascadeStochasticLP:

    def test_mass_balance_all_scenarios(self):
        """Mass balance at all three nodes for every scenario."""
        T, N = 8, N_TEST
        rng  = np.random.default_rng(10)
        total = rng.uniform(5, 15, (T, N))
        local = split_system_inflow(total)
        iJ = local["Joerundland"]
        iE = local["Evenstad"]
        iR = local["Rygene"]
        price = rng.uniform(200, 600, (T, N))
        weights = np.ones(N) / N
        s_init = 0.4 * S_MAX_J

        m = build_cascade_stochastic_lp(iJ, iE, iR, price, weights, s_init, T)
        m, _ = solve_model(m)

        for scen in range(N):
            for t in range(1, T + 1):
                # Jørundland
                s_t  = pyo.value(m.s_J[scen, t])
                s_t1 = pyo.value(m.s_J[scen, t + 1])
                gJ   = pyo.value(m.g_J[scen, t])
                xJ   = pyo.value(m.x_J[scen, t])
                err  = abs(s_t1 - (s_t + float(iJ[t-1, scen]) - gJ - xJ))
                assert err < 1e-4, f"J mass balance: scen={scen}, t={t}, err={err:.6f}"
                # Evenstad
                inE  = float(iE[t-1, scen]) + gJ + xJ
                outE = pyo.value(m.g_E[scen, t]) + pyo.value(m.x_E[scen, t])
                err  = abs(inE - outE)
                assert err < 1e-4, f"E pass-through: scen={scen}, t={t}, err={err:.6f}"
                # Rygene
                inR  = float(iR[t-1, scen]) + outE
                outR = pyo.value(m.g_R[scen, t]) + pyo.value(m.x_R[scen, t])
                err  = abs(inR - outR)
                assert err < 1e-4, f"R pass-through: scen={scen}, t={t}, err={err:.6f}"

    def test_nonanticipative_g_J_open_loop(self):
        """Open-loop: g_J[scen, t] must be identical across all scenarios."""
        T, N = 8, N_TEST
        rng  = np.random.default_rng(11)
        total = rng.uniform(5, 15, (T, N))
        local = split_system_inflow(total)
        price = rng.uniform(200, 600, (T, N))
        weights = np.ones(N) / N
        s_init = 0.4 * S_MAX_J

        m = build_cascade_stochastic_lp(
            local["Joerundland"], local["Evenstad"], local["Rygene"],
            price, weights, s_init, T,
            non_anticipative_stage_weeks=[list(range(1, T + 1))],
        )
        m, status = solve_model(m)
        assert status == "optimal"

        for t in range(1, T + 1):
            ref = pyo.value(m.g_J[0, t])
            for scen in range(1, N):
                val = pyo.value(m.g_J[scen, t])
                assert abs(val - ref) < 1e-4, (
                    f"g_J non-anticipativity violated: t={t}, scen={scen}, "
                    f"g_J[0]={ref:.4f} != g_J[{scen}]={val:.4f}"
                )

    def test_g_E_scenario_specific(self):
        """Evenstad gen is NOT constrained by non-anticipativity — must vary across scenarios.

        We use zero storage (s_init=0) and zero Jørundland inflow so that the
        only thing flowing to Evenstad is its own local inflow, which differs
        across scenarios. With g_J=0 forced by zero available water, g_E equals
        i_E which differs by construction.
        """
        T, N = 6, 4
        # Zero Jørundland inflow so g_J=x_J=0 is forced; vary i_E only
        total = np.zeros((T, N))
        # Scenarios 0,2 have low E-equivalent inflow; 1,3 have high
        total[:, 0] = 1.0   # i_E ~ 0.67 GWh/wk  (well below G_MAX_E=3.95)
        total[:, 1] = 4.0   # i_E ~ 2.69 GWh/wk
        total[:, 2] = 1.0
        total[:, 3] = 4.0
        local = split_system_inflow(total)
        price = np.full((T, N), 400.0)
        weights = np.ones(N) / N
        # Zero storage so Jørundland cannot compensate
        s_init = 0.0

        m = build_cascade_stochastic_lp(
            local["Joerundland"], local["Evenstad"], local["Rygene"],
            price, weights, s_init, T,
            non_anticipative_stage_weeks=[list(range(1, T + 1))],  # OL on g_J
        )
        m, status = solve_model(m)
        assert status == "optimal"

        # g_E should differ between scenarios 0 and 1 (different i_E)
        any_differ = any(
            abs(pyo.value(m.g_E[0, t]) - pyo.value(m.g_E[1, t])) > 1e-3
            for t in range(1, T + 1)
        )
        assert any_differ, "g_E does not vary across scenarios despite different local inflows"

    def test_nonanticipativity_two_stage_g_J(self):
        """Closed-loop: g_J within each stage equal; may differ across stages."""
        T, N = 6, 4
        rng  = np.random.default_rng(13)
        total = rng.uniform(5, 15, (T, N))
        local = split_system_inflow(total)
        price = rng.uniform(200, 600, (T, N))
        price[:3, :2] = 100.0
        price[:3, 2:] = 800.0
        weights = np.ones(N) / N
        s_init = 0.5 * S_MAX_J
        stage_weeks = [[1, 2, 3], [4, 5, 6]]

        m = build_cascade_stochastic_lp(
            local["Joerundland"], local["Evenstad"], local["Rygene"],
            price, weights, s_init, T,
            non_anticipative_stage_weeks=stage_weeks,
        )
        m, status = solve_model(m)
        assert status == "optimal"

        for sw in stage_weeks:
            for t in sw:
                ref = pyo.value(m.g_J[0, t])
                for scen in range(1, N):
                    assert abs(pyo.value(m.g_J[scen, t]) - ref) < 1e-4


# ── 3. Cascade realised revenue ───────────────────────────────────────────────

class TestCascadeRealisedRevenue:

    def test_no_clamp_moderate_inflow(self):
        """With moderate inflow no clamping at Jørundland."""
        T = 20
        total = np.full(T, 8.0)   # 8 GWh/wk total
        local = split_system_inflow(total)
        price = np.full(T, 400.0)
        s_init = 0.5 * S_MAX_J
        gen_J = np.full(T, local["Joerundland"].mean() * 0.9)

        res = cascade_realised_revenue(
            gen_J, local["Joerundland"], local["Evenstad"],
            local["Rygene"], price, s_init, T
        )
        assert res["n_clamp_min"] == 0
        assert res["n_clamp_max"] == 0
        assert res["rev_total"] > 0

    def test_evenstad_rygene_spill_at_capacity(self):
        """Evenstad/Rygene spill when inflow exceeds G_MAX."""
        T = 5
        total = np.full(T, 100.0)  # extremely high inflow
        local = split_system_inflow(total)
        price = np.full(T, 400.0)
        s_init = 0.3 * S_MAX_J
        gen_J = np.full(T, G_MAX_J)

        res = cascade_realised_revenue(
            gen_J, local["Joerundland"], local["Evenstad"],
            local["Rygene"], price, s_init, T
        )
        # Some spill should occur at E or R
        assert np.any(res["spill_E"] > 0) or np.any(res["spill_R"] > 0), (
            "Expected spill at Evenstad or Rygene with 100 GWh/wk total inflow"
        )

    def test_storage_conservation(self):
        """Jørundland storage should stay within [0, S_MAX_J]."""
        T = 52
        total = np.full(T, 10.0)
        local = split_system_inflow(total)
        price = np.full(T, 400.0)
        s_init = 0.5 * S_MAX_J
        gen_J = np.full(T, G_MAX_J)  # aggressive generation

        res = cascade_realised_revenue(
            gen_J, local["Joerundland"], local["Evenstad"],
            local["Rygene"], price, s_init, T
        )
        assert np.all(res["storage_J"] >= -1e-5), "Storage_J below 0"
        assert np.all(res["storage_J"] <= S_MAX_J + 1e-5), "Storage_J above S_MAX_J"


# ── 4. Inflow fractions ───────────────────────────────────────────────────────

class TestInflowFractions:

    def test_fractions_sum_to_one(self):
        total = sum(INFLOW_FRACTIONS.values())
        assert abs(total - 1.0) < 1e-4, f"Fractions sum to {total}"

    def test_split_conserves_total(self):
        total = np.random.default_rng(0).uniform(5, 15, 52)
        local = split_system_inflow(total)
        reconstructed = sum(local.values())
        np.testing.assert_allclose(reconstructed, total, rtol=1e-10)

    def test_split_shape_preserved(self):
        total_2d = np.ones((52, 10)) * 10.0
        local = split_system_inflow(total_2d)
        for p, arr in local.items():
            assert arr.shape == (52, 10), f"Shape mismatch for {p}: {arr.shape}"


# ── 5. Data cutoff ───────────────────────────────────────────────────────────

class TestCascadeDataCutoff:

    def test_nesvatn_no_data_after_cutoff(self):
        from pathlib import Path
        import pandas as pd
        path = Path("data/raw/nesvatn_volume.csv")
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            assert df["date"].max() <= pd.Timestamp("2025-12-31"), (
                "Nesvatn volume data contains dates after 2025-12-31"
            )
