"""
Tests for reservoir dynamics mass balance, bounds, and solver correctness.
"""
import numpy as np
import pytest
import pyomo.environ as pyo

from src.optimization.reservoir_model import (
    G_MAX,
    S_MAX,
    build_deterministic_lp,
    build_stochastic_lp,
    extract_schedule,
    solve_model,
)
from src.backtest.runner import realised_revenue


@pytest.fixture
def simple_path():
    """Simple deterministic inflow/price path for testing."""
    T = 52
    inflow = np.full(T, 20.0)   # 20 GWh/week constant inflow
    price = np.full(T, 400.0)   # 400 NOK/MWh constant price
    return inflow, price


@pytest.fixture
def solved_det_model(simple_path):
    inflow, price = simple_path
    s_init = 0.6 * S_MAX
    m = build_deterministic_lp(inflow, price, s_init, T=52)
    m, status = solve_model(m)
    return m, status, s_init


class TestDeterministicLP:
    def test_solves_optimally(self, solved_det_model):
        _, status, _ = solved_det_model
        assert status == "optimal"

    def test_mass_balance(self, solved_det_model, simple_path):
        m, _, s_init = solved_det_model
        inflow, _ = simple_path
        T = 52
        for t in range(1, T + 1):
            s_t = pyo.value(m.s[t])
            s_t1 = pyo.value(m.s[t + 1])
            g_t = pyo.value(m.g[t])
            x_t = pyo.value(m.x[t])
            i_t = float(inflow[t - 1])
            balance = s_t1 - (s_t + i_t - g_t - x_t)
            assert abs(balance) < 1e-4, (
                f"Mass balance violated at t={t}: "
                f"s[t+1]={s_t1:.4f}, s[t]+i-g-x={s_t+i_t-g_t-x_t:.4f}"
            )

    def test_storage_bounds(self, solved_det_model):
        m, _, _ = solved_det_model
        T = 52
        for t in range(1, T + 2):
            s = pyo.value(m.s[t])
            assert s >= -1e-5, f"Storage below 0 at t={t}: {s:.4f}"
            assert s <= S_MAX + 1e-5, f"Storage above S_max at t={t}: {s:.4f}"

    def test_generation_bounds(self, solved_det_model):
        m, _, _ = solved_det_model
        T = 52
        for t in range(1, T + 1):
            g = pyo.value(m.g[t])
            assert g >= -1e-6, f"Generation below 0 at t={t}: {g:.4f}"
            assert g <= G_MAX + 1e-5, f"Generation above G_max at t={t}: {g:.4f}"

    def test_spill_nonnegative(self, solved_det_model):
        m, _, _ = solved_det_model
        T = 52
        for t in range(1, T + 1):
            x = pyo.value(m.x[t])
            assert x >= -1e-6, f"Spill below 0 at t={t}: {x:.4f}"

    def test_initial_storage(self, solved_det_model):
        m, _, s_init = solved_det_model
        s1 = pyo.value(m.s[1])
        assert abs(s1 - s_init) < 1e-4, f"Initial storage mismatch: {s1} vs {s_init}"


class TestStochasticLP:
    def test_nonanticipative_open_loop(self):
        """
        Open-loop: all scenarios must share the same decisions at every week.
        """
        T = 10
        N = 3
        rng = np.random.default_rng(0)
        inflow = rng.uniform(10, 30, (T, N))
        price = rng.uniform(200, 600, (T, N))
        weights = np.ones(N) / N
        s_init = 0.5 * S_MAX

        m = build_stochastic_lp(
            inflow, price, weights, s_init=s_init, T=T,
            non_anticipative_stage_weeks=[list(range(1, T + 1))],
        )
        m, status = solve_model(m)
        assert status == "optimal"

        # All scenarios must have the same g[t] for all t
        for t in range(1, T + 1):
            g_ref = pyo.value(m.g[0, t])
            for scen in range(1, N):
                g_s = pyo.value(m.g[scen, t])
                assert abs(g_s - g_ref) < 1e-4, (
                    f"Non-anticipativity violated at t={t}, scen={scen}: "
                    f"g[0,t]={g_ref:.4f} != g[{scen},t]={g_s:.4f}"
                )

    def test_nonanticipative_closed_loop_stage_structure(self):
        """
        Closed-loop: decisions within each stage must be equal across scenarios;
        decisions in different stages may differ.
        """
        T = 6
        N = 4
        rng = np.random.default_rng(1)
        inflow = rng.uniform(10, 30, (T, N))
        price = rng.uniform(200, 600, (T, N))
        # Give scenarios very different prices to force different decisions
        price[:3, :2] = 100.0   # low price in stage 1, scenarios 0-1
        price[:3, 2:] = 900.0   # high price in stage 1, scenarios 2-3
        weights = np.ones(N) / N
        s_init = 0.5 * S_MAX

        stage_weeks = [[1, 2, 3], [4, 5, 6]]
        m = build_stochastic_lp(
            inflow, price, weights, s_init=s_init, T=T,
            non_anticipative_stage_weeks=stage_weeks,
        )
        m, status = solve_model(m)
        assert status == "optimal"

        # Within each stage, all scenarios share same g[t]
        for sw in stage_weeks:
            for t in sw:
                g_ref = pyo.value(m.g[0, t])
                for scen in range(1, N):
                    g_s = pyo.value(m.g[scen, t])
                    assert abs(g_s - g_ref) < 1e-4, (
                        f"Intra-stage non-anticipativity violated at t={t}, scen={scen}"
                    )

    def test_mass_balance_stochastic(self):
        """Mass balance must hold for every scenario."""
        T = 10
        N = 3
        rng = np.random.default_rng(2)
        inflow = rng.uniform(10, 30, (T, N))
        price = rng.uniform(200, 600, (T, N))
        weights = np.ones(N) / N
        s_init = 0.5 * S_MAX

        m = build_stochastic_lp(
            inflow, price, weights, s_init=s_init, T=T,
        )
        m, _ = solve_model(m)

        for scen in range(N):
            for t in range(1, T + 1):
                s_t = pyo.value(m.s[scen, t])
                s_t1 = pyo.value(m.s[scen, t + 1])
                g_t = pyo.value(m.g[scen, t])
                x_t = pyo.value(m.x[scen, t])
                i_t = float(inflow[t - 1, scen])
                balance = abs(s_t1 - (s_t + i_t - g_t - x_t))
                assert balance < 1e-4, (
                    f"Mass balance violated: scen={scen}, t={t}, error={balance:.6f}"
                )


class TestRealisedRevenue:
    def test_no_clamp_simple_case(self):
        """With constant inflow and moderate generation, no clamping should occur."""
        T = 52
        inflow = np.full(T, 20.0)
        price = np.full(T, 400.0)
        generation = np.full(T, 18.0)   # slightly less than inflow
        s_init = 0.5 * S_MAX

        rev, storage, n_cmin, n_cmax = realised_revenue(
            generation, inflow, price, s_init, T
        )
        assert n_cmin == 0, "Expected no min-clamp"
        assert n_cmax == 0, "Expected no max-clamp"
        assert rev > 0

    def test_storage_stays_bounded(self):
        """Storage must stay within [0, S_max] after applying realised revenue."""
        T = 52
        inflow = np.full(T, 50.0)   # high inflow → potential spill
        price = np.full(T, 400.0)
        generation = np.full(T, 10.0)   # low generation
        s_init = 0.8 * S_MAX

        _, storage, _, _ = realised_revenue(generation, inflow, price, s_init)
        assert np.all(storage >= -1e-5), "Storage below 0"
        assert np.all(storage <= S_MAX + 1e-5), "Storage above S_max"
