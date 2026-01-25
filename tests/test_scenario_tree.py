"""
Tests: scenario tree structure, non-anticipativity, and probability weights.
"""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.scenario_tree import (
    STAGE_WEEKS,
    N_STAGES,
    BRANCH_FACTOR,
    build_scenario_tree,
)


@pytest.fixture
def simple_scenarios():
    """Small deterministic scenario set for tree tests."""
    T = 52
    N = 100
    rng = np.random.default_rng(0)
    inflow = rng.uniform(5, 40, (T, N))
    price = rng.uniform(100, 800, (T, N))
    dates = pd.date_range("2020-01-06", periods=T, freq="W-MON")
    inflow_df = pd.DataFrame(inflow, index=dates,
                             columns=[f"s{j:03d}" for j in range(N)])
    price_df = pd.DataFrame(price, index=dates,
                            columns=[f"s{j:03d}" for j in range(N)])
    return inflow_df, price_df


class TestScenarioTree:
    def test_weights_sum_to_one(self, simple_scenarios):
        inflow_df, price_df = simple_scenarios
        tree = build_scenario_tree(inflow_df, price_df,
                                   forecast_start_week=inflow_df.index[0],
                                   n_raw=100)
        weights = tree["leaf_weights"]
        assert abs(weights.sum() - 1.0) < 1e-6, (
            f"Leaf weights do not sum to 1: sum={weights.sum():.8f}"
        )

    def test_leaf_counts_positive(self, simple_scenarios):
        inflow_df, price_df = simple_scenarios
        tree = build_scenario_tree(inflow_df, price_df,
                                   forecast_start_week=inflow_df.index[0],
                                   n_raw=100)
        n_leaves = tree["leaf_inflow"].shape[1]
        assert n_leaves > 0, "Zero leaf scenarios"

    def test_leaf_inflow_nonnegative(self, simple_scenarios):
        inflow_df, price_df = simple_scenarios
        tree = build_scenario_tree(inflow_df, price_df,
                                   forecast_start_week=inflow_df.index[0],
                                   n_raw=100)
        assert np.all(tree["leaf_inflow"] >= -1e-6), (
            "Negative inflow in leaf scenarios"
        )

    def test_leaf_price_positive(self, simple_scenarios):
        inflow_df, price_df = simple_scenarios
        tree = build_scenario_tree(inflow_df, price_df,
                                   forecast_start_week=inflow_df.index[0],
                                   n_raw=100)
        assert np.all(tree["leaf_price"] > 0), (
            "Non-positive price in leaf scenarios"
        )

    def test_n_leaf_weeks_equals_52(self, simple_scenarios):
        inflow_df, price_df = simple_scenarios
        tree = build_scenario_tree(inflow_df, price_df,
                                   forecast_start_week=inflow_df.index[0],
                                   n_raw=100)
        T = tree["leaf_inflow"].shape[0]
        assert T == 52, f"Expected 52 weeks per leaf path, got {T}"

    def test_stage_weeks_cover_all_52(self):
        all_weeks = sorted(set(w for stage in STAGE_WEEKS for w in stage))
        assert all_weeks == list(range(1, 53)), (
            f"STAGE_WEEKS do not cover weeks 1–52: {all_weeks}"
        )

    def test_stage_weeks_non_overlapping(self):
        seen = set()
        for stage in STAGE_WEEKS:
            for w in stage:
                assert w not in seen, f"Week {w} appears in multiple stages"
                seen.add(w)
