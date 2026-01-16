"""
Forecasting pipeline runner.

Run from project root:
    python -m src.forecasting.run_forecasting

Outputs:
    results/scenarios/inflow_idata.nc   — ArviZ InferenceData (inflow model)
    results/scenarios/price_idata.nc    — ArviZ InferenceData (price model)
    results/scenarios/scenario_tree_<year>.pkl  — per-year scenario trees
    results/figures/ppc_inflow_<year>.png
    results/figures/ppc_price_<year>.png
"""
from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecasting.price_model import fit_price_model, generate_price_scenarios
from src.forecasting.scenario_tree import build_scenario_tree, save_scenario_tree
from src.forecasting.seasonal_model import fit_inflow_model, generate_inflow_scenarios

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
SCENARIOS_DIR = Path("results/scenarios")
FIGURES_DIR = Path("results/figures")
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_RAW_SCENARIOS = 200


def _plot_ppc(observed: np.ndarray, ppc_samples: np.ndarray,
              dates: pd.DatetimeIndex, title: str, path: Path) -> None:
    """Plot observed vs posterior predictive interval."""
    lower = np.percentile(ppc_samples, 5, axis=0)
    upper = np.percentile(ppc_samples, 95, axis=0)
    median = np.percentile(ppc_samples, 50, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, lower, upper, alpha=0.3, label="90% PPD interval")
    ax.plot(dates, median, lw=1.0, label="PPD median")
    ax.plot(dates, observed, lw=0.8, color="k", alpha=0.8, label="Observed")
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOG.info("Saved PPC plot: %s", path)


def run(backtest_years: list[int] | None = None) -> None:
    """
    For each backtested year, fit models on pre-year data and build
    the scenario tree for that year.

    backtest_years: if None, uses all available full years in the panel.
    """
    panel = pd.read_csv(
        PROCESSED_DIR / "weekly_panel.csv",
        index_col="week_start",
        parse_dates=True,
    )

    all_years = sorted(panel["iso_year"].unique())
    # Need at least 5 years of training data; never look ahead
    if backtest_years is None:
        backtest_years = [y for y in all_years if y >= all_years[4]]

    LOG.info("Backtest years: %s", backtest_years)

    for year in backtest_years:
        train_end = year - 1
        LOG.info("=== Year %d (training on ≤ %d) ===", year, train_end)

        # ── Inflow model ──────────────────────────────────────────────────
        inf_nc = SCENARIOS_DIR / f"inflow_idata_{year}.nc"
        if inf_nc.exists():
            LOG.info("Loading cached inflow idata for year %d …", year)
            inflow_idata = az.from_netcdf(str(inf_nc))
        else:
            _, inflow_idata = fit_inflow_model(panel, train_end_year=train_end)
            inflow_idata.to_netcdf(str(inf_nc))

        # Posterior predictive check (held-out = test year)
        test_panel = panel[panel["iso_year"] == year]
        ppc_path = FIGURES_DIR / f"ppc_inflow_{year}.png"
        if not ppc_path.exists() and hasattr(inflow_idata, "posterior_predictive"):
            # Use the PPD samples from training fit
            ppc_obs_vals = inflow_idata.posterior_predictive.get("obs")
            if ppc_obs_vals is not None:
                train_panel = panel[panel["iso_year"] <= train_end]
                _plot_ppc(
                    np.log(train_panel["inflow_GWh_week"].values + 0.1),
                    ppc_obs_vals.values.reshape(-1, len(train_panel)),
                    train_panel.index,
                    f"Inflow PPC — training period (year ≤ {train_end})",
                    ppc_path,
                )

        # ── Price model ───────────────────────────────────────────────────
        pr_nc = SCENARIOS_DIR / f"price_idata_{year}.nc"
        if pr_nc.exists():
            LOG.info("Loading cached price idata for year %d …", year)
            price_idata = az.from_netcdf(str(pr_nc))
        else:
            _, price_idata = fit_price_model(
                panel, inflow_idata, train_end_year=train_end
            )
            price_idata.to_netcdf(str(pr_nc))

        # ── Scenario generation ───────────────────────────────────────────
        tree_path = SCENARIOS_DIR / f"scenario_tree_{year}.pkl"
        if tree_path.exists():
            LOG.info("Scenario tree for year %d already exists, skipping.", year)
            continue

        # Find the first Monday on or after the start of the test year
        year_start = panel[panel["iso_year"] == year].index.min()
        rng = np.random.default_rng(SEED + year)

        inflow_scen = generate_inflow_scenarios(
            inflow_idata, panel,
            forecast_start_week=year_start,
            n_weeks=52,
            n_scenarios=N_RAW_SCENARIOS,
            rng=rng,
        )
        price_scen = generate_price_scenarios(
            price_idata, inflow_idata, panel,
            forecast_start_week=year_start,
            inflow_scenarios=inflow_scen,
            n_weeks=52,
            rng=rng,
        )

        tree = build_scenario_tree(inflow_scen, price_scen, year_start,
                                   n_raw=N_RAW_SCENARIOS)
        save_scenario_tree(tree, tag=str(year))
        LOG.info("Scenario tree for year %d saved.", year)

    LOG.info("=== Forecasting pipeline complete. ===")


if __name__ == "__main__":
    run()
