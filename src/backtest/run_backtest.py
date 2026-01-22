"""
Main backtest runner.

Run from project root:
    python -m src.backtest.run_backtest

Outputs:
    results/tables/backtest_results.csv   — year-by-year results
    results/tables/backtest_summary.csv   — summary statistics
    results/figures/trajectory_<year>.png — diagnostic plots (one representative year)
"""
from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.backtest.runner import (
    run_closed_loop,
    run_open_loop,
    run_perfect_foresight,
)
from src.data_acquisition.magasin_client import get_no2_week1_filling
from src.forecasting.scenario_tree import load_scenario_tree
from src.optimization.reservoir_model import S_MAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
SCENARIOS_DIR = Path("results/scenarios")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

S_INIT_DEFAULT_FRACTION = 0.65   # fallback if Magasinstatistikk lookup fails

# Load NO2 week-1 filling fractions from NVE Magasinstatistikk (cached locally).
# Covers 1995–present; all 22 backtest years (2003–2024) are present.
try:
    _NO2_WEEK1_FILLING = get_no2_week1_filling()
    LOG.info("Loaded NO2 Magasinstatistikk week-1 filling for %d years "
             "(range: %.0f%%–%.0f%%)",
             len(_NO2_WEEK1_FILLING),
             min(_NO2_WEEK1_FILLING.values()) * 100,
             max(_NO2_WEEK1_FILLING.values()) * 100)
except Exception as _e:
    LOG.warning("Could not load Magasinstatistikk data (%s); "
                "falling back to fixed %.0f%% filling.", _e, S_INIT_DEFAULT_FRACTION * 100)
    _NO2_WEEK1_FILLING = {}


def _estimate_init_storage(
    panel: pd.DataFrame, year: int
) -> float:
    """
    Return start-of-year reservoir storage [GWh] for *year*.

    Uses the real NO2 elspot-zone filling percentage at ISO week 1 of *year*
    from NVE Magasinstatistikk, scaled to the model's S_MAX.  The NO2 zone
    encompasses Agder (where Arendalsvassdraget / Rygene is located) and is
    the closest publicly available aggregate that covers this system.

    Falls back to 65% of S_MAX if the lookup fails.
    """
    fraction = _NO2_WEEK1_FILLING.get(year, S_INIT_DEFAULT_FRACTION)
    return fraction * S_MAX


def _diagnostic_plot(year: int, weeks: pd.DatetimeIndex,
                     ol_res: dict, cl_res: dict, pf_res: dict,
                     realized_price: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Use 1-indexed week numbers throughout so sharex=True works correctly.
    # Mixing DatetimeIndex (for storage/price) with integer range (for generation)
    # caused matplotlib to treat integers as days-since-epoch and compress
    # all datetime data into a tiny right-edge sliver.
    week_numbers = np.arange(1, len(weeks) + 1)

    # Storage trajectories
    ax = axes[0]
    ax.plot(week_numbers, pf_res["storage_trajectory"][:-1], label="Perfect foresight", lw=1.5)
    ax.plot(week_numbers, cl_res["storage_trajectory"][:-1], label="Closed-loop", lw=1.5, ls="--")
    ax.plot(week_numbers, ol_res["storage_trajectory"][:-1], label="Open-loop", lw=1.2, ls=":")
    ax.axhline(S_MAX, color="grey", ls="-.", lw=0.8, label="S_max")
    ax.set_ylabel("Storage [GWh]")
    ax.legend(fontsize=8)
    ax.set_title(f"Year {year} — Reservoir trajectories and generation")

    # Generation
    ax2 = axes[1]
    ax2.step(week_numbers, pf_res["generation"], label="PF generation", lw=1.2)
    ax2.step(week_numbers, cl_res["realized_generation"], label="CL generation", lw=1.2, ls="--")
    ax2.step(week_numbers, ol_res["planned_generation"], label="OL generation", lw=1.0, ls=":")
    ax2.set_ylabel("Generation [GWh/wk]")
    ax2.legend(fontsize=8)

    # Realised price
    ax3 = axes[2]
    ax3.plot(week_numbers, realized_price, color="tab:red", lw=0.9)
    ax3.set_ylabel("Price [NOK/MWh]")
    ax3.set_xlabel("Week of year")
    ax3.set_xlim(1, len(weeks))

    fig.tight_layout()
    path = FIGURES_DIR / f"trajectory_{year}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOG.info("Saved diagnostic plot: %s", path)


def run(backtest_years: list[int] | None = None) -> pd.DataFrame:
    panel = pd.read_csv(
        PROCESSED_DIR / "weekly_panel.csv",
        index_col="week_start",
        parse_dates=True,
    )

    all_years = sorted(panel["iso_year"].unique())
    if backtest_years is None:
        backtest_years = [y for y in all_years if y >= all_years[4]]

    LOG.info("Running backtest for years: %s", backtest_years)
    records = []

    for year in backtest_years:
        LOG.info("=== Year %d ===", year)
        year_panel = panel[panel["iso_year"] == year].copy()
        if len(year_panel) < 50:
            LOG.warning("Year %d has only %d weeks, skipping.", year, len(year_panel))
            continue

        # Align to exactly 52 weeks
        year_panel = year_panel.iloc[:52]
        T = len(year_panel)

        realized_inflow = year_panel["inflow_GWh_week"].values.astype(float)
        realized_price  = year_panel["price_avg_NOK_MWh"].values.astype(float)
        weeks = year_panel.index

        s_init = _estimate_init_storage(panel, year)

        # Load pre-computed scenario tree (from forecasting pipeline)
        tree_path = SCENARIOS_DIR / f"scenario_tree_{year}.pkl"
        if not tree_path.exists():
            LOG.warning("Scenario tree for %d not found, skipping.", year)
            continue
        tree = load_scenario_tree(tag=str(year))

        # Load model idatas for closed-loop re-solves
        inf_nc = SCENARIOS_DIR / f"inflow_idata_{year}.nc"
        pr_nc  = SCENARIOS_DIR / f"price_idata_{year}.nc"
        if not inf_nc.exists() or not pr_nc.exists():
            LOG.warning("Model idatas for %d not found, skipping.", year)
            continue

        inflow_idata = az.from_netcdf(str(inf_nc))
        price_idata  = az.from_netcdf(str(pr_nc))

        # ── Open-loop ──────────────────────────────────────────────
        LOG.info("  Running open-loop …")
        try:
            ol = run_open_loop(tree, realized_inflow, realized_price, s_init)
        except Exception as e:
            LOG.error("Open-loop failed for %d: %s", year, e)
            ol = {"realized_revenue_NOK": np.nan}

        # ── Closed-loop ────────────────────────────────────────────
        LOG.info("  Running closed-loop …")
        try:
            cl = run_closed_loop(
                year, panel, inflow_idata, price_idata,
                realized_inflow, realized_price, s_init
            )
        except Exception as e:
            LOG.error("Closed-loop failed for %d: %s", year, e)
            cl = {"realized_revenue_NOK": np.nan}

        # ── Perfect foresight ──────────────────────────────────────
        LOG.info("  Running perfect foresight …")
        try:
            pf = run_perfect_foresight(realized_inflow, realized_price, s_init)
        except Exception as e:
            LOG.error("Perfect-foresight failed for %d: %s", year, e)
            pf = {"realized_revenue_NOK": np.nan}

        ol_rev = ol["realized_revenue_NOK"]
        cl_rev = cl["realized_revenue_NOK"]
        pf_rev = pf["realized_revenue_NOK"]

        vof = cl_rev - ol_rev     # value of flexibility (recourse value)
        regret = pf_rev - cl_rev  # residual regret (irreducible uncertainty)

        LOG.info("  OL=%.3f MNOK  CL=%.3f MNOK  PF=%.3f MNOK  "
                 "VoF=%.3f MNOK  Regret=%.3f MNOK",
                 ol_rev / 1e6, cl_rev / 1e6, pf_rev / 1e6,
                 vof / 1e6, regret / 1e6)

        records.append({
            "year": year,
            "open_loop_revenue_NOK": ol_rev,
            "closed_loop_revenue_NOK": cl_rev,
            "perfect_foresight_revenue_NOK": pf_rev,
            "value_of_flexibility_NOK": vof,
            "residual_regret_NOK": regret,
            "ol_clamp_min": ol.get("n_clamp_min", 0),
            "ol_clamp_max": ol.get("n_clamp_max", 0),
            "s_init_GWh": s_init,
        })

        # Diagnostic plot for every backtested year
        if "storage_trajectory" in ol and "storage_trajectory" in cl:
            _diagnostic_plot(year, weeks, ol, cl, pf, realized_price)

    if not records:
        LOG.error("No backtest results produced.")
        return pd.DataFrame()

    results = pd.DataFrame(records)
    results.to_csv(TABLES_DIR / "backtest_results.csv", index=False)

    # Summary statistics
    vof_arr = results["value_of_flexibility_NOK"].dropna().values
    regret_arr = results["residual_regret_NOK"].dropna().values
    summary = {
        "n_years": len(results),
        "mean_OL_revenue_MNOK": results["open_loop_revenue_NOK"].mean() / 1e6,
        "mean_CL_revenue_MNOK": results["closed_loop_revenue_NOK"].mean() / 1e6,
        "mean_PF_revenue_MNOK": results["perfect_foresight_revenue_NOK"].mean() / 1e6,
        "mean_VoF_MNOK": np.mean(vof_arr) / 1e6 if len(vof_arr) else np.nan,
        "std_VoF_MNOK": np.std(vof_arr) / 1e6 if len(vof_arr) else np.nan,
        "mean_residual_regret_MNOK": np.mean(regret_arr) / 1e6 if len(regret_arr) else np.nan,
    }

    # Statistical test: Wilcoxon signed-rank (non-parametric, appropriate for small n)
    if len(vof_arr) >= 5:
        try:
            stat, pval = wilcoxon(vof_arr, alternative="greater")
            summary["wilcoxon_stat"] = stat
            summary["wilcoxon_p_CL_gt_OL"] = pval
        except Exception:
            summary["wilcoxon_stat"] = np.nan
            summary["wilcoxon_p_CL_gt_OL"] = np.nan
    else:
        summary["wilcoxon_stat"] = np.nan
        summary["wilcoxon_p_CL_gt_OL"] = np.nan
        LOG.warning("Too few years (%d) for a meaningful Wilcoxon test. "
                    "Power is very low; statistical significance not claimed.", len(vof_arr))

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(TABLES_DIR / "backtest_summary.csv", index=False)

    LOG.info("=== Backtest complete ===")
    LOG.info("\n%s", summary_df.T.to_string())
    return results


if __name__ == "__main__":
    run()
