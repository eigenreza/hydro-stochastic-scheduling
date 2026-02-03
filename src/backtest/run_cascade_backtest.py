"""
Cascade backtest runner: three-plant Jørundland→Evenstad→Rygene.

Run from project root:
    python -m src.backtest.run_cascade_backtest

Outputs:
    results/tables/cascade_backtest_results.csv    — year-by-year results
    results/tables/cascade_backtest_summary.csv    — summary statistics
    results/figures/cascade_trajectory_<year>.png  — per-year diagnostic plots
"""
from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.backtest.cascade_runner import (
    run_cascade_closed_loop,
    run_cascade_open_loop,
    run_cascade_perfect_foresight,
)
from src.data_acquisition.cascade_panel import (
    INFLOW_FRACTIONS, get_cascade_panel
)
from src.data_acquisition.magasin_client import get_no2_week1_filling
from src.forecasting.scenario_tree import load_scenario_tree
from src.optimization.cascade_model import S_MAX_J

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)

SCENARIOS_DIR = Path("results/scenarios")
TABLES_DIR    = Path("results/tables")
FIGURES_DIR   = Path("results/figures")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

S_INIT_DEFAULT_FRACTION = 0.65

try:
    _NO2_WEEK1_FILLING = get_no2_week1_filling()
    LOG.info("Loaded Magasinstatistikk week-1 filling for %d years", len(_NO2_WEEK1_FILLING))
except Exception as _e:
    LOG.warning("Magasinstatistikk load failed (%s); using fixed %.0f%%", _e, S_INIT_DEFAULT_FRACTION * 100)
    _NO2_WEEK1_FILLING = {}


def _estimate_s_init_J(year: int) -> float:
    """Jørundland initial storage from NO2 Magasinstatistikk week-1 filling."""
    fraction = _NO2_WEEK1_FILLING.get(year, S_INIT_DEFAULT_FRACTION)
    return fraction * S_MAX_J


def _cascade_diagnostic_plot(
    year: int,
    weeks: pd.DatetimeIndex,
    ol_res: dict,
    cl_res: dict,
    pf_res: dict,
    path: Path,
) -> None:
    """Save diagnostic trajectory plot for the cascade."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    T_plot = len(ol_res["gen_J"])   # 52 (LP horizon, may differ from len(weeks) for 53-wk years)
    week_numbers = np.arange(1, T_plot + 1)

    ax1, ax2, ax3, ax4 = axes

    # Jørundland storage
    for res, label, ls in [
        (ol_res, "OL", "--"), (cl_res, "CL", "-"), (pf_res, "PF", ":")
    ]:
        ax1.plot(week_numbers, res["storage_J"][:T_plot], label=label, ls=ls)
    ax1.axhline(S_MAX_J, color="r", lw=0.8, ls=":", label="S_max_J")
    ax1.set_ylabel("Storage J [GWh]")
    ax1.set_title(f"Cascade {year}: Jørundland storage")
    ax1.legend(fontsize=8)
    ax1.set_xlim(1, T_plot)

    # Total system generation
    for res, label, ls in [
        (ol_res, "OL", "--"), (cl_res, "CL", "-"), (pf_res, "PF", ":")
    ]:
        total_gen = res["gen_J"][:T_plot] + res["gen_E"][:T_plot] + res["gen_R"][:T_plot]
        ax2.plot(week_numbers, total_gen, label=label, ls=ls)
    ax2.set_ylabel("Total gen [GWh/wk]")
    ax2.set_title("Total system generation (J+E+R)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(1, T_plot)

    # Per-plant generation (CL only)
    ax3.stackplot(
        week_numbers,
        cl_res["gen_J"][:T_plot], cl_res["gen_E"][:T_plot], cl_res["gen_R"][:T_plot],
        labels=["Joerundland", "Evenstad", "Rygene"],
        alpha=0.8,
    )
    ax3.set_ylabel("Gen [GWh/wk] (CL)")
    ax3.set_title("Per-plant generation (closed-loop)")
    ax3.legend(fontsize=8)
    ax3.set_xlim(1, T_plot)

    # Price
    panel_year = panel_global[panel_global["iso_year"] == year]
    ax4.plot(week_numbers, panel_year["price_avg_NOK_MWh"].values[:T_plot], color="k", lw=1)
    ax4.set_ylabel("Price [NOK/MWh]")
    ax4.set_xlabel("Week of year")
    ax4.set_title("Realised NO2 price")
    ax4.set_xlim(1, T_plot)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOG.info("Saved cascade trajectory plot: %s", path)


# Global panel reference for diagnostic plots
panel_global: pd.DataFrame = None   # type: ignore[assignment]


def run(backtest_years: list[int] | None = None) -> pd.DataFrame:
    """Run cascade backtest and return year-by-year results DataFrame."""
    global panel_global

    panel = get_cascade_panel()
    panel_global = panel
    all_years = sorted(panel["iso_year"].unique())
    if backtest_years is None:
        # Only backtest years with ≥52 weeks in panel and a scenario tree
        # (2025 is excluded: only partial data / no complete backtest year)
        backtest_years = [
            y for y in all_years
            if y >= all_years[4]
            and (panel["iso_year"] == y).sum() >= 52
            and (SCENARIOS_DIR / f"scenario_tree_{y}.pkl").exists()
            and y <= 2024
        ]

    LOG.info("Cascade backtest years: %s", backtest_years)

    rows = []
    for year in backtest_years:
        LOG.info("=== Cascade year %d ===", year)
        year_panel = panel[panel["iso_year"] == year]
        T = len(year_panel)

        # Per-plant realised inflows (clip to 52 weeks; ISO 53-week years handled by [:52])
        T_actual = len(year_panel)
        T = min(T_actual, 52)   # scenario trees are always built for 52 weeks
        inflow_J = year_panel["inflow_J_GWh_week"].values[:T].astype(float)
        inflow_E = year_panel["inflow_E_GWh_week"].values[:T].astype(float)
        inflow_R = year_panel["inflow_R_GWh_week"].values[:T].astype(float)
        price     = year_panel["price_avg_NOK_MWh"].values[:T].astype(float)

        s_init_J = _estimate_s_init_J(year)
        LOG.info("  s_init_J=%.1f GWh (%.1f%% of S_MAX_J=%.0f)",
                 s_init_J, s_init_J / S_MAX_J * 100, S_MAX_J)

        # Load scenario tree (same as used in single-reservoir backtest)
        tree = load_scenario_tree(tag=str(year))

        # Load Bayesian idatas for closed-loop re-solves
        inf_nc = SCENARIOS_DIR / f"inflow_idata_{year}.nc"
        pr_nc  = SCENARIOS_DIR / f"price_idata_{year}.nc"
        inflow_idata = az.from_netcdf(str(inf_nc))
        price_idata  = az.from_netcdf(str(pr_nc))

        # ── Perfect foresight ──────────────────────────────────────────────
        LOG.info("  Perfect foresight …")
        pf = run_cascade_perfect_foresight(inflow_J, inflow_E, inflow_R, price, s_init_J, T)
        LOG.info("  PF: total=%.1f NOK (J=%.1f E=%.1f R=%.1f)",
                 pf["rev_total"] / 1e6, pf["rev_J"] / 1e6, pf["rev_E"] / 1e6, pf["rev_R"] / 1e6)

        # ── Open loop ──────────────────────────────────────────────────────
        LOG.info("  Open loop …")
        ol = run_cascade_open_loop(
            tree, inflow_J, inflow_E, inflow_R, price, s_init_J
        )
        LOG.info("  OL: total=%.1f NOK (J=%.1f E=%.1f R=%.1f)",
                 ol["rev_total"] / 1e6, ol["rev_J"] / 1e6, ol["rev_E"] / 1e6, ol["rev_R"] / 1e6)

        # ── Closed loop ────────────────────────────────────────────────────
        LOG.info("  Closed loop …")
        cl = run_cascade_closed_loop(
            year, panel, inflow_idata, price_idata,
            inflow_J, inflow_E, inflow_R, price, s_init_J, T
        )
        LOG.info("  CL: total=%.1f NOK (J=%.1f E=%.1f R=%.1f)",
                 cl["rev_total"] / 1e6, cl["rev_J"] / 1e6, cl["rev_E"] / 1e6, cl["rev_R"] / 1e6)

        weeks = year_panel.index

        # ── Diagnostic plot ────────────────────────────────────────────────
        plot_path = FIGURES_DIR / f"cascade_trajectory_{year}.png"
        _cascade_diagnostic_plot(year, weeks, ol, cl, pf, plot_path)

        rows.append({
            "year": year,
            "s_init_J_GWh":     s_init_J,
            "OL_J_MNOK":        ol["rev_J"] / 1e6,
            "OL_E_MNOK":        ol["rev_E"] / 1e6,
            "OL_R_MNOK":        ol["rev_R"] / 1e6,
            "OL_total_MNOK":    ol["rev_total"] / 1e6,
            "CL_J_MNOK":        cl["rev_J"] / 1e6,
            "CL_E_MNOK":        cl["rev_E"] / 1e6,
            "CL_R_MNOK":        cl["rev_R"] / 1e6,
            "CL_total_MNOK":    cl["rev_total"] / 1e6,
            "PF_J_MNOK":        pf["rev_J"] / 1e6,
            "PF_E_MNOK":        pf["rev_E"] / 1e6,
            "PF_R_MNOK":        pf["rev_R"] / 1e6,
            "PF_total_MNOK":    pf["rev_total"] / 1e6,
            "VoF_J_MNOK":       (cl["rev_J"] - ol["rev_J"]) / 1e6,
            "VoF_E_MNOK":       (cl["rev_E"] - ol["rev_E"]) / 1e6,
            "VoF_R_MNOK":       (cl["rev_R"] - ol["rev_R"]) / 1e6,
            "VoF_total_MNOK":   (cl["rev_total"] - ol["rev_total"]) / 1e6,
            "regret_MNOK":      (pf["rev_total"] - cl["rev_total"]) / 1e6,
            "OL_n_clamp_min":   ol.get("n_clamp_min", 0),
            "OL_n_clamp_max":   ol.get("n_clamp_max", 0),
        })

    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "cascade_backtest_results.csv", index=False)
    LOG.info("Saved: results/tables/cascade_backtest_results.csv")

    # ── Summary statistics ─────────────────────────────────────────────────
    vof_total = results["VoF_total_MNOK"].values
    vof_J     = results["VoF_J_MNOK"].values
    vof_E     = results["VoF_E_MNOK"].values
    vof_R     = results["VoF_R_MNOK"].values

    p_total = np.nan
    if len(vof_total) >= 5:
        try:
            _, p_total = wilcoxon(vof_total, alternative="greater")
        except Exception:
            pass

    summary = {
        "mean_OL_total_MNOK":     results["OL_total_MNOK"].mean(),
        "mean_CL_total_MNOK":     results["CL_total_MNOK"].mean(),
        "mean_PF_total_MNOK":     results["PF_total_MNOK"].mean(),
        "mean_VoF_total_MNOK":    float(vof_total.mean()),
        "std_VoF_total_MNOK":     float(vof_total.std()),
        "median_VoF_total_MNOK":  float(np.median(vof_total)),
        "wilcoxon_p_CL_gt_OL":    float(p_total),
        "mean_VoF_J_MNOK":        float(vof_J.mean()),
        "mean_VoF_E_MNOK":        float(vof_E.mean()),
        "mean_VoF_R_MNOK":        float(vof_R.mean()),
        "mean_VoF_J_share":       float(vof_J.mean() / vof_total.mean()) if vof_total.mean() != 0 else np.nan,
        "mean_VoF_E_share":       float(vof_E.mean() / vof_total.mean()) if vof_total.mean() != 0 else np.nan,
        "mean_VoF_R_share":       float(vof_R.mean() / vof_total.mean()) if vof_total.mean() != 0 else np.nan,
        "mean_regret_MNOK":       results["regret_MNOK"].mean(),
        "n_years":                len(results),
    }
    pd.DataFrame([summary]).to_csv(TABLES_DIR / "cascade_backtest_summary.csv", index=False)
    LOG.info("Saved: results/tables/cascade_backtest_summary.csv")

    LOG.info("\n=== Cascade Backtest Summary (%d years) ===", len(results))
    LOG.info("Mean OL total:  %.2f MNOK/yr", summary["mean_OL_total_MNOK"])
    LOG.info("Mean CL total:  %.2f MNOK/yr", summary["mean_CL_total_MNOK"])
    LOG.info("Mean PF total:  %.2f MNOK/yr", summary["mean_PF_total_MNOK"])
    LOG.info("Mean VoF total: %.2f MNOK/yr (std=%.1f, p=%.3f)",
             summary["mean_VoF_total_MNOK"], summary["std_VoF_total_MNOK"], p_total)
    LOG.info("VoF decomposition: J=%.2f E=%.2f R=%.2f MNOK/yr (shares: J=%.1f%% E=%.1f%% R=%.1f%%)",
             summary["mean_VoF_J_MNOK"], summary["mean_VoF_E_MNOK"], summary["mean_VoF_R_MNOK"],
             summary["mean_VoF_J_share"] * 100, summary["mean_VoF_E_share"] * 100,
             summary["mean_VoF_R_share"] * 100)
    LOG.info("Mean regret: %.2f MNOK/yr", summary["mean_regret_MNOK"])

    return results


if __name__ == "__main__":
    run()
