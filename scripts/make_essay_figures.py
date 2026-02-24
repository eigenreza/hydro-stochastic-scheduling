"""
Generates the figures used in docs/essay.md from existing project outputs.

Reads only data already computed by the backtest pipeline
(results/tables/) and the weekly panel (data/processed/). Does not
recompute or approximate any underlying result; this script only
re-renders numbers that already exist elsewhere in the repository in
a form suitable for a short paper.

Run with:  python scripts/make_essay_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results/figures/essay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTREME_YEARS = [2018, 2021, 2022, 2023, 2024]

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_vof_decomposition():
    """Per-plant VoF decomposition across the 22-year cascade backtest."""
    cb = pd.read_csv(ROOT / "results/tables/cascade_backtest_results.csv")
    cb = cb.sort_values("year")

    fig, ax = plt.subplots(figsize=(8, 4))
    years = cb["year"].values
    crisis_mask = np.isin(years, EXTREME_YEARS)

    colors_j = ["#F59E0B" if c else "#1D4ED8" for c in crisis_mask]
    ax.bar(years, cb["VoF_J_MNOK"], color=colors_j, width=0.7)

    assert (cb["VoF_E_MNOK"] == 0.0).all() and (cb["VoF_R_MNOK"] == 0.0).all()

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Value of flexibility (MNOK/yr)")
    ax.set_title("Per-plant value of flexibility, three-plant cascade, 2003-2024")
    ax.text(
        0.015, 0.04,
        "Evenstad and Rygene (run-of-river): VoF = 0.000 in all 22 years, not plotted",
        transform=ax.transAxes, fontsize=8.5, color="#4B5563",
    )

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1D4ED8", label="Jorundland, normal years"),
        Patch(facecolor="#F59E0B", label="Jorundland, crisis years (2018, 2021-2024)"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_path = OUT_DIR / "fig1_vof_decomposition.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def fig2_regime_shift_mechanism():
    """
    Two-panel figure on the regime-shift mechanism:
      (a) 2022 realized weekly price vs. the 2003-2020 seasonal climatology
      (b) routing-lag VoF divergence vs. annual mean price, all 22 years
    """
    panel = pd.read_csv(ROOT / "data/processed/weekly_panel.csv", parse_dates=["week_start"])
    panel["year"] = panel["week_start"].dt.year
    panel["iso_week"] = panel["week_start"].dt.isocalendar().week

    hist = panel[panel["year"].between(2003, 2020)]
    climatology = hist.groupby("iso_week")["price_avg_NOK_MWh"].mean()

    yr2022 = panel[panel["year"] == 2022].set_index("iso_week")["price_avg_NOK_MWh"]
    common_weeks = climatology.index.intersection(yr2022.index)
    corr_2022 = yr2022.loc[common_weeks].corr(climatology.loc[common_weeks])

    lag = pd.read_csv(ROOT / "results/tables/cascade_lag1_sensitivity.csv")
    annual_price = {
        2003: 286.9, 2004: 245.7, 2005: 233.4, 2006: 396.8, 2007: 209.0,
        2008: 324.6, 2009: 294.2, 2010: 408.7, 2011: 357.7, 2012: 218.7,
        2013: 290.3, 2014: 227.9, 2015: 175.5, 2016: 233.9, 2017: 268.9,
        2018: 417.0, 2019: 383.2, 2020: 96.7, 2021: 768.1, 2022: 2128.9,
        2023: 903.5, 2024: 583.8,
    }
    lag["annual_price"] = lag["year"].map(annual_price)
    lag["abs_divergence"] = (lag["VoF_J_MNOK"] - lag["VoF_J_lag1"]).abs()
    r_lag, p_lag = stats.pearsonr(lag["annual_price"], lag["abs_divergence"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Both series normalized to their own mean so the seasonal SHAPE is
    # comparable on one axis; 2022's price level is ~7.6x the historical
    # mean and would otherwise swamp the climatology line entirely.
    clim_norm = climatology.values / climatology.values.mean()
    yr2022_common = yr2022.loc[common_weeks]
    yr2022_norm = yr2022_common.values / yr2022_common.values.mean()

    ax1.plot(climatology.index, clim_norm, color="#6B7280", linewidth=2,
             label="2003-2020 seasonal climatology")
    ax1.plot(yr2022_common.index, yr2022_norm, color="#DC2626", linewidth=1.8,
              label="2022 realized")
    ax1.set_xlabel("ISO week")
    ax1.set_ylabel("Weekly price, normalized to annual mean")
    ax1.set_title(f"(a) 2022 vs. historical seasonal shape\n(r = {corr_2022:.2f})")
    ax1.legend(fontsize=8, loc="upper left")

    crisis_mask = lag["year"].isin(EXTREME_YEARS)
    ax2.scatter(lag.loc[~crisis_mask, "annual_price"], lag.loc[~crisis_mask, "abs_divergence"],
                color="#1D4ED8", label="Normal years", zorder=3)
    ax2.scatter(lag.loc[crisis_mask, "annual_price"], lag.loc[crisis_mask, "abs_divergence"],
                color="#F59E0B", label="Crisis years", zorder=3)
    xs = np.linspace(lag["annual_price"].min(), lag["annual_price"].max(), 100)
    slope, intercept = np.polyfit(lag["annual_price"], lag["abs_divergence"], 1)
    ax2.plot(xs, slope * xs + intercept, color="#374151", linewidth=1, linestyle="--", zorder=2)
    ax2.set_xlabel("Annual mean price (NOK/MWh)")
    ax2.set_ylabel("|VoF_J(lag=0) - VoF_J(lag=1)|  (MNOK/yr)")
    ax2.set_title(f"(b) Routing-lag divergence vs. price level\n(r = {r_lag:.3f}, p < 0.001)")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path = OUT_DIR / "fig2_regime_shift_mechanism.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  2022 seasonal correlation r = {corr_2022:.3f}")
    print(f"  routing-lag divergence correlation r = {r_lag:.3f}, p = {p_lag:.2e}")


def fig3_storage_trajectory_2022():
    """
    Copies the existing, already-computed cascade trajectory diagnostic
    plot for 2022 (results/figures/cascade_trajectory_2022.png) into the
    essay figure folder. Not regenerated: this is the same plot produced
    by src/backtest/run_cascade_backtest.py as part of the official
    backtest pipeline. Its top panel (Jorundland storage under OL/CL/PF)
    is the evidence cited for ruling out reservoir saturation as the
    cause of 2022's VoF instability.
    """
    src = ROOT / "results/figures/cascade_trajectory_2022.png"
    dst = OUT_DIR / "fig3_storage_trajectory_2022.png"
    shutil.copyfile(src, dst)
    print(f"Copied {src} -> {dst}")


def fig4_phaseA_vs_phaseB():
    """
    Year-by-year VoF comparison between the Phase A single-reservoir
    model and the Phase B cascade model (Jorundland only), to make the
    cross-model consistency claim in Section 4.2 visual rather than only
    verbal. Both series come directly from the existing backtest result
    tables; no new computation.
    """
    pa = pd.read_csv(ROOT / "results/tables/backtest_results.csv")
    pb = pd.read_csv(ROOT / "results/tables/cascade_backtest_results.csv")

    pa = pa.sort_values("year")
    pb = pb.sort_values("year")
    pa_vof = pa["value_of_flexibility_NOK"].values / 1e6
    pb_vof = pb["VoF_J_MNOK"].values
    years = pa["year"].values
    crisis_mask = np.isin(years, EXTREME_YEARS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    colors_pa = ["#F59E0B" if c else "#1D4ED8" for c in crisis_mask]
    ax1.bar(years, pa_vof, color=colors_pa, width=0.7)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("VoF, Phase A\n(single reservoir, MNOK/yr)")
    ax1.set_title("(a) Phase A: single aggregated reservoir, S_max = 1300 GWh")

    colors_pb = ["#F59E0B" if c else "#1D4ED8" for c in crisis_mask]
    ax2.bar(years, pb_vof, color=colors_pb, width=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("VoF_J, Phase B\n(Jorundland, MNOK/yr)")
    ax2.set_title("(b) Phase B: three-plant cascade, S_max_J = 167 GWh")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1D4ED8", label="Normal years"),
        Patch(facecolor="#F59E0B", label="Crisis years (2018, 2021-2024)"),
    ]
    ax1.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_path = OUT_DIR / "fig4_phaseA_vs_phaseB.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    fig1_vof_decomposition()
    fig2_regime_shift_mechanism()
    fig3_storage_trajectory_2022()
    fig4_phaseA_vs_phaseB()
