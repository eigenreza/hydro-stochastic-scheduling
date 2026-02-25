"""
Interactive demo: Realized Gain from Adaptive Hydropower Reservoir Scheduling

Run with:  streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.data_loader import (
    ANNUAL_PRICES,
    BACKTEST_YEARS,
    EXTREME_THRESHOLD,
    EXTREME_YEARS,
    NORMAL_YEARS,
    PRICE_MEAN_REF,
    PRICE_STD_REF,
    get_year_panel,
    load_phase_a,
    load_phase_b,
    load_phase_b_lag1,
    load_scenario_tree,
    s_init_for_year,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hydropower Flexibility",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS for table styling and spacing
st.markdown("""
<style>
    .metric-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .section-title {font-size: 1.05rem; font-weight: 600; color: #1a1a2e; margin-bottom: 4px;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    st.markdown(
        "There are two versions of the model here. The simple one treats the "
        "whole river as a single big reservoir. The detailed one models three "
        "real power plants in a row along the same river, each with its own "
        "real capacity. The year slider picks which of the 22 historical years "
        "you're looking at in the Year detail and Capacity explorer tabs. The "
        "year filter narrows things down to all years, normal years only, or "
        "just the energy crisis years, so you can see how differently the "
        "results behave depending on which kind of year you pick."
    )
    st.divider()

    model_choice = st.radio(
        "Model",
        ["Phase A: Single reservoir", "Phase B: Three-plant cascade"],
        help=(
            "Phase A aggregates all 44 Arendalsvassdraget plants into a single "
            "1300 GWh reservoir. Phase B resolves the three-plant cascade "
            "Jorundland → Evenstad → Rygene at full physical detail."
        ),
    )
    is_cascade = "Phase B" in model_choice

    year = st.select_slider(
        "Backtest year",
        options=BACKTEST_YEARS,
        value=2015,
    )

    if is_cascade:
        lag = st.radio(
            "Routing lag (Phase B only)",
            ["lag = 0  (same-week)", "lag = 1  (one-week delay)"],
            help=(
                "Lag=0 assumes J's release reaches E/R in the same weekly step "
                "(realistic: travel time 4-8 h). Lag=1 assumes a one-week delay. "
                "Toggle to see how sensitive Jorundland's realized CL-OL gain is to this assumption."
            ),
        )
        use_lag1 = "lag = 1" in lag

    regime_filter = st.radio(
        "Year filter",
        ["All 22 years", "Normal years only", "Crisis era (2018-2024) only"],
        help=(
            f"Crisis years are those where the annual mean NO2 price exceeded "
            f"{EXTREME_THRESHOLD:.0f} NOK/MWh (mean + 1.5 std of 2003-2020 reference). "
            f"These years (2018, 2021-2024) sit outside the scenario tree's "
            f"training distribution, making realized CL-OL gain estimates unreliable."
        ),
    )

    st.divider()
    st.caption(
        "Data: NVE HydAPI (inflow) + Energi Data Service (NO2 price). "
        "Backtest period: 2003-2024. No data beyond 2025-12-31."
    )

# ── Header ───────────────────────────────────────────────────────────────────

st.title("Realized Gain from Adaptive Reservoir Scheduling")
st.caption(
    "Built by Reza Azad Gholami "
    "([@eigenreza](https://github.com/eigenreza)). "
    "Source code, data pipeline, and documentation: "
    "[github.com/eigenreza/hydro-stochastic-scheduling]"
    "(https://github.com/eigenreza/hydro-stochastic-scheduling)."
)
st.markdown(
    "A hydropower operator running a reservoir faces a genuinely hard problem "
    "every single week. Water keeps arriving from rain and snowmelt, but nobody "
    "knows exactly how much is coming next month or next season. Electricity "
    "prices move up and down with weather, demand, and events nobody can fully "
    "predict either. The operator has to decide, again and again, how much "
    "water to release now and sell as electricity, and how much to hold back "
    "in case prices rise later or a dry spell hits.\n\n"
    "This project looks at that problem on a real river in southern Norway, "
    "the Arendalsvassdraget, using 22 years of real historical data from 2003 "
    "to 2024, including river flow records, electricity prices from the NO2 "
    "price zone, and the actual capacity of the power plants involved.\n\n"
    "There are two very different ways an operator could handle this. You "
    "could plan the whole year ahead of time and stick to that plan no matter "
    "what actually happens with weather or prices. Or you could keep "
    "adjusting as the year goes on, reacting to whatever the weather and the "
    "market actually do. The second approach should, in theory, do better, "
    "since you are using more information as it becomes available. The "
    "question this project asks is how much better, in real terms, not just "
    "in theory, and where in the river system that benefit actually shows up.\n\n"
    "To answer that, this project builds two versions of the same underlying "
    "model. One treats the whole river as a single large reservoir. The "
    "other is more realistic and follows three actual power plants in "
    "sequence along the river, Jorundland, Evenstad, and Rygene, each with "
    "its own real size and capacity, since water released by one plant "
    "becomes the next plant's water supply downstream. Both versions are "
    "tested against all 22 historical years, comparing a plan made in "
    "advance, a plan that adapts as the year unfolds, and a third benchmark "
    "that assumes perfect knowledge of the future, just to see how much is "
    "left on the table even for the adaptive approach."
)

with st.expander("Quick reference: what do OL, CL, PF, and the other shorthand mean?"):
    st.markdown(
        "You'll see the same handful of abbreviations on almost every tab, so "
        "here they are in one place.\n\n"
        "**OL, open-loop**: the plan made once, at the start of the year, and "
        "never changed afterward, no matter what the weather or prices "
        "actually do.\n\n"
        "**CL, closed-loop**: the plan that gets re-checked and adjusted "
        "every quarter as the year goes on, reacting to what's actually "
        "happened so far.\n\n"
        "**PF, perfect foresight**: a benchmark that isn't a real strategy "
        "anyone could follow, it assumes you already knew exactly what the "
        "weather and prices would do for the whole year. It shows the best "
        "anyone could possibly have done, so the other two can be measured "
        "against it.\n\n"
        "**Realized CL-OL gain** (labelled VoF in the charts and underlying "
        "tables, for historical reasons): the closed-loop plan's revenue "
        "minus the open-loop plan's revenue, both measured against what "
        "actually happened that year. This is the main number the whole "
        "project is trying to estimate, in plain terms, how much it's worth "
        "to be able to react instead of committing up front. It is an "
        "after-the-fact comparison, not a guaranteed-nonnegative quantity: "
        "it can come out negative in a year where reacting turned out worse "
        "than the fixed plan.\n\n"
        "**MNOK**: million Norwegian kroner, the currency unit used for "
        "every revenue and value figure in the app.\n\n"
        "**GWh**: gigawatt hours, the unit used for both the water stored in "
        "the reservoir and the electricity generated each week, measured in "
        "terms of the energy it represents rather than its volume."
    )

with st.expander("How to use this app"):
    st.markdown(
        "There are three tabs below, and they're ordered to make sense in "
        "that order.\n\n"
        "Start with **Capacity explorer**. Pick a year, leave the slider "
        "where it is, and hit Recompute. That alone gives you a feel for "
        "the two extremes this whole project compares, a plan made in "
        "advance against the best possible outcome in hindsight. Then try "
        "dragging the slider and recomputing again to see how the gap "
        "between them changes.\n\n"
        "Once that clicks, move to **Year detail**. Pick any year from the "
        "slider in the sidebar and you'll see how that single year actually "
        "played out, week by week, with real numbers attached to each of "
        "the three approaches.\n\n"
        "Finally, **Realized Gain Over Time** zooms out to all 22 years at once, so "
        "you can see the bigger pattern rather than one year in isolation.\n\n"
        "The sidebar controls apply across tabs: the model choice, the year "
        "you're looking at, and the year filter all carry over wherever you "
        "go."
    )

# ── Load data ─────────────────────────────────────────────────────────────────

pa  = load_phase_a()
pb  = load_phase_b()
pb1 = load_phase_b_lag1()

# Apply regime filter
def filter_years(df, year_col="year"):
    if regime_filter == "Normal years only":
        return df[df[year_col].isin(NORMAL_YEARS)]
    elif "Crisis" in regime_filter:
        return df[df[year_col].isin(EXTREME_YEARS)]
    return df

# ── Colour palette (consistent across tabs) ──────────────────────────────────

CLR_POS    = "#2563EB"   # blue for positive realized gain
CLR_NEG    = "#DC2626"   # red for negative realized gain
CLR_CRISIS = "#F59E0B"   # amber for crisis years
CLR_NORMAL = "#6B7280"   # grey accent
CLR_OL     = "#1D4ED8"
CLR_CL     = "#059669"
CLR_PF     = "#7C3AED"

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_live, tab_year, tab_overview = st.tabs([
    "Capacity explorer (live)",
    f"Year detail: {year}",
    "Realized Gain Over Time",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: realized CL-OL gain time series
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown(
        "This tab steps back and shows the full picture across all 22 years at "
        "once. Each bar is one year, and its height shows how much extra revenue "
        "adapting the plan produced that year, or how much it cost if the bar dips "
        "below zero. The shaded years mark the 2021 to 2024 energy crisis, when "
        "prices moved in ways nothing in the historical record had prepared the "
        "model for. That's why those years look so much more erratic than the rest."
    )
    if is_cascade:
        # Phase B: show lag=0 vs lag=1 overlay, or per-plant decomposition
        subtab_ts, subtab_decomp = st.tabs(["Realized gain over time", "Per-plant decomposition"])

        with subtab_ts:
            df_filt = filter_years(pb1, "year")

            fig = go.Figure()

            # lag=0 bars
            vof0 = df_filt["VoF_J_MNOK"].values
            yr_vals = df_filt["year"].values
            crisis_mask = np.isin(yr_vals, EXTREME_YEARS)

            colors0 = [CLR_NEG if v < 0 else CLR_POS for v in vof0]
            opacities0 = [0.5 if cr else 0.85 for cr in crisis_mask]

            fig.add_trace(go.Bar(
                x=yr_vals, y=vof0, name="Jorundland gain, lag=0",
                marker_color=colors0,
                marker_opacity=opacities0,
                offsetgroup=0,
            ))

            if not use_lag1:
                # Also show lag=1 side-by-side
                vof1 = df_filt["VoF_J_lag1"].values
                colors1 = [CLR_NEG if v < 0 else "#10B981" for v in vof1]
                fig.add_trace(go.Bar(
                    x=yr_vals, y=vof1, name="Jorundland gain, lag=1",
                    marker_color=colors1,
                    marker_opacity=[0.5 if cr else 0.85 for cr in crisis_mask],
                    offsetgroup=1,
                ))
                fig.update_layout(barmode="group")
            else:
                vof_show = df_filt["VoF_J_lag1"].values
                colors_show = [CLR_NEG if v < 0 else CLR_POS for v in vof_show]
                fig.data = []
                fig.add_trace(go.Bar(
                    x=yr_vals, y=vof_show, name="Jorundland gain, lag=1",
                    marker_color=colors_show,
                    marker_opacity=opacities0,
                ))

            # Mark selected year
            fig.add_vline(
                x=year - 0.4, line_dash="dot", line_color="#888", line_width=1.5,
                annotation_text=str(year), annotation_position="top right",
                annotation_font_size=10,
            )

            # Shade crisis era
            for ext_yr in EXTREME_YEARS:
                if ext_yr in yr_vals:
                    fig.add_vrect(
                        x0=ext_yr - 0.5, x1=ext_yr + 0.5,
                        fillcolor=CLR_CRISIS, opacity=0.08, line_width=0,
                    )

            fig.add_hline(y=0, line_color="#999", line_width=1)
            fig.update_layout(
                title=f"Phase B: Annual realized gain at Jorundland, {lag.split(' ')[2]} routing",
                xaxis_title="Year",
                yaxis_title="Realized gain (MNOK/yr)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=380,
                margin=dict(l=50, r=20, t=60, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each bar is one year. Its height is how much extra revenue "
                "Jorundland earned that year from being able to adapt as "
                "things unfolded, compared to sticking to a plan made in "
                "advance. A bar below the zero line means adapting actually "
                "did worse that year. The unit is million Norwegian kroner "
                "per year (MNOK/yr)."
            )

            # Summary stats
            col1, col2, col3 = st.columns(3)
            sel_col = "VoF_J_lag1" if use_lag1 else "VoF_J_MNOK"
            sub = filter_years(pb1)[sel_col]
            nrm = pb1[pb1["year"].isin(NORMAL_YEARS)][sel_col]
            ext = pb1[pb1["year"].isin(EXTREME_YEARS)][sel_col]
            with col1:
                st.metric("All years: mean realized gain", f"{sub.mean():.2f} MNOK/yr",
                          f"std {sub.std():.1f}")
            with col2:
                st.metric("Normal years: mean realized gain", f"{nrm.mean():.2f} MNOK/yr",
                          f"std {nrm.std():.1f}  (n={len(nrm)})")
            with col3:
                st.metric("Crisis years: mean realized gain", f"{ext.mean():.2f} MNOK/yr",
                          f"std {ext.std():.1f}  (n={len(ext)})")

            st.caption(
                "Crisis years (amber shading) had annual mean prices above "
                f"{EXTREME_THRESHOLD:.0f} NOK/MWh, far outside the scenario "
                "tree's training distribution. Both OL and CL policies worked "
                "from miscalibrated forecasts, making realized CL-OL gain estimates unreliable."
            )

        with subtab_decomp:
            df_pb = filter_years(pb)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_pb["year"], y=df_pb["VoF_J_MNOK"], name="Jorundland (storage)",
                marker_color=CLR_POS,
                marker_opacity=[0.5 if yr in EXTREME_YEARS else 0.85 for yr in df_pb["year"]],
            ))
            fig2.add_trace(go.Bar(
                x=df_pb["year"], y=df_pb["VoF_E_MNOK"], name="Evenstad (run-of-river)",
                marker_color="#9CA3AF",
            ))
            fig2.add_trace(go.Bar(
                x=df_pb["year"], y=df_pb["VoF_R_MNOK"], name="Rygene (run-of-river)",
                marker_color="#6B7280",
            ))
            fig2.add_hline(y=0, line_color="#999", line_width=1)
            fig2.update_layout(
                barmode="stack",
                title="Phase B: Per-plant realized gain decomposition (lag=0)",
                xaxis_title="Year", yaxis_title="Realized gain (MNOK/yr)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=380,
                margin=dict(l=50, r=20, t=60, b=50),
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.info(
                "Evenstad and Rygene have a realized CL-OL gain of 0.000 MNOK "
                "in every single year, regardless of the routing assumption. "
                "Both plants' local inflow exceeds their turbine capacity in "
                "every week of the 26-year record. They run flat out regardless "
                "of policy. All the realized gain in the cascade lives at "
                "Jorundland, the only node with meaningful reservoir storage."
            )

    else:
        # Phase A
        df_pa = filter_years(pa)
        vof_vals = df_pa["VoF_MNOK"].values
        yr_vals  = df_pa["year"].values
        crisis_mask = np.isin(yr_vals, EXTREME_YEARS)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yr_vals, y=vof_vals, name="Realized gain (MNOK/yr)",
            marker_color=[CLR_NEG if v < 0 else CLR_POS for v in vof_vals],
            marker_opacity=[0.5 if cr else 0.85 for cr in crisis_mask],
        ))
        for ext_yr in EXTREME_YEARS:
            if ext_yr in yr_vals:
                fig.add_vrect(
                    x0=ext_yr - 0.5, x1=ext_yr + 0.5,
                    fillcolor=CLR_CRISIS, opacity=0.08, line_width=0,
                )
        fig.add_hline(y=0, line_color="#999", line_width=1)
        fig.add_vline(
            x=year - 0.4, line_dash="dot", line_color="#888", line_width=1.5,
            annotation_text=str(year), annotation_position="top right",
            annotation_font_size=10,
        )
        fig.update_layout(
            title="Phase A: Annual realized CL-OL gain (CL revenue minus OL revenue)",
            xaxis_title="Year", yaxis_title="Realized gain (MNOK/yr)",
            height=380,
            margin=dict(l=50, r=20, t=60, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each bar is one year. Its height is how much extra revenue the "
            "system earned that year from being able to adapt as things "
            "unfolded, compared to sticking to a plan made in advance. A bar "
            "below the zero line means adapting actually did worse that "
            "year. The unit is million Norwegian kroner per year (MNOK/yr)."
        )

        col1, col2, col3 = st.columns(3)
        nrm_vof = pa[pa["year"].isin(NORMAL_YEARS)]["VoF_MNOK"]
        ext_vof = pa[pa["year"].isin(EXTREME_YEARS)]["VoF_MNOK"]
        with col1:
            st.metric("All years: mean realized gain", f"{pa['VoF_MNOK'].mean():.1f} MNOK/yr",
                      f"std {pa['VoF_MNOK'].std():.1f}")
        with col2:
            st.metric("Normal years", f"{nrm_vof.mean():.1f} MNOK/yr",
                      f"std {nrm_vof.std():.1f}  (n={len(nrm_vof)})")
        with col3:
            st.metric("Crisis years", f"{ext_vof.mean():.1f} MNOK/yr",
                      f"std {ext_vof.std():.1f}  (n={len(ext_vof)})")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Year detail
# ═══════════════════════════════════════════════════════════════════════════════

with tab_year:
    st.markdown(
        "Here you can pick a single year out of the 22 studied and see exactly "
        "how the reservoir was managed that year, week by week, under each of "
        "the three approaches, plus what each one actually earned in revenue. "
        "The colored note below tells you whether the year was a normal market "
        "year or part of the energy crisis, which matters because the numbers "
        "mean something different depending on which kind of year you're in."
    )

    yr_price = ANNUAL_PRICES.get(year, 0)
    is_crisis = year in EXTREME_YEARS

    # Header ribbon
    if is_crisis:
        st.warning(
            f"**{year} is an elevated-price year** "
            f"(annual mean {yr_price:.0f} NOK/MWh, "
            f"above {EXTREME_THRESHOLD:.0f} NOK/MWh threshold). "
            "Both OL and CL policies were working from scenario trees "
            "calibrated on a fundamentally different price distribution."
        )
    else:
        st.success(
            f"**{year}: normal market year** "
            f"(annual mean {yr_price:.0f} NOK/MWh, "
            f"within the 2003-2020 reference distribution)."
        )

    # Revenue and realized-gain metrics
    try:
        if is_cascade:
            row_b = pb[pb["year"] == year].iloc[0]
            row_l = pb1[pb1["year"] == year].iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            vof_val = row_l["VoF_J_lag1"] if use_lag1 else row_b["VoF_J_MNOK"]
            ol_val  = row_l["OL_lag1"] if use_lag1 else row_b["OL_total_MNOK"]
            cl_val  = row_l["CL_lag1"] if use_lag1 else row_b["CL_total_MNOK"]
            pf_val  = row_l["PF_lag1"] if use_lag1 else row_b["PF_total_MNOK"]
            with col1:
                st.metric("OL revenue", f"{ol_val:.1f} MNOK")
            with col2:
                st.metric("CL revenue", f"{cl_val:.1f} MNOK")
            with col3:
                st.metric("PF revenue", f"{pf_val:.1f} MNOK")
            with col4:
                delta_sign = "+" if vof_val > 0 else ""
                st.metric(
                    f"Realized gain, Jorundland ({lag.split(' ')[2]})",
                    f"{delta_sign}{vof_val:.2f} MNOK",
                    help="CL - OL revenue at Jorundland"
                )

            # Per-plant realized-gain bar for this year
            st.subheader("Per-plant realized gain decomposition")
            plants = ["Jorundland", "Evenstad", "Rygene"]
            vof_plants = [row_b["VoF_J_MNOK"], row_b["VoF_E_MNOK"], row_b["VoF_R_MNOK"]]
            fig_decomp = go.Figure(go.Bar(
                x=vof_plants, y=plants, orientation="h",
                marker_color=[CLR_NEG if v < 0 else CLR_POS for v in vof_plants],
            ))
            fig_decomp.add_vline(x=0, line_color="#999", line_width=1)
            fig_decomp.update_layout(
                height=200,
                margin=dict(l=10, r=20, t=20, b=40),
                xaxis_title="Realized gain (MNOK)",
            )
            st.plotly_chart(fig_decomp, use_container_width=True)
            if row_b["VoF_E_MNOK"] == 0.0 and row_b["VoF_R_MNOK"] == 0.0:
                st.caption(
                    "Evenstad's and Rygene's realized gain is exactly zero: both "
                    "plants' local inflow exceeds their turbine capacity in every "
                    "week, so policy choice at Jorundland does not change their "
                    "generation."
                )

        else:
            row_a = pa[pa["year"] == year].iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("OL revenue", f"{row_a['OL_MNOK']:.1f} MNOK")
            with col2:
                st.metric("CL revenue", f"{row_a['CL_MNOK']:.1f} MNOK")
            with col3:
                st.metric("PF revenue", f"{row_a['PF_MNOK']:.1f} MNOK")
            with col4:
                v = row_a["VoF_MNOK"]
                st.metric("Realized gain", f"{'+'  if v > 0 else ''}{v:.1f} MNOK")
    except (IndexError, KeyError):
        st.warning(
            "There's no precomputed result for this year and model combination. "
            "Try a different year, or switch model."
        )

    # Trajectory figure
    st.subheader("Weekly trajectory")
    fig_prefix = "cascade_" if is_cascade else ""
    fig_path = ROOT / f"results/figures/{fig_prefix}trajectory_{year}.png"
    try:
        if fig_path.exists():
            img = Image.open(fig_path)
            st.image(img, use_container_width=True,
                     caption=(
                         f"{'Cascade' if is_cascade else 'Single-reservoir'} trajectory "
                         f"{year}: storage, generation, inflow, and price. "
                         "Blue = open-loop, green = closed-loop, purple = perfect foresight."
                     ))
            st.caption(
                "The storage line shows how full the reservoir is, in energy "
                "terms, week by week over the year. The dotted red line near "
                "the top marks the reservoir's maximum capacity. When a line "
                "goes flat at the top, the reservoir is full and any extra "
                "water has to spill or be released regardless of what the "
                "policy would otherwise choose. When it goes flat at zero, "
                "the reservoir is empty and there's simply no water left to "
                "release that week."
            )
        else:
            st.info(f"Trajectory figure not found for {year}.")
    except Exception:
        st.warning("The trajectory image for this year couldn't be loaded.")

    # Regime context: weekly price distribution for this year
    with st.expander("Weekly price distribution this year vs. historical"):
        try:
            panel_yr = get_year_panel(year)
            from app.data_loader import load_panel
            full_panel = load_panel()
            hist_prices = full_panel[
                full_panel["year"].between(2003, 2020)
            ]["price_avg_NOK_MWh"]
            yr_prices = panel_yr["price_avg_NOK_MWh"]

            fig_px = go.Figure()
            fig_px.add_trace(go.Box(
                y=hist_prices, name="2003-2020 reference",
                marker_color=CLR_NORMAL, boxmean=True,
            ))
            fig_px.add_trace(go.Box(
                y=yr_prices, name=str(year),
                marker_color=CLR_CRISIS if is_crisis else CLR_POS,
                boxmean=True,
            ))
            fig_px.update_layout(
                yaxis_title="Weekly price (NOK/MWh)",
                height=300,
                margin=dict(l=50, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_px, use_container_width=True)
            st.caption(
                "Each box covers the middle half of weekly prices that year "
                "(25th to 75th percentile), the line inside is the median, "
                "and the diamond is the average. The whiskers show the rest "
                "of the range. If the selected year's box sits much higher "
                "than the 2003-2020 reference box, or is much taller, prices "
                "that year were either persistently higher than usual, more "
                "volatile week to week, or both, which is exactly the "
                "condition under which the forecasting model is working "
                "outside the range it was built from."
            )
        except Exception:
            st.warning("The price comparison for this year couldn't be loaded.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Live recompute, capacity slider
# ═══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Capacity explorer: how does realized gain change with Jorundland's reservoir size?")
    st.markdown(
        "This tab lets you test the core idea of the project directly, on one of "
        "the three power plants studied: Jorundland, the one with real water "
        "storage behind it. Move the slider to change how big that plant's "
        "reservoir is assumed to be, then press recompute and the math gets "
        "solved fresh, right then, for whichever year you've picked. Two of the "
        "numbers you'll see are recomputed live: a release plan committed up "
        "front without knowing the future, and the best possible outcome if you "
        "somehow knew the future in advance. The gap between those two is "
        "roughly the value of having better information, not the realized "
        "CL-OL gain itself, since this open-loop plan is never revised. The "
        "fourth number, the closed-loop realized gain, is not part of "
        "this live experiment. It's pulled straight from the original 22 year "
        "backtest at the plant's actual real world capacity, shown here for "
        "reference so you can compare what the slider is telling you against "
        "the project's main finding. That one stays fixed no matter where you "
        "move the slider, since it isn't being recomputed."
    )

    colA, colB = st.columns([1, 1])
    with colA:
        live_year = st.selectbox(
            "Year for live recompute",
            options=BACKTEST_YEARS,
            index=BACKTEST_YEARS.index(2015),
            help="Choose a normal market year for the clearest signal.",
        )
    with colB:
        s_max_j = st.slider(
            "Jorundland reservoir capacity S_max (GWh)",
            min_value=20,
            max_value=400,
            value=167,
            step=5,
            help="Current design value: 167 GWh (from NVE HydAPI station 19.5.0, Nesvatn).",
        )

    st.caption(f"Current design capacity: **167 GWh**. Selected: **{s_max_j} GWh** ({s_max_j/167*100:.0f}% of actual).")

    if st.button("Recompute", type="primary", use_container_width=False):
        with st.spinner(f"Solving OL + PF LPs for {live_year} with S_max = {s_max_j} GWh..."):
            from app.live_solver import LiveSolveError, run_live_solve

            st.session_state.pop("live_result", None)
            try:
                tree      = load_scenario_tree(live_year)
                panel_yr  = get_year_panel(live_year)
                s_init    = s_init_for_year(live_year, "phase_b")
                result    = run_live_solve(live_year, float(s_max_j), tree, panel_yr, s_init)

                st.session_state["live_result"]  = result
                st.session_state["live_year"]    = live_year
                st.session_state["live_s_max_j"] = s_max_j
            except LiveSolveError as e:
                if e.infeasible:
                    st.error(
                        "That combination of settings doesn't leave a physically "
                        "possible way to manage the reservoir. The capacity you've "
                        "chosen is too small to hold the water this plant is "
                        "actually carrying at the start of the year, even running "
                        "flat out. Try a larger capacity, or pick a different year."
                    )
                else:
                    st.error(
                        "The solver couldn't find a workable schedule for this "
                        "combination of settings. Try a different capacity value "
                        "or a different year."
                    )
            except FileNotFoundError:
                st.error(
                    "The data needed for this year isn't available. Try a "
                    "different year from the slider."
                )
            except Exception:
                st.error(
                    "Something went wrong while preparing or running the "
                    "calculation for this year. Try a different year or "
                    "capacity value."
                )

    # Show result if available
    if "live_result" in st.session_state and st.session_state.get("live_year") == live_year:
        res = st.session_state["live_result"]
        s_used = st.session_state.get("live_s_max_j", 167)

        # Compare to reference (S_max = 167 GWh, from precomputed cascade results)
        row_ref = pb[pb["year"] == live_year]
        ref_ol  = float(row_ref["OL_total_MNOK"].iloc[0]) if len(row_ref) else None
        ref_pf  = float(row_ref["PF_total_MNOK"].iloc[0]) if len(row_ref) else None

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                f"OL revenue (S_max={s_used} GWh)",
                f"{res['ol_rev']:.1f} MNOK",
                delta=f"{res['ol_rev'] - ref_ol:.1f} vs 167 GWh" if ref_ol else None,
            )
        with c2:
            st.metric(
                f"PF revenue (S_max={s_used} GWh)",
                f"{res['pf_rev']:.1f} MNOK",
                delta=f"{res['pf_rev'] - ref_pf:.1f} vs 167 GWh" if ref_pf else None,
            )
        with c3:
            gap = res["pf_rev"] - res["ol_rev"]
            ref_gap = (ref_pf - ref_ol) if ref_ol and ref_pf else None
            st.metric(
                "PF - OL gap (info value)",
                f"{gap:.2f} MNOK",
                delta=f"{gap - ref_gap:.2f} vs 167 GWh" if ref_gap else None,
            )
        with c4:
            vof_col = "VoF_J_MNOK"
            ref_vof = float(pb[pb["year"] == live_year][vof_col].iloc[0]) if len(row_ref) else None
            st.metric(
                "CL - OL realized gain (from backtest)",
                f"{ref_vof:.2f} MNOK" if ref_vof is not None else "n/a",
                help="Pre-computed realized gain from the full closed-loop backtest at S_max=167 GWh",
            )

        # Storage trajectory comparison
        weeks = res["weeks"]
        st.subheader(f"Jorundland storage trajectory: {live_year}, S_max = {s_used} GWh")
        fig_traj = go.Figure()

        s_max_line = s_used
        fig_traj.add_hline(
            y=s_max_line, line_dash="dash", line_color="#9CA3AF",
            annotation_text=f"S_max = {s_max_line} GWh",
            annotation_position="right",
        )
        fig_traj.add_trace(go.Scatter(
            x=list(range(len(weeks) + 1)),
            y=res["ol_storage"],
            mode="lines",
            name=f"Open-loop (S_max={s_used})",
            line=dict(color=CLR_OL, width=2),
        ))
        fig_traj.add_trace(go.Scatter(
            x=list(range(len(weeks) + 1)),
            y=res["pf_storage"],
            mode="lines",
            name=f"Perfect foresight (S_max={s_used})",
            line=dict(color=CLR_PF, width=2, dash="dot"),
        ))

        # Also show reference OL trajectory from precomputed (if S_max != 167)
        if s_used != 167 and len(row_ref):
            # We don't have a pre-stored trajectory, so just note the reference revenue
            pass

        fig_traj.update_layout(
            xaxis_title="Week of year",
            yaxis_title="Storage (GWh)",
            yaxis_range=[-5, max(s_max_line * 1.1, 180)],
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=320,
            margin=dict(l=50, r=20, t=40, b=50),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        st.caption(
            "Both lines show how full Jorundland's reservoir is over the year "
            "at the capacity you've set with the slider, marked by the dashed "
            "grey line. The solid blue line is the committed plan, made "
            "without knowing how the year would turn out. The dotted purple "
            "line is the perfect foresight plan, which already knows the real "
            "prices and inflow for every week, so it can sometimes hold more "
            "water back, or release more of it, than the committed plan would "
            "dare to. That's what reacting to the real future, rather than "
            "guessing at it, looks like."
        )

        # Jorundland generation schedule
        st.subheader(f"Jorundland weekly generation: {live_year}")
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Scatter(
            x=weeks, y=res["ol_gen_j"], mode="lines+markers",
            name="Open-loop", line=dict(color=CLR_OL, width=1.5),
            marker=dict(size=3),
        ))
        fig_gen.add_trace(go.Scatter(
            x=weeks, y=res["pf_gen_j"], mode="lines",
            name="Perfect foresight", line=dict(color=CLR_PF, width=1.5, dash="dot"),
        ))
        fig_gen.add_hline(
            y=9.274, line_dash="dash", line_color="#9CA3AF",
            annotation_text="G_max_J", annotation_position="right",
        )
        fig_gen.update_layout(
            xaxis_title="ISO week",
            yaxis_title="Generation (GWh/wk)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=280,
            margin=dict(l=50, r=20, t=40, b=50),
        )
        st.plotly_chart(fig_gen, use_container_width=True)
        st.caption(
            "This is the release decision itself, week by week, for the "
            "same two policies as the storage chart above. The dashed line "
            "at G_max_J is Jorundland's turbine capacity, 9.274 GWh/week: a "
            "line touching it means the plant is running flat out that "
            "week and could not have generated more even if it wanted to. "
            "Weeks where the perfect-foresight line sits clearly above or "
            "below the open-loop line are weeks where knowing the real "
            "price in advance would have changed the release; compare "
            "those weeks against the price bars below to see whether that "
            "shift moved water toward the higher-price weeks."
        )

        # Price context
        panel_yr = get_year_panel(live_year)
        price_vals = panel_yr["price_avg_NOK_MWh"].values[:52]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(
            x=weeks, y=price_vals, name="Weekly price",
            marker_color=CLR_CRISIS, marker_opacity=0.7,
        ))
        fig_pr.update_layout(
            xaxis_title="ISO week",
            yaxis_title="Price (NOK/MWh)",
            height=200,
            margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption(
            "The realized weekly price for this year, on the same week axis "
            "as the generation chart above. Tall bars are expensive weeks; "
            "short bars are cheap ones."
        )
    else:
        st.info("Set the capacity and click Recompute to run a live LP solve.")
        st.caption(
            "Each click solves two linear programs (open-loop stochastic with a "
            "16-scenario sample + perfect-foresight deterministic) using HiGHS. "
            "Typical solve time: 1-3 seconds."
        )


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "If you want to go deeper than what's on this page, the full write-up "
    "is in the project's own files. `docs/methodology.md` covers exactly how "
    "the model is built and every assumption behind it. `docs/key_findings.md` "
    "is a short summary of what the 22 year backtest actually found. "
    "`data/raw/SOURCES.md` lists exactly where the river flow and price data "
    "came from, in case you want to check it yourself."
)
