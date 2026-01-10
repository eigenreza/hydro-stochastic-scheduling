"""
Main data acquisition pipeline.

Run from the project root:
    python -m src.data_acquisition.run_acquisition

Outputs:
    data/raw/selected_station.json
    data/raw/station_scores.csv
    data/raw/inflow_daily_raw.parquet
    data/raw/nordpool_no2_raw.parquet
    data/raw/SOURCES.md
    data/processed/weekly_panel.csv
    data/processed/DATA_DICTIONARY.md
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_acquisition.nve_client import fetch_daily_series
from src.data_acquisition.nordpool_client import fetch_no2_prices
from src.data_acquisition.station_selector import select_best_station
from src.data_acquisition.weekly_panel import build_weekly_panel, save_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)
RAW_DIR = Path("data/raw")


def run() -> None:
    pull_time = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ── 1. Station selection ────────────────────────────────────────────────
    LOG.info("=== Step 1: Station selection ===")
    station_json = RAW_DIR / "selected_station.json"
    if station_json.exists():
        LOG.info("Loading previously selected station from %s", station_json)
        best = json.loads(station_json.read_text())
    else:
        best = select_best_station()

    LOG.info("Station: %s — %s (river: %s, parameter: %s, completeness: %.3f)",
             best["station_id"], best["stationName"],
             best.get("riverName", ""), best["parameter"], best["completeness"])

    # ── 2. Inflow data pull ─────────────────────────────────────────────────
    LOG.info("=== Step 2: Inflow data pull ===")
    raw_inflow_path = RAW_DIR / "inflow_daily_raw.parquet"
    if raw_inflow_path.exists():
        LOG.info("Loading cached raw inflow from %s", raw_inflow_path)
        daily_inflow = pd.read_parquet(raw_inflow_path)
    else:
        daily_inflow = fetch_daily_series(
            station_id=best["station_id"],
            parameter=best["parameter"],
            start_date="1980-01-01",
            end_date="2025-12-31",
        )
        if daily_inflow.empty:
            raise RuntimeError(
                f"No inflow data returned for station {best['station_id']}. "
                "Check NVE_API_KEY and network connectivity."
            )
        daily_inflow.to_parquet(raw_inflow_path)
        LOG.info("Saved raw inflow: %s (%d days)", raw_inflow_path, len(daily_inflow))

    LOG.info("Inflow series: %s → %s, n=%d, missing=%.1f%%",
             daily_inflow.index.min().date(), daily_inflow.index.max().date(),
             len(daily_inflow),
             100 * daily_inflow["value"].isna().mean())

    # ── 3. Price data pull ──────────────────────────────────────────────────
    LOG.info("=== Step 3: Nord Pool NO2 price pull ===")
    raw_price_path = RAW_DIR / "nordpool_no2_raw.parquet"
    if raw_price_path.exists():
        LOG.info("Loading cached raw prices from %s", raw_price_path)
        daily_price = pd.read_parquet(raw_price_path)
    else:
        daily_price = fetch_no2_prices(
            start_date="2000-01-01",
            end_date="2025-12-31",
        )
    daily_price.index = pd.to_datetime(daily_price.index).tz_localize(None)

    LOG.info("Price series: %s → %s, n=%d",
             daily_price.index.min().date(), daily_price.index.max().date(),
             len(daily_price))

    # ── 4. Weekly panel ─────────────────────────────────────────────────────
    LOG.info("=== Step 4: Building weekly panel ===")
    panel = build_weekly_panel(daily_inflow, daily_price)
    save_panel(panel)

    # ── 5. SOURCES.md ───────────────────────────────────────────────────────
    _write_sources_md(best, pull_time, daily_inflow, daily_price, panel)

    LOG.info("=== Data acquisition complete. ===")
    LOG.info("Weekly panel: %d weeks (%s → %s)",
             len(panel), panel.index.min().date(), panel.index.max().date())


def _write_sources_md(best, pull_time, daily_inflow, daily_price, panel) -> None:
    scores_path = RAW_DIR / "station_scores.csv"
    scores_preview = ""
    if scores_path.exists():
        import pandas as pd
        sc = pd.read_csv(scores_path)
        scores_preview = sc.head(10).to_markdown(index=False) if len(sc) > 0 else ""

    md = f"""# Data Sources Manifest

Generated: {pull_time}
Data cutoff enforced throughout: 2025-12-31

---

## 1. Inflow / Discharge — NVE HydAPI v1

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Hydrological API
**URL:** https://hydapi.nve.no/api/v1/
**API key:** obtained programmatically via POST to https://hydapi.nve.no/Users (no email confirmation; key returned on-page)
**Endpoint used:** `/api/v1/Observations`
**Query parameters:**
- StationId: `{best["station_id"]}`
- Parameter: `{best["parameter"]}` (discharge/inflow, m³/s)
- ResolutionTime: 1440 (daily)
- ReferenceTime: 1980-01-01/2025-12-31
- Pull timestamp: {pull_time}

**Selected station:** {best["stationName"]} (`{best["station_id"]}`)
**River:** {best.get("riverName", "N/A")}
**County:** {best.get("countyName", "N/A")}
**Coverage:** {daily_inflow.index.min().date()} → {daily_inflow.index.max().date()}
**Completeness (from 2000-01-01):** {best["completeness"]:.3f}

**Selection process:** All NVE stations in NO2 price area counties (Rogaland, Agder, Vestland,
Telemark) and key regulated watercourses (Sira-Kvina, Otra, Suldalslågen, Ulla-Førre, etc.)
were scored by completeness of daily discharge coverage from 2000-01-01. The station with the
highest completeness was selected. Station scoring table: `data/raw/station_scores.csv`.

Top stations considered:
{scores_preview}

**Unit conversion:** Daily mean Q [m³/s] → weekly energy-equivalent [GWh/week] via fixed-head
approximation: η=0.88, H_net=440 m (Tonstad/Sira-Kvina), k={0.88*1000*9.81*440*604800/3.6e12:.4f} GWh/(m³/s).
This is a documented modeling simplification (see methodology note).

---

## 2. Electricity Prices — Nord Pool NO2

**Source:** Nord Pool Group, day-ahead market historical archive
**URL:** https://data.nordpoolgroup.com/ (public portal, no registration required)
**API endpoint:** https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices
**Query parameters:**
- market: DayAhead
- deliveryArea: NO2
- currency: NOK
- Pull timestamp: {pull_time}

**Coverage:** {daily_price.index.min().date()} → {daily_price.index.max().date()}
**Aggregation:** Daily average of hourly prices; weekly average and std. dev.

---

## 3. Final Aligned Weekly Panel

**File:** `data/processed/weekly_panel.csv`
**Window:** {panel.index.min().date()} → {panel.index.max().date()}
**Weeks:** {len(panel)}
**Gap handling:** Linear interpolation for gaps ≤ 3 consecutive weeks;
longer gaps dropped (see `src/data_acquisition/weekly_panel.py`).
"""
    (RAW_DIR / "SOURCES.md").write_text(md, encoding="utf-8")
    LOG.info("Wrote SOURCES.md")


if __name__ == "__main__":
    run()
