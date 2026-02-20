# Stochastic Hydropower Reservoir Scheduling

Quantifies the **value of operational flexibility** in weekly hydropower reservoir release scheduling under joint inflow and electricity price uncertainty, using real data from the Arendalsvassdraget system in southern Norway.

Three release policies are compared across a 22-year backtest (2003-2024):

| Policy | Description |
|--------|-------------|
| **Open-loop (OL)** | Full 52-week release plan committed at the year start, not revised as new information arrives |
| **Closed-loop (CL)** | Rolling-horizon policy, re-optimised each quarter as uncertainty resolves |
| **Perfect-foresight (PF)** | Deterministic hindsight optimum with full knowledge of realised inflow and price |

The key metrics are:

- **Value of flexibility (VoF)** = CL revenue minus OL revenue
- **Residual regret** = PF revenue minus CL revenue

Both are computed over an out-of-sample backtest on real historical data.

## Key findings

See [docs/key_findings.md](docs/key_findings.md) for the full summary. In brief:

1. In normal market conditions (2003-2020 price distribution), VoF at Jorundland is small but positive: approximately 3.4 MNOK/yr on average.
2. Flexibility value concentrates entirely at the only plant with meaningful reservoir storage. The two run-of-river plants in the cascade contribute VoF = 0.000 in every year.
3. During the 2021-2024 energy crisis, both OL and CL policies operated from scenario trees calibrated on a fundamentally different price distribution. VoF estimates for those years should not be interpreted as stable structural findings.

## Data sources

| Variable | Source | Resolution | Coverage |
|----------|---------|------------|----------|
| Inflow proxy (m3/s) | NVE HydAPI v1, station 19.127.0 (Rygene total, Arendalsvassdraget) | Daily | 1985-2025 |
| Day-ahead price (EUR/MWh) | Energi Data Service, NO2 bidding zone | Hourly | 2000-2025 |
| EUR/NOK exchange rate | Norges Bank SDMX API | Daily | 2000-2025 |

All data is real. No synthetic or placeholder values are used anywhere. Hard data cutoff: **2025-12-31**.

## Technical methodology

Full model specification, parameter sources, verification checks, and limitations: [docs/methodology.md](docs/methodology.md)

Key points:

- Effective head: H_eff = 264.5 m (cascade-level, derived from installed capacity and design flow for each plant)
- Reservoir capacity: 167 GWh at Jorundland (Nesvatn, NVE station 19.5.0), confirmed against registered volume
- Scenario trees: 256 leaf scenarios per year, K-means reduced from 2000 MCMC draws, Bayesian seasonal AR(1) models for inflow and price
- LP solver: HiGHS via Pyomo
- All MCMC randomness is seeded for exact reproducibility

## System requirements

- Python 3.11 or newer
- HiGHS LP solver (installed via `highspy`)
- NVE HydAPI key in a `.env` file (see below)

## Installation

```
git clone <this-repo>
cd hydro-stochastic-scheduling
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Configuration

Create a `.env` file at the project root (it is git-ignored) with your NVE API key:

```
NVE_API_KEY=your_key_here
```

Register for a free key at [hydapi.nve.no](https://hydapi.nve.no). The key is loaded via `python-dotenv` and never hardcoded.

## Running the pipeline

All commands are run from the project root.

**Step 1: Data acquisition**

```
python -m src.data_acquisition.run_acquisition
```

Fetches discharge from NVE HydAPI, day-ahead prices from Energi Data Service, and EUR/NOK from Norges Bank. Aggregates to ISO weekly panels. Saves `data/processed/weekly_panel.csv` and `data/raw/SOURCES.md`. Expected runtime: 5-10 minutes (API rate limits).

**Step 2: Forecasting**

```
python -m src.forecasting.run_forecasting
```

Fits Bayesian inflow and price models for each backtest year using only prior data. Saves scenario trees and posteriors to `results/scenarios/`. Expected runtime: 2-6 hours (MCMC, 2 models times 22 years).

**Step 3: Phase A backtest (single aggregate reservoir)**

```
python -m src.backtest.run_backtest
```

Runs OL, CL, and PF policies for the single-reservoir Phase A model. Saves results to `results/tables/backtest_results.csv`.

**Step 4: Phase B backtest (three-plant cascade)**

```
python -m src.backtest.run_cascade_backtest
```

Runs OL, CL, and PF policies for the Jorundland-Evenstad-Rygene cascade model. Saves results to `results/tables/cascade_backtest_results.csv`.

**Step 5: Tests**

```
python -m pytest tests/ -v
```

Covers reservoir mass balance, non-anticipativity constraints, data cutoff enforcement, and scenario tree structure. All tests should pass.

## Running the interactive demo

```
streamlit run app/main.py
```

Opens a browser-based tool for exploring the backtest results interactively. Features include year-by-year VoF breakdowns, lag sensitivity toggling, a normal-vs-crisis filter, and a live LP recompute for the Jorundland reservoir capacity slider.

## Project structure

```
hydro-stochastic-scheduling/
├── .env                            # NVE API key, git-ignored
├── requirements.txt
├── app/
│   ├── main.py                     # Streamlit interactive demo
│   ├── data_loader.py              # Cached data loading
│   └── live_solver.py              # Live LP recompute for capacity slider
├── data/
│   ├── raw/
│   │   └── SOURCES.md              # Auto-generated data provenance log
│   └── processed/
│       └── weekly_panel.csv        # 1,131 ISO weeks, inflow + price
├── docs/
│   ├── methodology.md              # Full technical specification
│   └── key_findings.md             # Summary of main results
├── results/
│   ├── scenarios/                  # Scenario trees and model posteriors
│   ├── tables/                     # Backtest revenue tables (CSV)
│   └── figures/                    # Reservoir trajectory plots (PNG)
├── src/
│   ├── data_acquisition/
│   │   ├── nve_client.py
│   │   ├── nordpool_client.py
│   │   ├── weekly_panel.py
│   │   ├── cascade_panel.py        # Phase B inflow splitting
│   │   ├── station_selector.py
│   │   └── run_acquisition.py
│   ├── forecasting/
│   │   ├── seasonal_model.py
│   │   ├── price_model.py
│   │   ├── scenario_tree.py
│   │   └── run_forecasting.py
│   ├── optimization/
│   │   ├── reservoir_model.py      # Phase A: single-reservoir LP
│   │   └── cascade_model.py        # Phase B: cascade LP
│   └── backtest/
│       ├── runner.py
│       ├── cascade_runner.py       # Phase B runner
│       ├── run_backtest.py
│       └── run_cascade_backtest.py
└── tests/
    ├── test_reservoir_dynamics.py
    ├── test_data_cutoff.py
    ├── test_scenario_tree.py
    └── test_cascade_model.py
```

## Reproducibility

All randomness is seeded:

- Inflow model: `SEED = 42`
- Price model: `SEED = 43`
- Scenario generation: `np.random.default_rng(42)`

Results are exactly reproducible given the same data and software versions listed in `requirements.txt`.

## Licence

- NVE HydAPI data: Norwegian Water Resources and Energy Directorate (NVE). Free for non-commercial use.
- Energi Data Service: CC BY 4.0
- Norges Bank exchange rates: public domain
