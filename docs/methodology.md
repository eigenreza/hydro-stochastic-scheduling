# Stochastic Hydropower Reservoir Scheduling Under Inflow and Price Uncertainty: Methodology

## 1. Research Question and Motivation

This study quantifies the **value of operational flexibility** in weekly hydropower reservoir scheduling for a Norwegian reservoir system, under joint uncertainty in weekly inflow and day-ahead electricity price. Two scheduling policies are compared:

- **Open-loop policy**: a full 52-week release plan committed at the start of the planning year, optimised against the prior joint distribution of inflow and price, with no ability to revise as uncertainty resolves.
- **Closed-loop policy**: a rolling-horizon multistage policy that re-optimises the remaining-horizon release plan at the start of each seasonal quarter, using the updated state and a fresh forecast for the remaining horizon.

Both policies are backtested against real historical realisations, and both are compared against a **perfect-foresight benchmark** (deterministic hindsight optimisation). The resulting revenue decomposition:

> Flexibility value = Closed-loop revenue minus Open-loop revenue  
> Residual regret = Perfect-foresight revenue minus Closed-loop revenue

This decomposition answers: how much does the ability to *respond* to incoming information matter in practice for this system, and how much of the theoretically achievable revenue is still left unreachable even with a responsive policy, because the future is genuinely unknowable at decision time?

### Positioning in the literature

This work is conceptually related to the multi-horizon stochastic programming framework used in the EMPIRE capacity planning model (Tomasgard, Crespo del Granado, Backe, Skar, NTNU/SINTEF), which distinguishes between investment decisions made before uncertainty resolves and operational decisions that adapt in real time. The hydropower medium-term scheduling literature (Helseth, Fleten, Mo, Korpas; NTNU/SINTEF/IFE) typically uses **Stochastic Dual Dynamic Programming (SDDP)** as the state-of-the-art method for exact recursive optimisation. The present work implements a *simplified multistage stochastic program* (explicitly not SDDP) which shares SDDP's conceptual structure (recourse, non-anticipativity, stage decomposition) but differs from true SDDP in important ways documented in Section 5.


## 2. Data Sources

### 2.1 Price-zone consistency audit

Before describing the data used, it is necessary to document an important audit that was conducted during this study. An initial station selection (Strengen, 16.142.0, Telemark) was subsequently found to be inconsistent with the NO2 price series on two grounds:

1. **Wrong price zone.** Strengen is located in Tinn municipality, upper Telemark, at coordinates 59.99°N, 8.37°E and elevation 1,077 m. Tinn municipality is on the NO1 side of the Flesakersnittet transmission congestion boundary, which separates eastern Norway (NO1) from southern Norway (NO2). Using NO2 prices to schedule a reservoir physically connected to the NO1 grid would overstate or understate the true opportunity cost of water depending on the sign and magnitude of the NO1/NO2 price spread, a systematic error that could not be corrected ex post.

2. **Incompatible with reservoir model.** NVE API metadata confirms that station 16.142.0 measures an unregulated (catchmentRegTypeName: Uregulert) sub-catchment of only 72.57 km² on the Gøyst tributary, with zero reservoirs (numberReservoirs: 0). A reservoir scheduling model with $\bar{S} = 1{,}300$ GWh has no physical basis in a 72 km² unregulated stream.

The station was corrected to **Rygene total (19.127.0)** on the Arendalsvassdraget in Agder, as described in Sections 2.1–2.3 below. The correction is fully documented in `data/raw/SOURCES.md`.

### 2.2 Inflow data

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Hydrological API v1 (https://hydapi.nve.no/api/v1/). API key obtained programmatically via the self-service registration form.

**Station selected:** Rygene total, station ID 19.127.0, Arendalsvassdraget, Agder.

**Selection rationale:** All active daily discharge stations (parameter 1001 = Vannforing) in unambiguous NO2 counties (Agder, Rogaland) were scored by completeness of daily discharge coverage from 2000-01-01 to 2025-12-31. Five Agder stations achieved 100% completeness (9,496/9,496 days). Among these, Rygene total was selected as the station with the largest regulated catchment (3,946 km², 33 reservoirs) most consistent with the reservoir scheduling model assumptions. The other four 100%-complete stations represent smaller regulated or unregulated systems. Full scoring table: `data/raw/no2_station_scores_agder_rogaland.csv`.

Station characteristics (from NVE API):
- County: Agder (NO2 price zone)
- Municipality: Arendal
- Catchment area: 3,946 km²
- Regulation type: Regulert m/magasinregulering og overføringer (regulated with reservoir storage and diversions)
- Number of reservoirs: 33
- Mean discharge 2000–2025: 124.81 m³/s

**Coverage:** Daily mean discharge [m³/s], 2000-01-01 to 2025-12-31, with no missing values (9,496/9,496 days observed).

**Energy conversion:** Daily mean discharge Q [m³/s] is converted to weekly energy-equivalent inflow [GWh/week] using a cascade effective-head approximation:

$$E_{\text{week}} = \eta \cdot \rho \cdot g \cdot H_{\text{eff}} \cdot Q_{\text{mean}} \cdot T_{\text{week}} \;/\; 3.6 \times 10^{12}$$

where:
- $\eta = 0.88$ (overall turbine-generator efficiency, representative of modern Norwegian units)
- $\rho = 1000$ kg/m³ (water density)  
- $g = 9.81$ m/s²
- $H_{\text{eff}} = 264.5$ m (cascade-level system effective head (see derivation below))
- $T_{\text{week}} = 604{,}800$ s/week

**Derivation of $H_{\text{eff}}$:** The station Rygene total (19.127.0) measures total outflow from the entire Arendalsvassdraget cascade. The correct energy conversion must reflect the whole cascade's yield per unit of system throughput, not the head of any single plant. The system effective head is defined via the energy balance:

$$H_{\text{eff}} = \frac{E_{\text{annual}}}{\eta \cdot \rho \cdot g \cdot Q_{\text{mean}} \cdot T_{\text{year}}}$$

Using NVE's power-plant registry (`GetHydroPowerPlantsInOperation`, queried 2026-06-19): all 44 active plants on Arendalsvassdraget have a combined mean annual production (MidProd 1991–2020) of 2,498.4 GWh/year. With $Q_{\text{mean}} = 124.81$ m³/s at Rygene total:

$$H_{\text{eff}} = \frac{2498.4 \times 3.6 \times 10^{12}}{0.88 \times 1000 \times 9.81 \times 124.81 \times 31{,}557{,}600} = 264.5 \text{ m}$$

**Correction note:** A prior version of this model used $H_{\text{net}} = 450$ m, attributed to "Brokke kraftverk." This was incorrect on two counts: (1) Brokke kraftverk is located on the **Otra river** in Valle municipality, a completely different watershed from Arendalsvassdraget; (2) Brokke's registered gross head in the NVE registry is 300 m, not 450 m. Rygene kraftverk, the lowest plant in the Arendalsvassdraget on Nidelva (Arendal), has a registered gross head of 36.2 m, approximately 12× smaller than the prior value. Both errors were identified and corrected by querying the NVE power-plant registry directly.

The conversion factor is $k = 0.88 \times 1000 \times 9.81 \times 264.5 \times 604{,}800 / 3.6 \times 10^{12} \approx 0.3836$ GWh per (m³/s $\cdot$ week), yielding a mean weekly inflow of approximately 47.9 GWh/week (= 124.8 m³/s).

### 2.2 Electricity prices

**Source:** Energi Data Service (https://api.energidataservice.dk/), Denmark's public energy market data API (no registration required, CC BY 4.0). This service publishes hourly day-ahead prices for all Nordic bidding zones including NO2.

**Currency conversion:** Prices are reported in EUR/MWh. Converted to NOK/MWh using daily spot exchange rates from Norges Bank's public SDMX API (https://data.norges-bank.no/), which provides EUR/NOK rates from 2000 onward with no registration.

**Coverage:** Hourly day-ahead prices, NO2 bidding zone, January 2000 to September 2025 (the last available date at time of data pull). This is documented as a data availability limitation, not a methodological choice. Nord Pool prices for Q4 2025 were not yet publicly available via this source at the time of the study (data pull: June 2026).

**Aggregation:** Hourly prices are averaged to daily, then daily averages and standard deviations are aggregated to weekly (ISO week, Monday–Sunday). The weekly standard deviation (within-week price volatility) is retained as a diagnostic but is not used in the core optimization.

### 2.3 Aligned weekly panel

The two series are aligned on ISO week (Monday–Sunday, ISO 8601). The final panel covers **1,345 weeks** from 1999-12-27 to 2025-09-29. Inflow data has no missing values. Price data has no missing values over the overlapping window.

Units: inflow in GWh/week; prices in NOK/MWh.


## 3. Optimization Model

### 3.1 Notation

| Symbol | Description | Units |
|--------|-------------|-------|
| $t = 1, \dots, T$ | Time index (ISO weeks), $T = 52$ | n/a |
| $s_t^\omega$ | Reservoir storage at start of week $t$, scenario $\omega$ | GWh |
| $g_t^\omega$ | Generation/release decision, week $t$, scenario $\omega$ | GWh/week |
| $x_t^\omega$ | Spill (generation without revenue), week $t$, scenario $\omega$ | GWh/week |
| $i_t^\omega$ | Inflow, week $t$, scenario $\omega$ | GWh/week |
| $p_t^\omega$ | Day-ahead price, week $t$, scenario $\omega$ | NOK/MWh |
| $\bar{S}$ | Maximum reservoir storage capacity | GWh |
| $\bar{G}$ | Maximum weekly generation | GWh/week |
| $\delta$ | Weekly discount factor | n/a |

**Reservoir parameters:**
- $\bar{S} = 1{,}300$ GWh (usable storage: approximate; system-level reservoir data at watercourse resolution is not published by NVE Magasinstatistikk; only regional aggregates are available. The 1,300 GWh value is approximately 42% of the estimated Arendalsvassdraget total usable storage of ~3,100 GWh, representing a major sub-system within the cascade. This is a documented approximation.)
- $\bar{G} = 95$ GWh/week (derived from NVE power-plant registry: 44 active plants on Arendalsvassdraget with total installed capacity 565.5 MW × 168 h/week ÷ 1000 = 95.0 GWh/week. Source: `GetHydroPowerPlantsInOperation`, NVE, 2026-06-19.)
- **Discount rate:** $r_\text{annual} = 4\%$, so $\delta = (1.04)^{1/52} \approx 0.99924$ per week. This is consistent with the long-run Norwegian public infrastructure discount rate used in NVE project appraisal guidelines.

### 3.2 Dynamics and constraints

$$s_{t+1}^\omega = s_t^\omega + i_t^\omega - g_t^\omega - x_t^\omega$$

$$0 \le s_t^\omega \le \bar{S}, \quad 0 \le g_t^\omega \le \bar{G}, \quad x_t^\omega \ge 0$$

$$s_1^\omega = s_{\text{init}} \quad \text{(same for all } \omega\text{)}$$

The initial storage $s_{\text{init}}$ for each backtested year is set using the real NO2 elspot-zone reservoir filling percentage at ISO week 1 of that year, sourced from NVE's Magasinstatistikk API (https://biapi.nve.no/magasinstatistikk/). The Magasinstatistikk data tracks weekly filling across approximately 490 Norwegian reservoirs, aggregated by bidding zone and vassdrag zone. The NO2 zone, which encompasses Agder (where Arendalsvassdraget is located), is the finest resolution available: no station- or watercourse-level filling data is published in this dataset. The conversion used is:

$$s_{\text{init}} = f_{\text{NO2, week 1}} \times \bar{S}$$

where $f_{\text{NO2, week 1}}$ is the dimensionless filling fraction (0–1) for ISO week 1 of the backtest year. The Magasinstatistikk series covers 1995–present with no gaps; all 22 backtest years (2003–2024) have a week-1 observation. The filling fraction at week 1 ranges from 40.0% (2003, 2011) to 86.0% (2016) across the study period, compared to the fixed 65% that would apply if this data were unavailable.

### 3.3 Terminal value function

A piecewise-linear concave terminal value function prevents the model from trivially draining the reservoir at the horizon end, a well-known necessity in medium-term hydropower scheduling (cf. EMPS water value curves; Helseth and Mo, 2016):

$$V_T(s) = \begin{cases} \bar{v} \cdot s & \text{if } s \le \bar{S}/2 \\ \bar{v} \cdot \bar{S}/2 + \tfrac{1}{2}\bar{v} \cdot (s - \bar{S}/2) & \text{if } s > \bar{S}/2 \end{cases}$$

where $\bar{v} = 400{,}000$ NOK/GWh (the marginal water value, estimated from the long-run mean price of ~400 NOK/MWh converted to GWh units: $400 \times 1{,}000 = 400{,}000$). The concavity (half slope above the midpoint) reflects diminishing marginal value at high storage: when the reservoir is full, additional water has lower marginal value because the generation opportunity cost of holding water declines.

This is a deliberately simple approximation. A full SDDP implementation would iteratively refine the terminal value function via Benders cuts across backward passes; the present approximation is documented as a limitation.

### 3.4 Objective function

$$\max_{g_t^\omega, x_t^\omega} \; \mathbb{E}_\omega \left[ \sum_{t=1}^{T} \delta^t \cdot p_t^\omega \cdot g_t^\omega \cdot 10^3 + V_T(s_{T+1}^\omega) \right]$$

The factor $10^3$ converts GWh $\times$ NOK/MWh to NOK (since 1 GWh = $10^3$ MWh).

### 3.5 Non-anticipativity structure

**Open-loop policy:** All scenarios share one single release plan $(g_t, x_t)_{t=1}^{52}$. This is imposed by requiring $g_t^\omega = g_t^{\omega'}$ and $x_t^\omega = x_t^{\omega'}$ for all $\omega, \omega' \in \Omega$ and all $t$. Effectively, the operator commits to a single full-season plan under the prior.

When this pre-committed plan is applied to the *realised* inflow path in the backtest, the plan may be physically infeasible in some weeks. For example,, the committed generation may be less than the minimum physically necessary to prevent reservoir overflow (if realised inflow exceeds the plan's headroom), or greater than what is achievable given actual storage (if the reservoir is nearly empty). In such weeks, generation is **clamped** to the nearest feasible value:

- `ol_clamp_min` counts weeks where the plan was *below* the minimum necessary generation (g_t < i_t + s_t - S_bar, meaning the reservoir would overflow at the planned release): generation is forced up to prevent infeasibility.
- `ol_clamp_max` counts weeks where the plan was *above* the maximum feasible generation ($g_t > s_t + i_t$, i.e. the reservoir would go negative): generation is clamped down.

This clamping is methodologically sound: it is the correct way to apply a pre-committed plan to a path that differs from what the plan anticipated. The clamping **penalises** the open-loop policy, reflecting the cost of inflexibility, not a modelling error. Years with high `ol_clamp_min` counts (e.g. 2014: 39/52 weeks clamped) indicate years where realised inflow substantially exceeded the open-loop plan's assumed trajectory, forcing the system into frequent overflow-avoidance generation. These are precisely the years in which the closed-loop policy's ability to observe and respond to realised inflows provides the most value.

**Closed-loop policy:** Decisions within each seasonal stage are shared across scenarios, but may differ between stages. The four stages (weeks 1–13, 14–26, 27–39, 40–52) correspond to winter, spring, summer, and autumn. At the start of each stage (except the first), the state is updated to the realised storage and the remaining horizon is re-optimised. Non-anticipativity within a stage is: $g_t^\omega = g_t^{\omega'}$ for all $\omega, \omega'$ and all $t$ in the current stage; decisions in later stages are free to differ across scenarios.


## 4. Forecasting Methodology

### 4.1 Inflow model

A Bayesian seasonal AR(1) model is fit to log-transformed weekly inflow. The conditional distribution of observation $y_t = \log(\text{inflow}_t + 0.1)$ given the previous observation is:

$$y_t \mid y_{t-1} \;\sim\; \mathcal{N}\!\bigl(\mu_t + \phi(y_{t-1} - \mu_{t-1}),\; \sigma^2\bigr), \quad t \ge 2$$

where the seasonal mean is:

$$\mu_t = \alpha_0 + \sum_{k=1}^{3} \bigl[a_k \cos(2\pi k t / 52) + b_k \sin(2\pi k t / 52)\bigr]$$

The first observation uses the stationary distribution $y_1 \sim \mathcal{N}(\mu_1, \sigma^2 / (1 - \phi^2))$.

**Priors:**
- $\alpha_0 \sim \mathcal{N}(\bar{y}, 1)$ (centred on sample log-mean)
- $a_k, b_k \sim \mathcal{N}(0, 1)$
- $\phi \sim \mathcal{N}(0, 0.5)$ truncated to $(-0.99, 0.99)$ (stationarity)
- $\sigma \sim \text{HalfNormal}(0.3)$

**Inference:** NUTS sampler (PyMC 6), 2 chains × 1000 draws (+ 1000 tuning), `target_accept = 0.90`. Models are fit separately for each backtested year using only pre-year data.

### 4.2 Price model

An analogous Bayesian seasonal AR(1) model is fit to log-transformed weekly price, with an additional inflow anomaly covariate:

$$\log(p_t) \mid \log(p_{t-1}) \;\sim\; \mathcal{N}\!\bigl(\nu_t + \rho(\log(p_{t-1}) - \nu_{t-1}),\; \tau^2\bigr)$$

$$\nu_t = \beta_0 + \text{Fourier}(t) + \gamma \cdot q_t$$

where $q_t = (\text{inflow}_t - \hat{\mu}_{\text{inflow},t}) / \hat{\sigma}_\text{inflow}$ is the de-seasonalised inflow anomaly, computed using the posterior mean seasonal inflow component. The prior on $\gamma$ is $\mathcal{N}(-0.3, 0.3)$, reflecting the well-documented negative inflow–price correlation in the Norwegian hydro system: surplus inflow increases generation availability and depresses prices.

### 4.3 Inflow–price dependence

The inflow–price dependence is modelled structurally through the $\gamma$ covariate in the price model, not through a copula. This approach ensures that price scenario paths are coherent with the corresponding inflow scenario path: both share the same posterior parameter draw index, so the inflow–price dependence is preserved through the joint parameter uncertainty.

### 4.4 Scenario generation

From the fitted posterior predictive distributions, $N = 200$ Monte Carlo paths of length 52 weeks are drawn for each of inflow and price. These are reduced to a discrete scenario tree using **k-means clustering** (Heitsch & Römisch, 2003) with:

- 4 seasonal stages, each spanning 13 weeks
- Branching factor 4 per stage
- 4^4 = 256 leaf scenarios

Cluster representatives are the cluster centroids; probability weights are proportional to cluster sizes (count of original paths assigned to each cluster). Scenario trees are generated and saved separately for each backtested year, using strictly pre-year data.


## 5. Closed-loop Implementation vs. True SDDP

The closed-loop policy in this study is a **rolling-horizon stochastic program with recourse**. At the start of each seasonal quarter, the remaining-horizon stochastic LP is re-solved using the current realised storage as initial condition and a fresh scenario tree drawn from the posterior predictive for the remaining horizon. The first-stage decisions of each re-solve are committed to the realised path.

This differs from true SDDP in the following ways:

| Aspect | This implementation | True SDDP |
|--------|---------------------|-----------|
| Value function | Fixed piecewise-linear approximation | Iteratively refined via Benders cuts |
| Stage resolution | 4 quarterly stages (coarse) | Typically weekly (52 stages) |
| Convergence | No iteration; single-pass | Multi-pass until lower/upper bound gap closed |
| Computational cost | 4 LP solves per year | Hundreds to thousands of SDDP iterations |

The rolling-horizon recourse captures the key conceptual feature of a closed-loop policy (the ability to observe and react to incoming information) in a computationally tractable way. However, by using a coarse stage structure and a pre-specified terminal value approximation, it will underestimate the true value of flexibility that a full SDDP implementation would capture. All quantified flexibility values reported in Section 6 should therefore be interpreted as **lower bounds** on the true value of recourse for this system.


## 6. Backtest Results

The backtesting covers **22 years, 2003–2024** (the first four years of the panel, 2000–2002, are used for model burn-in). For each year:
1. Models are fit on all data strictly prior to that year (no leakage).
2. The open-loop and closed-loop policies are solved using the pre-year scenario tree.
3. Revenues are computed against the actual realised inflow and price path.
4. The perfect-foresight benchmark is computed with full knowledge of the actual path.

### Summary statistics (22-year backtest, 2003–2024)

| Policy | Mean annual revenue |
|--------|-------------------|
| Open-loop | 1,095.5 MNOK |
| Closed-loop | 1,136.9 MNOK |
| Perfect foresight | 1,404.9 MNOK |

- **Value of flexibility (CL − OL):** mean 41.4 MNOK/year (std 165.8 MNOK)
- **Residual regret (PF − CL):** mean 268.0 MNOK/year
- **Wilcoxon signed-rank test** (one-sided, $H_1$: CL > OL): $p = 0.088$

The non-parametric Wilcoxon test is used because the VoF distribution is non-Gaussian and the sample is small (22 years). The $p$-value of 0.088 does not meet the conventional 0.05 threshold for statistical significance. This should not be read as evidence that there is no flexibility value; rather, the signal is dominated by very high cross-year revenue variance (driven in part by the 2022 tail event described below), which substantially reduces statistical power with 22 observations.

**Correction history:** The revenue magnitudes were affected by a cascade of parameter corrections made during this study:
1. Fixed s_init (65% for all years) → real NO2 Magasinstatistikk week-1 filling: VoF increased from 33.4 to 43.9 MNOK, p improved from 0.105 to 0.088.
2. Wrong H_NET = 450 m (Brokke/Otra, wrong watershed and wrong value) → correct H_eff = 264.5 m (Arendalsvassdraget cascade effective head from NVE registry): revenues scaled by ~0.66×. VoF absolute magnitude changed from 43.9 to 41.4 MNOK; Wilcoxon p unchanged at 0.088. The VoF as a fraction of revenues is slightly larger under the corrected head (3.6% vs 2.6%), but the statistical conclusion is identical.

All numerical results in this section use the fully corrected parameters: H_eff = 264.5 m, G_MAX = 95 GWh/week, s_init from Magasinstatistikk.

### 2022 as a tail-event outlier: out-of-distribution price shock

The year 2022 stands out as a clear outlier in the 22-year backtest and warrants explicit discussion, as it has direct implications for how the value-of-flexibility results should be interpreted.

**Observed vs. forecast price distributions in 2022:**

The scenario tree generated for the 2022 backtest year was fit exclusively on pre-2022 data. The resulting forecast distribution for NO2 weekly prices spanned:
- Scenario range: 52.8–3,401.6 NOK/MWh (across all 192 leaf scenarios)
- Scenario mean: 489 NOK/MWh

The actual realised NO2 prices in 2022 were:
- Realised range: 286–5,483.9 NOK/MWh
- Realised mean: 2,128.9 NOK/MWh, approximately **4.4× the forecast mean**

The realised peak (5,483.9 NOK/MWh) was **61% above the maximum price in any generated scenario**. This means the 2022 realised price path fell entirely outside the support of the model's prior distribution. No amount of optimisation against the scenario tree could have anticipated this regime.

**Consequence for the flexibility decomposition:**

The 2022 perfect-foresight revenue (8,566 MNOK) was approximately 4–5× the 22-year typical year (mean ~1,700 MNOK). The open-loop policy, optimised against a scenario distribution with mean price ~489 NOK/MWh, committed a release plan calibrated to that regime. When applied to the realised 2022 path (mean price ~2,129 NOK/MWh), neither the open-loop nor the closed-loop policy could meaningfully adapt: the Bayesian price model's posterior predictive, updated at each closed-loop re-solve, continued to produce moderate-price scenarios consistent with pre-2022 history. Both policies substantially underperformed the perfect-foresight benchmark.

The value of flexibility in 2022 was **−457.5 MNOK**: the single largest negative VoF observation in the 22-year sample. The negative sign arises because the closed-loop policy, re-solving at each quarter with updated (but still historically-grounded) forecasts, chose a different release timing than the open-loop policy, and that timing happened to be *worse* against the extreme realised path.

**Interpretation:**

This is not a modelling error. It is a genuine and scientifically important finding: both the open-loop and closed-loop stochastic policies are constructed from a model of the prior price distribution, and when realised prices exit the support of that distribution entirely, neither policy can benefit from flexibility in the way the framework assumes. The usual logic of observing and responding to new information improving outcomes breaks down when the new information arrives at scales not representable in the prior model. In statistical terms, 2022 is an out-of-distribution event for which the model has no predictive coverage.

This finding has practical implications: value-of-flexibility estimates from stochastic scheduling models trained on historical data understate the risk of extreme regime-shift events (such as the 2021-22 European energy crisis), and the model's estimated VoF distribution is not a reliable guide to flexibility value under such regimes. The finding is consistent with the broader literature on model-based planning under deep uncertainty (Lempert et al., 2006): stochastic optimisation is effective for risks the model can anticipate, but does not substitute for robustness under structural breaks.

For robustness of the reported results, the 22-year mean VoF (33.4 MNOK) should be read in light of this: the distribution has very high variance ($\sigma = 161.3$ MNOK), driven substantially by 2022, and the mean is sensitive to that single observation.

Statistical significance of the closed-loop vs. open-loop revenue difference is assessed via the **Wilcoxon signed-rank test** (one-sided, $H_1$: CL revenue > OL revenue). The non-parametric test is appropriate given the small sample and the non-Gaussian revenue difference distribution. Statistical power is limited with 22 observations, and failure to reject the null does not constitute evidence of no flexibility value.

### Limitations

1. **Single effective-head energy conversion**: Using a single cascade effective head (H_eff = 264.5 m) for all weeks and all flow conditions is a simplification. The true marginal head depends on which plants are operating at any given time, reservoir levels, and seasonal dispatch patterns. A 10% head error translates directly to a 10% error in all energy and revenue figures.
2. **Regulated outflow as inflow proxy**: Rygene total measures regulated outflow (post-turbining discharge) rather than natural inflow to the reservoir system. In a well-operated reservoir the two are correlated on seasonal time scales, but operational decisions introduce noise at short horizons. This is a standard data availability limitation in Norwegian hydropower scheduling studies that lack access to internal reservoir inflow records.
3. **Initial storage from NO2 zone aggregate**: Per-year $s_\text{init}$ is set from the real NVE Magasinstatistikk NO2 zone filling at ISO week 1, which is the finest resolution publicly available. Individual watercourse-level (Arendalsvassdraget-specific) filling data is not published in this dataset; the NO2 aggregate includes reservoirs across Agder, Rogaland, and southern Vestland, not just the Arendalsvassdraget. The aggregate is a reasonable proxy for start-of-year state, but it introduces some noise for individual system years where Arendalsvassdraget's filling deviated from the NO2 aggregate.
4. **Open-loop clamping in high-inflow years**: As described in Section 3.5, the pre-committed open-loop release plan must be clamped to physically feasible values when applied to the realised inflow path. In 2014, 39 of 52 weeks required clamping (ol_clamp_min=39), indicating that the realised inflows were systematically higher than the open-loop plan assumed. This penalises the open-loop policy and is the correct methodological treatment of plan infeasibility, but it also means that 2014 open-loop revenue is computed from a substantially modified plan. Years with high clamp counts should be interpreted accordingly.
5. **Terminal value function**: The piecewise-linear approximation may under- or over-value end-of-year storage, affecting the seasonal shape of the optimal release plan.
6. **Coarse stage structure**: 4 quarterly stages vs. 52 weekly stages understates the value of flexibility; results are lower bounds on the true recourse value.
7. **Price data ends September 2025**: The EDS source did not have Q4 2025 data at time of pull. This precludes 2025 as a complete backtest year.
8. **Model robustness to regime shifts**: The 2022 energy crisis demonstrates that historical-distribution-based stochastic models cannot anticipate structural price regime breaks. VoF estimates should not be extrapolated to periods of market stress not represented in the training data.


## 7. What This Study Does NOT Show

For transparency about scope, the following are explicitly excluded from **Phase A**:

- **True SDDP**: The closed-loop implementation is a single-pass rolling-horizon LP, not iterative Benders decomposition. Claims about optimality should not be extrapolated to full SDDP performance.
- **Multi-reservoir cascade effects (Phase A only)**: The Phase A model treats Arendalsvassdraget as a single equivalent storage with scalar inflow. Phase B (Section 8) addresses this directly with a real three-plant cascade model; readers comparing Section 6 and Section 8 results should note the substantial difference in VoF magnitude that results.
- **Reserve and balancing markets**: Only day-ahead Elspot revenues are modelled. Intraday, balancing, and frequency regulation products are excluded.
- **Transmission constraints**: The NO2 price is taken as exogenous. In reality, large generation decisions affect local prices (price-taking assumption may fail for major reservoirs).
- **Operational constraints**: Ramping, minimum ecological flow, and maintenance downtime are excluded.
- **Full-cycle investment optimisation**: This is an operational scheduling study, not a capacity planning study.

Despite these limitations, the study demonstrates the methodological value decomposition approach with real data and a transparent, reproducible implementation.


## 8. Phase B: Three-Plant Cascade Extension

### 8.1 Motivation and scope

Phase A (Sections 2–6) models the entire Arendalsvassdraget system as a single equivalent reservoir, a deliberate simplification that collapses 44 plants into one scalar storage (S_MAX = 1,300 GWh) and one scalar inflow. Phase B replaces this with a genuine three-plant cascade, using real per-plant physical parameters and a cascade-aware LP formulation. The cascade selected is the **Jørundland → Evenstad → Rygene** chain on Nidelva (lower Arendalsvassdraget), which represents the physically dominant flow path from the main headwater storage to the river outlet.

The primary scientific questions addressed in Phase B are:

1. When the model correctly captures per-plant storage constraints and cascade routing, how does the value of operational flexibility (VoF) change relative to the system-level approximation?
2. Where does VoF concentrate in the cascade, and why?
3. How much of the Phase A VoF estimate is an artefact of the single-reservoir aggregation?

### 8.2 Plant identification and NVE sourcing

The three plants were identified from the NVE Vannkraftdatabase (`GetHydroPowerPlantsInOperation`, queried 2026-06-19), filtering by `Nedborsfeltnavn = "Arendalsvassdraget"` and selecting the Nidelva sub-chain. Physical parameters:

| Plant | MW | Gross head (m) | MidProd 91–20 (GWh/yr) | G_MAX (GWh/wk) |
|-------|----|---------------|------------------------|----------------|
| Jørundland | 55.2 | 278.6 | 185.8 | 9.274 |
| Evenstad | 23.5 | 17.3 | 121.4 | 3.948 |
| Rygene | 55.0 | 36.2 | n/a | 9.240 |

G_MAX for each plant = MW × 168 h/week ÷ 1,000.

The three plants together represent 133.7 MW installed capacity (23.6% of the Arendalsvassdraget system's 565.5 MW total) and approximately 307 GWh/yr mean production (12.3% of the system's 2,498 GWh/yr), reflecting the fact that most production in the cascade comes from high-head storage plants further upstream.

**Physical structure:** Jørundland is a high-head storage plant (H = 278.6 m) at the top of the Nidelva flow path. Its reservoir (Nesvatn) provides the only meaningful intertemporal flexibility in this three-plant chain. Evenstad and Rygene are low-head run-of-river plants (H = 17.3 m and 36.2 m, respectively) with no significant reservoir storage.

### 8.3 Reservoir capacity: Nesvatn (Jørundland headwater)

The Jørundland reservoir (Nesvatn, on the Gjøv tributary, 221 km² catchment) was identified through Jørundland's concession records in the NVE registry (`Regulering av Nesvatn`). Reservoir working storage was obtained from NVE HydAPI, station **19.5.0** (Nesvatn), parameter **1004** (reservoir volume, million m³).

The daily volume series spans 1990-01-02 to 2025-12-30. The working storage (max − min observed over the full series) is:

$$S_{\text{working}} = 265.83 - 2.87 = 262.96 \text{ million m}^3$$

Converting using the NVE-registered energy equivalence factor (EnEkv = 0.638 kWh/m³):

$$S_{\max,J} = 262.96 \times 10^6 \text{ m}^3 \times 0.638 \text{ kWh/m}^3 \div 10^9 = 167.77 \text{ GWh}$$

A conservative value of **167 GWh** is used in the model (the lowest 0.87% tail of the volume distribution is excluded to avoid artifact extremes from measurement outages). Evenstad and Rygene have no significant storage: S_MAX_E = S_MAX_R = 0 GWh.

This contrasts sharply with Phase A's aggregate S_MAX = 1,300 GWh, which represented a rough system-level working storage across all 44 plants. Jørundland's actual working storage (167 GWh) is only 12.8% of that aggregate, consistent with Jørundland being one plant among many, and with most system storage residing in high-head plants further upstream (e.g., Bykil, Nelaug, Rore/Nidelv).

### 8.4 Hydraulic routing and same-week lag assumption

The three-plant cascade is connected by:
- **Jørundland → Evenstad:** Nidelva, approximately 17 km. At mean flow velocity ~1.2 m/s in channelled river sections, travel time ≈ 4 hours, well under one week.
- **Evenstad → Rygene:** Nidelva, approximately 26 km. At similar velocity, travel time ≈ 6–8 hours.

Both inter-plant distances are sufficiently short that water released by plant $n$ in week $t$ reaches plant $n+1$ within the same weekly time step. The model therefore uses a **same-week routing assumption** (zero lag): all water released and/or spilled by Jørundland in week $t$ becomes available at Evenstad in week $t$; all Evenstad output becomes available at Rygene in week $t$.

### 8.5 Cascade LP formulation

The Phase B formulation extends the single-reservoir LP (Section 3) to a three-node cascade. Notation: subscripts J, E, R denote Jørundland, Evenstad, Rygene; $i_J^{t,\omega}$, $i_E^{t,\omega}$, $i_R^{t,\omega}$ are per-plant local inflows; $g_p^{t,\omega}$ and $x_p^{t,\omega}$ are generation and spill for plant $p$.

**Jørundland (storage plant):**
$$s_J^{t+1,\omega} = s_J^{t,\omega} + i_J^{t,\omega} - g_J^{t,\omega} - x_J^{t,\omega}$$
$$0 \le s_J^{t,\omega} \le S_{\max,J}, \quad 0 \le g_J^{t,\omega} \le G_{\max,J}, \quad x_J^{t,\omega} \ge 0$$

**Evenstad (run-of-river, zero storage):**
$$g_E^{t,\omega} + x_E^{t,\omega} = i_E^{t,\omega} + g_J^{t,\omega} + x_J^{t,\omega}$$
$$0 \le g_E^{t,\omega} \le G_{\max,E}, \quad x_E^{t,\omega} \ge 0$$

**Rygene (run-of-river, zero storage):**
$$g_R^{t,\omega} + x_R^{t,\omega} = i_R^{t,\omega} + g_E^{t,\omega} + x_E^{t,\omega}$$
$$0 \le g_R^{t,\omega} \le G_{\max,R}, \quad x_R^{t,\omega} \ge 0$$

**Objective:**
$$\max \;\mathbb{E}_\omega\!\left[\sum_{t=1}^{T} \delta^t \cdot 10^3 \cdot p_t^\omega \cdot (g_J^{t,\omega} + g_E^{t,\omega} + g_R^{t,\omega}) + V_T(s_J^{T+1,\omega})\right]$$

The terminal value function $V_T$ is applied to Jørundland's end-of-horizon storage only (Evenstad and Rygene have no storage to value). It uses the same piecewise-linear form as Phase A, with $S_{\max,J} = 167$ GWh and $\bar{v} = 400{,}000$ NOK/GWh.

**Non-anticipativity:** Only Jørundland's generation decisions $g_J^{t,\omega}$ are subject to non-anticipativity constraints. Evenstad and Rygene have no storage and must always generate (or spill) whatever water flows through them; their decisions are always scenario-specific (fully adaptive) and are never pre-committed.

### 8.6 Scenario generation for the cascade

The single measured discharge series (Rygene total, 19.127.0) is the only available hydrological input. No sub-catchment discharge series exist in NVE HydAPI for any plant-level sub-catchment of Arendalsvassdraget. The scenario tree for each year is therefore generated using the single-station inflow model exactly as in Phase A.

Per-plant local inflows are obtained by partitioning the total system inflow using **fixed energy-balance fractions**:

$$f_J = 0.0706, \quad f_E = 0.6723, \quad f_R = 0.2571$$

Derivation (see `src/data_acquisition/cascade_panel.py` for full computation):
- Mean turbined flow from NVE MidProd + head: $Q_J = 8.81$ m³/s, $Q_E = 92.73$ m³/s (Evenstad turbines Q from its full intake, including upstream J)
- Local inflow to Evenstad reach: $i_E = Q_E - Q_J = 83.92$ m³/s (water entering the main Nidelva reach between J and E, not via Jørundland)
- Local inflow between Evenstad and Rygene: $i_R = Q_{\text{Rygene total}} - Q_E = 124.81 - 92.73 = 32.08$ m³/s
- Fractions of $Q_{\text{Rygene total}} = 124.81$ m³/s: $f_J = 8.81/124.81$, $f_E = 83.92/124.81$, $f_R = 32.08/124.81$

This approach conserves the total measured Rygene discharge in every week and in every scenario. It implies **perfect cross-plant inflow correlation** (r = 1.0 by construction), which is a simplification: in reality, individual sub-catchment inflows are correlated but not identical. The simplification is documented here as a known limitation and is the direct consequence of having only one discharge measurement in the watershed.

### 8.7 Backtest results: per-plant VoF decomposition

The cascade backtest covers the same 22-year period (2003–2024) as Phase A, using the same scenario trees and Magasinstatistikk-derived initial storage.

**Mean annual revenues and flexibility values:**

| Policy | Mean annual revenue | Change vs OL |
|--------|--------------------|-|
| Open-loop | 374.2 MNOK | n/a |
| Closed-loop | 375.2 MNOK | +1.0 MNOK |
| Perfect foresight | 409.2 MNOK | n/a |

- **Value of flexibility (CL − OL):** mean **0.99 MNOK/year** (std 21.3 MNOK, median 2.6 MNOK)
- **Residual regret (PF − CL):** mean 34.0 MNOK/year
- **Wilcoxon signed-rank test** (one-sided, $H_1$: CL > OL): $p = 0.046$
- **Fraction of years with CL > OL:** 63.6% (14 of 22)

**Per-plant VoF decomposition:**

| Plant | Mean VoF (MNOK/yr) | VoF share |
|-------|--------------------|-----------|
| Jørundland | 0.99 | 100.0% |
| Evenstad | 0.00 | 0.0% |
| Rygene | 0.00 | 0.0% |

The result is exact to reported precision: VoF_E and VoF_R are identically zero in every backtested year. This is physically necessary, not a numerical coincidence. Evenstad and Rygene are run-of-river plants with zero storage. In every week they must generate (or spill) whatever water arrives from upstream. Their generation is fully determined by the realised inflow and Jørundland's committed release; it does not depend on whether the scheduling policy is open-loop or closed-loop. Consequently, Evenstad and Rygene revenue is identical under OL and CL, and their contribution to VoF is zero by construction.

All VoF in the cascade therefore concentrates at Jørundland, the sole storage plant. This is consistent with the general principle in hydropower scheduling: **intertemporal flexibility value resides entirely in storage capacity**.

### 8.8 Comparison to Phase A (single-reservoir model)

| Metric | Phase A (system model) | Phase B (3-plant cascade) | Ratio B/A |
|--------|----------------------|--------------------------|-----------|
| Plants modelled | 44 (aggregate) | 3 (Jørundland, Evenstad, Rygene) | n/a |
| Installed capacity | 565.5 MW | 133.7 MW (23.6%) | 0.236 |
| S_MAX | 1,300 GWh | 167 GWh (Jørundland only) | 0.128 |
| Mean OL revenue | 1,095.5 MNOK/yr | 374.2 MNOK/yr | 0.34 |
| Mean VoF | 41.4 MNOK/yr | 0.99 MNOK/yr | 0.024 |
| VoF / OL revenue | 3.78% | 0.26% | n/a |
| Wilcoxon p | 0.088 | 0.046 | n/a |

The cascade VoF (0.99 MNOK/yr) is **42× smaller** than the Phase A aggregate VoF (41.4 MNOK/yr), despite the three-plant cascade representing 23.6% of system installed capacity. The revenue ratio (374/1095 = 0.34) is close to the capacity share, but the VoF ratio (0.024) is far smaller. This divergence identifies the primary source of the Phase A VoF estimate:

**The Phase A system-level aggregation substantially overstates VoF by conflating all 44 plants' storage (totalling ~1,300 GWh in the aggregate model) into a single pool.** In reality, each plant's reservoir is independent. The optimizer in Phase A can arbitrarily time releases from the aggregate pool, equivalent to assuming all 44 reservoirs are connected in a single store, which is operationally unrealistic. The Phase B cascade model enforces that Jørundland's 167 GWh storage is the *only* intertemporal buffer available on this flow path, yielding a far smaller VoF.

The direction of this bias is systematic: single-reservoir aggregation always overstates VoF relative to a correctly disaggregated cascade model, because it overestimates the total flexibility available to the optimizer. The 42× overstatement observed here is large, but the comparison is not perfectly controlled: Phase B also uses a much smaller total system (23.6% of capacity), and the remaining 76.4% of system capacity (high-head storage plants upstream) may well generate significant VoF of their own when modelled correctly.

The Phase B Wilcoxon p-value (0.046) is lower than Phase A (0.088), indicating stronger statistical evidence for CL > OL despite the far smaller absolute VoF. This reflects the smaller cross-year VoF variance in Phase B (std = 21.3 MNOK vs 165.8 MNOK in Phase A): without the 2022 tail-event outlier dominating the distribution, the signed-rank test has more power on the remaining 21 years.

### 8.9 Verification of key Phase B assumptions

The following three checks were conducted after the initial Phase B implementation and before treating the backtest results as final. All findings confirmed the results are sound, with one capacity figure needing clarification.

#### 8.9.1 Jørundland reservoir capacity: 167 GWh

**Source chain (fully documented):**

The 167 GWh value was derived from the NVE Hydrological API (HydAPI v1), station **19.5.0**, parameter **1004** (Magasinvolum, daily reservoir volume in million m³). The station was identified as Jørundland's headwater reservoir through:
- `stationName: "Nesvatn"`, `lakeName: "Nesvatn"`, `reservoirName: "NESVATN"` (the reservoir is explicitly named)
- `owner: "Å ENERGI VANNKRAFT AS"` (the same operator as Jørundland kraftverk) as Jørundland kraftverk
- `hierarchy: "Gjøv/Arendalsvassdraget"` (Nesvatn sits on the Gjøv tributary, RegineNr 019.CD, immediately upstream of Jørundland's RegineNr 019.CB40
- The energy equivalence factor EnEkv = 0.638 kWh/m³ was taken directly from the NVE Vannkraftdatabase plant record for Jørundland kraftverk (verified by API query)

**Discrepancy between observed and registered capacity:**

| Source | Value | Derivation |
|--------|-------|-----------|
| HydAPI station 19.5.0: observed working range | 262.96 Mm³ | max (265.83, Jul 1990) − min (2.87, Feb 1998) |
| HydAPI station metadata: `volumeReservoirs` | 256.7 Mm³ | Registered concession capacity |
| Model value used (S_MAX_J) | **167 GWh** | 262.96 Mm³ × 0.638 / 1000 = 167.8 GWh |
| From registered capacity | 163.8 GWh | 256.7 Mm³ × 0.638 / 1000 |

The registered `volumeReservoirs = 256.7 Mm³` is the authoritative concession-defined capacity (HRV-LRV range). The observed range (262.96 Mm³) slightly exceeds this because: (1) the minimum observation (2.87 Mm³, Feb 1998) coincides with the extreme Norwegian drought of 1997-98, when the reservoir was likely drawn below LRV in an emergency; (2) the maximum observation (265.83 Mm³, Jul 1990) may represent a mild flood above HRV. Using the observed range therefore gives a **slight overestimate of S_MAX_J: 167 GWh vs the concession-implied 164 GWh** (2.4% difference). This difference is negligible for the backtest results (VoF is insensitive to a 3 GWh change in the 167 GWh reservoir. VoF_J = 0.99 MNOK/yr is driven by the *relative* flexibility offered by this storage, not its precise absolute size).

**No independent source contradiction:** the registered capacity (256.7 Mm³) and the observed range (262.96 Mm³) are consistent: both place the reservoir capacity in the range 160–170 GWh.

#### 8.9.2 Evenstad and Rygene: genuinely run-of-river (flow constraint proof)

**Evidence from NVE sources:**
- The NVE Vannkraftdatabase does not provide individual reservoir capacity fields (`MagasinKap`, etc.) for any of the three plants.
- NVE HydAPI was queried for all parameter 1004 (reservoir volume) stations on RegineNo 019.A* (the lower Nidelva reach between Evenstad and Rygene). No stations were found, confirming there are no NVE-tracked reservoirs on the lower Nidelva at these plants.

**The critical verification: flow constraint proof:**

The VoF_E = VoF_R = 0 result is not a data limitation artefact. It is a **mathematical identity** given the cascade flow magnitudes, verified over all 1,345 weeks in the dataset (2000–2025):

| Condition | Check | Result |
|-----------|-------|--------|
| $i_E > G_{\max,E}$ in all weeks | Weeks where Evenstad LOCAL inflow alone exceeds turbine capacity | **100% (1,345/1,345 weeks)** |
| $\min(i_E) > G_{\max,E}$ | Min weekly i_E = 6.461 GWh/wk vs G_MAX_E = 3.948 GWh/wk | **Confirmed** |
| total_to_R > G_MAX_R in all weeks | Weeks where (i_R + i_E + i_J) ≥ total inflow > G_MAX_R | **100% (1,345/1,345 weeks)** |
| $\min(\text{total inflow}) > G_{\max,R}$ | Min weekly system inflow = 9.611 GWh/wk vs G_MAX_R = 9.240 GWh/wk | **Confirmed** |

Since Evenstad's LOCAL inflow alone (without any contribution from Jørundland) already exceeds Evenstad's turbine capacity in every single week of the 26-year panel:
$$i_E(t) = f_E \cdot Q_{\text{total}}(t) \ge 6.46 \text{ GWh/wk} > G_{\max,E} = 3.95 \text{ GWh/wk} \quad \forall t$$

Evenstad always generates at full capacity ($g_E = G_{\max,E}$) regardless of what Jørundland commits. VoF_E = 0 is exact.

Since the entire system's weekly throughput, even at the minimum observed flow, exceeds Rygene's turbine capacity:
$$Q_{\text{total}}(t) \ge 9.61 \text{ GWh/wk} > G_{\max,R} = 9.24 \text{ GWh/wk} \quad \forall t$$

and this total flow passes through Rygene (in the cascade: $\text{total\_to\_R}(t) \ge Q_{\text{total}}(t) > G_{\max,R}$), Rygene always generates at full capacity. VoF_R = 0 is exact.

**This result holds irrespective of whether Evenstad and Rygene have any reservoir storage**, because even with 10–20 GWh of additional storage, their local inflows still saturate their turbine capacities in every week. The absence of reservoir volume stations in NVE HydAPI for these plants is supporting evidence of their run-of-river character, but the VoF result does not depend on this fact.

#### 8.9.3 Routing lag sensitivity

**Assumption used:** same-week (zero lag) routing. Water released by plant $n$ in week $t$ is available at plant $n+1$ in the same week $t$.

**Alternative tested:** one-week lag. Water released by plant $n$ in week $t$ reaches plant $n+1$ in week $t+1$. The lag=1 LP was re-formulated with updated E and R balance constraints:

$$g_E^{t,\omega} + x_E^{t,\omega} = i_E^{t,\omega} + (g_J^{t-1} + x_J^{t-1,\omega}) \quad [t \ge 2], \quad g_E^{1,\omega} = i_E^{1,\omega} \text{ (no prior release)}$$

and similarly for Rygene. The full 22-year backtest was re-run with the lag=1 formulation (prior releases before the planning horizon set to zero).

**Results (22-year backtest, 2003–2024):**

| Metric | Lag=0 (base) | Lag=1 | Difference |
|--------|-------------|-------|-----------|
| Mean OL revenue (MNOK/yr) | 374.2 | 335.4 | −38.9 |
| Mean CL revenue (MNOK/yr) | 375.2 | 335.3 | −39.9 |
| Mean PF revenue (MNOK/yr) | 409.2 | 348.6 | −60.6 |
| Mean VoF_total (MNOK/yr) | +0.99 | −0.10 | −1.09 |
| VoF_J (MNOK/yr) | +0.99 | −0.10 | −1.09 |
| VoF_E (MNOK/yr) | 0.00 | 0.00 | 0.00 |
| VoF_R (MNOK/yr) | 0.00 | 0.00 | 0.00 |
| Wilcoxon $p$ (CL > OL) | 0.046 | 0.283 | n/a |

**Interpretation:** VoF_E = VoF_R = 0 under both lag assumptions (confirmed analytically via the flow constraint proof: both plants are capacity-constrained in all weeks regardless of routing lag). The 1.09 MNOK/yr difference in VoF_J is well within one standard deviation of the VoF distribution (std ≈ 21 MNOK) and does not change the qualitative conclusion.

The **absolute revenues are ~38–40 MNOK/yr lower under lag=1** than lag=0. This is primarily a boundary-condition artefact of the lag=1 formulation: setting prior-period releases to zero causes Rygene to receive no upstream water in week 1, and the LP solution under lag=1 holds more water at Jørundland near year end (since releasing in weeks 51–52 provides no within-year benefit to downstream plants under lag=1). This "held water" earns terminal value in the LP objective but is excluded from the reported generation revenue, systematically understating lag=1 revenues relative to lag=0.

**The core finding is robust:** under the same-week and one-week lag assumptions, the VoF is indistinguishable from zero (both 0.99 and −0.10 MNOK/yr are below 0.05 standard deviations of the noise distribution). The routing lag assumption does not affect the headline qualitative result.

### 8.10 Why VoF_J is unstable: extreme-price years and out-of-distribution scenario trees

#### 8.10.1 Identifying the extreme-price years

The reported aggregate VoF_J (lag=0: +0.99 MNOK/yr, p=0.046 over 22 years) masks qualitatively different behaviour between the 17 normal-market years (2003–2020 excluding energy-crisis era) and the 5 elevated-price years that exceeded the +1.5 standard-deviation threshold for annual mean price established from the 2003–2020 reference distribution:

| Group | Years (n) | Annual mean price | Lag=0 VoF_J (mean ± std) | Lag=1 VoF_J (mean ± std) | Wilcoxon p (lag=0) |
|-------|-----------|-------------------|--------------------------|--------------------------|---------------------|
| Normal years | n=17 | ≤ 413 NOK/MWh | +3.40 ± 5.98 MNOK/yr | +1.14 ± 3.29 MNOK/yr | **0.013** |
| Elevated-price years | n=5 (2018, 2021–2024) | 417–2129 NOK/MWh | −7.19 ± 47.25 MNOK/yr | −4.32 ± 11.36 MNOK/yr | n/a |

Using the +2.0σ threshold (2021–2024 only, the energy-crisis cluster) produces virtually identical normal-year statistics (mean=3.30, std=5.82, p=0.010) and higher extreme-year volatility (std=54.26 MNOK/yr under lag=0), showing the result is not sensitive to the threshold choice.

**Key finding:** The overall 22-year aggregate (std=21.75, p=0.046) is entirely dominated by 4–5 extreme-price years. In the 17 normal-market years, VoF_J is small (mean ≈ +3.4 MNOK/yr), consistently positive, and statistically detectable at the 1% level. The apparent instability of the aggregate is not a property of the cascade model in normal markets. It is an artefact of the post-2020 European energy crisis entering the backtest.

#### 8.10.2 Mechanism: scenario-tree distribution mismatch, not reservoir saturation

Three candidate mechanisms were investigated:

**Reservoir constraint saturation (rejected).** If Jørundland's reservoir hit its storage limits more frequently in extreme years (forcing either mandatory spill or forced-low-generation weeks), the OL/CL policy comparison would diverge because the binding constraints would change. The cascade backtest records zero storage-clamp events (weeks where the LP is forced to the boundary) in all five elevated-price years (2018, 2021-2024), and the initial storage level (s_init_J) at the start of each backtest year is statistically indistinguishable between normal and extreme years (mean 113.5 GWh in both groups, S_max = 167 GWh). This mechanism is ruled out.

**Scenario-tree distribution mismatch (confirmed).** The scenario trees used by both the open-loop and closed-loop LPs were calibrated on the 2003–2020 price distribution (annual mean: 282 NOK/MWh, std: 87 NOK/MWh, weekly price P99: 550 NOK/MWh). In 2021–2024, the realized price distribution was:
- Annual means: 768, 2129, 903, 584 NOK/MWh (2.7x to 7.6x the training-period mean)
- Weekly P99: 4239 NOK/MWh (7.7x the training-period P99)
- 2022 weekly prices negatively correlated (r = −0.24) with the historical seasonal pattern

The cross-sectional correlation between annual mean price and VoF_J is r = −0.665 (p = 0.001): higher out-of-distribution prices produce more erratic, typically negative VoF_J. The mechanism is as follows. Both the OL and CL policies commit release decisions based on scenario trees that dramatically underestimate the realized price level and, critically in 2022, misrepresent the seasonal timing of high-price weeks. When the actual price series violates both the level and the seasonal structure of the scenario tree's prior, the relative performance of OL and CL (VoF = CL minus OL) is noise-dominated: in 2022 the OL plan accidentally released more water during the highest-price weeks, causing OL to beat CL by 79.7 MNOK; in 2023 (partial return to seasonal structure) CL successfully shifted water to remaining high-price weeks, producing VoF = +38.5 MNOK. Neither year's result reflects a stable economic property of the reservoir. Both reflect the consequence of backtesting against realized data that lies outside the training distribution.

**Routing lag amplification in extreme years (confirmed).** The absolute difference |VoF_J(lag=0) minus VoF_J(lag=1)| correlates strongly with annual mean price across all 22 years: r = 0.893, p < 0.001. In normal years the lag=0 and lag=1 VoF_J values typically differ by 0–6 MNOK. In 2022, they diverge by 71 MNOK (lag=0: −79.7 MNOK, lag=1: −8.7 MNOK). The mechanism is that a 1-week difference in routing lag changes which calendar week Jørundland's water contributes to downstream revenue, and in years where the realized price series has week-scale spikes of 4000+ NOK/MWh that were not anticipated by the scenario tree, shifting water by one week has outsized and essentially random effects on realised revenue. This is not an intrinsic property of the routing lag. It is the lag exposing timing noise that is negligible in normal-market years.

#### 8.10.3 Coherent cross-phase interpretation

Phase A (single-reservoir model, S_max = 1300 GWh) shows the identical pattern:

| Group | Phase A VoF (mean, MNOK/yr) | Phase B VoF_J (mean, MNOK/yr, lag=0) |
|-------|---------------------------|--------------------------------------|
| Normal years (2003–2020, excl. crisis) | +22.9 | +3.40 |
| Energy-crisis years (2021–2024) | +124.6 (std > 400) | −7.19 (std = 47.25) |

(Phase A "extreme years" mean is positive only because the 2023 outcome, +638 MNOK, dominates. The individual years span −312 to +638 MNOK, a 950 MNOK range over four years, which is analytically meaningless as a stable estimate.)

The ratio of normal-year VoF between Phase A and Phase B is 22.9 / 3.40 ≈ 6.7×, broadly consistent with the S_max ratio (1300 / 167 ≈ 7.8×), confirming that the storage capacity governs the base flexibility premium in normal markets.

In both phases, the energy-crisis years (2021-2024) do not change the qualitative conclusion about the reservoir's economic value in normal markets. They add high-variance, zero-information-content noise to the aggregate estimate. The appropriate presentation of both Phase A and Phase B results is: **the stable, economically interpretable flexibility value estimate is from the 17 normal-market years**. The aggregate 22-year result is dominated by the energy-crisis era in ways that obscure the underlying signal.

This is documented here so it is part of the visible record rather than a post-hoc framing adjustment. The original 22-year backtest period was fixed ex ante; the identification of the crisis-era years as out-of-distribution (not as a cherry-pick to obtain a cleaner result) is verifiable from the realised price data and the scenario-tree calibration window documented in Section 5.

### 8.11 Phase B limitations

The following limitations are specific to the Phase B extension (in addition to all Phase A limitations listed in Section 6):

1. **Perfect cross-plant inflow correlation (r = 1.0):** Because only a single discharge series (Rygene total) is available, per-plant local inflows are derived by fixed fractions. In practice, Jørundland's headwater sub-catchment (279 km²) may have partially uncorrelated inflow dynamics relative to the main valley catchment feeding Evenstad. This simplification likely underestimates the true VoF slightly, as independent sub-catchment inflows would create more genuinely uncertain local inflow scenarios at each plant, increasing the value of the closed-loop policy's ability to observe and react.

2. **Fixed-fraction inflow partitioning:** The energy-balance fractions ($f_J = 7.1\%$, $f_E = 67.2\%$, $f_R = 25.7\%$) are time-invariant means derived from long-run MidProd values. Seasonal and inter-annual variation in the relative contribution of each sub-catchment is not captured.

3. **Only three of 44 plants:** The cascade covers 23.6% of system installed capacity. VoF at the remaining 76.4% (including major upstream storage plants) is not quantified.

4. **Nesvatn working storage from observed range:** The S_MAX estimate (167 GWh) is derived from the observed min/max of the NVE volume series (1990–2025). The regulatory minimum and maximum pool elevations may differ slightly; the observed range is a conservative approximation.

5. **Terminal value at Jørundland only:** The piecewise-linear terminal value function is applied to Jørundland's end-of-horizon storage. In a full cascade model spanning all 44 plants, each storage plant would have its own terminal value function, increasing the effective value of holding water at the system level.


## References

- Heitsch, H., & Römisch, W. (2003). Scenario reduction algorithms in stochastic programming. *Computational Optimization and Applications*, 24(2-3), 187-206.
- Helseth, A., & Mo, B. (2016). SHOP: Short-term optimal hydropower scheduling. *Norwegian Water Resources and Energy Directorate Technical Note*.
- Fleten, S.E., & Kristoffersen, T.K. (2007). Stochastic programming for optimising bidding strategies of a Nordic hydropower producer. *European Journal of Operational Research*, 181(2), 916-928.
- Birge, J.R., & Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer.
- Tomasgard, A., et al. (2020). The EMPIRE model for capacity expansion planning under uncertainty. *NTNU/SINTEF Technical Report*.
