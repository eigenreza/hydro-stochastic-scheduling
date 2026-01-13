# Data Sources Manifest

Generated: 2026-06-19 (updated from 2026-06-18 original after price-zone audit)
Data cutoff enforced throughout: 2025-12-31

---

## Price-Zone Audit Note (2026-06-19)

The original station selection (Strengen, 16.142.0, Telemark) was found to be
incorrect on two grounds:

1. **Wrong price zone.** Strengen is in Tinn municipality, upper Telemark
   (coordinates 59.99°N, 8.37°E). Tinn feeds the eastern Norwegian grid (NO1).
   The Flesakersnittet transmission congestion boundary separating NO1 from NO2
   runs through Telemark; Tinn is on the NO1 side. The price data (EDS "NO2")
   therefore did not correspond to the physical location of the modelled station.

2. **Incompatible with reservoir model.** NVE API metadata confirms that
   Strengen (16.142.0) measures only a 72.57 km² unregulated alpine sub-catchment
   at 1077 m elevation on the Gøyst tributary. It has zero reservoirs
   (`numberReservoirs: 0`, `catchmentRegTypeName: Uregulert`). A modelled
   reservoir with S_MAX = 1300 GWh and G_MAX = 100 GWh/week has no physical
   correspondence to a 72 km² unregulated stream.

The station was re-selected to **Rygene total (19.127.0)** on the
Arendalsvassdraget in Agder. Agder is unambiguously in the NO2 bidding zone.
Rygene total has a 3,946 km² regulated catchment with 33 reservoirs and 100%
data completeness from 2000-01-01 to 2025-12-31. The net-head parameter was
updated from 300 m (Tokke/Telemarksvassdraget) to 450 m (Brokke/Arendalsvassdraget).

---

## 1. Inflow / Discharge: NVE HydAPI v1

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Hydrological API
**URL:** https://hydapi.nve.no/api/v1/
**Endpoint used:** `/api/v1/Observations`
**Query parameters:**
- StationId: `19.127.0`
- Parameter: `1001` (discharge/inflow, m³/s)
- ResolutionTime: 1440 (daily)
- ReferenceTime: 2000-01-01/2025-12-31
- Pull timestamp: 2026-06-19

**Selected station:** Rygene total (`19.127.0`)
**River:** Arendalsvassdraget
**County:** Agder
**Municipality:** Arendal
**Coverage:** 2000-01-01 → 2025-12-31
**Completeness (from 2000-01-01):** 1.000 (9,496/9,496 days)
**Mean discharge (2000–2025):** 124.81 m³/s
**Catchment area:** 3,946 km² (regulated)
**Number of reservoirs:** 33
**Regulation type:** Regulert m/magasinregulering og overføringer
**Price zone:** NO2 (Agder, southern Norway)

**Selection process:** All NVE stations in unambiguous NO2 counties (Agder, Rogaland)
were scored by completeness of daily discharge coverage from 2000-01-01 to 2025-12-31.
Full scoring table: `data/raw/no2_station_scores_agder_rogaland.csv`.
Rygene total and four other Agder stations tied at 1.000 completeness; Rygene total was
chosen as the largest regulated catchment (3,946 km², 33 reservoirs) most consistent
with the reservoir scheduling model assumptions.

**Unit conversion:** Daily mean Q [m³/s] → weekly energy-equivalent [GWh/week] via
cascade-effective-head approximation: η=0.88, H_eff=264.5 m (Arendalsvassdraget system
effective head; see derivation in methodology.md Section 2.2 and NVE power-plant registry
source below). k = 0.88 × 1000 × 9.81 × 264.5 × 604800 / 3.6e12 ≈ 0.3836 GWh/(m³/s·week).

**Correction note (2026-06-19):** A prior version used H_net = 450 m attributed to "Brokke
kraftverk." Brokke is on the Otra river (NVE registry head: 300 m), not Arendalsvassdraget.
The correct system effective head was recomputed from the NVE power-plant registry (see below).

---

## 2. Electricity Prices: Energi Data Service, NO2

**Source:** Energi Data Service (Danish Energy Agency / Energinet)
**URL:** https://api.energidataservice.dk/dataset/Elspotprices
**Query parameters:**
- filter: {"PriceArea": "NO2"}
- columns: HourUTC, SpotPriceEUR
- Fetched in three batches: 2000-01-01–2008-12-31, 2009-01-01–2016-12-31, 2017-01-01–2025-12-31
- Pull timestamp: 2026-06-19

**Currency conversion:** EUR/MWh → NOK/MWh via Norges Bank EUR/NOK daily
mid-rates (series B4 code: EURNOK). Source: https://data.norges-bank.no/
**Coverage (after conversion):** 2000-01-03 → 2025-12-31
**Aggregation:** Daily average of hourly prices; weekly average and std. dev.

**Note on EDS zone coverage:** EDS tracks electricity markets connected to the
Danish grid. NO2 (Kristiansand/Agder) connects to Denmark via the Skagerrak
high-voltage DC cable and is available in EDS from 2000. NO1 (Oslo/Eastern Norway)
is not available in EDS. This makes NO2 the only Norwegian bidding zone for which
a 2000-onset EDS price series can be constructed, and reinforces the choice of an
Agder (NO2) inflow station.

---

## 3. EUR/NOK Exchange Rates: Norges Bank

**Source:** Norges Bank, Statistical data
**URL:** https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP?format=sdmx-json&startPeriod=2000-01-01&endPeriod=2025-12-31&locale=en
**Coverage:** 2000-01-03 → 2025-12-31 (business days; weekends/holidays forward-filled)

---

## 4. Power Plant Registry: NVE Vannkraftdatabase

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Hydropower Plant Registry
**URL:** https://api.nve.no/web/Powerplant/GetHydroPowerPlantsInOperation/?mediaType=json
**Authentication:** None required (public API, NLOD licence)
**Query timestamp:** 2026-06-19

**Use:** Queried to obtain the system-level effective head H_eff for Arendalsvassdraget
and the total installed capacity for the G_MAX parameter. Key results:
- 44 active plants on Arendalsvassdraget (Nedborsfeltnavn = "Arendalsvassdraget")
- Total MidProd_91_20 = 2,498.4 GWh/year (mean annual production, 1991–2020 reference period)
- Total installed capacity = 565.5 MW → G_MAX = 565.5 × 168 / 1000 = 95.0 GWh/week
- H_eff = 264.5 m (derived via energy balance; see methodology.md Section 2.2)

**Rygene kraftverk (NVE VannKraftverkID confirmed):**
- Watershed: Arendalsvassdraget
- Gross head: 36.2 m
- Installed capacity: 55 MW
- Slukeevne (max discharge): 172.5 m³/s
- In operation since 1978

**Brokke kraftverk (verified for correction note):**
- Watershed: **Otra** (NOT Arendalsvassdraget)
- Gross head: 300 m
- Installed capacity: 330 MW
- The prior H_NET = 450 m was attributed to Brokke; both the watercourse and the value were wrong.

---

## 5. Reservoir Filling Statistics: NVE Magasinstatistikk

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Magasinstatistikk API
**URL:** https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData
**Authentication:** None required (public API, NLOD licence, CC BY 3.0)
**Area used:** NO2 elspot zone (omrType='EL', omrnr=2)
**Coverage:** ISO weeks from 1995 through present
**Pull timestamp:** 2026-06-19
**Cached file:** `data/raw/no2_magasin_weekly.csv`

**Description:** Weekly reservoir filling percentage (fyllingsgrad, 0–1) and absolute
filling (fylling_TWh) for the NO2 bidding zone, aggregated from approximately 490 of
Norway's major reservoirs tracked by NVE. The NO2 zone covers Agder, most of Rogaland,
southern Vestland, Vestfold, and Telemark.

**Use in backtest:** The ISO week-1 fyllingsgrad for each backtested year (2003–2024)
is used as the initial reservoir filling fraction for that year's backtest, replacing
the previous fixed 65% assumption. Conversion: s_init = fyllingsgrad_NO2_week1 × S_MAX.
This is the finest geographic resolution available in Magasinstatistikk; individual
watercourse-level data is not provided; the dataset aggregates to vassdrag zones
(VASS1–4) and elspot zones (NO1–5). The NO2 zone is the correct match since it
aligns with both the physical location of Arendalsvassdraget and the price series used.

**Note on resolution:** No station-level or watercourse-specific (Arendalsvassdraget)
filling data is available from NVE Magasinstatistikk. The VASS1 vassdrag zone was
also investigated; it covers Østlandet, Agder, and parts of Rogaland, broader than
NO2 and a worse match for a southern Agder station. NO2 was therefore selected as the
most appropriate available aggregate.

---

## 6. Final Aligned Weekly Panel

**File:** `data/processed/weekly_panel.csv`
**Inflow station:** Rygene total (19.127.0), Arendalsvassdraget, Agder
**Price zone:** NO2 (EDS)
**Gap handling:** Linear interpolation for gaps ≤ 3 consecutive weeks;
longer gaps dropped (see `src/data_acquisition/weekly_panel.py`).

---

## Phase B: Cascade Extension Sources

The following sources support the three-plant cascade extension (Jørundland → Evenstad → Rygene). See `docs/methodology.md` Section 8 for full technical documentation.

---

## 7. Per-Plant Physical Parameters: NVE Vannkraftdatabase

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Hydropower Plant Registry
**URL:** https://api.nve.no/web/Powerplant/GetHydroPowerPlantsInOperation/?mediaType=json
**Authentication:** None required (public API, NLOD licence)
**Query timestamp:** 2026-06-19
**Full extract:** `data/raw/nve_arendalsvassdraget_plants.csv`

Cascade plants extracted from the full Arendalsvassdraget registry:

| Plant | MW (MaksYtelse) | Gross head (m) | MidProd 91–20 (GWh/yr) | Max discharge (m³/s) | In operation |
|-------|-----------------|---------------|------------------------|----------------------|-------------|
| Jørundland | 55.2 | 278.6 | 185.8 | 22.1 | 1973 |
| Evenstad | 23.5 | 17.3 | 121.4 | 144.1 | 1973 |
| Rygene | 55.0 | 36.2 | n/a | 172.5 | 1978 |

**Derived model parameters:**
- G_MAX = MW × 168 h/week ÷ 1,000: G_MAX_J = 9.274 GWh/wk, G_MAX_E = 3.948 GWh/wk, G_MAX_R = 9.240 GWh/wk
- Energy conversion constant per plant: k_p = η × ρ × g × H_p × T_week / 3.6×10¹² GWh/(m³/s·wk), where η=0.88

**NVE vs. other sources cross-check (Jørundland):**
Statkraft published documents and the Norwegian Power Plant Register (NVE licence data, 2022 edition) both list Jørundland at 55.2 MW with gross head 278.6 m, consistent with the API values used. EnEkv for Nesvatn is 0.638 kWh/m³ per NVE HydAPI station metadata (station 19.5.0).

---

## 8. Nesvatn Reservoir Volume: NVE HydAPI

**Source:** Norwegian Water Resources and Energy Directorate (NVE), Hydrological API v1
**URL:** https://hydapi.nve.no/api/v1/Observations
**Authentication:** API key (stored in `.env`, not committed)
**Query parameters:**
- StationId: `19.5.0` (Nesvatn, on Gjøv, Arendalsvassdraget)
- Parameter: `1004` (reservoir volume, million m³)
- ResolutionTime: 1440 (daily)
- ReferenceTime: 1990-01-01/2025-12-31
- Pull timestamp: 2026-06-19

**File:** `data/raw/nesvatn_volume.csv`
**Coverage:** 1990-01-02 to 2025-12-30 (no significant gaps)
**Volume range observed:** 2.87 – 265.83 million m³
**Working storage:** max − min = 262.96 million m³

**Station identification:** Jørundland kraftverk's headwater reservoir was identified as
Nesvatn by cross-referencing the NVE concession database ("Regulering av Nesvatn" listed
in Jørundland's concession record) and confirming the station's river assignment matches
the Gjøv tributary that feeds Jørundland's penstock.

**Energy capacity conversion:**
- EnEkv (energy equivalence factor): 0.638 kWh/m³ (from NVE HydAPI station 19.5.0 metadata)
- S_MAX = 262.96 × 10⁶ m³ × 0.638 kWh/m³ ÷ 10⁹ = 167.77 GWh
- Model value used: **167 GWh** (conservative; lowest 0.87% of observations excluded)

**Data cutoff compliance:** Volume series ends 2025-12-30 ≤ 2025-12-31 (enforced cutoff). ✓

---

## 9. Inflow Fraction Derivation: Energy-Balance Method

No sub-catchment discharge series are available in NVE HydAPI for any individual
plant-level catchment within Arendalsvassdraget. The total measured system discharge
(Rygene total, 19.127.0, 3,946 km²) is partitioned into per-plant local inflows using
the following energy-balance method.

**Step 1: Mean turbined flow from NVE MidProd and head:**

$$Q_p = \frac{\text{MidProd}_p \times 3.6 \times 10^{12}}{\eta \cdot \rho \cdot g \cdot H_p \cdot T_{\text{year}}}$$

where $T_{\text{year}} = 31{,}557{,}600$ s/yr.

- $Q_J = 185.8 \times 3.6 \times 10^{12} \,/\, (0.88 \times 1000 \times 9.81 \times 278.6 \times 31{,}557{,}600) = 8.81$ m³/s
- $Q_E = 121.4 \times 3.6 \times 10^{12} \,/\, (0.88 \times 1000 \times 9.81 \times 17.3 \times 31{,}557{,}600) = 92.73$ m³/s

**Step 2: Local inflows (water entering each reach not already counted upstream):**

- $i_J = Q_J = 8.81$ m³/s (Jørundland headwater only)
- $i_E = Q_E - Q_J = 92.73 - 8.81 = 83.92$ m³/s (main Nidelva valley between Jørundland and Evenstad)
- $i_R = Q_{\text{Rygene,measured}} - Q_E = 124.81 - 92.73 = 32.08$ m³/s (between Evenstad and Rygene)

**Step 3: Fractions of total Rygene discharge (124.81 m³/s):**

$$f_J = \frac{8.81}{124.81} = 0.0706, \quad f_E = \frac{83.92}{124.81} = 0.6723, \quad f_R = \frac{32.08}{124.81} = 0.2571$$

Sum: $0.0706 + 0.6723 + 0.2571 = 1.0000$ (conserves total measured flow). ✓

**Limitation:** The fractions are derived from long-run mean turbined flows, not from direct
sub-catchment discharge measurements. They are time-invariant and imply perfect cross-plant
inflow correlation (r = 1.0). In practice, inflows to different sub-catchments are correlated
but not identical; seasonal and inter-annual deviations from the mean fractions are not captured.
