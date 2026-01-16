"""
Bayesian seasonal AR(1) model for weekly NO2 day-ahead electricity prices.

Model specification
-------------------
    p_t = mu_t^p + v_t
    mu_t^p = beta_0 + Fourier(k=1..K) + gamma * q_t
    q_t = (inflow_t - seasonal_mean_inflow_t) / std(inflow)

    v_t = rho * v_{t-1} + eta_t,  eta_t ~ N(0, tau^2)

Conditional distribution:
    p_t | p_{t-1} ~ N(mu_t + rho*(p_{t-1} - mu_{t-1}), tau^2)

The first observation uses the stationary distribution.

gamma is expected to be negative (high inflow -> low price), encoding the
well-documented inverse inflow-price relationship in the Norwegian system.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt

LOG = logging.getLogger(__name__)

SEED = 43
PROCESSED_DIR = Path("data/processed")
K_HARMONICS = 3
N_DRAWS = 1000
N_CHAINS = 2
N_TUNE = 1000


def _fourier_features(week_nums: np.ndarray, K: int) -> np.ndarray:
    cols = []
    for k in range(1, K + 1):
        cols.append(np.cos(2 * np.pi * k * week_nums / 52))
        cols.append(np.sin(2 * np.pi * k * week_nums / 52))
    return np.column_stack(cols)


def _compute_seasonal_inflow_mean(
    weeks: np.ndarray,
    inflow_idata: az.InferenceData,
) -> np.ndarray:
    post = inflow_idata.posterior
    alpha_mean = float(post["alpha"].mean())
    coef_mean = post["fourier_coefs"].mean(dim=["chain", "draw"]).values
    log_seas = alpha_mean + _fourier_features(weeks, K_HARMONICS) @ coef_mean
    return np.exp(log_seas)


def fit_price_model(
    panel: pd.DataFrame,
    inflow_idata: az.InferenceData,
    train_end_year: int | None = None,
) -> tuple[pm.Model, az.InferenceData]:
    df = panel.copy()
    if train_end_year is not None:
        df = df[df["iso_year"] <= train_end_year]

    p_raw = df["price_avg_NOK_MWh"].values.astype(float)
    p = np.log(np.maximum(p_raw, 1.0))
    n = len(p)
    weeks = df["iso_week"].values.astype(float)
    X_fft = _fourier_features(weeks, K_HARMONICS)
    X_fft_tensor = pt.as_tensor_variable(X_fft.astype("float64"))
    p_tensor = pt.as_tensor_variable(p.astype("float64"))

    # Inflow anomaly covariate
    seas_inflow = _compute_seasonal_inflow_mean(weeks, inflow_idata)
    inflow_vals = df["inflow_GWh_week"].values.astype(float)
    inflow_std = float(np.std(inflow_vals))
    q = ((inflow_vals - seas_inflow) / (inflow_std + 1e-8)).astype("float64")
    q_tensor = pt.as_tensor_variable(q)

    mu_prior = float(np.mean(p))
    LOG.info("Fitting Bayesian price model: n=%d, train_end=%s", n, train_end_year)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=mu_prior, sigma=1.0)
        price_coefs = pm.Normal("price_fourier_coefs", mu=0.0, sigma=1.0,
                                shape=2 * K_HARMONICS)
        gamma = pm.Normal("gamma", mu=-0.3, sigma=0.3)
        rho = pm.TruncatedNormal("rho", mu=0.0, sigma=0.5, lower=-0.99, upper=0.99)
        tau = pm.HalfNormal("tau", sigma=0.3)

        seasonal_mean_p = beta + pt.dot(X_fft_tensor, price_coefs) + gamma * q_tensor
        tau_stat = tau / pt.sqrt(1.0 - rho ** 2)

        # Initial observation
        pm.Normal("p0", mu=seasonal_mean_p[0], sigma=tau_stat, observed=p[0])

        # Conditional AR(1) observations
        cond_mean_p = seasonal_mean_p[1:] + rho * (p_tensor[:-1] - seasonal_mean_p[:-1])
        pm.Normal("p_ar", mu=cond_mean_p, sigma=tau, observed=p[1:])

        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            random_seed=SEED,
            progressbar=True,
            target_accept=0.90,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=SEED)

    LOG.info("Price model fitted.")
    return model, idata


def generate_price_scenarios(
    price_idata: az.InferenceData,
    inflow_idata: az.InferenceData,
    panel: pd.DataFrame,
    forecast_start_week: pd.Timestamp,
    inflow_scenarios: pd.DataFrame,
    n_weeks: int = 52,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(SEED)

    n_scenarios = inflow_scenarios.shape[1]
    forecast_dates = inflow_scenarios.index
    forecast_iso_weeks = np.array([d.isocalendar().week for d in forecast_dates], dtype=float)
    X_f = _fourier_features(forecast_iso_weeks, K_HARMONICS)

    seas_inflow = _compute_seasonal_inflow_mean(forecast_iso_weeks, inflow_idata)
    inflow_std = float(panel["inflow_GWh_week"].std())

    post = price_idata.posterior
    beta_draws = post["beta"].values.flatten()
    pcoef_draws = post["price_fourier_coefs"].values.reshape(-1, 2 * K_HARMONICS)
    gamma_draws = post["gamma"].values.flatten()
    rho_draws = post["rho"].values.flatten()
    tau_draws = post["tau"].values.flatten()

    n_post = len(beta_draws)
    idx = rng.integers(0, n_post, size=n_scenarios)

    price_paths = np.zeros((n_weeks, n_scenarios))
    for j, i in enumerate(idx):
        inflow_path = inflow_scenarios.iloc[:, j].values
        q = (inflow_path - seas_inflow) / (inflow_std + 1e-8)

        seas_p = beta_draws[i] + X_f @ pcoef_draws[i] + gamma_draws[i] * q
        rho_i = float(rho_draws[i])
        tau_i = float(tau_draws[i])

        tau_stat = tau_i / np.sqrt(max(1 - rho_i ** 2, 1e-6))
        p_prev = seas_p[0] + rng.normal(0, tau_stat)
        price_paths[0, j] = np.exp(p_prev)

        for t in range(1, n_weeks):
            cond_mean = seas_p[t] + rho_i * (p_prev - seas_p[t - 1])
            p_t = cond_mean + rng.normal(0, tau_i)
            price_paths[t, j] = np.exp(p_t)
            p_prev = p_t

    return pd.DataFrame(
        np.maximum(price_paths, 1.0),
        index=forecast_dates,
        columns=inflow_scenarios.columns,
    )
