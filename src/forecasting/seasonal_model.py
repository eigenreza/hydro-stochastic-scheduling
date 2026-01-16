"""
Bayesian seasonal AR(1) model for weekly inflow.

Model specification
-------------------
Let y_t = log(inflow_t + 0.1) be log-transformed weekly inflow.

    y_t = mu_t + u_t
    mu_t = alpha_0 + sum_{k=1}^{K} [a_k cos(2*pi*k*t/52) + b_k sin(2*pi*k*t/52)]
    u_t = phi * u_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)

Equivalently, the conditional distribution of y_t given y_{t-1} is:
    y_t | y_{t-1} ~ N(mu_t + phi * (y_{t-1} - mu_{t-1}), sigma^2)   for t >= 2

This formulation observes the original data y directly, which is valid in PyMC
(observed data are constant numpy arrays, not symbolic expressions).

The first observation uses the stationary distribution:
    y_1 ~ N(mu_1, sigma_stat^2)   where sigma_stat = sigma / sqrt(1 - phi^2)

K=3 harmonics capture Norwegian snowmelt-driven hydrology.

Priors:
    alpha_0    ~ Normal(log_y_mean, 1)
    a_k, b_k  ~ Normal(0, 1)
    phi        ~ TruncatedNormal(0, 0.5, lower=-0.99, upper=0.99)
    sigma      ~ HalfNormal(0.3)
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

SEED = 42
PROCESSED_DIR = Path("data/processed")
K_HARMONICS = 3
N_DRAWS = 1000
N_CHAINS = 2
N_TUNE = 1000


def _load_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "weekly_panel.csv"
    df = pd.read_csv(path, index_col="week_start", parse_dates=True)
    return df.dropna(subset=["inflow_GWh_week"])


def _fourier_features(week_nums: np.ndarray, K: int) -> np.ndarray:
    cols = []
    for k in range(1, K + 1):
        cols.append(np.cos(2 * np.pi * k * week_nums / 52))
        cols.append(np.sin(2 * np.pi * k * week_nums / 52))
    return np.column_stack(cols)


def fit_inflow_model(
    panel: pd.DataFrame,
    train_end_year: int | None = None,
) -> tuple[pm.Model, az.InferenceData]:
    """
    Fit the Bayesian seasonal AR(1) model.
    Returns (model, idata).
    """
    df = panel.copy()
    if train_end_year is not None:
        df = df[df["iso_year"] <= train_end_year]

    y_raw = df["inflow_GWh_week"].values.astype(float)
    y = np.log(y_raw + 0.1)
    n = len(y)
    weeks = df["iso_week"].values.astype(float)
    X_fft = _fourier_features(weeks, K_HARMONICS)
    X_fft_tensor = pt.as_tensor_variable(X_fft.astype("float64"))
    y_tensor = pt.as_tensor_variable(y.astype("float64"))
    mu_prior = float(np.mean(y))

    LOG.info("Fitting Bayesian inflow model: n=%d, train_end=%s", n, train_end_year)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=mu_prior, sigma=1.0)
        coefs = pm.Normal("fourier_coefs", mu=0.0, sigma=1.0, shape=2 * K_HARMONICS)
        phi = pm.TruncatedNormal("phi", mu=0.0, sigma=0.5, lower=-0.99, upper=0.99)
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        seasonal_mean = alpha + pt.dot(X_fft_tensor, coefs)   # shape (n,)
        sigma_stat = sigma / pt.sqrt(1.0 - phi ** 2)

        # Initial observation: stationary distribution
        pm.Normal("y0", mu=seasonal_mean[0], sigma=sigma_stat, observed=y[0])

        # Conditional: y_t | y_{t-1} ~ N(mu_t + phi*(y_{t-1} - mu_{t-1}), sigma^2)
        cond_mean = seasonal_mean[1:] + phi * (y_tensor[:-1] - seasonal_mean[:-1])
        pm.Normal("y_ar", mu=cond_mean, sigma=sigma, observed=y[1:])

        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            random_seed=SEED,
            progressbar=True,
            target_accept=0.90,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=SEED)

    LOG.info("Inflow model fitted.")
    return model, idata


def generate_inflow_scenarios(
    idata: az.InferenceData,
    panel: pd.DataFrame,
    forecast_start_week: pd.Timestamp,
    n_weeks: int = 52,
    n_scenarios: int = 200,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Draw n_scenarios paths from the posterior predictive.
    Returns DataFrame of shape (n_weeks, n_scenarios) of inflow in GWh/week.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    forecast_dates = pd.date_range(forecast_start_week, periods=n_weeks, freq="W-MON")
    forecast_iso_weeks = np.array([d.isocalendar().week for d in forecast_dates], dtype=float)
    X_f = _fourier_features(forecast_iso_weeks, K_HARMONICS)

    post = idata.posterior
    alpha_draws = post["alpha"].values.flatten()
    coef_draws = post["fourier_coefs"].values.reshape(-1, 2 * K_HARMONICS)
    phi_draws = post["phi"].values.flatten()
    sigma_draws = post["sigma"].values.flatten()

    n_post = len(alpha_draws)
    idx = rng.integers(0, n_post, size=n_scenarios)

    paths = np.zeros((n_weeks, n_scenarios))
    for j, i in enumerate(idx):
        seas = alpha_draws[i] + X_f @ coef_draws[i]
        phi_i = float(phi_draws[i])
        sig_i = float(sigma_draws[i])

        # Simulate from stationary initial distribution
        sig_stat = sig_i / np.sqrt(max(1 - phi_i ** 2, 1e-6))
        y_prev = seas[0] + rng.normal(0, sig_stat)
        paths[0, j] = np.exp(y_prev) - 0.1

        for t in range(1, n_weeks):
            cond_mean = seas[t] + phi_i * (y_prev - seas[t - 1])
            y_t = cond_mean + rng.normal(0, sig_i)
            paths[t, j] = np.exp(y_t) - 0.1
            y_prev = y_t

    return pd.DataFrame(
        np.maximum(paths, 0.0),
        index=forecast_dates,
        columns=[f"s{j:03d}" for j in range(n_scenarios)],
    )
