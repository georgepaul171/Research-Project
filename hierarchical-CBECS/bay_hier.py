# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Hierarchical Models for EUI Prediction
# This notebook fits three different Bayesian models for Energy Use Intensity (EUI) using hierarchical structures and Student-t likelihoods.

# %%
## Import libraries
import pandas as pd
import bambi as bmb
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# %%
## Load and prepare data
df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data-CBECS/Data_Ready_to_Model.csv')
features = [
    'EUI_kWh_per_sqmt', 'SQMT', 'NFLOOR', 'FLCEILHT',
    'MONUSE', 'OCCUPYP', 'WKHRS', 'NWKER',
    'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65',
    'YRCONC', 'PUBCLIM'
]
df_model = df[features].dropna()

# Scale numeric features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df_model.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])),
    columns=[col for col in df_model.columns if col not in ['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']]
)
df_ready = pd.concat([X_scaled, df_model[['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']].reset_index(drop=True)], axis=1)

# Make group variables categorical
df_ready['YRCONC'] = df_ready['YRCONC'].astype("category")
df_ready['PUBCLIM'] = df_ready['PUBCLIM'].astype("category")

# %%
## Three Models

# Model 1: Add random intercepts by YRCONC + PUBCLIM
formula1 = 'EUI_kWh_per_sqmt ~ SQMT + NFLOOR + FLCEILHT + MONUSE + OCCUPYP + WKHRS + NWKER + HEATP + COOLP + DAYLTP + HDD65 + CDD65 + (1|YRCONC) + (1|PUBCLIM)'
model1 = bmb.Model(formula=formula1, data=df_ready, family='gaussian')
idata1 = model1.fit(draws=1000, tune=1000, target_accept=0.95, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

# Model 2: Add interactions
formula2 = 'EUI_kWh_per_sqmt ~ SQMT + HEATP + SQMT:HEATP + NWKER + FLCEILHT + NWKER:FLCEILHT + HDD65 + CDD65 + (1|YRCONC) + (1|PUBCLIM)'
model2 = bmb.Model(formula=formula2, data=df_ready, family='gaussian')
idata2 = model2.fit(draws=1000, tune=1000, target_accept=0.95, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

# Model 3: Crossed groups (nested-like)
formula3 = 'EUI_kWh_per_sqmt ~ SQMT + FLCEILHT + MONUSE + OCCUPYP + NWKER + HEATP + COOLP + HDD65 + CDD65 + (1|YRCONC:PUBCLIM)'
model3 = bmb.Model(formula=formula3, data=df_ready, family='gaussian')
idata3 = model3.fit(draws=1000, tune=1000, target_accept=0.95, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

# %%
# Compare using LOO
loo_comparison = az.compare(
    {"M1": idata1, "M2": idata2, "M3": idata3},
    ic="loo",
    method="stacking",
    scale="deviance"
)

print(loo_comparison)
az.plot_compare(loo_comparison)
plt.title("Model Comparison via LOO")
plt.show()

# %% [markdown]
# ## Hierarchical Intercepts with Student-t Likelihood

# %%
import numpy as np

# Encode group variables
df_ready['YRCONC_code'] = df_ready['YRCONC'].cat.codes
df_ready['PUBCLIM_code'] = df_ready['PUBCLIM'].cat.codes

X = df_ready.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])
y = df_ready['EUI_kWh_per_sqmt'].values
n, p = X.shape

# %%
# PyMC model
coords = {
    "obs_id": np.arange(n),
    "feature": X.columns.tolist(),
    "YRCONC": df_ready['YRCONC_code'].unique(),
    "PUBCLIM": df_ready['PUBCLIM_code'].unique()
}

with pm.Model(coords=coords) as model:

    sigma_yrconc = pm.HalfNormal("sigma_yrconc", 10)
    sigma_pubclim = pm.HalfNormal("sigma_pubclim", 10)

    a_yrconc = pm.Normal("a_yrconc", mu=0, sigma=sigma_yrconc, dims="YRCONC")
    a_pubclim = pm.Normal("a_pubclim", mu=0, sigma=sigma_pubclim, dims="PUBCLIM")

    beta = pm.Normal("beta", mu=0, sigma=5, dims="feature")
    intercept = pm.Normal("intercept", mu=0, sigma=5)

    mu = (
        intercept
        + pm.math.dot(X.values, beta)
        + a_yrconc[df_ready['YRCONC_code'].values]
        + a_pubclim[df_ready['PUBCLIM_code'].values]
    )

    sigma = pm.HalfNormal("sigma", 10)
    nu = pm.Exponential("nu", 1 / 30)
    y_obs = pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y, dims="obs_id")

    trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

# %%
az.plot_trace(trace, var_names=["intercept", "sigma", "nu", "sigma_yrconc", "sigma_pubclim"])
plt.show()

az.summary(trace, var_names=["intercept", "sigma", "nu", "sigma_yrconc", "sigma_pubclim"], round_to=2)

# %% [markdown]
# ### Bayesian Robust Hierarchical Model Summary (Student-t)
# - Student-t handles outliers
# - `sigma_pubclim` > `sigma_yrconc` — climate zone has more influence
# - All `r_hat` ≈ 1.0 — convergence is good

# %%
# Posterior predictive check
ppc = pm.sample_posterior_predictive(trace, model=model)
y_pred_samples = ppc["y_obs"]
y_pred_mean = y_pred_samples.mean(axis=0)

# %%
# Performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y, y_pred_mean))
mae = mean_absolute_error(y, y_pred_mean)
r2 = r2_score(y, y_pred_mean)

print(f"RMSE: {rmse:.2f} kWh/m²")
print(f"MAE: {mae:.2f} kWh/m²")
print(f"R² Score: {r2:.3f}")

# %% [markdown]
# ## Model Performance – Robust Hierarchical Student-t
# | Metric | Value | Interpretation |
# |--------|-------|----------------|
# | **RMSE** | ~113.8 kWh/m² | Moderate prediction error |
# | **MAE** | ~76.9 kWh/m² | Mean absolute error is more robust to outliers |
# | **R²** | 0.027 | Model explains ~2.7% of variance |
#
# ### Conclusion
# - Useful for uncertainty quantification and understanding group effects.
# - Needs further refinement for improved predictive performance.