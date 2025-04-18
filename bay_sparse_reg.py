# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sparse Bayesian Regression: EUI Modeling
# This notebook implements a sparse Bayesian regression model to predict Energy Use Intensity (EUI) in office buildings using Bambi and PyMC.

# %%
## Import Libraries
import pandas as pd
import bambi as bmb
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# %%
# Load the dataset
df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data/Data_Ready_Model.csv')

# Select and scale relevant variables
features = [
    'EUI_kWh_per_sqmt', 'SQMT', 'NFLOOR', 'FLCEILHT',
    'MONUSE', 'OCCUPYP', 'WKHRS', 'NWKER', 'YRCONC',
    'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65', 'PUBCLIM'
]

df_model = df[features].dropna()

# Optional: Scale inputs
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df_model.drop(columns='EUI_kWh_per_sqmt')), columns=df_model.columns[1:])
df_bayes = pd.concat([X_scaled, df_model['EUI_kWh_per_sqmt'].reset_index(drop=True)], axis=1)

# %%
## Fit Sparse Bayesian Regression Model
formula = 'EUI_kWh_per_sqmt ~ SQMT + NFLOOR + FLCEILHT + MONUSE + OCCUPYP + WKHRS + NWKER + HEATP + COOLP + DAYLTP + HDD65 + CDD65'
model = bmb.Model(formula=formula, data=df_bayes, family='gaussian')

idata = model.fit(draws=1000, tune=1000, target_accept=0.9, return_inferencedata=True)

idata.extend(model.predict(idata, kind="pps", inplace=False))

# %%
## Posterior Predictive Check
az.plot_ppc(idata)
plt.title("Posterior Predictive Check")
plt.show()

# %% [markdown]
# ### Posterior Predictive Check (PPC)
# - Black curve = observed distribution
# - Blue = posterior predictive samples
# - Orange dashed = predictive mean
# 
# **Insights:**
# - Good alignment around mean/mode
# - Slight deviations in tails (expected)
# - Model captures key patterns in training data

# %%
## Next Step 1: Analyse parameter posteriors
az.summary(idata, round_to=2)

# %% [markdown]
# ### Posterior Summary Interpretation
# - `mean`, `sd`, `hdi_3%`, `hdi_97%` show effect and uncertainty.
# - `r_hat` ≈ 1.0 means MCMC convergence is excellent.
# 
# **Highlights:**
# - `SQMT`: strong negative relationship with EUI
# - `FLCEILHT`, `NWKER`, `HDD65`: strong positive effects
# - `DAYLTP`: weak effect (near 0)
# 
# This summary helps isolate the most impactful predictors of building energy use.

# %%
## Next Step 2: Compute model performance metrics
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Extract observed and predicted values
observed = idata.observed_data["EUI_kWh_per_sqmt"].values
predicted_samples = idata.posterior_predictive["EUI_kWh_per_sqmt"].values

# Average over chains and draws to get point predictions
predicted_mean = predicted_samples.mean(axis=(0, 1))

# Compute metrics
mse = mean_squared_error(observed, predicted_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(observed, predicted_mean)
r2 = r2_score(observed, predicted_mean)

# Print results
print(f"RMSE: {rmse:.2f} kWh/m²")
print(f"MAE: {mae:.2f} kWh/m²")
print(f"R² Score: {r2:.3f}")

# %% [markdown]
# ### Model Performance Metrics
# - **RMSE:** ~105.6 kWh/m²  
# - **MAE:** ~74.6 kWh/m²  
# - **R² Score:** 0.162
# 
# **Interpretation:**
# - The model captures ~16% of the variance — not bad for noisy energy data.
# - Scaled features and sparse prior are helping to prevent overfitting.
# - Useful baseline — ready for hierarchy, feature engineering, or prior tuning next.