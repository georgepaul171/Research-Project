# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Neural Network (PyMC) on Seattle Energy Data
# This notebook trains a Bayesian Neural Network on imputed energy data using PyMC, JAX, and NumPyro.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import arviz as az
import pymc.sampling.jax
import os

import jax
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

print("\n--- JAX ---")
print("JAX devices:", jax.devices())

print("\n--- PyTorch ---")
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA device available.")

# Output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## 2. Load and Prepare Data

# %%
X_train = pd.read_csv("data/X_train_imputed.csv")
y_train = pd.read_csv("data/y_train_imputed.csv")
X_test = pd.read_csv("data/X_test_imputed.csv")
y_test = pd.read_csv("data/y_test_imputed.csv")

# Rename columns if needed
if y_train.columns[0] != "SiteEUI(kBtu/sf)":
    y_train.columns = ["SiteEUI(kBtu/sf)"]
if y_test.columns[0] != "SiteEUI(kBtu/sf)":
    y_test.columns = ["SiteEUI(kBtu/sf)"]

y_train = y_train.squeeze()
y_test = y_test.squeeze()

# %% [markdown]
# ## 3. Standardize Features

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

n_features = X_train.shape[1]
trace_file = os.path.join(output_dir, "bnn_trace_seattle.nc")

# %% [markdown]
# ## 4. Build and Train Model

# %%
if os.path.exists(trace_file):
    print(f"Loading trace from {trace_file}...")
    trace = az.from_netcdf(trace_file)
    with pm.Model() as bnn_model:
        X_data = pm.Data("X_data", X_train_scaled)
        y_data = pm.Data("y_data", y_train_np)

        w1 = pm.Normal("w1", mu=0, sigma=1, shape=(n_features, 32))
        b1 = pm.Normal("b1", mu=0, sigma=1, shape=(32,))
        z1 = pt.tanh(pt.dot(X_data, w1) + b1)

        w2 = pm.Normal("w2", mu=0, sigma=1, shape=(32, 16))
        b2 = pm.Normal("b2", mu=0, sigma=1, shape=(16,))
        z2 = pt.tanh(pt.dot(z1, w2) + b2)

        w_out = pm.Normal("w_out", mu=0, sigma=1, shape=(16,))
        b_out = pm.Normal("b_out", mu=0, sigma=1)
        mu = pt.dot(z2, w_out) + b_out

        sigma = pm.HalfNormal("sigma", sigma=1)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_np)
else:
    with pm.Model() as bnn_model:
        X_data = pm.Data("X_data", X_train_scaled)
        y_data = pm.Data("y_data", y_train_np)

        w1 = pm.Normal("w1", mu=0, sigma=1, shape=(n_features, 32))
        b1 = pm.Normal("b1", mu=0, sigma=1, shape=(32,))
        z1 = pt.tanh(pt.dot(X_data, w1) + b1)

        w2 = pm.Normal("w2", mu=0, sigma=1, shape=(32, 16))
        b2 = pm.Normal("b2", mu=0, sigma=1, shape=(16,))
        z2 = pt.tanh(pt.dot(z1, w2) + b2)

        w_out = pm.Normal("w_out", mu=0, sigma=1, shape=(16,))
        b_out = pm.Normal("b_out", mu=0, sigma=1)
        mu = pt.dot(z2, w_out) + b_out

        sigma = pm.HalfNormal("sigma", sigma=1)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_np)

        trace = pm.sampling.jax.sample_numpyro_nuts(
            draws=500,
            tune=500,
            target_accept=0.9,
            chains=1,
            random_seed=42
        )
        trace = az.from_dict(trace)
        trace.to_netcdf(trace_file)
        print(f"Saved trace to {trace_file}")

# %% [markdown]
# ## 5. Posterior Prediction

# %%
with bnn_model:
    pm.set_data({"X_data": X_test_scaled})
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

mu_pred_eval = ppc["y_obs"]
pred_mean = mu_pred_eval.mean(axis=0)
pred_std = mu_pred_eval.std(axis=0)

# %% [markdown]
# ## 6. Evaluation Metrics

# %%
r2 = r2_score(y_test_np, pred_mean)
rmse = mean_squared_error(y_test_np, pred_mean, squared=False)
mae = mean_absolute_error(y_test_np, pred_mean)

print("=== Evaluation Metrics ===")
print(f"R²     : {r2:.3f}")
print(f"RMSE   : {rmse:.2f}")
print(f"MAE    : {mae:.2f}")
print(f"Avg σ  : {np.mean(pred_std):.2f}")
print(f"Max σ  : {np.max(pred_std):.2f}")

with open(os.path.join(output_dir, "bnn_seattle_metrics.txt"), "w") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"R²     : {r2:.3f}\n")
    f.write(f"RMSE   : {rmse:.2f}\n")
    f.write(f"MAE    : {mae:.2f}\n")
    f.write(f"Avg σ  : {np.mean(pred_std):.2f}\n")
    f.write(f"Max σ  : {np.max(pred_std):.2f}\n")

# %% [markdown]
# ## 7. Visualizations

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].errorbar(range(len(pred_mean)), pred_mean, yerr=pred_std, fmt='o', alpha=0.5, label="Pred ± σ")
axes[0, 0].plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.6, label="Actual")
axes[0, 0].set_title("Prediction vs Actual")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].hist(pred_mean, bins=30, alpha=0.7, label="Predicted")
axes[0, 1].hist(y_test_np, bins=30, alpha=0.7, label="Actual")
axes[0, 1].set_title("Prediction vs Actual Distribution")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].scatter(pred_mean, pred_std, alpha=0.5)
axes[1, 0].set_title("Uncertainty vs Prediction")
axes[1, 0].grid(True)

residuals = y_test_np - pred_mean
axes[1, 1].scatter(pred_mean, residuals, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title("Residual Plot")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bnn_seattle_results.png"))
plt.close()

# %% [markdown]
# ## 8. Trace Summary

# %%
summary_df = az.summary(trace)
summary_df.to_csv(os.path.join(output_dir, "bnn_trace_summary.csv"))
summary_df