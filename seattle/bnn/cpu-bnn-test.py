# %%
# BNN on Small Subset (CPU) — Seattle Energy Data

import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# %%
# Output directory
output_dir = "results_cpu_test"
os.makedirs(output_dir, exist_ok=True)

# %%
# Load data
X_train = pd.read_csv("data/X_train_imputed.csv")
y_train = pd.read_csv("data/y_train_imputed.csv")
X_test = pd.read_csv("data/X_test_imputed.csv")
y_test = pd.read_csv("data/y_test_imputed.csv")
print("✅ Loaded and subsampled data")

y_train.columns = ["SiteEUI(kBtu/sf)"]
y_test.columns = ["SiteEUI(kBtu/sf)"]

# Subsample (for CPU testing)
X_train = X_train.sample(n=500, random_state=42)
y_train = y_train.loc[X_train.index].squeeze()
X_test = X_test.sample(n=200, random_state=42)
y_test = y_test.loc[X_test.index].squeeze()

# %%
# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

n_features = X_train.shape[1]

# %%
# Build and train model
print("✅ Starting model...")
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
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    trace = pm.sample(draws=200, tune=200, chains=1, target_accept=0.9, random_seed=42)
    
print("✅ Sampling complete")

# %%
# 🛠 Reset model context safely
print("✅ Dataset shapes:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"y_train_np:     {y_train_np.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")
print(f"y_test_np:      {y_test_np.shape}")

# Make sure X_data and y_data exist once
with bnn_model:
    # Switch to test inputs
    pm.set_data({"X_data": X_test_scaled})
    
    # Detach training targets to avoid shape mismatch (use None or dummy)
    bnn_model["y_data"].set_value(np.zeros_like(y_test_np))

    # Generate predictive samples
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

# Convert to numpy arrays
mu_pred_eval = ppc.posterior_predictive["y_obs"].values
pred_mean = mu_pred_eval.mean(axis=0)
pred_std = mu_pred_eval.std(axis=0)

print("✅ Posterior prediction complete")

# %%
print("✅ Calculating metrics...")

# Posterior prediction
mu_pred_eval = ppc.posterior_predictive["y_obs"].values
print("mu_pred_eval shape:", mu_pred_eval.shape)  # Expect (1, 200, 200)

# Collapse chain+draws → single mean per test point
pred_mean = mu_pred_eval.mean(axis=(0, 1))  # shape: (200,)
pred_std = mu_pred_eval.std(axis=(0, 1))    # shape: (200,)

print("pred_mean shape:", pred_mean.shape)
print("y_test_np shape:", y_test_np.shape)

# Metrics
r2 = r2_score(y_test_np, pred_mean)
rmse = np.sqrt(mean_squared_error(y_test_np, pred_mean))
mae = mean_absolute_error(y_test_np, pred_mean)

print("=== Evaluation Metrics ===")
print(f"R²     : {r2:.3f}")
print(f"RMSE   : {rmse:.2f}")
print(f"MAE    : {mae:.2f}")
print(f"Avg σ  : {np.mean(pred_std):.2f}")
print(f"Max σ  : {np.max(pred_std):.2f}")

# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"R²     : {r2:.3f}\n")
    f.write(f"RMSE   : {rmse:.2f}\n")
    f.write(f"MAE    : {mae:.2f}\n")
    f.write(f"Avg σ  : {np.mean(pred_std):.2f}\n")
    f.write(f"Max σ  : {np.max(pred_std):.2f}\n")

# %%
# Visualizations
print("✅ Visualizing...")
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
plt.savefig(os.path.join(output_dir, "bnn_results.png"))
plt.close()

# %%
# Trace summary
print("✅ Summarizing trace...")
summary_df = az.summary(trace)
summary_df.to_csv(os.path.join(output_dir, "trace_summary.csv"))
summary_df