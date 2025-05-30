# debugbnn.py — Bayesian Neural Network + Variant Experiments on Seattle Energy Subset

import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge

# === Setup ===
output_dir = "results_cpu_test"
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
X_train = pd.read_csv("data/X_train_imputed.csv")
y_train = pd.read_csv("data/y_train_imputed.csv")
X_test = pd.read_csv("data/X_test_imputed.csv")
y_test = pd.read_csv("data/y_test_imputed.csv")
print("\u2705 Loaded and subsampled data")

y_train.columns = ["SiteEUI(kBtu/sf)"]
y_test.columns = ["SiteEUI(kBtu/sf)"]

X_train = X_train.sample(n=500, random_state=42)
y_train = y_train.loc[X_train.index].squeeze()
X_test = X_test.sample(n=200, random_state=42)
y_test = y_test.loc[X_test.index].squeeze()

# === Standardize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

n_features = X_train.shape[1]

# === Build and Train Model ===
print("\u2705 Starting model...")
with pm.Model() as bnn_model:
    X_data = pm.Data("X_data", X_train_scaled)
    y_data = pm.Data("y_data", y_train_np)

    # Toggle activation function
    activation = pt.tanh  # Change to pt.nnet.relu to compare

    w1 = pm.Normal("w1", mu=0, sigma=1, shape=(n_features, 32))
    b1 = pm.Normal("b1", mu=0, sigma=1, shape=(32,))
    z1 = activation(pt.dot(X_data, w1) + b1)

    w2 = pm.Normal("w2", mu=0, sigma=1, shape=(32, 16))
    b2 = pm.Normal("b2", mu=0, sigma=1, shape=(16,))
    z2 = activation(pt.dot(z1, w2) + b2)

    w_out = pm.Normal("w_out", mu=0, sigma=1, shape=(16,))
    b_out = pm.Normal("b_out", mu=0, sigma=1)
    mu = pt.dot(z2, w_out) + b_out

    sigma = pm.HalfNormal("sigma", sigma=1)

    # Toggle likelihood distribution
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)
    # y_obs = pm.StudentT("y_obs", nu=3, mu=mu, sigma=sigma, observed=y_data)

    trace = pm.sample(draws=200, tune=200, chains=1, target_accept=0.9, random_seed=42)
print("\u2705 Sampling complete")

# === Posterior Prediction ===
print("\u2705 Dataset shapes:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"y_train_np:     {y_train_np.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")
print(f"y_test_np:      {y_test_np.shape}")

with bnn_model:
    pm.set_data({"X_data": X_test_scaled})
    bnn_model["y_data"].set_value(np.zeros_like(y_test_np))
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

mu_pred_eval = ppc.posterior_predictive["y_obs"].values
pred_mean = mu_pred_eval.mean(axis=(0, 1))
pred_std = mu_pred_eval.std(axis=(0, 1))
print("\u2705 Posterior prediction complete")

# === Evaluation ===
print("\u2705 Calculating metrics...")
r2 = r2_score(y_test_np, pred_mean)
rmse = np.sqrt(mean_squared_error(y_test_np, pred_mean))
mae = mean_absolute_error(y_test_np, pred_mean)

print("=== Evaluation Metrics ===")
print(f"R²     : {r2:.3f}")
print(f"RMSE   : {rmse:.2f}")
print(f"MAE    : {mae:.2f}")
print(f"Avg σ  : {np.mean(pred_std):.2f}")
print(f"Max σ  : {np.max(pred_std):.2f}")

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"R²     : {r2:.3f}\n")
    f.write(f"RMSE   : {rmse:.2f}\n")
    f.write(f"MAE    : {mae:.2f}\n")
    f.write(f"Avg σ  : {np.mean(pred_std):.2f}\n")
    f.write(f"Max σ  : {np.max(pred_std):.2f}\n")

# === Visualizations ===
print("\u2705 Visualizing...")
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

# === Trace Summary ===
print("\u2705 Summarizing trace...")
summary_df = az.summary(trace)
summary_df.to_csv(os.path.join(output_dir, "trace_summary.csv"))

# === Baseline Ridge Comparison ===
print("\n✅ Baseline Ridge Regression...")
ridge = Ridge()
ridge.fit(X_train_scaled, y_train_np)
ridge_preds = ridge.predict(X_test_scaled)
print("Ridge RMSE:", np.sqrt(mean_squared_error(y_test_np, ridge_preds)))
print("Ridge R²:", r2_score(y_test_np, ridge_preds))

# On a small sample (500 training, 200 test), the Bayesian Neural Network yielded 
# an RMSE of 36.81 and an R² of −0.027. In contrast, Ridge Regression performed 
# slightly better with an RMSE of 36.04 and R² of 0.015. The BNN also produced 
# predictive uncertainty estimates with average standard deviation of 25.55, 
# highlighting its utility for uncertainty-aware tasks despite weaker performance 
# under small data regimes