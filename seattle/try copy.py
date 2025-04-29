# bnn_seattle.py
# This is a simple BNN model for the Seattle dataset.
import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import arviz as az
import pymc.sampling.jax
import os

# ========= 1. Load Imputed Data =========
data_path = "/Users/georgepaul/Desktop/Research-Project/seattle/data/seattle-data-imputed.csv"
df = pd.read_csv(data_path)

# Define feature and target columns
numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "SteamUse(kBtu)", "GHGEmissionsIntensity"
]
target_col = "SiteEUI(kBtu/sf)"

X = df[numeric_columns]
y = df[target_col]

# ========= 2. Preprocess =========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

n_features = X_train.shape[1]
trace_file = "bnn_trace_seattle.nc"

# ========= 3. Build or Load BNN =========
if os.path.exists(trace_file):
    print(f"Loading existing trace from {trace_file}...")
    trace = az.from_netcdf(trace_file)
    print("Trace loaded successfully.")
    with pm.Model() as bnn_model:
        X_data = pm.Data("X_data", X_train_scaled)
        y_data = pm.Data("y_data", y_train_np)

        # Rebuild model structure (must match original)
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
    print("Training new BNN model...")
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
        print(f"Trace saved to {trace_file}")

# ========= 4. Posterior Predictive =========
with bnn_model:
    pm.set_data({"X_data": X_test_scaled})
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

mu_pred_eval = ppc["y_obs"]
pred_mean = mu_pred_eval.mean(axis=0)
pred_std = mu_pred_eval.std(axis=0)

# ========= 5. Evaluation =========
r2 = r2_score(y_test_np, pred_mean)
rmse = mean_squared_error(y_test_np, pred_mean, squared=False)
mae = mean_absolute_error(y_test_np, pred_mean)

print("\n========= Evaluation Metrics =========")
print(f"R² Score                  : {r2:.3f}")
print(f"RMSE                      : {rmse:.2f}")
print(f"MAE                       : {mae:.2f}")
print(f"Avg Std Dev (Uncertainty) : {np.mean(pred_std):.2f}")
print(f"Max Std Dev               : {np.max(pred_std):.2f}")

# ========= 6. Visualizations =========
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predictions with uncertainty
axes[0, 0].errorbar(range(len(pred_mean)), pred_mean, yerr=pred_std, fmt='o', alpha=0.5, label="Predicted ± std")
axes[0, 0].plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.6, label="Actual")
axes[0, 0].set_title("Predicted vs Actual")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Prediction distribution
axes[0, 1].hist(pred_mean, bins=30, alpha=0.7, label="Predicted")
axes[0, 1].hist(y_test_np, bins=30, alpha=0.7, label="Actual")
axes[0, 1].set_title("Distribution of Predictions vs Actual")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Uncertainty vs prediction
axes[1, 0].scatter(pred_mean, pred_std, alpha=0.5)
axes[1, 0].set_title("Uncertainty vs Prediction")
axes[1, 0].grid(True)

# 4. Residual plot
residuals = y_test_np - pred_mean
axes[1, 1].scatter(pred_mean, residuals, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title("Residual Plot")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("bnn_seattle_results.png")
plt.close()

print("\nResults saved to 'bnn_seattle_results.png'")