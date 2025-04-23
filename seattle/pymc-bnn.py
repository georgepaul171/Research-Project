# bnn_seattle_model.py

# ============================================================
# Virtual Environment Setup (run once in terminal):
# ============================================================
# python3 -m venv bpd_env
# source bpd_env/bin/activate
# pip install pandas scikit-learn pymc arviz matplotlib
# ============================================================

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ========== 1. Load Data ==========
X_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_train.csv")
X_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_test.csv")
y_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_train.csv").squeeze()
y_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_test.csv").squeeze()

# ========== 2. Ensure Numeric and Clean ==========
X_train = X_train.apply(pd.to_numeric, errors="coerce").dropna()
X_test = X_test.apply(pd.to_numeric, errors="coerce").dropna()
y_train = pd.to_numeric(y_train, errors="coerce")[:len(X_train)]
y_test = pd.to_numeric(y_test, errors="coerce")[:len(X_test)]

X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

n_features = X_train_np.shape[1]

# ========== 3. Build and Train the Bayesian Neural Network ==========
with pm.Model() as bnn_model:
    X_data = pm.Data("X_data", X_train_np)

    w1 = pm.Normal("w1", mu=0, sigma=1, shape=(n_features, 128))
    b1 = pm.Normal("b1", mu=0, sigma=1, shape=(128,))
    z1 = pt.tanh(pt.dot(X_data, w1) + b1)

    w2 = pm.Normal("w2", mu=0, sigma=1, shape=(128, 64))
    b2 = pm.Normal("b2", mu=0, sigma=1, shape=(64,))
    z2 = pt.tanh(pt.dot(z1, w2) + b2)

    w_out = pm.Normal("w_out", mu=0, sigma=1, shape=(64,))
    b_out = pm.Normal("b_out", mu=0, sigma=1)
    mu = pt.dot(z2, w_out) + b_out

    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_np)

    trace = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9,
                      return_inferencedata=True, cores=1, random_seed=42)

# ========== 4. Make Predictions Using Posterior Predictive ==========
with bnn_model:
    pm.set_data({"X_data": X_test_np})
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

mu_pred_eval = ppc["y_obs"]  # shape: (n_samples, n_test_points)

# ========== 5. Evaluation ==========
pred_mean = mu_pred_eval.mean(axis=0)
pred_std = mu_pred_eval.std(axis=0)

r2 = r2_score(y_test_np, pred_mean)
rmse = np.sqrt(np.mean((y_test_np - pred_mean) ** 2))
mae = mean_absolute_error(y_test_np, pred_mean)
mape = np.mean(np.abs((y_test_np - pred_mean) / y_test_np)) * 100
avg_uncertainty = np.mean(pred_std)
max_uncertainty = np.max(pred_std)

print("\n========= Evaluation Metrics =========")
print(f"R² Score                  : {r2:.3f}")
print(f"RMSE                      : {rmse:.2f}")
print(f"MAE                       : {mae:.2f}")
print(f"MAPE (%)                 : {mape:.2f}%")
print(f"Avg Std Dev (Uncertainty) : {avg_uncertainty:.2f}")
print(f"Max Std Dev (Uncertainty) : {max_uncertainty:.2f}")

# ========== 6. Plot Predictions with Uncertainty ==========
plt.figure(figsize=(10, 5))
plt.errorbar(range(len(pred_mean)), pred_mean, yerr=pred_std, fmt='o', alpha=0.5, label="Predicted ± std")
plt.plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.6, label="Actual")
plt.xlabel("Sample Index")
plt.ylabel("Site EUI")
plt.title("PyMC BNN: Predicted vs Actual with Uncertainty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
