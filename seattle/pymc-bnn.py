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
from sklearn.impute import KNNImputer
import os
import arviz as az

# Flag to control whether to train or load existing model
TRAIN_NEW_MODEL = True  # Set to True to train new model, False to load existing

# ========== 1. Load Data ==========
# Use relative paths
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("\nInitial shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ========== 2. Clean and Impute Missing Values ==========
print("\nBefore cleaning:")
print(f"Missing values in X_train: {X_train.isna().sum().sum()}")
print(f"Missing values in y_train: {y_train.isna().sum()}")
print(f"Missing values in X_test: {X_test.isna().sum().sum()}")
print(f"Missing values in y_test: {y_test.isna().sum()}")

# Convert to numeric, keeping NaN values
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")
y_train = pd.to_numeric(y_train, errors="coerce")
y_test = pd.to_numeric(y_test, errors="coerce")

# Create imputer
imputer = KNNImputer(n_neighbors=5, weights='distance')

# Fit imputer on training data and transform both train and test
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# For y values, use forward fill then backward fill as a simple approach
y_train = y_train.fillna(method='ffill').fillna(method='bfill')
y_test = y_test.fillna(method='ffill').fillna(method='bfill')

print("\nAfter cleaning:")
print(f"Missing values in X_train: {X_train.isna().sum().sum()}")
print(f"Missing values in y_train: {y_train.isna().sum()}")
print(f"Missing values in X_test: {X_test.isna().sum().sum()}")
print(f"Missing values in y_test: {y_test.isna().sum()}")

# Convert to numpy arrays
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

print("\nFinal numpy array shapes:")
print(f"X_train_np shape: {X_train_np.shape}")
print(f"y_train_np shape: {y_train_np.shape}")
print(f"X_test_np shape: {X_test_np.shape}")
print(f"y_test_np shape: {y_test_np.shape}")

n_features = X_train_np.shape[1]

# ========== 3. Build and Train the Bayesian Neural Network ==========
trace_file = "bnn_trace.nc"

if TRAIN_NEW_MODEL:
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
        print("\n========= Training Progress =========")
        print(f"Number of samples: {len(trace.posterior.draw)}")
        print(f"Number of chains: {len(trace.posterior.chain)}")
        print("Training completed successfully!")
        
        # Save the trace
        trace.to_netcdf(trace_file)
        print(f"Trace saved to {trace_file}")
else:
    # Load the existing trace
    if os.path.exists(trace_file):
        trace = az.from_netcdf(trace_file)
        print(f"Loaded existing trace from {trace_file}")
    else:
        raise FileNotFoundError(f"No saved trace found at {trace_file}. Set TRAIN_NEW_MODEL=True to train a new model.")

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

# ========== 6. Enhanced Visualization ==========
# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predictions vs Actual with Uncertainty
axes[0, 0].errorbar(range(len(pred_mean)), pred_mean, yerr=pred_std, fmt='o', alpha=0.5, label="Predicted ± std")
axes[0, 0].plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.6, label="Actual")
axes[0, 0].set_xlabel("Sample Index")
axes[0, 0].set_ylabel("Site EUI")
axes[0, 0].set_title("PyMC BNN: Predicted vs Actual with Uncertainty")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Distribution of Predictions
axes[0, 1].hist(pred_mean, bins=30, alpha=0.7, label="Predicted")
axes[0, 1].hist(y_test_np, bins=30, alpha=0.7, label="Actual")
axes[0, 1].set_xlabel("Site EUI")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Distribution of Predictions vs Actual")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Uncertainty Analysis
axes[1, 0].scatter(pred_mean, pred_std, alpha=0.5)
axes[1, 0].set_xlabel("Predicted Value")
axes[1, 0].set_ylabel("Standard Deviation")
axes[1, 0].set_title("Uncertainty vs Prediction")
axes[1, 0].grid(True)

# 4. Residual Plot
residuals = y_test_np - pred_mean
axes[1, 1].scatter(pred_mean, residuals, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel("Predicted Value")
axes[1, 1].set_ylabel("Residuals")
axes[1, 1].set_title("Residual Plot")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('bnn_results.png')  # Save the plot instead of showing it
plt.close()

# Additional summary statistics
print("\n========= Additional Statistics =========")
print(f"Mean of predictions: {np.mean(pred_mean):.2f}")
print(f"Std of predictions: {np.std(pred_mean):.2f}")
print(f"Mean of residuals: {np.mean(residuals):.2f}")
print(f"Std of residuals: {np.std(residuals):.2f}")
