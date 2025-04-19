# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # PyMC Bayesian Neural Network (128-64) for EUI Estimation (No JAX, Fixed Activation)

# %%
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# %%
# ## Load Data

X_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_train.csv")
X_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_test.csv")
y_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_train.csv").squeeze()
y_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_test.csv").squeeze()

X_train = X_train.apply(pd.to_numeric, errors="coerce").dropna()
X_test = X_test.apply(pd.to_numeric, errors="coerce").dropna()
y_train = pd.to_numeric(y_train, errors="coerce")[:len(X_train)]
y_test = pd.to_numeric(y_test, errors="coerce")[:len(X_test)]

X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

n_features = X_train_np.shape[1]

# %%
# ## Define PyMC Model (128–64) with ReLU Replaced

with pm.Model() as bnn_128_64:
    # First hidden layer
    w1 = pm.Normal("w1", mu=0, sigma=1, shape=(n_features, 128))
    b1 = pm.Normal("b1", mu=0, sigma=1, shape=(128,))
    z1_pre = pm.math.dot(X_train_np, w1) + b1
    z1 = pm.math.maximum(z1_pre, 0)  # ReLU replacement

    # Second hidden layer
    w2 = pm.Normal("w2", mu=0, sigma=1, shape=(128, 64))
    b2 = pm.Normal("b2", mu=0, sigma=1, shape=(64,))
    z2_pre = pm.math.dot(z1, w2) + b2
    z2 = pm.math.maximum(z2_pre, 0)

    # Output layer
    w_out = pm.Normal("w_out", mu=0, sigma=1, shape=(64,))
    b_out = pm.Normal("b_out", mu=0, sigma=1)
    mu = pm.math.dot(z2, w_out) + b_out

    # Likelihood
    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_np)

    # Sampling
    trace = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True,
        cores=1,
        random_seed=42
    )

# %%
# ## Posterior Analysis

az.plot_trace(trace, var_names=["w1", "w_out", "sigma"])
plt.tight_layout()
plt.show()

# %%
# ## Predict on Test Set

with bnn_128_64:
    posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y_obs"], predictions=True)

predictions = posterior_pred.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
pred_std = posterior_pred.posterior_predictive["y_obs"].std(dim=["chain", "draw"]).values

# %%
# ## Evaluation

r2 = r2_score(y_test_np, predictions)
rmse = mean_squared_error(y_test_np, predictions, squared=False)

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# %%
# ## Plot Predictions with Uncertainty

plt.figure(figsize=(10, 5))
plt.errorbar(range(len(predictions)), predictions, yerr=pred_std, fmt='o', alpha=0.5, label="Predicted ± std")
plt.plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.7, label="Actual")
plt.xlabel("Sample Index")
plt.ylabel("Site EUI")
plt.title("Predicted vs Actual EUI with Uncertainty (PyMC 128–64 BNN)")
plt.legend()
plt.show()
# %%
