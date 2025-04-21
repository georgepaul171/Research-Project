import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data
X_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_train.csv")
X_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_test.csv")
y_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_train.csv").squeeze()
y_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_test.csv").squeeze()

# Clean and preprocess
X_train = X_train.apply(pd.to_numeric, errors="coerce").dropna()
X_test = X_test.apply(pd.to_numeric, errors="coerce").dropna()
y_train = pd.to_numeric(y_train, errors="coerce")[:len(X_train)]
y_test = pd.to_numeric(y_test, errors="coerce")[:len(X_test)]

# Standardize features for GP (helps convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

# GP Model
with pm.Model() as gp_model:
    X_data = pm.Data("X_data", X_train_scaled)
    y_data = pm.Data("y_data", y_train_np)

    # Define covariance function (kernel)
    r = pm.Gamma("r", alpha=2, beta=1)
    n = pm.HalfNormal("n", sigma=1)
    cov = n**2 * pm.gp.cov.ExpQuad(input_dim=X_train.shape[1], ls=r)

    # GP prior
    gp = pm.gp.Marginal(cov_func=cov)
    sigma = pm.HalfNormal("sigma", sigma=1)

    y_ = gp.marginal_likelihood("y", X=X_data, y=y_data, noise=sigma)

    trace = pm.sample(1000, tune=1000, target_accept=0.9, chains=2, random_seed=42)

# Predictive mean and uncertainty
with gp_model:
    pm.set_data({"X_data": X_test_scaled})
    mu_pred, var_pred = gp.predict(X_test_scaled, point=trace.posterior.mean(dim=["chain", "draw"]).to_dict(), diag=True, pred_noise=True)
    std_pred = np.sqrt(var_pred)

# Evaluation
r2 = r2_score(y_test_np, mu_pred)
rmse = np.sqrt(y_test_np, mu_pred, squared=False)**0.5

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# Plot predictions with uncertainty
plt.figure(figsize=(10, 5))
plt.errorbar(range(len(mu_pred)), mu_pred, yerr=std_pred, fmt='o', alpha=0.5, label="Predicted ± std")
plt.plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.6, label="Actual")
plt.xlabel("Sample Index")
plt.ylabel("Site EUI")
plt.title("Bayesian Gaussian Process Regression: Predicted vs Actual with Uncertainty")
plt.legend()
plt.grid(True)
plt.show()