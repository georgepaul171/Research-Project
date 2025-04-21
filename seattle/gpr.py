import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define columns
numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "SteamUse(kBtu)", "GHGEmissionsIntensity", "SiteEUI(kBtu/sf)"
]

target_col = "SiteEUI(kBtu/sf)"

# Load and clean
df = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/seattle-data-cleaned.csv")[numeric_columns]
df_clean = df.apply(pd.to_numeric, errors="coerce").dropna()

# Split into features/target
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

# Build GP model
with pm.Model() as gp_model:
    X_data = pm.Data("X_data", X_train_scaled)
    y_data = pm.Data("y_data", y_train_np)

    r = pm.Gamma("r", alpha=2, beta=1)
    n = pm.HalfNormal("n", sigma=1)
    cov = n**2 * pm.gp.cov.ExpQuad(input_dim=X_train.shape[1], ls=r)

    gp = pm.gp.Marginal(cov_func=cov)
    sigma = pm.HalfNormal("sigma", sigma=1)

    y_ = gp.marginal_likelihood("y", X=X_data, y=y_data, noise=sigma)

    trace = pm.sample(1000, tune=1000, target_accept=0.9, chains=1, cores= 1, random_seed=42)

# Predict on test data
with gp_model:
    f_pred = gp.conditional("f_pred", X_test_scaled)
    posterior_pred = pm.sample_posterior_predictive(
        trace,
        var_names=["f_pred"],
        predictions=True,
        random_seed=42
    )


f_samples = posterior_pred.predictions["f_pred"].values
print("y_test_np shape:", y_test_np.shape)
print("f_samples shape:", f_samples.shape)

# Collapse (chain, draw) into one axis
f_samples = f_samples.reshape(-1, f_samples.shape[-1])  # shape: (1000, 134)

mu_pred = f_samples.mean(axis=0)  # → shape: (134,)
std_pred = f_samples.std(axis=0)
# Evaluation
r2 = r2_score(y_test_np, mu_pred)
rmse = np.sqrt(np.mean((y_test_np - mu_pred) ** 2))

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.errorbar(range(len(mu_pred)), mu_pred, yerr=std_pred, fmt='o', alpha=0.5, label="Predicted ± std")
plt.plot(range(len(y_test_np)), y_test_np, 'k.', alpha=0.6, label="Actual")
plt.xlabel("Sample Index")
plt.ylabel("Site EUI")
plt.title("Bayesian Gaussian Process Regression: Predicted vs Actual with Uncertainty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# python gpr.py

# Initializing NUTS using jitter+adapt_diag...
# Sequential sampling (1 chains in 1 job)
# NUTS: [r, n, sigma]
                                                                                                                                  
#   Progress                                   Draws   Divergences   Step size   Grad evals   Sampling Speed   Elapsed   Remaining  
#  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
#   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   2000    0             0.62        3            33.30 draws/s    0:01:00   0:00:00    
                                                                                                                                  
# Sampling 1 chain for 1_000 tune and 1_000 draw iterations (1_000 + 1_000 draws total) took 60 seconds.
# Only one chain was sampled, this makes it impossible to run some convergence checks
# Sampling: [f_pred]
# Sampling ... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 / 0:00:01
# y_test_np shape: (134,)
# f_samples shape: (1, 1000, 134)
# R² Score: 0.893
# RMSE: 12.86