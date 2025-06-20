import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import norm
from AREHap_groupprior import AdaptivePriorARD, AdaptivePriorConfig

# --- Feature engineering function (copied from AREHap_groupprior.py) ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    df['building_age'] = 2025 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    df['age_energy_star_interaction'] = df['building_age_log'] * df['energy_star_rating_normalized']
    df['area_energy_star_interaction'] = df['floor_area_log'] * df['energy_star_rating_normalized']
    df['age_ghg_interaction'] = df['building_age_log'] * df['ghg_emissions_int_log']
    return df

# Paths
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
AEH_METRICS_PATH = os.path.join(RESULTS_DIR, 'results_groupprior', 'metrics.json')
DATA_PATH = os.path.join(RESULTS_DIR, '..', 'cleaned_office_buildings.csv')

FEATURES = [
    "ghg_emissions_int_log",
    "floor_area_log",
    "electric_eui",
    "fuel_eui",
    "energy_star_rating_normalized",
    "energy_mix",
    "building_age_log",
    "floor_area_squared",
    "energy_intensity_ratio",
    "building_age_squared",
    "energy_star_rating_squared",
    "ghg_per_area",
    "age_energy_star_interaction",
    "area_energy_star_interaction",
    "age_ghg_interaction"
]
TARGET = "site_eui"

# Helper functions
def picp(y_true, y_pred, y_std, level=0.9):
    z = norm.ppf(1 - (1 - level) / 2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return np.mean((y_true >= lower) & (y_true <= upper))

def crps_gaussian(y_true, y_pred, y_std):
    # Approximate CRPS for Gaussian predictive distribution
    return np.mean(np.abs(y_pred - y_true)) - 0.5 * np.mean(np.abs(y_std))

def evaluate_model(model, X, y, n_splits=3, random_state=42, is_bayesian=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses, maes, r2s, crpss, picps = [], [], [], [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        if is_bayesian:
            y_pred, y_std = model.predict(X_test, return_std=True)
        else:
            y_pred = model.predict(X_test)
            y_std = np.std(y_train - model.predict(X_train)) * np.ones_like(y_pred)
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
        crpss.append(crps_gaussian(y_test, y_pred, y_std))
        picps.append(picp(y_test, y_pred, y_std, level=0.9))
    return {
        "rmse": np.mean(rmses),
        "mae": np.mean(maes),
        "r2": np.mean(r2s),
        "crps": np.mean(crpss),
        "picp_90": np.mean(picps)
    }

# Load and engineer data
df = pd.read_csv(DATA_PATH)
df = feature_engineering(df)
df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# Configurations for Bayesian priors
config_hierarchical = AdaptivePriorConfig(prior_type='hierarchical', group_prior_types={'energy': 'hierarchical'})
config_horseshoe = AdaptivePriorConfig(prior_type='horseshoe', group_prior_types={'energy': 'horseshoe'})
config_spike_slab = AdaptivePriorConfig(prior_type='spike_slab', group_prior_types={'energy': 'spike_slab'})

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "Hierarchical ARD": AdaptivePriorARD(config_hierarchical),
    "Horseshoe": AdaptivePriorARD(config_horseshoe),
    "Spike-and-Slab": AdaptivePriorARD(config_spike_slab)
}

# Evaluate baselines
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    is_bayesian = isinstance(model, AdaptivePriorARD)
    results[name] = evaluate_model(model, X, y, is_bayesian=is_bayesian)

# AEH metrics
with open(AEH_METRICS_PATH, "r") as f:
    aeh_metrics = json.load(f)
results["AEH (Ours)"] = {
    "rmse": aeh_metrics.get("rmse", np.nan),
    "mae": aeh_metrics.get("mae", np.nan),
    "r2": aeh_metrics.get("r2", np.nan),
    "crps": aeh_metrics.get("crps", np.nan),
    "picp_90": aeh_metrics.get("picp_90", np.nan)
}

# Save table as CSV
df_results = pd.DataFrame(results).T[["rmse", "mae", "r2", "crps", "picp_90"]]
output_csv = os.path.join(RESULTS_DIR, 'results_groupprior', 'baseline_comparison.csv')
df_results.to_csv(output_csv)
print(f"Saved comparison table to {output_csv}")

# Plot bar chart for each metric
import matplotlib
matplotlib.use('Agg')
fig, axes = plt.subplots(1, 5, figsize=(24, 6), sharey=False)
metrics = ["rmse", "mae", "r2", "crps", "picp_90"]
for i, metric in enumerate(metrics):
    ax = axes[i]
    df_results[metric].plot(kind='bar', ax=ax, color=['#1f77b4' if m != 'AEH (Ours)' else '#ff7f0e' for m in df_results.index])
    ax.set_title(metric.upper())
    ax.set_ylabel(metric.upper())
    ax.set_xticklabels(df_results.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
output_png = os.path.join(RESULTS_DIR, 'results_groupprior', 'baseline_comparison.png')
plt.savefig(output_png, dpi=300)
print(f"Saved comparison bar chart to {output_png}") 