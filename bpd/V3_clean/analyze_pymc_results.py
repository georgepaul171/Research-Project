#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import arviz as az
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import xarray as xr

# Load the results
results_dir = "results_pymc_quick"
summary_file = f"{results_dir}/summary.csv"
trace_file = f"{results_dir}/trace.nc"

print("Analyzing PyMC Results...")

# Load summary
summary = pd.read_csv(summary_file, index_col=0)
print(f"Loaded summary with {len(summary)} parameters")

# Load trace
idata = az.from_netcdf(trace_file)
trace = idata
print(f"Loaded trace with shape: {trace.posterior.dims}")

# Extract key diagnostics
rhat_values = summary['r_hat'].values
ess_bulk_values = summary['ess_bulk'].values
ess_tail_values = summary['ess_tail'].values

diagnostics = {
    'rhat_max': float(np.max(rhat_values)),
    'rhat_mean': float(np.mean(rhat_values)),
    'ess_min': float(np.min(ess_bulk_values)),
    'ess_mean': float(np.mean(ess_bulk_values)),
    'n_eff': int(np.sum(ess_bulk_values))
}

print(f"\nConvergence Diagnostics:")
print(f"R-hat max: {diagnostics['rhat_max']:.3f}")
print(f"R-hat mean: {diagnostics['rhat_mean']:.3f}")
print(f"ESS min: {diagnostics['ess_min']:.0f}")
print(f"ESS mean: {diagnostics['ess_mean']:.0f}")
print(f"Total effective samples: {diagnostics['n_eff']}")

# Extract coefficient results
coeff_params = [col for col in summary.index if 'reordered_coeffs' in col]
coeff_summary = summary.loc[coeff_params]

feature_names = [
    "ghg_emissions_int_log",
    "electric_eui", 
    "fuel_eui",
    "energy_star_rating_normalized",
    "floor_area_log",
    "building_age_log",
    "energy_intensity_ratio",
    "ghg_per_area",
    "energy_mix",
    "energy_star_rating_squared",
    "floor_area_squared",
    "building_age_squared"
]

print(f"\nFeature Importance (Posterior Mean ± Std):")
for i, (param, feature) in enumerate(zip(coeff_params, feature_names)):
    mean_val = coeff_summary.loc[param, 'mean']
    std_val = coeff_summary.loc[param, 'sd']
    rhat_val = coeff_summary.loc[param, 'r_hat']
    ess_val = coeff_summary.loc[param, 'ess_bulk']
    print(f"{i+1:2d}. {feature:30s}: {mean_val:8.3f} ± {std_val:6.3f} (R-hat: {rhat_val:.3f}, ESS: {ess_val:.0f})")

# Extract key model parameters
intercept = summary.loc['adaptive_prior::intercept', 'mean']
sigma = summary.loc['adaptive_prior::sigma', 'mean']
tau_energy = summary.loc['adaptive_prior::tau_energy', 'mean']

print(f"\nKey Model Parameters:")
print(f"Intercept: {intercept:.3f}")
print(f"Sigma (noise): {sigma:.3f}")
print(f"Energy group shrinkage (tau_energy): {tau_energy:.3f}")

# Save diagnostics
with open(f"{results_dir}/diagnostics.json", 'w') as f:
    json.dump(diagnostics, f, indent=4)
print(f"Diagnostics saved to {results_dir}/diagnostics.json")

# --- Additional Metrics ---
# Load y_true from data
print("\nCalculating additional model fit metrics...")
data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
df = pd.read_csv(data_csv_path)
from V3 import feature_engineering_no_interactions
df = feature_engineering_no_interactions(df)
y_true = df['site_eui'].values.astype(float)

# Posterior predictive mean and std
has_ppc = hasattr(trace, 'posterior_predictive') and hasattr(trace.posterior_predictive, 'likelihood')
if has_ppc:
    y_pred = trace.posterior_predictive['likelihood'].mean(dim=('chain', 'draw')).values
    y_std = trace.posterior_predictive['likelihood'].std(dim=('chain', 'draw')).values
else:
    y_pred = None
    y_std = None
    print("WARNING: No posterior predictive samples found in trace.nc. Metrics requiring predictions will be skipped.")

# Metrics
if y_pred is not None and y_std is not None:
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    def picp_90(y_true, y_pred, y_std):
        z = norm.ppf(1 - 0.1/2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        return float(np.mean((y_true >= lower) & (y_true <= upper)))
    picp = picp_90(y_true, y_pred, y_std)
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"PICP90: {picp:.3f}")
    overall_metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'picp_90': picp
    }
else:
    overall_metrics = {
        'r2': None,
        'rmse': None,
        'mae': None,
        'picp_90': None,
        'note': 'Posterior predictive samples not found in trace.nc. Metrics requiring predictions are unavailable.'
    }
    print("R², RMSE, MAE, PICP90: Not available (no posterior predictive samples found)")

with open(f"{results_dir}/metrics.json", 'w') as f:
    json.dump(overall_metrics, f, indent=4)
print(f"All metrics saved to {results_dir}/metrics.json")

# --- Divergences and Tree Depth Warnings ---
print("\nChecking for divergences and tree depth warnings...")
divergences = 0
tree_depth_warnings = 0
if hasattr(trace, 'sample_stats'):
    for chain in trace.sample_stats:
        if 'diverging' in trace.sample_stats[chain]:
            divergences += int(np.sum(trace.sample_stats[chain]['diverging'].values))
        if 'tree_depth' in trace.sample_stats[chain]:
            max_depth = int(np.max(trace.sample_stats[chain]['tree_depth'].values))
            if max_depth >= 10:
                tree_depth_warnings += 1
print(f"Divergences after tuning: {divergences}")
print(f"Chains hitting max tree depth: {tree_depth_warnings}")

# Save to a summary file
with open(f"{results_dir}/sampling_warnings.json", 'w') as f:
    json.dump({'divergences': divergences, 'tree_depth_warnings': tree_depth_warnings}, f, indent=4)
print(f"Sampling warnings saved to {results_dir}/sampling_warnings.json")

print(f"\nAnalysis complete! Results saved in {results_dir}/") 