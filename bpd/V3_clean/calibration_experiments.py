import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from V3 import AdaptivePriorARD, AdaptivePriorConfig, feature_engineering_no_interactions
from scipy.optimize import minimize_scalar

# --- Load data and features (same as V3.py) ---
data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
target = "site_eui"
na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
df = feature_engineering_no_interactions(df)
features = [
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
    "ghg_per_area"
]
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1)
feature_names = features.copy()

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_calibration_experiments')
os.makedirs(results_dir, exist_ok=True)

# --- Helper: Calibration curve ---
def calibration_curve(y_true, y_pred, y_std, coverages=[0.5, 0.8, 0.9, 0.95, 0.99], scale=1.0):
    from scipy.stats import norm
    empirical = []
    for c in coverages:
        z = norm.ppf(1 - (1-c)/2)
        lower = y_pred - z * y_std * scale
        upper = y_pred + z * y_std * scale
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        empirical.append(coverage)
    return coverages, empirical

def plot_calibration(coverages, empirical, label, outpath):
    plt.figure(figsize=(8,5))
    plt.plot(coverages, empirical, 'o-', label=label)
    plt.plot([0,1],[0,1],'r--',label='Perfect Calibration')
    plt.xlabel('Nominal Coverage')
    plt.ylabel('Empirical Coverage')
    plt.title('Uncertainty Calibration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# --- 1. Baseline: No calibration ---
config_nocal = AdaptivePriorConfig(
    beta_0=10.0,
    group_sparsity=False,
    dynamic_shrinkage=False,
    hmc_steps=20,
    hmc_leapfrog_steps=3,
    hmc_epsilon=0.0001,
    max_iter=1000,
    tol=1e-8,
    use_hmc=False,
    robust_noise=False,
    uncertainty_calibration=False,
    group_prior_types={
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    }
)
model_nocal = AdaptivePriorARD(config_nocal)
model_nocal.fit(X, y, feature_names=feature_names, output_dir=results_dir)
y_pred, y_std = model_nocal.predict(X, return_std=True)
coverages, empirical = calibration_curve(y, y_pred, y_std)
plot_calibration(coverages, empirical, 'No Calibration', os.path.join(results_dir, 'calibration_nocal.png'))

# --- 2. Built-in model calibration ---
config_cal = AdaptivePriorConfig(
    beta_0=10.0,
    group_sparsity=False,
    dynamic_shrinkage=False,
    hmc_steps=20,
    hmc_leapfrog_steps=3,
    hmc_epsilon=0.0001,
    max_iter=1000,
    tol=1e-8,
    use_hmc=False,
    robust_noise=False,
    uncertainty_calibration=True,
    group_prior_types={
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    }
)
model_cal = AdaptivePriorARD(config_cal)
model_cal.fit(X, y, feature_names=feature_names, output_dir=results_dir)
y_pred_cal, y_std_cal = model_cal.predict(X, return_std=True)
coverages_cal, empirical_cal = calibration_curve(y, y_pred_cal, y_std_cal)
plot_calibration(coverages_cal, empirical_cal, 'Model Calibration', os.path.join(results_dir, 'calibration_modelcal.png'))

# --- 3. Post-hoc scaling calibration ---
def calibration_loss(scale):
    _, empirical = calibration_curve(y, y_pred, y_std, scale=scale)
    return np.mean((np.array(empirical) - np.array(coverages))**2)
res = minimize_scalar(calibration_loss, bounds=(0.1, 10), method='bounded')
optimal_scale = res.x
coverages_post, empirical_post = calibration_curve(y, y_pred, y_std, scale=optimal_scale)
plot_calibration(coverages_post, empirical_post, f'Post-hoc Scale={optimal_scale:.2f}', os.path.join(results_dir, 'calibration_posthoc.png'))

with open(os.path.join(results_dir, 'posthoc_scale.txt'), 'w') as f:
    f.write(f'Optimal post-hoc scale: {optimal_scale}\n')

# --- Summary ---
print("Calibration experiments complete. See results_calibration_experiments for plots and optimal scaling factor.")
print(f"Optimal post-hoc scale: {optimal_scale}") 