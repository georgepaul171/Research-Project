import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.preprocessing import StandardScaler, RobustScaler

# Import the model from V3_clean
from V3 import AdaptivePriorARD, AdaptivePriorConfig, feature_engineering_no_interactions

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

# --- Results directory ---
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_debug_v3_intervals')
os.makedirs(results_dir, exist_ok=True)

# --- Helper: Fit and analyze model ---
def fit_and_analyze(config, label):
    print(f"\n--- Fitting model: {label} ---")
    model = AdaptivePriorARD(config)
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    y_pred, y_std = model.predict(X, return_std=True)
    print(f"alpha: {model.alpha}")
    print(f"beta (mean, min, max): {np.mean(model.beta)}, {np.min(model.beta)}, {np.max(model.beta)}")
    print(f"weights (mean, min, max): {np.mean(model.m)}, {np.min(model.m)}, {np.max(model.m)}")
    print(f"y_std (mean, min, max): {np.mean(y_std)}, {np.min(y_std)}, {np.max(y_std)}")
    # Save stats
    stats = {
        'alpha': float(model.alpha),
        'beta_mean': float(np.mean(model.beta)),
        'beta_min': float(np.min(model.beta)),
        'beta_max': float(np.max(model.beta)),
        'weights_mean': float(np.mean(model.m)),
        'weights_min': float(np.min(model.m)),
        'weights_max': float(np.max(model.m)),
        'y_std_mean': float(np.mean(y_std)),
        'y_std_min': float(np.min(y_std)),
        'y_std_max': float(np.max(y_std)),
    }
    with open(os.path.join(results_dir, f'stats_{label}.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    # Plot histogram of y_std
    plt.figure(figsize=(8, 5))
    plt.hist(y_std, bins=50, alpha=0.7)
    plt.xlabel('Predicted Uncertainty (std)')
    plt.ylabel('Count')
    plt.title(f'Histogram of Prediction Uncertainty (y_std) - {label}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'y_std_hist_{label}.png'))
    plt.close()
    return model, y_pred, y_std

# --- 1. Baseline: Use config from V3.py ---
config_baseline = AdaptivePriorConfig(
    beta_0=1.0,
    group_sparsity=False,
    dynamic_shrinkage=False,
    hmc_steps=20,
    hmc_leapfrog_steps=3,
    hmc_epsilon=0.0001,
    max_iter=1000,
    tol=1e-8,
    use_hmc=False,
    robust_noise=False,
    group_prior_types={
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    }
)
fit_and_analyze(config_baseline, 'baseline')

# --- 2. Experiment: Increase beta_0 (weaker prior) ---
config_beta10 = AdaptivePriorConfig(
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
    group_prior_types={
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    }
)
fit_and_analyze(config_beta10, 'beta10')

# --- 3. Experiment: Disable uncertainty calibration ---
config_nocal = AdaptivePriorConfig(
    beta_0=1.0,
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
fit_and_analyze(config_nocal, 'nocalibration')

# --- 4. Experiment: Use all hierarchical priors (no AEH) ---
config_allhier = AdaptivePriorConfig(
    beta_0=1.0,
    group_sparsity=False,
    dynamic_shrinkage=False,
    hmc_steps=20,
    hmc_leapfrog_steps=3,
    hmc_epsilon=0.0001,
    max_iter=1000,
    tol=1e-8,
    use_hmc=False,
    robust_noise=False,
    group_prior_types={
        'energy': 'hierarchical',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    }
)
fit_and_analyze(config_allhier, 'allhierarchical')

print("\nCheck the results_debug_v3_intervals folder for stats and plots. Compare y_std histograms and stats to see which config reduces the excessive uncertainty.") 