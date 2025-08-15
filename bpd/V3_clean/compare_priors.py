import os
import json
import numpy as np
import pandas as pd
from V3 import AdaptivePriorARD, AdaptivePriorConfig, feature_engineering_no_interactions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

# Set up paths and parameters
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_compare_priors')
os.makedirs(results_dir, exist_ok=True)
data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
target = "site_eui"
na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']

# Load and preprocess data
print("Loading data...")
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
feature_names = features.copy()
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1)

# Helper to compute PICP 90%
def picp_90(y_true, y_pred, y_std):
    z = norm.ppf(1 - 0.1/2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

# Define prior configs to compare
prior_configs = {
    'AEH': {
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    },
    'Horseshoe': {
        'energy': 'horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    },
    'ElasticNet': {
        'energy': 'elastic_net',  
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    }
}

results = {}

for prior_name, group_priors in prior_configs.items():
    print(f"\nRunning model with {prior_name} prior for energy group...")
    
    # For Elastic Net, use AEH with alpha=1.0, beta=1.0, gamma=0.0, rho=0.0 (no horseshoe, pure elastic net)
    if prior_name == 'ElasticNet':
        config = AdaptivePriorConfig(
            group_prior_types={
                'energy': 'adaptive_elastic_horseshoe',
                'building': 'hierarchical',
                'interaction': 'hierarchical'
            },
            max_iter=50,
            use_hmc=False
        )
        model = AdaptivePriorARD(config)
        # Patch AEH parameters for pure elastic net
        def patch_aeh_to_elastic_net(model):

            if hasattr(model, 'group_prior_hyperparams') and 'energy' in model.group_prior_hyperparams:
                params = model.group_prior_hyperparams['energy']
                params['alpha'] = 1.0  # L1 only
                params['beta'] = 1.0  # Elastic net only
                params['gamma'] = 0.0  # No adaptation
                params['rho'] = 0.0  # No momentum
        # Fit model and patch after initialisation
        model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
        patch_aeh_to_elastic_net(model)
    else:
        config = AdaptivePriorConfig(
            group_prior_types=group_priors,
            max_iter=50,
            use_hmc=False
        )
        model = AdaptivePriorARD(config)
        model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    # Get predictions and uncertainty
    y_pred, y_std = model.predict(X, return_std=True)
    # Compute metrics
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    picp = picp_90(y, y_pred, y_std)
    results[prior_name] = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'picp_90': picp
    }
    print(f"{prior_name} results: R2={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}, PICP90={picp:.2f}")

# Save results
with open(os.path.join(results_dir, 'compare_priors_results.json'), 'w') as f:
    json.dump(results, f, indent=4)
print("\nAll prior comparison results saved to compare_priors_results.json.") 