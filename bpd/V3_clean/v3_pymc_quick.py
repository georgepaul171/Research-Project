#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from V3 import feature_engineering_no_interactions

# Set up paths and parameters
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_pymc_quick')
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

# Define feature groups for hierarchical priors
energy_features = [0, 2, 3, 4, 5, 8, 10, 11]  # energy-related features
building_features = [1, 6, 9]  # building-related features
interaction_features = [7]  # interaction features

print(f"Data shape: {X.shape}")
print(f"Energy features: {len(energy_features)}")
print(f"Building features: {len(building_features)}")
print(f"Interaction features: {len(interaction_features)}")

# Helper to compute PICP 90%
def picp_90(y_true, y_pred, y_std):
    z = norm.ppf(1 - 0.1/2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def create_simplified_adaptive_prior_model(X, y, feature_names, model_name="adaptive_prior"):
    """
    Create a simplified PyMC model with group-specific priors.
    """
    print("Creating PyMC model...")
    
    with pm.Model(name=model_name) as model:
        # Global scale for the model
        sigma = pm.HalfNormal('sigma', sigma=1.0)
        
        # Energy group: Horseshoe-like prior (simplified)
        tau_energy = pm.HalfCauchy('tau_energy', beta=1.0)
        lambda_energy = pm.HalfCauchy('lambda_energy', beta=1.0, shape=len(energy_features))
        energy_coeffs = pm.Normal('energy_coeffs', 
                                mu=0, 
                                sigma=tau_energy * lambda_energy,
                                shape=len(energy_features))
        
        # Building group: Hierarchical prior
        sigma_building = pm.HalfNormal('sigma_building', sigma=1.0)
        building_coeffs = pm.Normal('building_coeffs', 
                                  mu=0, 
                                  sigma=sigma_building,
                                  shape=len(building_features))
        
        # Interaction group: Hierarchical prior
        sigma_interaction = pm.HalfNormal('sigma_interaction', sigma=1.0)
        interaction_coeffs = pm.Normal('interaction_coeffs', 
                                     mu=0, 
                                     sigma=sigma_interaction,
                                     shape=len(interaction_features))
        
        # Intercept
        intercept = pm.Normal('intercept', mu=0, sigma=10.0)
        
        # Combine coefficients in the right order
        all_coeffs = pm.Deterministic('all_coeffs', 
                                    pm.math.concatenate([
                                        energy_coeffs,
                                        building_coeffs, 
                                        interaction_coeffs
                                    ]))
        
        # Reorder to match original feature order
        feature_order = energy_features + building_features + interaction_features
        reordered_coeffs = pm.Deterministic('reordered_coeffs', 
                                          all_coeffs[feature_order])
        
        # Linear predictor
        mu = intercept + pm.math.dot(X, reordered_coeffs)
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)
    
    print("Model created successfully!")
    return model

def run_bayesian_inference(model, X, y, feature_names, results_dir):
    """
    Run Bayesian inference with NUTS sampling - QUICK VERSION.
    """
    print("Running Bayesian inference with NUTS (quick version)...")
    
    # Run NUTS sampling with fewer draws for speed
    with model:
        trace = pm.sample(
            draws=500,   # HALF the original (was 1000)
            tune=500,    # HALF the original (was 1000)
            chains=4,
            cores=1,     # Use single core to avoid issues
            return_inferencedata=True,
            random_seed=42,
            target_accept=0.8,
            max_treedepth=10
        )
    
    print("Sampling completed!")
    
    # Save trace
    trace_file = os.path.join(results_dir, 'trace.nc')
    trace.to_netcdf(trace_file)
    print(f"Trace saved to {trace_file}")
    
    # Comprehensive diagnostics
    print("\nComputing diagnostics...")
    
    # ArviZ diagnostics
    summary = az.summary(trace, round_to=4)
    summary.to_csv(os.path.join(results_dir, 'summary.csv'))
    
    # R-hat and ESS
    rhat = az.rhat(trace)
    ess = az.ess(trace)
    
    # Save diagnostics
    diagnostics = {
        'rhat_max': float(rhat.max()),
        'rhat_mean': float(rhat.mean()),
        'ess_min': float(ess.min()),
        'ess_mean': float(ess.mean()),
        'n_eff': int(ess.sum())
    }
    
    with open(os.path.join(results_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diagnostics, f, indent=4)
    
    # Posterior predictive checks
    print("Computing posterior predictive samples...")
    with model:
        ppc = pm.sample_posterior_predictive(trace, samples=250, random_seed=42)  # HALF the samples
    
    # Compute predictions and uncertainty
    y_pred = ppc.posterior_predictive['likelihood'].mean(dim=('chain', 'draw')).values
    y_std = ppc.posterior_predictive['likelihood'].std(dim=('chain', 'draw')).values
    
    # Metrics
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    picp = picp_90(y, y_pred, y_std)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'picp_90': picp
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model Performance: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}, PICP90={picp:.3f}")
    print(f"Diagnostics: R-hat max={diagnostics['rhat_max']:.3f}, ESS min={diagnostics['ess_min']:.0f}")
    
    return trace, ppc, metrics, diagnostics

def create_visualizations(trace, ppc, X, y, feature_names, results_dir):
    """
    Create comprehensive visualizations for the Bayesian model.
    """
    print("Creating visualizations...")
    
    # 1. Trace plots for key parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    az.plot_trace(trace, var_names=['intercept', 'sigma', 'tau_energy', 'sigma_building'], 
                  axes=axes, compact=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'trace_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Posterior predictive checks
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # PPC scatter
    y_pred = ppc.posterior_predictive['likelihood'].mean(dim=('chain', 'draw')).values
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Observed')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Posterior Predictive vs Observed')
    
    # Residuals
    residuals = y - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    
    # Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residuals Distribution')
    
    # PPC histogram
    y_ppc_samples = ppc.posterior_predictive['likelihood'].values.flatten()
    axes[1, 1].hist(y_ppc_samples, bins=30, alpha=0.7, density=True, label='PPC')
    axes[1, 1].hist(y, bins=30, alpha=0.7, density=True, label='Observed')
    axes[1, 1].set_xlabel('Site EUI')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Posterior Predictive vs Observed Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'posterior_predictive_checks.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance (coefficient posterior means and intervals)
    coeff_names = ['intercept'] + feature_names
    coeff_samples = np.column_stack([
        trace.posterior['intercept'].values.flatten(),
        trace.posterior['reordered_coeffs'].values.reshape(-1, len(feature_names))
    ])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    coeff_means = coeff_samples.mean(axis=0)
    coeff_stds = coeff_samples.std(axis=0)
    
    # Sort by absolute importance
    importance_order = np.argsort(np.abs(coeff_means))[::-1]
    
    y_pos = np.arange(len(coeff_names))
    ax.barh(y_pos, coeff_means[importance_order], 
            xerr=coeff_stds[importance_order], 
            capsize=5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([coeff_names[i] for i in importance_order])
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance (Posterior Mean ± Std)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved!")

def main():
    """
    Main function to run the quick Bayesian analysis.
    """
    print("Starting PyMC Bayesian Analysis (QUICK VERSION) for Adaptive Prior ARD...")
    
    try:
        # Create the model
        model = create_simplified_adaptive_prior_model(X, y, feature_names)
        
        # Run inference
        trace, ppc, metrics, diagnostics = run_bayesian_inference(model, X, y, feature_names, results_dir)
        
        # Create visualizations
        create_visualizations(trace, ppc, X, y, feature_names, results_dir)
        
        # Save final summary
        summary_text = f"""
PyMC Bayesian Analysis Results (QUICK VERSION)
=============================================

Model Performance:
- R²: {metrics['r2']:.3f}
- RMSE: {metrics['rmse']:.2f}
- MAE: {metrics['mae']:.2f}
- PICP90: {metrics['picp_90']:.3f}

Sampling Diagnostics:
- R-hat max: {diagnostics['rhat_max']:.3f}
- R-hat mean: {diagnostics['rhat_mean']:.3f}
- ESS min: {diagnostics['ess_min']:.0f}
- ESS mean: {diagnostics['ess_mean']:.0f}
- Total effective samples: {diagnostics['n_eff']}

Sampling Configuration (QUICK):
- Chains: 4
- Draws per chain: 500 (reduced from 1000)
- Tuning steps: 500 (reduced from 1000)
- Sampler: NUTS with adaptive tuning

Files saved:
- trace.nc: Full posterior samples
- summary.csv: Parameter summaries
- diagnostics.json: Convergence diagnostics
- metrics.json: Model performance metrics
- Various visualization plots
"""
        
        with open(os.path.join(results_dir, 'analysis_summary.md'), 'w') as f:
            f.write(summary_text)
        
        print("\n" + summary_text)
        print(f"\nAll results saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 