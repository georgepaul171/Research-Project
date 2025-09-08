#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from V3 import feature_engineering_no_interactions

results_dir = "results_pymc_quick"
summary_file = f"{results_dir}/summary.csv"
trace_file = f"{results_dir}/trace.nc"

def picp_90(y_true, y_pred, y_std):
    z = norm.ppf(1 - 0.1/2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def compute_predictions_from_posterior(idata, X, y):
    """Compute predictions directly from posterior samples"""
    print("Computing predictions from posterior samples...")
    
    # Extract posterior samples
    posterior = idata.posterior
    
    # Get parameter samples with correct names
    intercept_samples = posterior['adaptive_prior::intercept'].values  # shape: (chain, draw)
    sigma_samples = posterior['adaptive_prior::sigma'].values
    energy_coeffs_samples = posterior['adaptive_prior::energy_coeffs'].values  # shape: (chain, draw, 8)
    building_coeffs_samples = posterior['adaptive_prior::building_coeffs'].values  # shape: (chain, draw, 3)
    interaction_coeffs_samples = posterior['adaptive_prior::interaction_coeffs'].values  # shape: (chain, draw, 1)
    
    # Combine coefficients in the correct order
    n_chains, n_draws = intercept_samples.shape
    n_samples = n_chains * n_draws
    
    # Reshape to combine chains and draws
    energy_coeffs_flat = energy_coeffs_samples.reshape(n_samples, 8)
    building_coeffs_flat = building_coeffs_samples.reshape(n_samples, 3)
    interaction_coeffs_flat = interaction_coeffs_samples.reshape(n_samples, 1)
    intercept_flat = intercept_samples.reshape(n_samples)
    
    # Combine all coefficients in the correct order
    all_coeffs = np.concatenate([energy_coeffs_flat, building_coeffs_flat, interaction_coeffs_flat], axis=1)
    feature_order = [0,2,3,4,5,8,10,11,1,6,9,7]
    reordered_coeffs = all_coeffs[:, feature_order]
    
    # Compute predictions for all samples
    y_pred_samples = np.zeros((n_samples, len(y)))
    for i in range(n_samples):
        y_pred_samples[i] = intercept_flat[i] + X @ reordered_coeffs[i]
    
    # Reshape back to (chain, draw, n_obs)
    y_pred_reshaped = y_pred_samples.reshape(n_chains, n_draws, len(y))
    
    # Compute mean and std across samples
    y_pred_mean = np.mean(y_pred_reshaped, axis=(0, 1))
    y_pred_std = np.std(y_pred_reshaped, axis=(0, 1))
    
    return y_pred_mean, y_pred_std, y_pred_reshaped

def main():
    print("\n--- PyMC Results: Complete Analysis ---\n")
    # Load summary
    summary = pd.read_csv(summary_file, index_col=0)
    print(f"Loaded summary with {len(summary)} parameters")
    # Load trace
    idata = az.from_netcdf(trace_file)
    # Load data
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    df = pd.read_csv(data_csv_path)
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
    y = df["site_eui"].values.astype(np.float32).reshape(-1)
    
    # Compute predictions from posterior samples
    y_pred, y_std, y_pred_samples = compute_predictions_from_posterior(idata, X, y)
    
    # Diagnostics
    rhat_values = summary['r_hat'].values
    ess_bulk_values = summary['ess_bulk'].values
    diagnostics = {
        'rhat_max': float(np.max(rhat_values)),
        'rhat_mean': float(np.mean(rhat_values)),
        'ess_min': float(np.min(ess_bulk_values)),
        'ess_mean': float(np.mean(ess_bulk_values)),
        'n_eff': int(np.sum(ess_bulk_values))
    }
    with open(f"{results_dir}/diagnostics.json", 'w') as f:
        json.dump(diagnostics, f, indent=4)
    print(f"Diagnostics saved to {results_dir}/diagnostics.json")
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    picp = picp_90(y, y_pred, y_std)
    overall_metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'picp_90': picp
    }
    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"All metrics saved to {results_dir}/metrics.json")
    
    # Divergences and tree depth
    divergences = 0
    tree_depth_warnings = 0
    if hasattr(idata, 'sample_stats'):
        for chain in idata.sample_stats:
            if 'diverging' in idata.sample_stats[chain]:
                divergences += int(np.sum(idata.sample_stats[chain]['diverging'].values))
            if 'tree_depth' in idata.sample_stats[chain]:
                max_depth = int(np.max(idata.sample_stats[chain]['tree_depth'].values))
                if max_depth >= 10:
                    tree_depth_warnings += 1
    with open(f"{results_dir}/sampling_warnings.json", 'w') as f:
        json.dump({'divergences': divergences, 'tree_depth_warnings': tree_depth_warnings}, f, indent=4)
    print(f"Sampling warnings saved to {results_dir}/sampling_warnings.json")
    
    # Feature importance
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
    # Save feature importance as CSV
    coeff_summary['feature'] = feature_names
    coeff_summary.to_csv(f"{results_dir}/feature_importance.csv")
    
    # Visualizations
    print("Creating visualizations...")
    # 1. Trace plots
    az.plot_trace(idata, var_names=['adaptive_prior::intercept', 'adaptive_prior::sigma', 'adaptive_prior::tau_energy', 'adaptive_prior::sigma_building'])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'trace_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Posterior predictive checks
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Observed')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Posterior Predictive vs Observed')
    
    residuals = y - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residuals Distribution')
    
    # Use the posterior predictive samples for distribution comparison
    y_ppc_samples = y_pred_samples.flatten()
    axes[1, 1].hist(y_ppc_samples, bins=30, alpha=0.7, density=True, label='PPC')
    axes[1, 1].hist(y, bins=30, alpha=0.7, density=True, label='Observed')
    axes[1, 1].set_xlabel('Site EUI')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Posterior Predictive vs Observed Distribution')
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'posterior_predictive_checks.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance
    coeff_means = coeff_summary['mean'].values
    coeff_stds = coeff_summary['sd'].values
    importance_order = np.argsort(np.abs(coeff_means))[::-1]
    coeff_names = feature_names
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(coeff_names))
    ax.barh(y_pos, coeff_means[importance_order], xerr=coeff_stds[importance_order], capsize=5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([coeff_names[i] for i in importance_order])
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance (Posterior Mean ± Std)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualizations saved!")
    
    # Documentation
    doc_md = os.path.join(results_dir, 'visualization_documentation.md')
    with open(doc_md, 'w') as f:
        f.write("""# PyMC Results: Visualization Documentation\n\nThis document describes the visualizations generated by your Bayesian regression analysis using PyMC.\n\n---\n\n## 1. Trace Plots (`trace_plots.png`)\n- **Purpose:** Assess convergence and mixing of MCMC chains for key parameters (e.g., intercept, sigma, tau_energy, sigma_building).\n- **What to look for:**\n  - Chains should overlap and look like 'hairy caterpillars'.\n  - No trends or drifts; good mixing.\n  - R-hat values close to 1 indicate convergence.\n\n---\n\n## 2. Posterior Predictive Checks (`posterior_predictive_checks.png`)\n- **Observed vs. Predicted Scatter:**\n  - **Purpose:** Compare model predictions to actual observed values.\n  - **Interpretation:** Points should cluster around the y=x line. Large deviations may indicate model misspecification.\n- **Residuals vs. Predicted:**\n  - **Purpose:** Check for systematic errors in predictions.\n  - **Interpretation:** Residuals should be randomly scattered around zero. Patterns may indicate bias.\n- **Residuals Histogram:**\n  - **Purpose:** Assess the distribution of errors.\n  - **Interpretation:** Should be roughly normal and centered at zero.\n- **PPC vs. Observed Histogram:**\n  - **Purpose:** Compare the distribution of model predictions to observed data.\n  - **Interpretation:** Overlap indicates good model fit. Large discrepancies suggest model issues.\n\n---\n\n## 3. Feature Importance (`feature_importance.png`)\n- **Purpose:** Show the posterior mean and standard deviation (uncertainty) for each regression coefficient.\n- **Interpretation:**\n  - Large magnitude (positive or negative) = more important feature.\n  - Wide error bars = more uncertainty in the effect estimate.\n\n---\n\n## 4. Additional Notes\n- **Divergences and Tree Depth Warnings:**\n  - If present, these are reported in `sampling_warnings.json`.\n  - Divergences may indicate model geometry issues; consider increasing `target_accept` or reparameterizing.\n  - Tree depth warnings suggest the sampler is hitting its maximum allowed depth; consider increasing `max_treedepth`.\n\n---\n\n**For best results:**\n- Use trace plots and R-hat/ESS diagnostics to confirm convergence.\n- Use posterior predictive checks to validate model fit.\n- Use feature importance plots to interpret model drivers.\n\nAll plots are saved as PNGs in this results folder.\n""")
    print(f"\nAnalysis complete! Results saved in {results_dir}/\n")

if __name__ == "__main__":
    main() 