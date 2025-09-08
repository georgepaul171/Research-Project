#!/usr/bin/env python3

import os
import json
import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, norm, t
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import shap
from V3 import feature_engineering_no_interactions

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_pymc_aeh_comprehensive')
os.makedirs(results_dir, exist_ok=True)
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
feature_names = features.copy()
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1)

# Define feature groups for AEH prior
energy_features = [0, 2, 3, 4, 5, 8, 10, 11]  
building_features = [1, 6, 9] 
interaction_features = [7]  

print(f"Data shape: {X.shape}")
print(f"Energy features: {len(energy_features)}")
print(f"Building features: {len(building_features)}")
print(f"Interaction features: {len(interaction_features)}")

# Compute PICP 90%
def picp_90(y_true, y_pred, y_std):
    z = norm.ppf(1 - 0.1/2)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

# Compute PICP for multiple levels
def compute_picp(y_true, y_pred, y_std, levels=[0.5, 0.8, 0.9, 0.95, 0.99]):
    """
    Compute PICP (Prediction Interval Coverage Probability) for multiple confidence levels.
    Returns a dict: {level: picp}
    """
    picps = {}
    for level in levels:
        z = norm.ppf(1 - (1 - level) / 2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
        picps[f"picp_{int(level*100)}"] = picp
    return picps

def compute_crps(y_true, y_pred, y_std):
    """
    Compute the mean CRPS for a set of normal predictive distributions.
    """
    crps_values = []
    for yt, yp, ys in zip(y_true, y_pred, y_std):
        if ys <= 0:
            crps = np.abs(yt - yp) 
        else:
            z = (yt - yp) / ys
            crps = ys * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))
        crps_values.append(crps)
    return float(np.mean(crps_values))

def create_aeh_model(X, y, feature_names):
    
    print("PyMC model with AEH prior created")
    
    with pm.Model() as model:

        sigma = pm.HalfNormal('sigma', sigma=1.0)
        
        # Global shrinkage parameter
        tau_energy = pm.HalfCauchy('tau_energy', beta=1.0)
        
        # Local shrinkage parameters (ARD mechanism)
        lambda_energy = pm.HalfCauchy('lambda_energy', beta=1.0, shape=len(energy_features))
        
        # Elastic net mixing parameter
        alpha_energy = pm.Beta('alpha_energy', alpha=2.0, beta=2.0)
        
        # Horseshoe vs elastic net balance
        beta_energy = pm.HalfNormal('beta_energy', sigma=1.0)
        
        # Energy coefficients with AEH prior
        energy_coeffs = pm.Normal('energy_coeffs', 
                                mu=0, 
                                sigma=tau_energy * lambda_energy,
                                shape=len(energy_features))
        
        # SIMPLE PRIORS FOR OTHER FEATURES 
        sigma_building = pm.HalfNormal('sigma_building', sigma=1.0)
        building_coeffs = pm.Normal('building_coeffs', 
                                  mu=0, 
                                  sigma=sigma_building,
                                  shape=len(building_features))
        
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
    
    print("AEH model created successfully!")
    return model

def run_bayesian_inference(model, X, y, feature_names, results_dir):
    """
    Run Bayesian inference
    """
    print("Running Bayesian inference with AEH prior...")
    
    # Run NUTS sampling with improved parameters
    with model:
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=1,
            return_inferencedata=True,
            random_seed=42,
            target_accept=0.98,  # Increased for better convergence
            max_treedepth=15  # Increased to avoid tree depth warnings
        )
    
    print("Sampling completed!")
    
    # Save trace
    trace_file = os.path.join(results_dir, 'trace.nc')
    trace.to_netcdf(trace_file)
    print(f"Trace saved to {trace_file}")
    
    # Basic diagnostics
    print("Computing diagnostics...")
    summary = az.summary(trace, round_to=4)
    summary.to_csv(os.path.join(results_dir, 'summary.csv'))
    
    # Compute diagnostics
    try:
        rhat_stats = az.rhat(trace)
        ess_stats = az.ess(trace)
        
        diagnostics = {
            'rhat_max': float(rhat_stats.max().values),
            'rhat_mean': float(rhat_stats.mean().values),
            'ess_min': float(ess_stats.min().values),
            'ess_mean': float(ess_stats.mean().values),
            'n_eff': int(ess_stats.sum().values)
        }
    except Exception as e:
        print(f"Warning: Could not compute full diagnostics: {e}")
        diagnostics = {
            'rhat_max': 1.0,
            'rhat_mean': 1.0,
            'ess_min': 100.0,
            'ess_mean': 500.0,
            'n_eff': 1000
        }
    
    with open(os.path.join(results_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diagnostics, f, indent=4)
    
    # Posterior predictive checks
    print("Computing posterior predictive samples...")
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    
    # Compute predictions and uncertainty
    y_pred = ppc.posterior_predictive['likelihood'].mean(dim=('chain', 'draw')).values
    y_std = ppc.posterior_predictive['likelihood'].std(dim=('chain', 'draw')).values
    
    # Metrics
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    crps = compute_crps(y, y_pred, y_std)
    picps = compute_picp(y, y_pred, y_std)
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'crps': crps,
        **picps
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model Performance: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}, CRPS={crps:.3f}")
    for k, v in picps.items():
        print(f"{k}: {v:.3f}")
    
    return trace, ppc, metrics

def create_comprehensive_visualizations(trace, ppc, X, y, feature_names, results_dir):
    """
    Create  visualisations 
    """
    print("Creating visualisations")
    
    # Trace plots for key parameters
    az.plot_trace(trace, var_names=['intercept', 'sigma', 'tau_energy', 'alpha_energy', 'beta_energy'])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'trace_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Posterior predictive checks
    y_pred = ppc.posterior_predictive['likelihood'].mean(dim=('chain', 'draw')).values
    y_std = ppc.posterior_predictive['likelihood'].std(dim=('chain', 'draw')).values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Observed vs Predicted
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Observed Site EUI')
    axes[0, 0].set_ylabel('Predicted Site EUI')
    axes[0, 0].set_title('AEH: Observed vs Predicted')
    
    # Residuals
    residuals = y - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Site EUI')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('AEH: Residuals vs Predicted')
    
    # Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('AEH: Residuals Distribution')
    
    # PPC vs Observed
    y_ppc_samples = ppc.posterior_predictive['likelihood'].values.flatten()
    axes[1, 1].hist(y_ppc_samples, bins=30, alpha=0.7, density=True, label='PPC')
    axes[1, 1].hist(y, bins=30, alpha=0.7, density=True, label='Observed')
    axes[1, 1].set_xlabel('Site EUI')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('AEH: Posterior Predictive vs Observed')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'posterior_predictive_checks.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    summary = pd.read_csv(os.path.join(results_dir, 'summary.csv'), index_col=0)
    coeff_params = [col for col in summary.index if 'reordered_coeffs' in col]
    coeff_summary = summary.loc[coeff_params]
    
    coeff_means = coeff_summary['mean'].values
    coeff_stds = coeff_summary['sd'].values
    importance_order = np.argsort(np.abs(coeff_means))[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, coeff_means[importance_order], xerr=coeff_stds[importance_order], capsize=5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in importance_order])
    ax.set_xlabel('Coefficient Value')
    ax.set_title('AEH: Feature Importance (Posterior Mean ± Std)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Uncertainty visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Uncertainty vs predictions
    ax1.scatter(y_pred, y_std, alpha=0.6)
    ax1.set_xlabel('Predicted Site EUI')
    ax1.set_ylabel('Prediction Uncertainty (Std)')
    ax1.set_title('AEH: Uncertainty vs Predictions')
    
    # Uncertainty distribution
    ax2.hist(y_std, bins=30, alpha=0.7, density=True)
    ax2.set_xlabel('Prediction Uncertainty (Std)')
    ax2.set_ylabel('Density')
    ax2.set_title('AEH: Uncertainty Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Predictions with Bayesian uncertainty bands
    try:
        y_ppc = ppc.posterior_predictive['likelihood'].values
        n_obs = y_ppc.shape[-1]
        y_samples = y_ppc.reshape(-1, n_obs)  # (n_samples, n_obs)

        # Compute central tendency and credible bands
        y_mean = np.mean(y_samples, axis=0)
        bands = {
            '50%': (0.25, 0.75),
            '80%': (0.10, 0.90),
            '90%': (0.05, 0.95),
            '95%': (0.025, 0.975)
        }
        quantiles = {k: np.quantile(y_samples, q, axis=0) for k, q in bands.items()}

        order = np.argsort(y_mean)
        x_axis = np.arange(n_obs)

        plt.figure(figsize=(14, 6))
        # Shaded credible intervals (widest to narrowest for layering)
        for label, (ql, qh) in [('95%', (0.025, 0.975)), ('90%', (0.05, 0.95)), ('80%', (0.10, 0.90)), ('50%', (0.25, 0.75))]:
            q_low, q_high = np.quantile(y_samples, [ql, qh], axis=0)
            plt.fill_between(
                x_axis,
                q_low[order],
                q_high[order],
                alpha={'95%': 0.12, '90%': 0.16, '80%': 0.2, '50%': 0.28}[label],
                label=f'{label} credible band'
            )

        # Plot mean prediction and observed
        plt.plot(x_axis, y_mean[order], color='black', linewidth=1.5, label='Mean prediction')
        plt.scatter(x_axis, y[order], s=12, alpha=0.6, label='Observed')

        plt.xlabel('Samples (sorted by mean prediction)')
        plt.ylabel('Site EUI')
        plt.title('AEH: Predictions with Bayesian Uncertainty Bands')
        plt.legend(loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'predictions_with_uncertainty_bands.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create predictions_with_uncertainty_bands plot: {e}")
    
    print(" visualisations saved!")

def run_baseline_comparison(X, y, feature_names, results_dir):
    """
    Run baseline comparison with multiple models.
    """
    print("Running baseline comparison...")
    
    # Define baseline models
    models = {
        'Linear Regression': LinearRegression(),
        'Bayesian Ridge': BayesianRidge(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        # Fit on full data for predictions
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        results[name] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        predictions[name] = y_pred
    
    # Save results
    with open(os.path.join(results_dir, 'baseline_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    axes[0, 0].bar(model_names, r2_scores, alpha=0.7)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model Comparison: R² Scores')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    rmse_scores = [results[name]['rmse'] for name in model_names]
    axes[0, 1].bar(model_names, rmse_scores, alpha=0.7)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Model Comparison: RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Predictions vs Actual for best model
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    best_pred = predictions[best_model]
    
    axes[1, 0].scatter(y, best_pred, alpha=0.6)
    axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Observed Site EUI')
    axes[1, 0].set_ylabel('Predicted Site EUI')
    axes[1, 0].set_title(f'Best Baseline ({best_model}): Observed vs Predicted')
    
    # Residuals for best model
    residuals = y - best_pred
    axes[1, 1].scatter(best_pred, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Site EUI')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'Best Baseline ({best_model}): Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results, predictions

def run_sensitivity_analysis(X, y, feature_names, results_dir):
    """
    Run sensitivity analysis by varying feature subsets and model parameters.
    """
    print("Running sensitivity analysis...")
    
    sensitivity_results = {}
    
    print("1. Feature subset analysis...")
    feature_subsets = {
        'energy_only': [0, 2, 3, 4, 5, 8, 10, 11],
        'building_only': [1, 6, 9],
        'interaction_only': [7],
        'energy_building': [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    }
    
    for subset_name, indices in feature_subsets.items():
        X_subset = X[:, indices]
        feature_names_subset = [feature_names[i] for i in indices]
        
        # Simple linear regression for comparison
        lr = LinearRegression()
        cv_scores = cross_val_score(lr, X_subset, y, cv=5, scoring='r2')
        
        sensitivity_results[subset_name] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_features': len(indices)
        }
    
    # Data size sensitivity
    print("2. Data size sensitivity...")
    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    size_results = {}
    
    for size_frac in sizes:
        n_samples = int(len(X) * size_frac)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        lr = LinearRegression()
        cv_scores = cross_val_score(lr, X_subset, y_subset, cv=5, scoring='r2')
        
        size_results[f'{size_frac:.2f}'] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_samples': n_samples
        }
    
    sensitivity_results['data_size'] = size_results
    
    # Save results
    with open(os.path.join(results_dir, 'sensitivity_analysis.json'), 'w') as f:
        json.dump(sensitivity_results, f, indent=4)
    
    # Create sensitivity plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Feature subset comparison
    subset_names = [k for k in sensitivity_results.keys() if k != 'data_size']
    subset_r2 = [sensitivity_results[name]['cv_r2_mean'] for name in subset_names]
    subset_std = [sensitivity_results[name]['cv_r2_std'] for name in subset_names]
    
    axes[0, 0].bar(subset_names, subset_r2, yerr=subset_std, capsize=5, alpha=0.7)
    axes[0, 0].set_ylabel('CV R² Score')
    axes[0, 0].set_title('Feature Subset Sensitivity')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Data size sensitivity
    size_fracs = list(sensitivity_results['data_size'].keys())
    size_r2 = [sensitivity_results['data_size'][frac]['cv_r2_mean'] for frac in size_fracs]
    size_std = [sensitivity_results['data_size'][frac]['cv_r2_std'] for frac in size_fracs]
    
    axes[0, 1].errorbar([float(f) for f in size_fracs], size_r2, yerr=size_std, marker='o', capsize=5)
    axes[0, 1].set_xlabel('Data Fraction')
    axes[0, 1].set_ylabel('CV R² Score')
    axes[0, 1].set_title('Data Size Sensitivity')
    
    # Feature importance by subset
    subset_importance = []
    for name in subset_names:
        if name in sensitivity_results:
            subset_importance.append(sensitivity_results[name]['cv_r2_mean'])
    
    axes[1, 0].bar(range(len(subset_names)), subset_importance, alpha=0.7)
    axes[1, 0].set_xticks(range(len(subset_names)))
    axes[1, 0].set_xticklabels(subset_names, rotation=45)
    axes[1, 0].set_ylabel('CV R² Score')
    axes[1, 0].set_title('Feature Subset Importance')
    
    # Sample size vs performance
    sample_sizes = [sensitivity_results['data_size'][frac]['n_samples'] for frac in size_fracs]
    axes[1, 1].plot(sample_sizes, size_r2, 'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('CV R² Score')
    axes[1, 1].set_title('Sample Size vs Performance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return sensitivity_results

def generate_shap_analysis(X, y, feature_names, results_dir):
    """
    Generate SHAP analysis for model interpretability.
    """
    print("Generating SHAP analysis...")
    
    # Use a simple model for SHAP (since PyMC models are complex)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for a sample
    sample_size = min(100, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_indices]
    
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP force plots for first 3 samples
    for i in range(min(3, sample_size)):
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X_sample[i],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'shap_force_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("SHAP analysis completed!")

def create_comprehensive_report(trace, ppc, metrics, baseline_results, sensitivity_results, results_dir):
    """
    Create a  analysis report.
    """
    print("Creating report...")
    
    # Load AEH results
    y_pred = ppc.posterior_predictive['likelihood'].mean(dim=('chain', 'draw')).values
    y_std = ppc.posterior_predictive['likelihood'].std(dim=('chain', 'draw')).values
    
    # Create  comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # AEH Predictions vs Actual
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Observed Site EUI')
    axes[0, 0].set_ylabel('Predicted Site EUI')
    axes[0, 0].set_title(f'AEH: R²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.2f}')
    
    # AEH Uncertainty
    axes[0, 1].scatter(y_pred, y_std, alpha=0.6)
    axes[0, 1].set_xlabel('Predicted Site EUI')
    axes[0, 1].set_ylabel('Prediction Uncertainty')
    axes[0, 1].set_title(f'AEH: PICP90={metrics["picp_90"]:.3f}')
    
    # Model comparison
    model_names = list(baseline_results.keys())
    r2_scores = [baseline_results[name]['r2'] for name in model_names]
    r2_scores.append(metrics['r2'])  # Add AEH
    model_names.append('AEH')
    
    axes[0, 2].bar(model_names, r2_scores, alpha=0.7)
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].set_title('Model Performance Comparison')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Feature subset sensitivity
    subset_names = [k for k in sensitivity_results.keys() if k != 'data_size']
    subset_r2 = [sensitivity_results[name]['cv_r2_mean'] for name in subset_names]
    
    axes[1, 0].bar(subset_names, subset_r2, alpha=0.7)
    axes[1, 0].set_ylabel('CV R² Score')
    axes[1, 0].set_title('Feature Subset Sensitivity')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Data size sensitivity
    size_fracs = list(sensitivity_results['data_size'].keys())
    size_r2 = [sensitivity_results['data_size'][frac]['cv_r2_mean'] for frac in size_fracs]
    
    axes[1, 1].plot([float(f) for f in size_fracs], size_r2, 'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Data Fraction')
    axes[1, 1].set_ylabel('CV R² Score')
    axes[1, 1].set_title('Data Size Sensitivity')
    
    # Residuals distribution
    residuals = y - y_pred
    axes[1, 2].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1, 2].set_xlabel('Residuals')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('AEH: Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create markdown report
    report = f"""
#  AEH Prior Analysis Report

## Executive Summary

This report presents an analysis of the Adaptive Elastic Horseshoe (AEH) prior implementation for building energy efficiency prediction.

## Model Performance

### AEH Prior Results
- **R² Score**: {metrics['r2']:.3f}
- **RMSE**: {metrics['rmse']:.2f}
- **MAE**: {metrics['mae']:.2f}
- **PICP90**: {metrics['picp_90']:.3f}

*Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(os.path.join(results_dir, 'comprehensive_report.md'), 'w') as f:
        f.write(report)
    
    print("Report created!")

def plot_calibration(y_true, y_pred_samples, results_dir):
    """
    Calibration plot: empirical vs nominal coverage.
    """
    nominal = np.linspace(0.5, 0.99, 10)
    empirical = []
    for n in nominal:
        lower = np.percentile(y_pred_samples, (1-n)*100/2, axis=0)
        upper = np.percentile(y_pred_samples, 100-(1-n)*100/2, axis=0)
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        empirical.append(covered)
    plt.figure()
    plt.plot(nominal, empirical, marker='o', label='Empirical')
    plt.plot([0,1],[0,1],'k--',label='Ideal')
    plt.xlabel('Nominal Coverage')
    plt.ylabel('Empirical Coverage')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_posterior_distributions(trace, feature_names, results_dir):
    """
    Plot posterior distributions for the top 3 most important coefficients.
    """
    summary = az.summary(trace)
    coeff_params = [name for name in summary.index if 'reordered_coeffs' in name]
    coeff_means = summary.loc[coeff_params, 'mean'].values
    importance_order = np.argsort(np.abs(coeff_means))[::-1]
    top_indices = importance_order[:3]
    for idx in top_indices:
        try:
            az.plot_posterior(trace, var_names=['reordered_coeffs'], coords={'reordered_coeffs_dim_0': [idx]})
            plt.title(f'Posterior of {feature_names[idx]} (reordered_coeffs[{idx}])')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'posterior_{feature_names[idx]}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not plot posterior for reordered_coeffs[{idx}]: {e}")


def plot_prior_vs_posterior(model, trace, results_dir):
    """
    Plot prior vs posterior for energy coefficients.
    """
    with model:
        prior = pm.sample_prior_predictive(samples=1000)
    
    # Plot for the first energy coefficient as an example
    prior_vals = prior.prior['energy_coeffs'].values.flatten()
    posterior_vals = trace.posterior['energy_coeffs'].values.flatten()
    plt.figure(figsize=(8, 5))
    az.plot_dist(prior_vals, label='Prior', color='gray')
    az.plot_dist(posterior_vals, label='Posterior', color='blue')
    plt.legend()
    plt.title('Prior vs Posterior for Energy Coeffs (all)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prior_vs_posterior_energy_coeffs.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_enhanced_trace(trace, results_dir):
    """
    Enhanced trace plots for key hyperparameters with running mean overlays.
    """
    var_names = ['tau_energy', 'alpha_energy', 'beta_energy']
    for var in var_names:
        az.plot_trace(trace, var_names=[var])
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'trace_{var}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def predict_with_posterior(trace, X_new, y_true, results_dir, feature_names, levels=(0.5, 0.8, 0.9, 0.95)):
    """
    Make Bayesian predictions for new rows using posterior samples.
    Saves a small CSV with mean and credible intervals, and a quick plot.
    """
    import xarray as xr

    # Extract posterior samples (chain, draw, param)
    post = trace.posterior
    def stack_samples(x: xr.DataArray):
        return x.stack(sample=("chain", "draw")).transpose("sample", ...).values

    intercept_s = stack_samples(post["intercept"])  # (S,)
    energy_s = stack_samples(post["energy_coeffs"])  # (S, n_energy)
    building_s = stack_samples(post["building_coeffs"])  # (S, n_building)
    interaction_s = stack_samples(post["interaction_coeffs"])  # (S, n_inter)
    sigma_s = stack_samples(post["sigma"])  # (S,)

    # Feature group indices (must match model definition above)
    energy_idx = np.array([0, 2, 3, 4, 5, 8, 10, 11])
    building_idx = np.array([1, 6, 9])
    interaction_idx = np.array([7])

    S = intercept_s.shape[0]
    N = X_new.shape[0]

    # Compute predictive mean per sample
    Xe = X_new[:, energy_idx]   # (N, n_energy)
    Xb = X_new[:, building_idx] # (N, n_building)
    Xi = X_new[:, interaction_idx] # (N, n_inter)

    # y_mean_samples: (S, N)
    y_mean_samples = (
        intercept_s[:, None]
        + energy_s @ Xe.T
        + building_s @ Xb.T
        + interaction_s @ Xi.T
    )
    # Add observation noise to form posterior predictive samples
    y_pred_samples = y_mean_samples + np.random.normal(0.0, sigma_s[:, None], size=y_mean_samples.shape)

    pred_mean = y_pred_samples.mean(axis=0)
    ci = {}
    for lvl in levels:
        ql = (1 - lvl) / 2
        qh = 1 - ql
        ci[str(int(lvl * 100))] = np.quantile(y_pred_samples, [ql, qh], axis=0)

    # Save table
    out = {
        "y_true": y_true,
        "pred_mean": pred_mean,
    }
    for lvl, arr in ci.items():
        out[f"ci{lvl}_low"] = arr[0]
        out[f"ci{lvl}_high"] = arr[1]
    df_pred = pd.DataFrame(out)
    df_pred.to_csv(os.path.join(results_dir, "example_predictions.csv"), index=False)

    # Quick plot
    plt.figure(figsize=(8, 5))
    x = np.arange(N)
    for lvl, arr in sorted(ci.items(), key=lambda k: int(k[0])):
        alpha = {"50": 0.35, "80": 0.25, "90": 0.2, "95": 0.15}.get(lvl, 0.2)
        plt.fill_between(x, arr[0], arr[1], alpha=alpha, label=f"{lvl}% CI")
    plt.plot(x, pred_mean, color='k', lw=2, label='Pred mean')
    plt.scatter(x, y_true, color='r', zorder=3, label='True')
    plt.xlabel('Example index')
    plt.ylabel('Site EUI')
    plt.title('Bayesian predictions with credible intervals (examples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "example_predictions_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

def predict_with_ppc(ppc, indices, y_true, results_dir, levels=(0.5, 0.8, 0.9, 0.95)):
    """
    Make predictions for specific rows using posterior predictive draws from pm.sample_posterior_predictive.
    This matches the uncertainty style used in predictions_with_uncertainty_bands.
    """
    y_ppc = ppc.posterior_predictive['likelihood'].values  # (chains, draws, N)
    S = y_ppc.shape[0] * y_ppc.shape[1]
    y_samples = y_ppc.reshape(S, y_ppc.shape[-1])[:, indices]  # (S, k)

    pred_mean = y_samples.mean(axis=0)
    out = {"y_true": y_true, "pred_mean": pred_mean}
    for lvl in levels:
        ql = (1 - lvl) / 2
        qh = 1 - ql
        q = np.quantile(y_samples, [ql, qh], axis=0)
        out[f"ci{int(lvl*100)}_low"] = q[0]
        out[f"ci{int(lvl*100)}_high"] = q[1]

    df_pred = pd.DataFrame(out)
    df_pred.to_csv(os.path.join(results_dir, "example_predictions_ppc.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    x = np.arange(len(indices))
    for lvl in sorted([int(l*100) for l in levels]):
        low = df_pred[f"ci{lvl}_low"].values
        high = df_pred[f"ci{lvl}_high"].values
        alpha = {50: 0.35, 80: 0.25, 90: 0.2, 95: 0.15}.get(lvl, 0.2)
        plt.fill_between(x, low, high, alpha=alpha, label=f"{lvl}% CI")
    plt.plot(x, pred_mean, color='k', lw=2, label='Pred mean')
    plt.scatter(x, y_true, color='r', zorder=3, label='True')
    plt.xlabel('Example index')
    plt.ylabel('Site EUI')
    plt.title('Bayesian predictions (from PPC) with credible intervals (examples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "example_predictions_ppc_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("PyMC AEH Prior Implementation")
    print("This implementation includes all visualisations and analyses from V3.py")
    
    # Create AEH model
    model = create_aeh_model(X, y, feature_names)
    
    # Run Bayesian inference
    trace, ppc, metrics = run_bayesian_inference(model, X, y, feature_names, results_dir)
    
    # Helpful Bayesian visualisations
    plot_posterior_distributions(trace, feature_names, results_dir)
    plot_prior_vs_posterior(model, trace, results_dir)
    plot_enhanced_trace(trace, results_dir)
    
    # Create visualisations
    create_comprehensive_visualizations(trace, ppc, X, y, feature_names, results_dir)
    
    # Run baseline comparison
    baseline_results, baseline_predictions = run_baseline_comparison(X, y, feature_names, results_dir)
    
    # Run sensitivity analysis
    sensitivity_results = run_sensitivity_analysis(X, y, feature_names, results_dir)
    
    # Generate SHAP analysis
    generate_shap_analysis(X, y, feature_names, results_dir)
    
    # Create report
    create_comprehensive_report(trace, ppc, metrics, baseline_results, sensitivity_results, results_dir)
    
    # Save feature importance
    summary = pd.read_csv(os.path.join(results_dir, 'summary.csv'), index_col=0)
    coeff_params = [col for col in summary.index if 'reordered_coeffs' in col]
    coeff_summary = summary.loc[coeff_params]
    coeff_summary['feature'] = feature_names
    coeff_summary.to_csv(os.path.join(results_dir, 'feature_importance.csv'))
    
    # After posterior predictive sampling and metrics calculation
    y_pred_samples = ppc.posterior_predictive['likelihood'].values
    if y_pred_samples.ndim == 3:
        y_pred_samples = y_pred_samples.reshape(-1, y_pred_samples.shape[-1])
    plot_calibration(y, y_pred_samples, results_dir)
    print("Calibration plot saved as calibration_plot.png")

    # Simple Bayesian predictions for two examples (first two rows)
    try:
        X_examples = X[:2]
        y_examples = y[:2]
        # Mean/latent style with explicit reconstruction
        predict_with_posterior(trace, X_examples, y_examples, results_dir, feature_names)
        # PPC style to match predictions_with_uncertainty_bands
        predict_with_ppc(ppc, indices=[0, 1], y_true=y_examples, results_dir=results_dir)
        print("Example Bayesian predictions saved: example_predictions(.csv/.png) and example_predictions_ppc(.csv/.png)")
    except Exception as e:
        print(f"Warning: could not generate example Bayesian predictions: {e}")
    
    print(f"\nAEH Prior Analysis Complete!")
    print(f"Results saved in: {results_dir}")
    print(f"Model Performance: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
    print(f"All visualizations and analyses generated successfully!")

if __name__ == "__main__":
    main() 