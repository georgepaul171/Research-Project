import numpy as np
import pandas as pd
import os
import json
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler
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

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_debug_model_range')
os.makedirs(results_dir, exist_ok=True)

def print_and_save_stats(label, y_true, y_pred, weights=None, scaler_y=None):
    stats = {}
    stats['true_min'] = float(y_true.min())
    stats['true_max'] = float(y_true.max())
    stats['pred_scaled_min'] = float(y_pred.min())
    stats['pred_scaled_max'] = float(y_pred.max())
    if scaler_y is not None:
        y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        stats['pred_unscaled_min'] = float(y_pred_unscaled.min())
        stats['pred_unscaled_max'] = float(y_pred_unscaled.max())
    else:
        y_pred_unscaled = y_pred
        stats['pred_unscaled_min'] = float(y_pred.min())
        stats['pred_unscaled_max'] = float(y_pred.max())
    if weights is not None:
        stats['weights'] = [float(w) for w in weights]
    print(f"\n[{label}] True min/max: {stats['true_min']} {stats['true_max']}")
    print(f"[{label}] Predicted (scaled) min/max: {stats['pred_scaled_min']} {stats['pred_scaled_max']}")
    print(f"[{label}] Predicted (unscaled) min/max: {stats['pred_unscaled_min']} {stats['pred_unscaled_max']}")
    if weights is not None:
        print(f"[{label}] Weights: {weights}")
    with open(os.path.join(results_dir, f'stats_{label}.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    return y_pred_unscaled

def fit_with_logging(self, X, y, feature_names=None, output_dir=None):
    y = np.asarray(y).reshape(-1)
    n_samples, n_features = X.shape
    self.alpha = np.clip(self.config.alpha_0, 1e-10, None)
    self.beta = np.ones(n_features) * np.clip(self.config.beta_0, 1e-10, None)
    self.beta[-1] = 1e-10  # Do not regularize the bias term from the start (if bias present)
    self.m = np.zeros(n_features)
    self.S = np.eye(n_features)
    self._initialize_adaptive_priors(n_features)
    X_train_scaled = self.scaler_X.fit_transform(X)
    y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    log_path = os.path.join(results_dir, 'em_debug_log.txt')
    with open(log_path, 'w') as log_f:
        for iteration in range(self.config.max_iter):
            try:
                self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                     np.diag(np.clip(self.beta, 1e-10, None)))
            except np.linalg.LinAlgError:
                jitter = 1e-6 * np.eye(n_features)
                self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                     np.diag(np.clip(self.beta, 1e-10, None)) + jitter)
            self.m = self.alpha * self.S @ X_train_scaled.T @ y_train_scaled
            # Log alpha, beta, m
            log_str = (f"Iter {iteration}: alpha={self.alpha:.4e}, "
                       f"beta_mean={np.mean(self.beta):.4e}, beta_min={np.min(self.beta):.4e}, beta_max={np.max(self.beta):.4e}, "
                       f"m_mean={np.mean(self.m):.4e}, m_min={np.min(self.m):.4e}, m_max={np.max(self.m):.4e}")
            print(log_str)
            log_f.write(log_str + '\n')
            # Standard EM update for alpha, beta
            residuals = y_train_scaled - X_train_scaled @ self.m
            alpha_new = n_samples / (np.sum(residuals**2) + np.trace(X_train_scaled @ self.S @ X_train_scaled.T))
            beta_new = np.zeros_like(self.beta)
            for j in range(n_features):
                beta_new[j] = 1 / (np.clip(self.m[j]**2, 1e-10, None) + np.clip(np.diag(self.S)[j], 1e-10, None) + 1e-6)
            beta_new[-1] = 1e-10  # Do not regularize the bias term (if bias present)
            beta_diff = np.abs(np.clip(beta_new, 1e-10, None) - np.clip(self.beta, 1e-10, None))
            alpha_diff = np.abs(alpha_new - self.alpha)
            if (alpha_diff < self.config.tol and np.all(beta_diff < self.config.tol)):
                print(f"[DEBUG] EM converged at iteration {iteration}")
                log_f.write(f"[DEBUG] EM converged at iteration {iteration}\n")
                break
            self.alpha = alpha_new
            self.beta = np.clip(beta_new, 1e-10, None)
    # Final log
    log_str = (f"FINAL: alpha={self.alpha:.4e}, "
               f"beta_mean={np.mean(self.beta):.4e}, beta_min={np.min(self.beta):.4e}, beta_max={np.max(self.beta):.4e}, "
               f"m_mean={np.mean(self.m):.4e}, m_min={np.min(self.m):.4e}, m_max={np.max(self.m):.4e}")
    print(log_str)
    with open(log_path, 'a') as log_f:
        log_f.write(log_str + '\n')
    # After fitting, run diagnostics as before
    y_pred, y_std = self.predict(X, return_std=True)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    mean_std = float(np.mean(y_std))
    crps = float(np.mean(np.abs(y - y_pred)) - 0.5 * np.mean(np.abs(y_std)))
    from scipy.stats import norm
    confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    picp_scores = []
    for level in confidence_levels:
        z_score = norm.ppf(1 - (1 - level) / 2)
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        coverage = np.mean((y >= lower) & (y <= upper))
        picp_scores.append(float(coverage))
    self.cv_results = pd.DataFrame([{
        'fold': 1,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_std': mean_std,
        'crps': crps,
        'picp_50': picp_scores[0],
        'picp_80': picp_scores[1],
        'picp_90': picp_scores[2],
        'picp_95': picp_scores[3],
        'picp_99': picp_scores[4]
    }])
    return self

# 1. LinearRegression
lr = LinearRegression()
lr.fit(X, y)
y_pred_lr = lr.predict(X)
print_and_save_stats('LinearRegression', y, y_pred_lr, weights=lr.coef_)

# 2. BayesianRidge
br = BayesianRidge()
br.fit(X, y)
y_pred_br = br.predict(X)
print_and_save_stats('BayesianRidge', y, y_pred_br, weights=br.coef_)

# 3. AdaptivePriorARD (AEH only on 'energy', hierarchical on others)
config_aeh_energy = AdaptivePriorConfig(
    beta_0=10.0,
    group_sparsity=False,
    dynamic_shrinkage=False,
    hmc_steps=1,
    hmc_leapfrog_steps=1,
    hmc_epsilon=0.0001,
    max_iter=20,
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
model_aeh_energy = AdaptivePriorARD(config_aeh_energy)
fit_with_logging(model_aeh_energy, X, y, feature_names=feature_names, output_dir=results_dir)
y_pred_aeh_energy, y_std_aeh_energy = model_aeh_energy.predict(X, return_std=True)
# print_scaling_diagnostics(model_aeh_energy, X, y, y_pred_aeh_energy, 'AdaptivePriorARD_AEH_energy')
print_and_save_stats('AdaptivePriorARD_AEH_energy', y, y_pred_aeh_energy, weights=model_aeh_energy.m, scaler_y=getattr(model_aeh_energy, 'scaler_y', None))

# 4. AdaptivePriorARD (AEH on all groups)
config_aeh_all = AdaptivePriorConfig(
    beta_0=10.0,
    group_sparsity=False,
    dynamic_shrinkage=False,
    hmc_steps=1,
    hmc_leapfrog_steps=1,
    hmc_epsilon=0.0001,
    max_iter=20,
    tol=1e-8,
    use_hmc=False,
    robust_noise=False,
    uncertainty_calibration=False,
    group_prior_types={
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'adaptive_elastic_horseshoe',
        'interaction': 'adaptive_elastic_horseshoe'
    }
)
model_aeh_all = AdaptivePriorARD(config_aeh_all)
fit_with_logging(model_aeh_all, X, y, feature_names=feature_names, output_dir=results_dir)
y_pred_aeh_all, y_std_aeh_all = model_aeh_all.predict(X, return_std=True)
# print_scaling_diagnostics(model_aeh_all, X, y, y_pred_aeh_all, 'AdaptivePriorARD_AEH_all')
print_and_save_stats('AdaptivePriorARD_AEH_all', y, y_pred_aeh_all, weights=model_aeh_all.m, scaler_y=getattr(model_aeh_all, 'scaler_y', None))

# 5. Fit on extremes only (lowest 10 and highest 10 targets) with AEH on all groups
extreme_idx = np.concatenate([np.argsort(y)[:10], np.argsort(y)[-10:]])
X_extreme = X[extreme_idx]
y_extreme = y[extreme_idx]
model_extreme = AdaptivePriorARD(config_aeh_all)
fit_with_logging(model_extreme, X_extreme, y_extreme, feature_names=feature_names, output_dir=results_dir)
y_pred_extreme, _ = model_extreme.predict(X_extreme, return_std=True)
print_and_save_stats('AdaptivePriorARD_AEH_all_extremes', y_extreme, y_pred_extreme, weights=model_extreme.m, scaler_y=getattr(model_extreme, 'scaler_y', None))

print("\nDiagnostics complete. See results_debug_model_range/ for details.") 