import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, norm, t
import warnings
from datetime import datetime
import joblib
import shap

warnings.filterwarnings('ignore')

POSTHOC_UNCERTAINTY_SCALE = 10.0

@dataclass
class AdaptivePriorConfig:
    alpha_0: float = 1.0
    beta_0: float = 1.0
    max_iter: int = 100
    tol: float = 1e-4
    n_splits: int = 2
    random_state: int = 42
    prior_type: str = 'hierarchical'
    adaptation_rate: float = 0.1
    uncertainty_threshold: float = 0.1
    group_sparsity: bool = True
    dynamic_shrinkage: bool = True
    use_hmc: bool = True
    # FIXED HMC PARAMETERS
    hmc_steps: int = 50  # Increased from 20
    hmc_epsilon: float = 0.001  # Increased from 0.0001
    hmc_leapfrog_steps: int = 5  # Increased from 3
    uncertainty_calibration: bool = True
    calibration_factor: float = 0.03
    robust_noise: bool = True
    student_t_df: float = 3.0
    group_prior_types: dict = field(default_factory=lambda: {
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    })

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        return super(NumpyEncoder, self).default(obj)

class AdaptivePriorARD:
    def __init__(self, config: Optional[AdaptivePriorConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            if kwargs:
                self.config = AdaptivePriorConfig()
                for key, value in kwargs.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            else:
                self.config = AdaptivePriorConfig()
        
        self.alpha = None
        self.beta = None
        self.m = None
        self.S = None
        self.prior_hyperparams = None
        self.feature_groups = None
        self.shrinkage_params = None
        self.scaler_X = RobustScaler()
        self.scaler_y = StandardScaler()
        self.cv_results = None
        self.uncertainty_calibration_factor = self.config.calibration_factor
        self.uncertainty_calibration_history = []
        self.r_hat_history = []
        self.convergence_history = []
    
    def get_params(self, deep=True):
        return {
            'alpha_0': self.config.alpha_0,
            'beta_0': self.config.beta_0,
            'max_iter': self.config.max_iter,
            'tol': self.config.tol,
            'n_splits': self.config.n_splits,
            'random_state': self.config.random_state,
            'prior_type': self.config.prior_type,
            'adaptation_rate': self.config.adaptation_rate,
            'uncertainty_threshold': self.config.uncertainty_threshold,
            'group_sparsity': self.config.group_sparsity,
            'dynamic_shrinkage': self.config.dynamic_shrinkage,
            'use_hmc': self.config.use_hmc,
            'hmc_steps': self.config.hmc_steps,
            'hmc_epsilon': self.config.hmc_epsilon,
            'hmc_leapfrog_steps': self.config.hmc_leapfrog_steps,
            'uncertainty_calibration': self.config.uncertainty_calibration,
            'calibration_factor': self.config.calibration_factor,
            'robust_noise': self.config.robust_noise,
            'student_t_df': self.config.student_t_df
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self

    def _initialize_adaptive_priors(self, n_features: int):
        self.feature_groups = self._create_feature_groups(n_features)
        self.group_prior_hyperparams = {}
        self.shrinkage_params = {'kappa': 0.1}
        
        for group, indices in self.feature_groups.items():
            prior_type = self.config.group_prior_types.get(group, 'hierarchical')
            if prior_type == 'hierarchical':
                self.group_prior_hyperparams[group] = {
                    'tau': 1.0,
                    'lambda': np.ones(len(indices))
                }
            elif prior_type == 'spike_slab':
                self.group_prior_hyperparams[group] = {
                    'pi': np.ones(len(indices)) * 0.5,
                    'sigma2_0': np.ones(len(indices)) * 0.01,
                    'sigma2_1': np.ones(len(indices)) * 1.0
                }
            elif prior_type == 'horseshoe':
                self.group_prior_hyperparams[group] = {
                    'tau': 1.0,
                    'lambda': np.ones(len(indices))
                }
            elif prior_type == 'adaptive_elastic_horseshoe':
                self.group_prior_hyperparams[group] = {
                    'tau': 1.0,
                    'lambda': np.ones(len(indices)),
                    'alpha': 0.5,  # Elastic net mixing
                    'beta': 0.5,   # Horseshoe mixing
                    'momentum': np.zeros(len(indices))
                }

    def _create_feature_groups(self, n_features: int) -> Dict[str, List[int]]:
        if n_features == 12:  # Based on your feature set
            return {
                'energy': [0, 2, 3, 4, 5, 8, 11],  # Energy-related features
                'building': [1, 6, 7, 9, 10],      # Building-related features
                'interaction': []                   # No interaction features in this case
            }
        else:
            return {'default': list(range(n_features))}

    def _update_uncertainty_calibration(self, X: np.ndarray, y: np.ndarray):
        if not self.config.uncertainty_calibration:
            return
        
        y_pred, y_std = self.predict(X, return_std=True)
        residuals = np.abs(y - y_pred)
        calibration_factor = np.mean(residuals) / np.mean(y_std)
        self.uncertainty_calibration_factor = calibration_factor
        self.uncertainty_calibration_history.append(calibration_factor)

    def _update_adaptive_priors(self, iteration: int):
        if not hasattr(self, 'group_prior_hyperparams'):
            return
        
        for group, params in self.group_prior_hyperparams.items():
            prior_type = self.config.group_prior_types.get(group, 'hierarchical')
            if prior_type == 'adaptive_elastic_horseshoe':
                # Update AEH parameters
                alpha = params['alpha']
                beta = params['beta']
                momentum = params['momentum']
                
                # Adaptive updates
                alpha_new = alpha * (1 - self.config.adaptation_rate) + 0.1 * self.config.adaptation_rate
                beta_new = beta * (1 - self.config.adaptation_rate) + 0.1 * self.config.adaptation_rate
                
                # Momentum updates
                momentum_new = momentum * 0.9 + 0.1 * np.random.randn(len(momentum))
                
                params['alpha'] = np.clip(alpha_new, 0.01, 0.99)
                params['beta'] = np.clip(beta_new, 0.01, 0.99)
                params['momentum'] = momentum_new

    def _hmc_step(self, X: np.ndarray, y: np.ndarray, current_w: np.ndarray, 
                  current_momentum: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        # Improved HMC step with better numerical stability
        epsilon = self.config.hmc_epsilon
        n_leapfrog = self.config.hmc_leapfrog_steps
        
        w = current_w.copy()
        momentum = current_momentum.copy()
        
        # Half step for momentum
        gradient = self._calculate_gradient(X, y, w)
        momentum = momentum - 0.5 * epsilon * gradient
        
        # Full steps for position and momentum
        for _ in range(n_leapfrog):
            w = w + epsilon * momentum
            gradient = self._calculate_gradient(X, y, w)
            momentum = momentum - epsilon * gradient
        
        # Final half step for momentum
        gradient = self._calculate_gradient(X, y, w)
        momentum = momentum - 0.5 * epsilon * gradient
        
        # Calculate acceptance probability
        current_hamiltonian = self._calculate_hamiltonian(X, y, current_w, current_momentum)
        proposed_hamiltonian = self._calculate_hamiltonian(X, y, w, momentum)
        
        log_acceptance = current_hamiltonian - proposed_hamiltonian
        acceptance_prob = min(1.0, np.exp(log_acceptance))
        
        return w, momentum, acceptance_prob

    def _calculate_hamiltonian(self, X: np.ndarray, y: np.ndarray, 
                             w: np.ndarray, momentum: np.ndarray) -> float:
        # Improved Hamiltonian calculation
        n_samples = X.shape[0]
        
        # Potential energy (negative log posterior)
        residuals = y - X @ w
        log_likelihood = -0.5 * n_samples * np.log(2 * np.pi) - 0.5 * np.sum(residuals**2)
        
        # Prior energy (assuming Gaussian prior)
        log_prior = -0.5 * np.sum(w**2)  # Simplified prior
        
        # Kinetic energy
        kinetic_energy = 0.5 * np.sum(momentum**2)
        
        return -(log_likelihood + log_prior) + kinetic_energy

    def _calculate_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Improved gradient calculation with regularization
        n_samples = X.shape[0]
        residuals = y - X @ w
        
        # Likelihood gradient
        likelihood_grad = -X.T @ residuals / n_samples
        
        # Prior gradient (simplified)
        prior_grad = -w
        
        return likelihood_grad + prior_grad

    def _calculate_gelman_rubin(self, chains: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        if len(chains) < 2:
            return 1.0, {}
        
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        n_params = chains[0].shape[1]
        
        # Stack all chains
        all_samples = np.vstack(chains)
        
        # Calculate within-chain variance
        within_var = np.zeros(n_params)
        for i in range(n_params):
            chain_vars = [np.var(chains[j][:, i]) for j in range(n_chains)]
            within_var[i] = np.mean(chain_vars)
        
        # Calculate between-chain variance
        between_var = np.zeros(n_params)
        for i in range(n_params):
            chain_means = [np.mean(chains[j][:, i]) for j in range(n_chains)]
            grand_mean = np.mean(chain_means)
            between_var[i] = n_samples * np.var(chain_means)
        
        # Calculate R-hat
        r_hat_stats = {}
        for i in range(n_params):
            if within_var[i] > 0:
                r_hat = np.sqrt((between_var[i] / within_var[i] + 1) / n_chains)
            else:
                r_hat = 1.0
            r_hat_stats[f'weight_{i}'] = r_hat
        
        mean_r_hat = np.mean(list(r_hat_stats.values()))
        return mean_r_hat, r_hat_stats

    def _calculate_effective_sample_size(self, chains: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        if len(chains) < 2:
            return 1.0, {}
        
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        n_params = chains[0].shape[1]
        
        ess_stats = {}
        for i in range(n_params):
            # Calculate autocorrelation for each chain
            chain_ess = []
            for j in range(n_chains):
                chain_data = chains[j][:, i]
                # Simple ESS calculation based on autocorrelation
                autocorr = np.correlate(chain_data, chain_data, mode='full')
                autocorr = autocorr[autocorr.size//2:]
                autocorr = autocorr / autocorr[0]
                
                # Find first zero crossing
                zero_crossing = np.where(np.diff(np.sign(autocorr)))[0]
                if len(zero_crossing) > 0:
                    lag = zero_crossing[0]
                else:
                    lag = n_samples // 2
                
                ess = n_samples / (1 + 2 * np.sum(autocorr[1:lag+1]))
                chain_ess.append(ess)
            
            ess_stats[f'weight_{i}'] = np.mean(chain_ess)
        
        mean_ess = np.mean(list(ess_stats.values()))
        return mean_ess, ess_stats

    def _hmc_sampling(self, X: np.ndarray, y: np.ndarray, 
                     initial_w: np.ndarray, n_chains: int = 4, return_chains: bool = False) -> Tuple[np.ndarray, List[float], Dict[str, float], Dict[str, float], Optional[List[np.ndarray]]]:
        # Improved HMC sampling with better initialization and diagnostics
        n_steps = self.config.hmc_steps
        n_params = len(initial_w)
        
        chains = []
        acceptance_probs = []
        
        for chain in range(n_chains):
            # Initialize chain with different starting points
            if chain == 0:
                w = initial_w.copy()
            else:
                w = initial_w + 0.1 * np.random.randn(n_params)
            
            chain_samples = []
            chain_acceptance = []
            
            for step in range(n_steps):
                # Initialize momentum
                momentum = np.random.randn(n_params)
                
                # HMC step
                w_new, momentum_new, acceptance_prob = self._hmc_step(X, y, w, momentum)
                
                # Accept or reject
                if np.random.random() < acceptance_prob:
                    w = w_new
                
                chain_samples.append(w.copy())
                chain_acceptance.append(acceptance_prob)
            
            chains.append(np.array(chain_samples))
            acceptance_probs.extend(chain_acceptance)
        
        # Calculate diagnostics
        r_hat_mean, r_hat_stats = self._calculate_gelman_rubin(chains)
        ess_mean, ess_stats = self._calculate_effective_sample_size(chains)
        
        # Return mean of final samples from all chains
        final_samples = np.array([chain[-1] for chain in chains])
        mean_w = np.mean(final_samples, axis=0)
        
        if return_chains:
            return mean_w, acceptance_probs, r_hat_stats, ess_stats, chains
        else:
            return mean_w, acceptance_probs, r_hat_stats, ess_stats, None

    def _plot_trace_chains(self, chains, param_names, output_dir, top_indices, fold=1):
        os.makedirs(output_dir, exist_ok=True)
        n_chains = len(chains)
        n_steps = chains[0].shape[0]
        
        for idx in top_indices:
            plt.figure(figsize=(12, 6))
            
            # Plot all chains
            for c in range(n_chains):
                plt.plot(range(n_steps), chains[c][:, idx], 
                        label=f'Chain {c+1}', alpha=0.7, linewidth=1)
            
            plt.title(f'Trace Plot: {param_names[idx]} (Fold {fold})')
            plt.xlabel('HMC Step')
            plt.ylabel('Parameter Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'trace_{param_names[idx]}_fold{fold}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None, output_dir: Optional[str] = None) -> 'AdaptivePriorARD':
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape
        self.alpha = np.clip(self.config.alpha_0, 1e-10, None)
        self.beta = np.ones(n_features) * np.clip(self.config.beta_0, 1e-10, None)
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        self._initialize_adaptive_priors(n_features)
        
        X_train_scaled = self.scaler_X.fit_transform(X)
        y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Initialize log files
        beta_tau_log = None
        em_progress_log = None
        aeh_log = None
        
        if output_dir is not None:
            beta_tau_log_path = os.path.join(output_dir, 'beta_tau_log.txt')
            em_progress_log_path = os.path.join(output_dir, 'em_progress_log.txt')
            aeh_hyperparams_log_path = os.path.join(output_dir, 'aeh_hyperparams_log.txt')
            beta_tau_log = open(beta_tau_log_path, 'w')
            em_progress_log = open(em_progress_log_path, 'w')
            aeh_log = open(aeh_hyperparams_log_path, 'w')
        
        for iteration in range(self.config.max_iter):
            try:
                self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                     np.diag(np.clip(self.beta, 1e-10, None)))
            except np.linalg.LinAlgError:
                jitter = 1e-6 * np.eye(n_features)
                self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                     np.diag(np.clip(self.beta, 1e-10, None)) + jitter)
            
            self.m = self.alpha * self.S @ X_train_scaled.T @ y_train_scaled
            
            if self.config.use_hmc:
                print(f"[DEBUG] HMC is running! Iteration: {iteration}")
                self.m, acceptance_probs, r_hat_stats, ess_stats, chains = self._hmc_sampling(
                    X_train_scaled, y_train_scaled, self.m, return_chains=True
                )
                
                mean_acceptance = np.mean(acceptance_probs)
                mean_r_hat = np.mean(list(r_hat_stats.values()))
                mean_ess = np.mean(list(ess_stats.values()))
                
                print(f"Iteration {iteration}: Acceptance={mean_acceptance:.3f}, R-hat={mean_r_hat:.3f}, ESS={mean_ess:.1f}")
                
                if not hasattr(self, 'convergence_history'):
                    self.convergence_history = []
                
                self.convergence_history.append({
                    'iteration': iteration,
                    'r_hat_mean': mean_r_hat,
                    'r_hat_std': np.std(list(r_hat_stats.values())),
                    'r_hat_stats': r_hat_stats,
                    'ess_mean': mean_ess,
                    'ess_std': np.std(list(ess_stats.values())),
                    'ess_stats': ess_stats,
                    'acceptance_mean': mean_acceptance
                })
                
                # Save diagnostics
                if output_dir is not None:
                    with open(os.path.join(output_dir, f'hmc_rhat_ess_iter{iteration+1}.json'), 'w') as f:
                        json.dump({'r_hat': r_hat_stats, 'ess': ess_stats}, f, indent=4)
                    
                    with open(os.path.join(output_dir, f'hmc_acceptance_iter{iteration+1}.json'), 'w') as f:
                        json.dump({'acceptance_probs': acceptance_probs}, f, indent=4)
                    
                    # Plot trace plots
                    if chains is not None and feature_names is not None:
                        top_indices = list(range(len(feature_names)))
                        self._plot_trace_chains(chains, feature_names, output_dir, top_indices, fold=iteration+1)
            
            # Log beta and tau values
            if beta_tau_log is not None:
                beta_tau_log.write(f"Iter {iteration}: beta={self.beta.tolist()}\n")
                if hasattr(self, 'group_prior_hyperparams'):
                    for group, params in self.group_prior_hyperparams.items():
                        if 'tau' in params:
                            beta_tau_log.write(f"  group {group} tau={params['tau']}\n")
            
            # Log weights and predictions
            if em_progress_log is not None:
                y_pred_em = X_train_scaled @ self.m
                y_pred_em_unscaled = self.scaler_y.inverse_transform(y_pred_em.reshape(-1, 1)).ravel()
                em_progress_log.write(f"Iter {iteration}: min_w={self.m.min()}, max_w={self.m.max()}, min_pred={y_pred_em.min()}, max_pred={y_pred_em.max()}, min_pred_unscaled={y_pred_em_unscaled.min()}, max_pred_unscaled={y_pred_em_unscaled.max()}\n")
                em_progress_log.flush()
            
            # Update parameters (rest of EM algorithm)
            residuals = y_train_scaled - X_train_scaled @ self.m
            
            if self.config.robust_noise:
                df = self.config.student_t_df
                weights = (df + 1) / (df + residuals**2)
                alpha_new = np.sum(weights) / (np.sum(weights * residuals**2) + 
                                            np.trace(X_train_scaled @ self.S @ X_train_scaled.T))
            else:
                alpha_new = n_samples / (np.sum(residuals**2) + 
                                       np.trace(X_train_scaled @ self.S @ X_train_scaled.T))
            
            alpha_new = np.clip(alpha_new, 1e-10, None)
            beta_new = np.zeros_like(self.beta)
            
            for group, indices in self.feature_groups.items():
                prior_type = self.config.group_prior_types.get(group, 'hierarchical')
                if prior_type == 'hierarchical':
                    for idx, j in enumerate(indices):
                        beta_new[j] = 1 / (np.clip(self.m[j]**2, 1e-10, None) +
                                               np.clip(np.diag(self.S)[j], 1e-10, None))
                        beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                elif prior_type == 'spike_slab':
                    for idx, j in enumerate(indices):
                        pi = self.group_prior_hyperparams[group]['pi'][idx]
                        sigma2_0 = self.group_prior_hyperparams[group]['sigma2_0'][idx]
                        sigma2_1 = self.group_prior_hyperparams[group]['sigma2_1'][idx]
                        beta_new[j] = (pi / np.clip(sigma2_1, 1e-10, None) +
                                       (1 - pi) / np.clip(sigma2_0, 1e-10, None))
                        beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                elif prior_type == 'horseshoe':
                    for idx, j in enumerate(indices):
                        tau = self.group_prior_hyperparams[group]['tau']
                        lambd = self.group_prior_hyperparams[group]['lambda'][idx]
                        beta_new[j] = 1 / (np.clip(self.m[j]**2, 1e-10, None) / (2 * tau) + lambd)
                        beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                elif prior_type == 'adaptive_elastic_horseshoe':
                    for idx, j in enumerate(indices):
                        alpha = self.group_prior_hyperparams[group]['alpha']
                        beta = self.group_prior_hyperparams[group]['beta']
                        tau = self.group_prior_hyperparams[group]['tau']
                        lambd = self.group_prior_hyperparams[group]['lambda'][idx]
                        
                        m2 = np.clip(self.m[j]**2, 1e-10, None)
                        horseshoe_term = m2 / (2 * tau) + lambd
                        elastic_term = alpha * np.abs(self.m[j]) + (1 - alpha) * m2
                        beta_new[j] = 1 / (horseshoe_term * (1 - beta) + elastic_term * beta)
                        beta_new[j] = np.clip(beta_new[j], 1e-10, None)
            
            if self.config.group_sparsity:
                for group, indices in self.feature_groups.items():
                    group_beta = np.mean(beta_new[indices])
                    beta_new[indices] = group_beta
            
            if self.config.dynamic_shrinkage:
                kappa = np.clip(self.shrinkage_params['kappa'], 0, 1)
                beta_new = beta_new * (1 - kappa) + self.beta * kappa
            
            self._update_adaptive_priors(iteration)
            
            # AEH diagnostics
            if hasattr(self, 'group_prior_hyperparams'):
                for group, params in self.group_prior_hyperparams.items():
                    if self.config.group_prior_types.get(group) == 'adaptive_elastic_horseshoe':
                        if aeh_log is not None:
                            aeh_log.write(f"Iter {iteration}, group {group}:\n")
                            for pname in ['lambda', 'tau', 'alpha', 'beta', 'momentum']:
                                if pname in params:
                                    aeh_log.write(f"  {pname}: {params[pname]}\n")
                            aeh_log.flush()
            
            # Check convergence
            beta_diff = np.abs(np.clip(beta_new, 1e-10, None) - np.clip(self.beta, 1e-10, None))
            alpha_diff = np.abs(alpha_new - self.alpha)
            
            if (alpha_diff < self.config.tol and np.all(beta_diff < self.config.tol)):
                print(f"[DEBUG] EM converged at iteration {iteration}")
                if em_progress_log is not None:
                    em_progress_log.write(f"[DEBUG] EM converged at iteration {iteration}\n")
                break
            
            self.alpha = alpha_new
            self.beta = np.clip(beta_new, 1e-10, None)
        
        # Close log files
        if beta_tau_log is not None:
            beta_tau_log.close()
        if em_progress_log is not None:
            em_progress_log.close()
        if aeh_log is not None:
            aeh_log.close()
        
        # Compute final metrics
        y_pred, y_std = self.predict(X, return_std=True)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mean_std = float(np.mean(y_std))
        crps = float(np.mean(np.abs(y - y_pred)) - 0.5 * np.mean(np.abs(y_std)))
        
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
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = X_scaled @ self.m
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            # Calculate uncertainty
            epistemic_var = np.diag(X_scaled @ self.S @ X_scaled.T)
            aleatoric_var = 1.0 / self.alpha
            total_var = epistemic_var + aleatoric_var
            
            # Apply calibration
            calibrated_var = total_var * (self.uncertainty_calibration_factor ** 2)
            y_std = np.sqrt(calibrated_var)
            
            return y_pred, y_std
        else:
            return y_pred, None
    
    def get_feature_importance(self) -> np.ndarray:
        return np.abs(self.m)
    
    def save_model(self, path: str):
        model_data = {
            'config': self.config,
            'alpha': self.alpha,
            'beta': self.beta,
            'm': self.m,
            'S': self.S,
            'prior_hyperparams': self.prior_hyperparams,
            'feature_groups': self.feature_groups,
            'shrinkage_params': self.shrinkage_params,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'uncertainty_calibration_factor': self.uncertainty_calibration_factor
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'AdaptivePriorARD':
        model_data = joblib.load(path)
        model = cls(model_data['config'])
        model.alpha = model_data['alpha']
        model.beta = model_data['beta']
        model.m = model_data['m']
        model.S = model_data['S']
        model.prior_hyperparams = model_data['prior_hyperparams']
        model.feature_groups = model_data['feature_groups']
        model.shrinkage_params = model_data['shrinkage_params']
        model.scaler_X = model_data['scaler_X']
        model.scaler_y = model_data['scaler_y']
        model.uncertainty_calibration_factor = model_data['uncertainty_calibration_factor']
        return model

def feature_engineering_no_interactions(df: pd.DataFrame) -> pd.DataFrame:
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
    return df

if __name__ == "__main__":
    # Configure model with FIXED HMC parameters
    config = AdaptivePriorConfig(
        beta_0=0.1,
        group_sparsity=True,
        dynamic_shrinkage=True,
        max_iter=50,
        tol=1e-4,
        use_hmc=True,  # Enable HMC
        hmc_steps=50,  # FIXED: Increased from 20
        hmc_epsilon=0.001,  # FIXED: Increased from 0.0001
        hmc_leapfrog_steps=5,  # FIXED: Increased from 3
        robust_noise=True,
        uncertainty_calibration=True,
        group_prior_types={
            'energy': 'adaptive_elastic_horseshoe',
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        }
    )
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultshmc_fixed')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    
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
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Fit model
    model = AdaptivePriorARD(config)
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    
    # Save results
    y_pred, y_std = model.predict(X, return_std=True)
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred))
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model
    model.save_model(os.path.join(results_dir, 'adaptive_prior_model.joblib'))
    
    # Save feature importance
    feature_importance = model.get_feature_importance()
    importance_dict = {name: float(importance) for name, importance in zip(feature_names, feature_importance)}
    
    with open(os.path.join(results_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_dict, f, indent=4)
    
    # Save convergence history
    if hasattr(model, 'convergence_history') and model.convergence_history:
        with open(os.path.join(results_dir, 'convergence_history.json'), 'w') as f:
            json.dump(model.convergence_history, f, indent=4, cls=NumpyEncoder)
    
    print(f"Results saved to {results_dir}")
    print(f"Final metrics: {metrics}")
    print(f"Feature importance: {importance_dict}") 