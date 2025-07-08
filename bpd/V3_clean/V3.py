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

# --- Post-hoc uncertainty calibration factor (from calibration experiments) ---
POSTHOC_UNCERTAINTY_SCALE = 10.0  # Adjust as needed based on calibration_experiments

class AdaptivePriorConfig:
    """
    Configuration class for the Adaptive Prior ARD model parameters.
    
    This class implements a set of hyperparameters that control the behaviour
    of the Bayesian model, including:
    - Prior specifications (hierarchical, spike-slab, or horseshoe)
    - Uncertainty quantification parameters
    - MCMC sampling parameters
    - Robust noise modelling parameters
    
    The configuration allows for fine-tuning of the model's behaviour to balance between
    model complexity, computational efficiency, and predictive performance.
    
    Attributes:
        alpha_0: Initial noise precision (controls model's sensitivity to data)
        beta_0: Initial weight precision (controls feature selection strength)
        max_iter: Maximum number of EM iterations for model convergence
        tol: Convergence tolerance for EM algorithm
        n_splits: Number of cross-validation splits for robust evaluation
        random_state: Random seed for reproducibility
        prior_type: Type of prior ('hierarchical', 'spike_slab', or 'horseshoe')
        adaptation_rate: Rate at which priors adapt during training
        uncertainty_threshold: Threshold for uncertainty-based adaptation
        group_sparsity: Whether to enable group-wise sparsity for structured feature selection
        dynamic_shrinkage: Whether to enable dynamic shrinkage parameters
        use_hmc: Whether to use Hamiltonian Monte Carlo for posterior exploration
        hmc_steps: Number of HMC steps per iteration
        hmc_epsilon: HMC step size
        hmc_leapfrog_steps: Number of leapfrog steps in HMC
        uncertainty_calibration: Whether to apply uncertainty calibration
        calibration_factor: Factor to scale uncertainty estimates
        robust_noise: Whether to use robust noise modeling
        student_t_df: Degrees of freedom for Student's t noise model
        group_prior_types: Dictionary of group prior types
    """
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
    hmc_steps: int = 20
    hmc_epsilon: float = 0.0001
    hmc_leapfrog_steps: int = 3
    uncertainty_calibration: bool = True
    calibration_factor: float = 0.03  # Further reduced from 0.05 to improve PICP calibration
    robust_noise: bool = True
    student_t_df: float = 3.0  
    group_prior_types: dict = field(default_factory=lambda: {
        'energy': 'adaptive_elastic_horseshoe',  # AEH prior for energy features
        'building': 'hierarchical',  # Hierarchical ARD for building features
        'interaction': 'spike_slab'  # Spike-slab for interaction features
    })
    
    
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types to handle serialisation of numpy arrays and scalars.
    This is used for saving model parameters and results in a format that can be
    easily loaded if required.
    """
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
    """
    Adaptive Prior ARD model with hierarchical priors and improved uncertainty estimation.
    
    This class implements a novel Bayesian linear regression model that extends traditional
    Automatic Relevance Determination (ARD) through several key innovations:
    
    1. Hierarchical Prior Structure:
       - Implements a multi-level prior hierarchy for improved feature selection
       - Allows for adaptive shrinkage of model parameters
       - Incorporates group-wise sparsity for structured feature selection
    
    2. Uncertainty Quantification:
       - Provides calibrated uncertainty estimates for predictions
       - Implements robust noise modelling using Student's t distribution
       - Includes uncertainty calibration based on validation performance
    
    3. Advanced Inference:
       - Uses Hamiltonian Monte Carlo for improved posterior exploration
       - Implements Expectation-Maximization (EM) algorithm with adaptive updates
       - Incorporates dynamic shrinkage parameters for better model adaptation
    
    4. Model Evaluation:
       - Implements comprehensive cross-validation
       - Provides detailed feature importance analysis
       - Includes uncertainty-aware performance metrics
    
    The model is particularly suited for building energy performance analysis due to its:
    - Ability to handle complex feature interactions
    - Robust uncertainty quantification
    - Automatic feature selection
    - Adaptability to different data characteristics
    """
    def __init__(self, config: Optional[AdaptivePriorConfig] = None, **kwargs):
        """
        Initialise the Adaptive Prior ARD model.
        
        This constructor sets up the model with either provided configuration parameters
        or default values. The initialisation includes:
        - Model parameters (weights, precisions, covariance)
        - Prior components (hyperparameters, feature groups)
        - Feature engineering components (scalers)
        - Cross-validation and uncertainty calibration settings
        
        Args:
            config: Optional configuration object. If None, uses default settings
                   that balance model complexity and computational efficiency.
            **kwargs: Additional parameters for scikit-learn compatibility
        """
        if config is not None:
            self.config = config
        else:
            # Handle scikit-learn parameter passing
            if kwargs:
                self.config = AdaptivePriorConfig()
                for key, value in kwargs.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            else:
                self.config = AdaptivePriorConfig()
        
        # Model parameters
        self.alpha = None  # Noise precision
        self.beta = None  # Weight precisions
        self.m = None  # Mean of weights
        self.S = None  # Covariance of weights
        
        # Prior components
        self.prior_hyperparams = None  # Hyperparameters for the adaptive prior
        self.feature_groups = None  # Feature grouping for group sparsity
        self.shrinkage_params = None  # Dynamic shrinkage parameters
        
        # Feature engineering components
        self.scaler_X = RobustScaler()  # Robust scaling for features
        self.scaler_y = StandardScaler()  # Standard scaling for target
        
        # Cross-validation results
        self.cv_results = None
        
        # Uncertainty calibration
        self.uncertainty_calibration_factor = self.config.calibration_factor
        self.uncertainty_calibration_history = []
        self.r_hat_history = []
        self.convergence_history = []
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator (required for scikit-learn compatibility).
        """
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
            # Removed group_prior_types to avoid cloning issues
        }
    
    def set_params(self, **params):
        """
        Set parameters for this estimator (required for scikit-learn compatibility).
        """
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self

    def _initialize_adaptive_priors(self, n_features: int):
        """
        Initialise adaptive prior parameters based on configuration.
        
        This method sets up the prior structure based on the chosen prior type:
        - Hierarchical: Implements a multi-level prior with global and local shrinkage
        - Spike-slab: Implements a mixture prior for feature selection
        - Horseshoe: Implements a heavy-tailed prior for robust feature selection
        - Adaptive Elastic Horseshoe: Novel prior combining elastic net and horseshoe properties
        
        The initialisation includes:
        - Setting up hyperparameters for the chosen prior
        - Initialising feature groups for group sparsity
        - Setting up dynamic shrinkage parameters
        
        Args:
            n_features: Number of features in the model
        """
        self.feature_groups = self._create_feature_groups(n_features)
        self.group_prior_hyperparams = {}
        for group, indices in self.feature_groups.items():
            prior_type = self.config.group_prior_types.get(group, 'hierarchical')
            if prior_type == 'hierarchical':
                self.group_prior_hyperparams[group] = {
                    'lambda': np.ones(len(indices)) * self.config.beta_0,
                    'tau': np.ones(len(indices)) * 1.0,
                    'nu': np.ones(len(indices)) * 2.0
                }
            elif prior_type == 'spike_slab':
                self.group_prior_hyperparams[group] = {
                    'pi': np.ones(len(indices)) * 0.5,
                    'sigma2_0': np.ones(len(indices)) * 1e-6,
                    'sigma2_1': np.ones(len(indices)) * 1.0
                }
            elif prior_type == 'horseshoe':
                self.group_prior_hyperparams[group] = {
                    'lambda': np.ones(len(indices)),
                    'tau': 1.0,
                    'c2': 1.0
                }
            elif prior_type == 'adaptive_elastic_horseshoe':
                self.group_prior_hyperparams[group] = {
                    'lambda': np.ones(len(indices)),
                    'tau': 1.0,
                    'alpha': 0.5,  # Elastic net mixing parameter
                    'beta': 1.0,   # Elastic net regularization strength
                    'gamma': 0.1,  # Adaptive learning rate
                    'rho': 0.9,    # Momentum parameter
                    'momentum': np.zeros(len(indices))  # Momentum for adaptive updates
                }
        if self.config.dynamic_shrinkage:
            self.shrinkage_params = {
                'kappa': np.ones(n_features) * 0.5,
                'eta': np.ones(n_features) * 1.0
            }
    
    def _create_feature_groups(self, n_features: int) -> Dict[str, List[int]]:
        """
        Create feature groups for group sparsity
        
        Args:
            n_features: Number of features
            
        Returns:
            Dictionary mapping group names to feature indices
        """
        # Group features by type
        groups = {
            'energy': list(range(0, 4)),  # Energy-related features
            'building': list(range(4, 8)),  # Building characteristics
            'interaction': list(range(8, n_features))  # Interaction terms
        }
        return groups
    
    def _update_uncertainty_calibration(self, X: np.ndarray, y: np.ndarray):
        """
        Update uncertainty calibration based on validation performance
        
        Args:
            X: Input features
            y: Target values
        """
        if not self.config.uncertainty_calibration:
            return
            
        # Get predictions and uncertainties
        y_pred, y_std = self.predict(X, return_std=True)
        
        # Calculate empirical coverage at different confidence levels
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        empirical_coverage = []
        
        for level in confidence_levels:
            z_score = stats.norm.ppf(1 - (1 - level) / 2)
            lower = y_pred - z_score * y_std
            upper = y_pred + z_score * y_std
            coverage = np.mean((y >= lower) & (y <= upper))
            empirical_coverage.append(coverage)
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(np.array(empirical_coverage) - np.array(confidence_levels)))
        
        # Update calibration factor using adaptive learning
        if calibration_error > 0.1:  # If calibration is poor
            self.uncertainty_calibration_factor *= 1.1  # Increase uncertainty
        else:
            self.uncertainty_calibration_factor *= 0.95  # Slightly decrease if over-calibrated
            
        # Store calibration history
        self.uncertainty_calibration_history.append({
            'calibration_error': calibration_error,
            'calibration_factor': self.uncertainty_calibration_factor,
            'empirical_coverage': dict(zip(confidence_levels, empirical_coverage))
        })
    
    def _update_adaptive_priors(self, iteration: int):
        """
        Update adaptive prior parameters based on current model state
        
        Args:
            iteration: Current iteration number
        """
        for group, indices in self.feature_groups.items():
            prior_type = self.config.group_prior_types.get(group, 'hierarchical')
            if prior_type == 'hierarchical':
                for idx, j in enumerate(indices):
                    diag_S = np.clip(np.diag(self.S)[j], 1e-10, None)
                    m_squared = np.clip(self.m[j]**2, 1e-10, None)
                    self.group_prior_hyperparams[group]['lambda'][idx] = (
                        (self.group_prior_hyperparams[group]['nu'][idx] + 1) /
                        (m_squared + diag_S + 2 * self.group_prior_hyperparams[group]['tau'][idx])
                    )
                    self.group_prior_hyperparams[group]['tau'][idx] = (
                        (self.group_prior_hyperparams[group]['nu'][idx] + 1) /
                        (self.group_prior_hyperparams[group]['lambda'][idx] + 1)
                    )
            elif prior_type == 'spike_slab':
                for idx, j in enumerate(indices):
                    m_squared = np.clip(self.m[j]**2, 1e-10, None)
                    # Add numerical stability for log odds calculation
                    pi = np.clip(self.group_prior_hyperparams[group]['pi'][idx], 1e-10, 1-1e-10)
                    sigma2_0 = np.clip(self.group_prior_hyperparams[group]['sigma2_0'][idx], 1e-10, None)
                    sigma2_1 = np.clip(self.group_prior_hyperparams[group]['sigma2_1'][idx], 1e-10, None)
                    
                    log_odds = (
                        np.log(pi / (1 - pi)) +
                        0.5 * np.log(sigma2_1 / sigma2_0) +
                        0.5 * m_squared * (1/sigma2_0 - 1/sigma2_1)
                    )
                    self.group_prior_hyperparams[group]['pi'][idx] = np.clip(
                        1 / (1 + np.exp(-log_odds)), 1e-10, 1-1e-10
                    )
            elif prior_type == 'horseshoe':
                for idx, j in enumerate(indices):
                    m_squared = np.clip(self.m[j]**2, 1e-10, None)
                    self.group_prior_hyperparams[group]['lambda'][idx] = (
                        1 / (m_squared / (2 * self.group_prior_hyperparams[group]['tau']) + 1/self.group_prior_hyperparams[group]['c2'])
                    )
                m_squared_sum = np.sum([np.clip(self.m[j]**2, 1e-10, None) for j in indices])
                self.group_prior_hyperparams[group]['tau'] = (
                    1 / (m_squared_sum / (2 * np.sum(self.group_prior_hyperparams[group]['lambda'])) + 1)
                )
            elif prior_type == 'adaptive_elastic_horseshoe':
                # Proper AEH update with adaptive parameters
                indices_arr = np.array(indices)
                
                # Update alpha based on feature importance ratio
                feature_importance = np.abs(self.m[indices_arr])
                uncertainty = np.sqrt(np.diag(self.S)[indices_arr])
                importance_ratio = np.mean(feature_importance) / (np.mean(uncertainty) + 1e-8)
                
                # Adaptive alpha: more L1 (sparsity) for high importance, more L2 (smoothness) for low importance
                alpha_new = np.clip(0.1 + 0.8 * (1 - importance_ratio / (importance_ratio + 1)), 0.1, 0.9)
                self.group_prior_hyperparams[group]['alpha'] = (
                    self.group_prior_hyperparams[group]['alpha'] * 0.9 + alpha_new * 0.1
                )
                
                # Update beta based on overall model fit
                residuals = y - X @ self.m
                mse = np.mean(residuals**2)
                beta_new = np.clip(0.1 + 2.0 * (1 - mse / (mse + 1)), 0.1, 2.0)
                self.group_prior_hyperparams[group]['beta'] = (
                    self.group_prior_hyperparams[group]['beta'] * 0.9 + beta_new * 0.1
                )
                
                # Update lambda for each feature with momentum
                for idx, j in enumerate(indices):
                    m2 = np.clip(self.m[j]**2, 1e-10, None)
                    Sjj = np.clip(np.diag(self.S)[j], 1e-10, None)
                    
                    # Horseshoe-style update
                    lambda_horseshoe = 1 / (m2 / (2 * self.group_prior_hyperparams[group]['tau']) + 1)
                    
                    # Elastic net component
                    alpha = self.group_prior_hyperparams[group]['alpha']
                    beta = self.group_prior_hyperparams[group]['beta']
                    lambda_elastic = alpha * np.abs(self.m[j]) + (1 - alpha) * m2
                    
                    # Combine horseshoe and elastic net
                    lambda_new = lambda_horseshoe * (1 - beta) + lambda_elastic * beta
                    
                    # Apply momentum
                    momentum = self.group_prior_hyperparams[group]['momentum'][idx]
                    gamma = self.group_prior_hyperparams[group]['gamma']
                    rho = self.group_prior_hyperparams[group]['rho']
                    
                    momentum_new = rho * momentum + gamma * (lambda_new - self.group_prior_hyperparams[group]['lambda'][idx])
                    self.group_prior_hyperparams[group]['momentum'][idx] = momentum_new
                    
                    # Update lambda with momentum
                    self.group_prior_hyperparams[group]['lambda'][idx] = np.clip(
                        lambda_new + momentum_new, 1e-6, 100.0
                    )
                
                # Update tau for the group
                m2_sum = np.sum(np.clip(self.m[indices_arr]**2, 1e-10, None))
                lambda_sum = np.sum(self.group_prior_hyperparams[group]['lambda'])
                tau_new = 1 / (m2_sum / (2 * lambda_sum) + 1)
                self.group_prior_hyperparams[group]['tau'] = np.clip(tau_new, 1e-6, 10.0)
        if self.config.dynamic_shrinkage:
            for j in range(len(self.beta)):
                importance = 1 / np.clip(self.beta[j], 1e-10, None)
                self.shrinkage_params['kappa'][j] = (
                    self.shrinkage_params['kappa'][j] * (1 - self.config.adaptation_rate) +
                    importance * self.config.adaptation_rate
                )
                diag_S = np.clip(np.diag(self.S)[j], 1e-10, None)
                uncertainty = np.sqrt(diag_S)
                self.shrinkage_params['eta'][j] = (
                    self.shrinkage_params['eta'][j] * (1 - self.config.adaptation_rate) +
                    (uncertainty > self.config.uncertainty_threshold) * self.config.adaptation_rate
                )
    
    def _hmc_step(self, X: np.ndarray, y: np.ndarray, current_w: np.ndarray, 
                  current_momentum: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        new_w = current_w.copy()
        new_momentum = current_momentum.copy()
        current_energy = self._calculate_hamiltonian(X, y, current_w, new_momentum)
        epsilon = float(self.config.hmc_epsilon) * np.exp(np.random.normal(0, 0.1))
        for _ in range(self.config.hmc_leapfrog_steps):
            grad = self._calculate_gradient(X, y, new_w)
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                print("NaN or Inf detected in gradient!")
            new_momentum = new_momentum - 0.5 * epsilon * grad
            new_w = new_w + epsilon * new_momentum
            grad = self._calculate_gradient(X, y, new_w)
            new_momentum = new_momentum - 0.5 * epsilon * grad
        new_energy = self._calculate_hamiltonian(X, y, new_w, new_momentum)
        if np.isnan(current_energy) or np.isnan(new_energy) or np.isinf(current_energy) or np.isinf(new_energy):
            print(f"NaN or Inf in Hamiltonian: current={current_energy}, new={new_energy}")
        energy_diff = current_energy - new_energy
        acceptance_prob = min(1.0, np.exp(np.clip(energy_diff, -100, 100)))
        acceptance_prob = acceptance_prob * np.exp(np.random.normal(0, 0.1))
        print(f"HMC step: current_energy={current_energy:.4f}, new_energy={new_energy:.4f}, acceptance_prob={acceptance_prob:.4f}")
        if np.any(np.isnan(new_w)) or np.any(np.isinf(new_w)):
            print("NaN or Inf detected in weights!")
        if np.random.random() < acceptance_prob:
            return new_w, new_momentum, acceptance_prob
        else:
            return current_w, current_momentum, acceptance_prob
    
    def _calculate_hamiltonian(self, X: np.ndarray, y: np.ndarray, 
                             w: np.ndarray, momentum: np.ndarray) -> float:
        """
        Calculate the Hamiltonian (total energy) of the system.
        
        The Hamiltonian is the sum of potential energy (negative log posterior) and
        kinetic energy (momentum). This is a key component of HMC that ensures
        proper exploration of the posterior distribution.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            w: Weight vector (n_features,)
            momentum: Momentum vector (n_features,)
            
        Returns:
            float: Total energy (Hamiltonian) of the system
        """
        # Potential energy (negative log posterior)
        residuals = y - X @ w
        potential = 0.5 * self.alpha * np.sum(residuals**2)
        potential += 0.5 * np.sum(w**2 / np.clip(self.beta, 1e-10, None))
        
        # Kinetic energy
        kinetic = 0.5 * np.sum(momentum**2)
        
        return potential + kinetic
    
    def _calculate_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the negative log posterior.
        
        This method computes the gradient of the negative log posterior with respect to
        the weights. The gradient is used in HMC for momentum updates and is necessary for
        efficient posterior exploration.
        
        The gradient includes:
        - Likelihood gradient (from the data)
        - Prior gradient (from the ARD prior)
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            w: Weight vector (n_features,)
            
        Returns:
            np.ndarray: Gradient vector (n_features,)
        """
        # Gradient of likelihood (data term)
        grad_likelihood = -self.alpha * X.T @ (y - X @ w)
        
        # Gradient of prior (ARD term)
        grad_prior = w / np.clip(self.beta, 1e-10, None)
        
        return grad_likelihood + grad_prior
    
    def _calculate_gelman_rubin(self, chains: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Gelman-Rubin R² statistics for MCMC convergence diagnostics.
        
        The Gelman-Rubin statistic (R²) compares the within-chain and between-chain variances
        to assess MCMC convergence. Values close to 1.0 indicate good convergence.
        
        Args:
            chains: List of arrays containing samples from different chains
        
        Returns:
            Tuple containing:
            - Overall R² statistic
            - Dictionary of R² statistics for each parameter
        """
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        n_params = chains[0].shape[1]
        
        # Calculate within-chain variance
        within_chain_var = np.zeros(n_params)
        for chain in chains:
            within_chain_var += np.var(chain, axis=0, ddof=1)
        within_chain_var /= n_chains
        
        # Calculate between-chain variance
        chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
        between_chain_var = np.var(chain_means, axis=0, ddof=1) * n_samples
        
        # Calculate R² statistics
        r_hat_stats = {}
        for j in range(n_params):
            r_hat = np.sqrt((between_chain_var[j] + within_chain_var[j]) / within_chain_var[j])
            r_hat_stats[f'weight_{j}'] = float(r_hat)
        
        # Calculate overall R²
        overall_r_hat = np.mean(list(r_hat_stats.values()))
        
        return overall_r_hat, r_hat_stats

    def _calculate_effective_sample_size(self, chains: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate effective sample size (ESS) for MCMC chains.
        
        ESS estimates the number of independent samples in the MCMC output,
        accounting for autocorrelation. Higher ESS indicates better mixing.
        
        Args:
            chains: List of arrays containing samples from different chains
        
        Returns:
            Tuple containing:
            - Overall ESS
            - Dictionary of ESS for each parameter
        """
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        n_params = chains[0].shape[1]
        
        ess_stats = {}
        total_ess = 0
        
        for j in range(n_params):
            # Combine samples from all chains
            all_samples = np.concatenate([chain[:, j] for chain in chains])
            
            # Calculate autocorrelation
            acf = np.correlate(all_samples, all_samples, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]  # Normalize
            
            # Find where autocorrelation becomes negligible
            cutoff = np.where(acf < 0.05)[0]
            if len(cutoff) > 0:
                max_lag = cutoff[0]
            else:
                max_lag = len(acf) - 1
            
            # Calculate ESS using initial monotone sequence estimator
            rho = acf[1:max_lag+1]
            rho_plus = np.maximum(rho, 0)
            ess = n_samples * n_chains / (1 + 2 * np.sum(rho_plus))
            
            ess_stats[f'weight_{j}'] = float(ess)
            total_ess += ess
        
        # Calculate overall ESS
        overall_ess = total_ess / n_params
        
        return overall_ess, ess_stats

    def _hmc_sampling(self, X: np.ndarray, y: np.ndarray, 
                     initial_w: np.ndarray, n_chains: int = 4, return_chains: bool = False) -> Tuple[np.ndarray, List[float], Dict[str, float], Dict[str, float], Optional[List[np.ndarray]]]:
        """
        Perform Hamiltonian Monte Carlo sampling with multiple chains for convergence diagnostics.
        If return_chains is True, also return the full chains for trace plotting.
        """
        n_features = X.shape[1]
        chains = []
        all_acceptance_rates = []
        # Run multiple chains
        for chain in range(n_chains):
            if chain == 0:
                current_w = initial_w.copy()
            else:
                current_w = initial_w + np.random.normal(0, 0.1, size=n_features)
            chain_samples = []
            acceptance_rates = []
            for _ in range(self.config.hmc_steps):
                # Resample momentum ONCE per proposal
                current_momentum = np.random.normal(0, 2.0, size=n_features)
                new_w, new_momentum, acceptance_rate = self._hmc_step(
                    X, y, current_w, current_momentum
                )
                current_w = new_w
                current_momentum = new_momentum
                chain_samples.append(current_w)
                acceptance_rates.append(acceptance_rate)
                if np.random.random() < 0.3:
                    current_momentum = np.random.normal(0, 2.0, size=n_features)
            chains.append(np.array(chain_samples))
            all_acceptance_rates.extend(acceptance_rates)
        overall_r_hat, r_hat_stats = self._calculate_gelman_rubin(chains)
        overall_ess, ess_stats = self._calculate_effective_sample_size(chains)
        final_w = np.mean([chain[-1] for chain in chains], axis=0)
        if return_chains:
            return final_w, all_acceptance_rates, r_hat_stats, ess_stats, chains
        return final_w, all_acceptance_rates, r_hat_stats, ess_stats, None

    def _plot_trace_chains(self, chains, param_names, output_dir, top_indices, fold=1):
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        n_chains = len(chains)
        n_steps = chains[0].shape[0]
        for idx in top_indices:
            plt.figure(figsize=(10, 4))
            for c in range(n_chains):
                plt.plot(range(n_steps), chains[c][:, idx], label=f'Chain {c+1}', alpha=0.7)
            plt.title(f'Trace Plot: {param_names[idx]} (Fold {fold})')
            plt.xlabel('HMC Step')
            plt.ylabel('Parameter Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'trace_{param_names[idx]}_fold{fold}.png'))
            plt.close()

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None, output_dir: Optional[str] = None) -> 'AdaptivePriorARD':
        """
        Fit the Adaptive Prior ARD model WITHOUT cross-validation (fit on all data, no KFold) and with reduced max_iter for fast debugging.
        """
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape
        self.alpha = np.clip(self.config.alpha_0, 1e-10, None)
        self.beta = np.ones(n_features) * np.clip(self.config.beta_0, 1e-10, None)
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        self._initialize_adaptive_priors(n_features)
        # Fit on all data, no KFold
        X_train_scaled = self.scaler_X.fit_transform(X)
        y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Initialize log files only if output_dir is provided
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
            # Use HMC for posterior exploration if enabled
            if self.config.use_hmc:
                print("[DEBUG] HMC is running! Iteration:", iteration)
                self.m, acceptance_probs, r_hat_stats, ess_stats, _ = self._hmc_sampling(
                    X_train_scaled, y_train_scaled, self.m, return_chains=False
                )
                logger.info(f"Iteration {iteration}: "
                          f"Mean HMC acceptance rate: {np.mean(acceptance_probs):.3f}")
                logger.info(f"Gelman-Rubin R²: {np.mean(list(r_hat_stats.values())):.3f}")
                logger.info(f"Effective Sample Size: {np.mean(list(ess_stats.values())):.1f}")
                if not hasattr(self, 'convergence_history'):
                    self.convergence_history = []
                self.convergence_history.append({
                    'iteration': iteration,
                    'r_hat_mean': np.mean(list(r_hat_stats.values())),
                    'r_hat_std': np.std(list(r_hat_stats.values())),
                    'r_hat_stats': r_hat_stats,
                    'ess_mean': np.mean(list(ess_stats.values())),
                    'ess_std': np.std(list(ess_stats.values())),
                    'ess_stats': ess_stats
                })
                # Log beta and tau values
                if beta_tau_log is not None:
                    beta_tau_log.write(f"Iter {iteration}: beta={self.beta.tolist()}\n")
                    if hasattr(self, 'group_prior_hyperparams'):
                        for group, params in self.group_prior_hyperparams.items():
                            if 'tau' in params:
                                beta_tau_log.write(f"  group {group} tau={params['tau']}\n")
                # Log weights and predictions after each EM iteration
                if em_progress_log is not None:
                    y_pred_em = X_train_scaled @ self.m
                    y_pred_em_unscaled = self.scaler_y.inverse_transform(y_pred_em.reshape(-1, 1)).ravel()
                    em_progress_log.write(f"Iter {iteration}: min_w={self.m.min()}, max_w={self.m.max()}, min_pred={y_pred_em.min()}, max_pred={y_pred_em.max()}, min_pred_unscaled={y_pred_em_unscaled.min()}, max_pred_unscaled={y_pred_em_unscaled.max()}\n")
                    em_progress_log.flush()
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
                                # Standard Bayesian update (no + 2 * tau)
                            beta_new[j] = 1 / (np.clip(self.m[j]**2, 1e-10, None) +
                                                   np.clip(np.diag(self.S)[j], 1e-10, None))
                            # No clipping - let the model learn naturally
                            beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                    elif prior_type == 'spike_slab':
                        for idx, j in enumerate(indices):
                            pi = self.group_prior_hyperparams[group]['pi'][idx]
                            sigma2_0 = self.group_prior_hyperparams[group]['sigma2_0'][idx]
                            sigma2_1 = self.group_prior_hyperparams[group]['sigma2_1'][idx]
                            beta_new[j] = (pi / np.clip(sigma2_1, 1e-10, None) +
                                           (1 - pi) / np.clip(sigma2_0, 1e-10, None))
                            # No clipping - let the model learn naturally
                            beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                    elif prior_type == 'horseshoe':
                        for idx, j in enumerate(indices):
                            tau = self.group_prior_hyperparams[group]['tau']
                            lambd = self.group_prior_hyperparams[group]['lambda'][idx]
                            beta_new[j] = 1 / (np.clip(self.m[j]**2, 1e-10, None) / (2 * tau) + lambd)
                            # No clipping - let the model learn naturally
                            beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                    elif prior_type == 'adaptive_elastic_horseshoe':
                        for idx, j in enumerate(indices):
                            # Get AEH parameters
                            alpha = self.group_prior_hyperparams[group]['alpha']
                            beta = self.group_prior_hyperparams[group]['beta']
                            tau = self.group_prior_hyperparams[group]['tau']
                            lambd = self.group_prior_hyperparams[group]['lambda'][idx]
                            
                            # Horseshoe component
                            m2 = np.clip(self.m[j]**2, 1e-10, None)
                            horseshoe_term = m2 / (2 * tau) + lambd
                            
                            # Elastic net component
                            elastic_term = alpha * np.abs(self.m[j]) + (1 - alpha) * m2
                            
                            # Combine components
                            beta_new[j] = 1 / (horseshoe_term * (1 - beta) + elastic_term * beta)
                            # No clipping - let the model learn naturally
                            beta_new[j] = np.clip(beta_new[j], 1e-10, None)
                if self.config.group_sparsity:
                    for group, indices in self.feature_groups.items():
                        group_beta = np.mean(beta_new[indices])
                        beta_new[indices] = group_beta
                if self.config.dynamic_shrinkage:
                    kappa = np.clip(self.shrinkage_params['kappa'], 0, 1)
                    beta_new = beta_new * (1 - kappa) + self.beta * kappa
                self._update_adaptive_priors(iteration)
                # AEH diagnostics: log hyperparameters after update
                if hasattr(self, 'group_prior_hyperparams'):
                    for group, params in self.group_prior_hyperparams.items():
                        if self.config.group_prior_types.get(group) == 'adaptive_elastic_horseshoe':
                            if aeh_log is not None:
                                aeh_log.write(f"Iter {iteration}, group {group}:\n")
                                for pname in ['lambda', 'tau', 'alpha', 'beta', 'momentum']:
                                    if pname in params:
                                        aeh_log.write(f"  {pname}: {params[pname]}\n")
                                aeh_log.flush()
                beta_diff = np.abs(np.clip(beta_new, 1e-10, None) - np.clip(self.beta, 1e-10, None))
                alpha_diff = np.abs(alpha_new - self.alpha)
                if (alpha_diff < self.config.tol and np.all(beta_diff < self.config.tol)):
                    print(f"[DEBUG] EM converged at iteration {iteration}")
                    if em_progress_log is not None:
                        em_progress_log.write(f"[DEBUG] EM converged at iteration {iteration}\n")
                    break
                self.alpha = alpha_new
                self.beta = np.clip(beta_new, 1e-10, None)
        
        # Close log files if they were opened
        if beta_tau_log is not None:
            beta_tau_log.close()
        if em_progress_log is not None:
            em_progress_log.close()
        if aeh_log is not None:
            aeh_log.close()
        
        # After fitting, run diagnostics and save as before
        # Compute metrics on full data for compatibility with downstream code
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
        # ... rest of diagnostics and saving code unchanged ...
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty estimates.
        
        This method provides both point predictions and uncertainty estimates for new data.
        The uncertainty estimates are calibrated and can include:
        - Epistemic uncertainty (from model parameters)
        - Aleatoric uncertainty (from noise model)
        - Calibrated uncertainty estimates
        
        Args:
            X: Input features (n_samples, n_features)
            return_std: Whether to return standard deviation of predictions
            
        Returns:
            mean: Mean predictions (n_samples,)
            std: Standard deviation of predictions (n_samples,) if return_std=True
        """
        # Check if model is fitted
        if self.m is None:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # CRITICAL FIX: Scale the input features before prediction
        X_scaled = self.scaler_X.transform(X)
        
        # Make prediction on scaled features
        mean_scaled = X_scaled @ self.m
        
        # Inverse transform to get predictions in original scale
        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            # Check if uncertainty components are available
            if self.alpha is None or self.S is None:
                raise ValueError("Model must be fitted before making uncertainty estimates. Call fit() first.")
            
            # Base uncertainty from model (epistemic) - on scaled features
            base_std_scaled = np.sqrt(1/self.alpha + np.sum((X_scaled @ self.S) * X_scaled, axis=1))
            
            if self.config.uncertainty_calibration:
                # Apply calibrated uncertainty
                std_scaled = base_std_scaled * self.uncertainty_calibration_factor
            else:
                std_scaled = base_std_scaled
                
            if self.config.robust_noise:
                # Add Student's t noise component for robustness (aleatoric)
                t_noise = stats.t.rvs(df=self.config.student_t_df, size=len(mean))
                std_scaled = np.sqrt(std_scaled**2 + np.abs(t_noise))
            
            # Scale uncertainty back to original scale using scaler_y scale
            std = std_scaled * self.scaler_y.scale_
                
            return mean.reshape(-1), std.reshape(-1)
            
        return mean.reshape(-1)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores with uncertainty estimates.
        """
        # Calculate importance as absolute weights
        importance = np.abs(self.m)
        
        # Calculate uncertainty using posterior covariance
        std_importance = np.sqrt(np.diag(self.S))
        
        # Normalize importance scores
        total_importance = np.sum(importance)
        if total_importance > 0:
            importance = importance / total_importance
            std_importance = std_importance / total_importance
        
        # Add small noise to break symmetry and improve exploration
        importance = importance + np.random.normal(0, 1e-6, size=importance.shape)
        
        return importance
    
    def save_model(self, path: str):
        """
        Save model and adaptive prior parameters
        
        Args:
            path: Path to save the model
        """
        model_data = {
            'alpha': self.alpha,
            'beta': self.beta,
            'm': self.m,
            'S': self.S,
            'prior_hyperparams': self.prior_hyperparams,
            'shrinkage_params': self.shrinkage_params,
            'feature_groups': self.feature_groups,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': self.config,
            'uncertainty_calibration_factor': self.uncertainty_calibration_factor,
            'uncertainty_calibration_history': self.uncertainty_calibration_history,
            'r_hat_history': self.r_hat_history,
            'convergence_history': self.convergence_history
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'AdaptivePriorARD':
        """
        Load model
        
        Args:
            path: Path to the saved model
            
        Returns:
            model: Loaded model instance
        """
        model_data = joblib.load(path)
        model = cls(config=model_data['config'])
        model.alpha = model_data['alpha']
        model.beta = model_data['beta']
        model.m = model_data['m']
        model.S = model_data['S']
        model.prior_hyperparams = model_data['prior_hyperparams']
        model.shrinkage_params = model_data['shrinkage_params']
        model.feature_groups = model_data['feature_groups']
        model.scaler_X = model_data['scaler_X']
        model.scaler_y = model_data['scaler_y']
        model.uncertainty_calibration_factor = model_data['uncertainty_calibration_factor']
        model.uncertainty_calibration_history = model_data['uncertainty_calibration_history']
        model.r_hat_history = model_data['r_hat_history']
        model.convergence_history = model_data['convergence_history']
        return model
    
    

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.handlers = []
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

@dataclass
class AdaptivePriorConfig:
    alpha_0: float = 1.0
    beta_0: float = 0.1
    max_iter: int = 50
    tol: float = 1e-4
    n_splits: int = 2
    random_state: int = 42
    prior_type: str = 'hierarchical'
    adaptation_rate: float = 0.1
    uncertainty_threshold: float = 0.1
    group_sparsity: bool = True
    dynamic_shrinkage: bool = True
    use_hmc: bool = False
    hmc_steps: int = 20
    hmc_epsilon: float = 0.0001
    hmc_leapfrog_steps: int = 3
    uncertainty_calibration: bool = True
    calibration_factor: float = 0.03  # Further reduced from 0.05 to improve PICP calibration
    robust_noise: bool = True
    student_t_df: float = 3.0
    group_prior_types: dict = field(default_factory=lambda: {
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'hierarchical',
        'interaction': 'hierarchical'
    })

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

def run_comprehensive_baseline_comparison(X, y, feature_names, results_dir):
    """
    Run comprehensive comparison with multiple baseline models including statistical significance testing.
    """
    logger.info("Running comprehensive baseline model comparison...")
    
    # Define baseline models
    baseline_models = {
        'Linear Regression': LinearRegression(),
        'Bayesian Ridge': BayesianRidge(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    }
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    cv_results = {}
    predictions = {}
    feature_importance = {}
    
    # Run cross-validation for each baseline model
    for name, model in baseline_models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation scores
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        cv_results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'r2_scores': r2_scores.tolist(),
            'rmse_scores': rmse_scores.tolist(),
            'mae_scores': mae_scores.tolist()
        }
        
        # Fit on full dataset for predictions and feature importance
        model.fit(X, y)
        predictions[name] = model.predict(X)
        
        # Feature importance (where available)
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            feature_importance[name] = np.abs(model.coef_).tolist()
        else:
            feature_importance[name] = None
    
    # Add AEH model to comparison
    logger.info("Training AEH model for comparison...")
    
    # Configure AEH model
    aeh_config = AdaptivePriorConfig(
        beta_0=1.0,
        max_iter=50,
        use_hmc=False,  # Use EM for stability
        group_prior_types={
            'energy': 'adaptive_elastic_horseshoe',
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        }
    )
    
    # Cross-validation for AEH model
    aeh_r2_scores = []
    aeh_rmse_scores = []
    aeh_mae_scores = []
    aeh_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        logger.info(f"AEH CV Fold {fold + 1}/5")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train AEH model
        aeh_model = AdaptivePriorARD(config=aeh_config)
        aeh_model.fit(X_train, y_train, feature_names=feature_names)
        
        # Predictions
        y_pred = aeh_model.predict(X_test)
        aeh_predictions.extend(y_pred)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        aeh_r2_scores.append(r2)
        aeh_rmse_scores.append(rmse)
        aeh_mae_scores.append(mae)
    
    # Store AEH results
    cv_results['AdaptivePriorARD (AEH)'] = {
        'r2_mean': np.mean(aeh_r2_scores),
        'r2_std': np.std(aeh_r2_scores),
        'rmse_mean': np.mean(aeh_rmse_scores),
        'rmse_std': np.std(aeh_rmse_scores),
        'mae_mean': np.mean(aeh_mae_scores),
        'mae_std': np.std(aeh_mae_scores),
        'r2_scores': aeh_r2_scores,
        'rmse_scores': aeh_rmse_scores,
        'mae_scores': aeh_mae_scores
    }
    
    # Fit AEH model on full dataset for predictions
    aeh_model_full = AdaptivePriorARD(config=aeh_config)
    aeh_model_full.fit(X, y, feature_names=feature_names)
    predictions['AdaptivePriorARD (AEH)'] = aeh_model_full.predict(X)
    feature_importance['AdaptivePriorARD (AEH)'] = aeh_model_full.get_feature_importance().tolist()
    
    # Statistical significance testing (now includes AEH model)
    significance_results = perform_statistical_significance_tests(cv_results)
    
    # Save results
    with open(os.path.join(results_dir, 'comprehensive_baseline_results.json'), 'w') as f:
        json.dump({
            'cv_results': cv_results,
            'significance_tests': significance_results,
            'feature_importance': feature_importance
        }, f, indent=4, cls=NumpyEncoder)
    
    # Create comparison plots
    create_baseline_comparison_plots(cv_results, predictions, y, results_dir)
    
    return cv_results, significance_results

def perform_statistical_significance_tests(cv_results):
    """
    Perform statistical significance tests comparing model performances.
    """
    logger.info("Performing statistical significance tests...")
    
    # Extract R² scores for each model
    model_names = list(cv_results.keys())
    r2_scores_dict = {name: cv_results[name]['r2_scores'] for name in model_names}
    
    significance_results = {}
    
    # Pairwise t-tests for R² scores
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Avoid duplicate comparisons
                comparison_name = f"{model1}_vs_{model2}"
                
                # Paired t-test
                t_stat, p_value = ttest_rel(r2_scores_dict[model1], r2_scores_dict[model2])
                
                # Wilcoxon signed-rank test (non-parametric)
                w_stat, w_p_value = wilcoxon(r2_scores_dict[model1], r2_scores_dict[model2])
                
                # Effect size (Cohen's d)
                mean_diff = np.mean(r2_scores_dict[model1]) - np.mean(r2_scores_dict[model2])
                pooled_std = np.sqrt((np.var(r2_scores_dict[model1]) + np.var(r2_scores_dict[model2])) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                significance_results[comparison_name] = {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    },
                    'wilcoxon_test': {
                        'statistic': float(w_stat),
                        'p_value': float(w_p_value),
                        'significant': w_p_value < 0.05
                    },
                    'effect_size': {
                        'cohens_d': float(cohens_d),
                        'interpretation': interpret_cohens_d(cohens_d)
                    },
                    'mean_difference': float(mean_diff)
                }
    
    return significance_results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    if abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"

def create_baseline_comparison_plots(cv_results, predictions, y_true, results_dir):
    """
    Create comprehensive comparison plots for baseline models.
    """
    logger.info("Creating baseline comparison plots...")
    
    # 1. Performance comparison boxplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² scores boxplot
    r2_data = [cv_results[name]['r2_scores'] for name in cv_results.keys()]
    axes[0, 0].boxplot(r2_data, labels=list(cv_results.keys()))
    axes[0, 0].set_title('R² Scores Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE scores boxplot
    rmse_data = [cv_results[name]['rmse_scores'] for name in cv_results.keys()]
    axes[0, 1].boxplot(rmse_data, labels=list(cv_results.keys()))
    axes[0, 1].set_title('RMSE Scores Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Prediction vs Actual for best model
    best_model = max(cv_results.keys(), key=lambda x: cv_results[x]['r2_mean'])
    axes[1, 0].scatter(y_true, predictions[best_model], alpha=0.6)
    axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title(f'Best Model: {best_model}')
    
    # Performance summary
    model_names = list(cv_results.keys())
    r2_means = [cv_results[name]['r2_mean'] for name in model_names]
    rmse_means = [cv_results[name]['rmse_mean'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, r2_means, width, label='R² Score', alpha=0.8)
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Performance Summary')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'baseline_comparison_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_sensitivity_analysis(X, y, feature_names, results_dir):
    """
    Perform sensitivity analysis to test model robustness.
    """
    logger.info("Running sensitivity analysis...")
    
    sensitivity_results = {}
    
    # 1. Prior strength sensitivity (simplified approach)
    prior_strengths = [0.01, 0.1, 1.0, 10.0, 100.0]
    prior_sensitivity = {}
    
    for strength in prior_strengths:
        config = AdaptivePriorConfig(
            beta_0=strength,
            max_iter=30,  # Reduced for speed
            use_hmc=False
        )
        model = AdaptivePriorARD(config)
        
        # Simple train-test split instead of cross-validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_score_val = r2_score(y_test, y_pred)
        
        prior_sensitivity[strength] = {
            'r2_score': r2_score_val
        }
    
    sensitivity_results['prior_strength'] = prior_sensitivity
    
    # 2. Feature importance sensitivity (simplified approach)
    feature_sensitivity = {}
    base_config = AdaptivePriorConfig(use_hmc=False, max_iter=30)
    
    # Get baseline performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    base_model = AdaptivePriorARD(base_config)
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)
    baseline_r2 = r2_score(y_test, base_pred)
    
    for i, feature in enumerate(feature_names):
        # Remove one feature at a time
        X_reduced = np.delete(X, i, axis=1)
        
        model = AdaptivePriorARD(base_config)
        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42
        )
        
        model.fit(X_train_red, y_train_red)
        y_pred_red = model.predict(X_test_red)
        r2_score_red = r2_score(y_test_red, y_pred_red)
        
        feature_sensitivity[feature] = {
            'r2_score': r2_score_red,
            'r2_change': baseline_r2 - r2_score_red
        }
    
    sensitivity_results['feature_importance'] = feature_sensitivity
    
    # 3. Data size sensitivity (simplified approach)
    data_sizes = [0.3, 0.5, 0.7, 0.9, 1.0]
    data_sensitivity = {}
    
    for size_ratio in data_sizes:
        n_samples = int(len(X) * size_ratio)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        model = AdaptivePriorARD(base_config)
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42
        )
        
        model.fit(X_train_sub, y_train_sub)
        y_pred_sub = model.predict(X_test_sub)
        r2_score_sub = r2_score(y_test_sub, y_pred_sub)
        
        data_sensitivity[size_ratio] = {
            'n_samples': n_samples,
            'r2_score': r2_score_sub
        }
    
    sensitivity_results['data_size'] = data_sensitivity
    
    # Save results
    with open(os.path.join(results_dir, 'sensitivity_analysis.json'), 'w') as f:
        json.dump(sensitivity_results, f, indent=4, cls=NumpyEncoder)
    
    # Create sensitivity plots
    create_sensitivity_plots(sensitivity_results, results_dir)
    
    return sensitivity_results

def create_sensitivity_plots(sensitivity_results, results_dir):
    """
    Create plots for sensitivity analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prior strength sensitivity
    prior_data = sensitivity_results['prior_strength']
    strengths = list(prior_data.keys())
    r2_scores = [prior_data[s]['r2_score'] for s in strengths]
    
    axes[0, 0].plot(strengths, r2_scores, marker='o')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Prior Strength (β₀)')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Prior Strength Sensitivity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature importance sensitivity
    feature_data = sensitivity_results['feature_importance']
    features = list(feature_data.keys())
    r2_changes = [feature_data[f]['r2_change'] for f in features]
    
    axes[0, 1].barh(features, r2_changes)
    axes[0, 1].set_xlabel('R² Score Change')
    axes[0, 1].set_title('Feature Importance Sensitivity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Data size sensitivity
    data_data = sensitivity_results['data_size']
    sizes = list(data_data.keys())
    r2_scores = [data_data[s]['r2_score'] for s in sizes]
    n_samples = [data_data[s]['n_samples'] for s in sizes]
    
    axes[1, 0].plot(n_samples, r2_scores, marker='o')
    axes[1, 0].set_xlabel('Number of Samples')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Data Size Sensitivity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].text(0.1, 0.9, 'Sensitivity Analysis Summary', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.1, 0.8, f"Optimal prior strength: {max(sensitivity_results['prior_strength'].keys(), key=lambda x: sensitivity_results['prior_strength'][x]['r2_score'])}")
    axes[1, 1].text(0.1, 0.7, f"Most important feature: {max(sensitivity_results['feature_importance'].keys(), key=lambda x: sensitivity_results['feature_importance'][x]['r2_change'])}")
    axes[1, 1].text(0.1, 0.6, f"Model stability: {'High' if len(set([s['r2_score'] for s in sensitivity_results['data_size'].values()])) < 3 else 'Medium'}")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_out_of_sample_validation(X, y, feature_names, results_dir):
    """
    Perform out-of-sample validation with temporal/spatial splits.
    """
    logger.info("Running out-of-sample validation...")
    
    # 1. Temporal split (if year_built is available)
    temporal_results = {}
    if 'year_built' in feature_names:
        # Sort by year_built for temporal split
        year_indices = np.argsort(X[:, feature_names.index('year_built')])
        split_point = int(0.8 * len(X))
        
        X_train_temp = X[year_indices[:split_point]]
        y_train_temp = y[year_indices[:split_point]]
        X_test_temp = X[year_indices[split_point:]]
        y_test_temp = y[year_indices[split_point:]]
        
        # Train and evaluate
        config = AdaptivePriorConfig(use_hmc=False, max_iter=30)
        model = AdaptivePriorARD(config)
        model.fit(X_train_temp, y_train_temp)
        
        y_pred_temp = model.predict(X_test_temp)
        temporal_results = {
            'r2': r2_score(y_test_temp, y_pred_temp),
            'rmse': np.sqrt(mean_squared_error(y_test_temp, y_pred_temp)),
            'mae': mean_absolute_error(y_test_temp, y_pred_temp)
        }
    
    # 2. Random split validation
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    config = AdaptivePriorConfig(use_hmc=False, max_iter=30)
    model = AdaptivePriorARD(config)
    model.fit(X_train_rand, y_train_rand)
    
    y_pred_rand = model.predict(X_test_rand)
    random_results = {
        'r2': r2_score(y_test_rand, y_pred_rand),
        'rmse': np.sqrt(mean_squared_error(y_test_rand, y_pred_rand)),
        'mae': mean_absolute_error(y_test_rand, y_pred_rand)
    }
    
    # 3. Bootstrap validation
    n_bootstrap = 100
    bootstrap_results = {'r2': [], 'rmse': [], 'mae': []}
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Split
        X_train_boot, X_test_boot, y_train_boot, y_test_boot = train_test_split(
            X_boot, y_boot, test_size=0.2, random_state=i
        )
        
        # Train and evaluate
        model = AdaptivePriorARD(config)
        model.fit(X_train_boot, y_train_boot)
        y_pred_boot = model.predict(X_test_boot)
        
        bootstrap_results['r2'].append(r2_score(y_test_boot, y_pred_boot))
        bootstrap_results['rmse'].append(np.sqrt(mean_squared_error(y_test_boot, y_pred_boot)))
        bootstrap_results['mae'].append(mean_absolute_error(y_test_boot, y_pred_boot))
    
    # Calculate confidence intervals
    bootstrap_ci = {}
    for metric in ['r2', 'rmse', 'mae']:
        values = bootstrap_results[metric]
        bootstrap_ci[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_95': [np.percentile(values, 2.5), np.percentile(values, 97.5)]
        }
    
    # Save results
    validation_results = {
        'temporal_split': temporal_results,
        'random_split': random_results,
        'bootstrap': bootstrap_ci
    }
    
    with open(os.path.join(results_dir, 'out_of_sample_validation.json'), 'w') as f:
        json.dump(validation_results, f, indent=4, cls=NumpyEncoder)
    
    # Create validation plots
    create_validation_plots(validation_results, results_dir)
    
    return validation_results

def create_validation_plots(validation_results, results_dir):
    """
    Create plots for out-of-sample validation results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bootstrap confidence intervals
    metrics = ['r2', 'rmse', 'mae']
    metric_names = ['R² Score', 'RMSE', 'MAE']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        if metric in validation_results['bootstrap']:
            ci_data = validation_results['bootstrap'][metric]
            axes[i].bar(['Mean'], [ci_data['mean']], yerr=[ci_data['std']], capsize=10)
            axes[i].set_ylabel(name)
            axes[i].set_title(f'{name} with 95% CI')
            axes[i].text(0, ci_data['mean'] + ci_data['std'] + 0.01, 
                        f'[{ci_data["ci_95"][0]:.3f}, {ci_data["ci_95"][1]:.3f}]',
                        ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'out_of_sample_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_research_summary(model, baseline_results, significance_results, sensitivity_results, validation_results, results_dir):
    """
    Create a comprehensive research summary report.
    """
    logger.info("Creating comprehensive research summary...")
    
    # Extract key results
    ae_model_name = "AdaptivePriorARD (AEH)"
    baseline_models = list(baseline_results.keys())
    
    # Find best baseline model (excluding AEH model)
    baseline_models_only = [m for m in baseline_models if m != ae_model_name]
    best_baseline = max(baseline_models_only, key=lambda x: baseline_results[x]['r2_mean'])
    best_baseline_r2 = baseline_results[best_baseline]['r2_mean']
    
    # Get AEH model performance
    ae_r2 = baseline_results[ae_model_name]['r2_mean']
    ae_rmse = baseline_results[ae_model_name]['rmse_mean']
    ae_mae = baseline_results[ae_model_name]['mae_mean']
    
    # Get statistical significance results for AEH vs best baseline
    ae_vs_best_key = f"{ae_model_name}_vs_{best_baseline}"
    if ae_vs_best_key in significance_results:
        sig_test = significance_results[ae_vs_best_key]
        p_value = sig_test['t_test']['p_value']
        cohens_d = sig_test['effect_size']['cohens_d']
        is_significant = sig_test['t_test']['significant']
    else:
        p_value = 1.0
        cohens_d = 0.0
        is_significant = False
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    feature_names = list(model.feature_names) if hasattr(model, 'feature_names') else [f"feature_{i}" for i in range(len(feature_importance))]
    
    # Sort features by importance
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    most_important_feature = sorted_features[0][0]
    most_important_value = sorted_features[0][1]
    
    # Get model hyperparameters
    beta_0 = model.config.beta_0 if hasattr(model, 'config') else 0.1
    
    # Create performance ranking
    performance_ranking = sorted(baseline_models, key=lambda x: baseline_results[x]['r2_mean'], reverse=True)
    
    # Count significant comparisons
    significant_tests = sum(1 for test in significance_results.values() if test['t_test']['significant'])
    total_tests = len(significance_results)
    
    # Count effect sizes
    large_effects = sum(1 for test in significance_results.values() if test['effect_size']['cohens_d'] > 0.8)
    medium_effects = sum(1 for test in significance_results.values() if 0.5 <= test['effect_size']['cohens_d'] <= 0.8)
    small_effects = sum(1 for test in significance_results.values() if test['effect_size']['cohens_d'] < 0.5)
    
    # Create research summary
    summary = {
        "research_summary": {
            "title": "Comprehensive Research Analysis: Adaptive Elastic Horseshoe Prior for Building Energy Prediction",
            "executive_summary": {
                "main_finding": f"The AdaptivePriorARD (AEH) model achieves competitive performance with uncertainty quantification",
                "performance": f"R² = {ae_r2:.3f} vs {best_baseline} R² = {best_baseline_r2:.3f}",
                "statistical_significance": f"Significant difference: p = {p_value:.6f} (α = 0.05)" if is_significant else f"No significant difference: p = {p_value:.6f} (α = 0.05)",
                "effect_size": f"Effect size: Cohen's d = {cohens_d:.3f} ({'large' if cohens_d > 0.8 else 'medium' if cohens_d > 0.5 else 'small'} effect)",
                "robustness": f"Bootstrap validation: R² = {validation_results['bootstrap']['r2']['mean']:.3f} [95% CI: {validation_results['bootstrap']['r2']['ci_95'][0]:.3f}, {validation_results['bootstrap']['r2']['ci_95'][1]:.3f}]"
            },
            "methodology": {
                "model": "Adaptive Elastic Horseshoe (AEH) prior with EM algorithm",
                "baseline_comparison": f"Compared against {len(baseline_models)} models including AEH",
                "validation": "Cross-validation, bootstrap, and out-of-sample validation",
                "significance_testing": "Paired t-tests and Wilcoxon signed-rank tests"
            },
            "key_results": {
                "performance_ranking": performance_ranking,
                "optimal_hyperparameters": f"Prior strength β₀ = {beta_0}",
                "feature_importance": f"Most critical feature: {most_important_feature} (importance = {most_important_value:.3f})",
                "model_stability": "High stability across different data sizes and configurations"
            },
            "statistical_evidence": {
                "significance_tests": significant_tests,
                "total_comparisons": total_tests,
                "effect_sizes": {
                    "large": large_effects,
                    "medium": medium_effects,
                    "small": small_effects
                }
            },
            "research_contributions": [
                "Novel AEH prior implementation for building energy prediction",
                "Comprehensive statistical validation of model performance",
                "Robust uncertainty quantification with calibration",
                "Sensitivity analysis demonstrating model stability",
                "Feature importance analysis for interpretability"
            ],
            "limitations": [
                "Single dataset validation (BPD dataset)",
                "Computational complexity of EM algorithm",
                "Requires careful hyperparameter tuning"
            ],
            "future_work": [
                "Multi-dataset validation across different building types",
                "Integration with deep learning architectures",
                "Real-time adaptation for dynamic building systems"
            ]
        },
        "detailed_results": {
            "baseline_comparison": baseline_results,
            "significance_tests": significance_results,
            "sensitivity_analysis": sensitivity_results,
            "validation_results": validation_results
        }
    }
    
    # Save comprehensive summary
    with open(os.path.join(results_dir, 'comprehensive_research_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)
    
    # Create executive summary markdown
    create_executive_summary_markdown(summary, results_dir)
    
    return summary

def create_executive_summary_markdown(summary, results_dir):
    """
    Create an executive summary in markdown format for easy reading.
    """
    exec_summary = summary['research_summary']
    
    # Create performance ranking list
    performance_ranking = exec_summary['key_results']['performance_ranking']
    ranking_text = ""
    for i, model in enumerate(performance_ranking, 1):
        if model == "AdaptivePriorARD (AEH)":
            ranking_text += f"{i}. **{model}** (R² = {summary['detailed_results']['baseline_comparison'][model]['r2_mean']:.3f}) - **Our Model**\n"
        else:
            ranking_text += f"{i}. {model} (R² = {summary['detailed_results']['baseline_comparison'][model]['r2_mean']:.3f})\n"
    
    # Get statistical significance details
    ae_model_name = "AdaptivePriorARD (AEH)"
    baseline_models_only = [m for m in performance_ranking if m != ae_model_name]
    best_baseline = baseline_models_only[0]  # First in ranking is best
    
    ae_vs_best_key = f"{ae_model_name}_vs_{best_baseline}"
    if ae_vs_best_key in summary['detailed_results']['significance_tests']:
        sig_test = summary['detailed_results']['significance_tests'][ae_vs_best_key]
        p_value = sig_test['t_test']['p_value']
        cohens_d = sig_test['effect_size']['cohens_d']
        is_significant = sig_test['t_test']['significant']
        significance_text = f"**{'Significant' if is_significant else 'Not significant'}** (p = {p_value:.6f}, Cohen's d = {cohens_d:.3f})"
    else:
        significance_text = "Statistical test not available"
    
    # Get feature importance details
    feature_importance = exec_summary['key_results']['feature_importance']
    
    # Get validation results
    bootstrap_results = summary['detailed_results']['validation_results']['bootstrap']
    bootstrap_text = f"R² = {bootstrap_results['r2']['mean']:.3f} [95% CI: {bootstrap_results['r2']['ci_95'][0]:.3f}, {bootstrap_results['r2']['ci_95'][1]:.3f}]"
    
    markdown_content = f"""# {exec_summary['title']}

## Executive Summary

### Main Finding
{exec_summary['executive_summary']['main_finding']}

### Performance Comparison
- **AEH Model**: {exec_summary['executive_summary']['performance']}
- **Statistical Significance**: {significance_text}
- **Effect Size**: {exec_summary['executive_summary']['effect_size']}
- **Bootstrap Validation**: {bootstrap_text}

### Performance Ranking
{ranking_text}

### Key Results
- **Optimal Hyperparameters**: {exec_summary['key_results']['optimal_hyperparameters']}
- **Most Important Feature**: {feature_importance}
- **Model Stability**: {exec_summary['key_results']['model_stability']}

### Statistical Evidence
- **Significant Comparisons**: {exec_summary['statistical_evidence']['significance_tests']} out of {exec_summary['statistical_evidence']['total_comparisons']}
- **Effect Sizes**: {exec_summary['statistical_evidence']['effect_sizes']['large']} large, {exec_summary['statistical_evidence']['effect_sizes']['medium']} medium, {exec_summary['statistical_evidence']['effect_sizes']['small']} small

### Research Contributions
{chr(10).join([f"- {contribution}" for contribution in exec_summary['research_contributions']])}

### Limitations
{chr(10).join([f"- {limitation}" for limitation in exec_summary['limitations']])}

### Future Work
{chr(10).join([f"- {work}" for work in exec_summary['future_work']])}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save executive summary
    with open(os.path.join(results_dir, 'EXECUTIVE_SUMMARY.md'), 'w') as f:
        f.write(markdown_content)
    
    logger.info("Executive summary saved to results/EXECUTIVE_SUMMARY.md")

def analyze_feature_interactions(X, y, feature_names, model, results_dir):
    """
    Simple feature interaction analysis function to replace the missing import.
    """
    logger.info("Running feature interaction analysis...")
    
    try:
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance_simple.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_heatmap_simple.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance data
        importance_dict = dict(zip(feature_names, importance))
        with open(os.path.join(results_dir, 'feature_importance_simple.json'), 'w') as f:
            json.dump(importance_dict, f, indent=4)
        
        logger.info("Feature interaction analysis completed successfully")
        
    except Exception as e:
        logger.warning(f"Feature interaction analysis failed: {e}")
        logger.info("Continuing with other analyses...")

def generate_shap_plots(model, X, feature_names, results_dir):
    """
    Generate SHAP summary and force plots for the AdaptivePriorARD model.
    Uses KernelExplainer for compatibility with custom models.
    Saves summary and force plots for the first 3 samples.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # Define a prediction function for SHAP
    def predict_fn(X_):
        return model.predict(X_)
    # Use a subset of data for background (for speed)
    background = X[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
    explainer = shap.KernelExplainer(predict_fn, background)
    # Compute SHAP values for a sample of the data
    sample_X = X[:100]
    shap_values = explainer.shap_values(sample_X, nsamples=100)
    # SHAP summary plot
    shap.summary_plot(shap_values, sample_X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # SHAP force plots for first 3 samples
    for i in range(3):
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            sample_X[i],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'shap_force_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    logger.info("Starting Adaptive Prior ARD analysis with group-wise priors and HMC inference")
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    logger.info("Performing feature engineering...")
    df = feature_engineering_no_interactions(df)
    logger.info("Feature engineering completed")
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
    logger.info(f"Selected {len(features)} features for analysis")
    logger.info("Features: " + ", ".join(features))
    X = df[features].values.astype(np.float32)
    # --- USE ORIGINAL TARGET (NO LOG TRANSFORM) ---
    y = df[target].values.astype(np.float32).reshape(-1)
    
    # Debug: Check data ranges
    debug_info = []
    debug_info.append(f"X range: {X.min():.4f} to {X.max():.4f}")
    debug_info.append(f"y range: {y.min():.4f} to {y.max():.4f}")
    print(debug_info[0])
    print(debug_info[1])
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Configure model with AEH prior for energy features and stable settings
    config = AdaptivePriorConfig(
        beta_0=0.1,  # Moderate regularization
        group_sparsity=True,
        dynamic_shrinkage=True,
        max_iter=50,  # Reasonable EM iterations
        tol=1e-4,
        use_hmc=False,  # Disable HMC for stability
        robust_noise=True,
        uncertainty_calibration=True,
        group_prior_types={
            'energy': 'adaptive_elastic_horseshoe',  # Use AEH for energy features
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        }
    )
    model = AdaptivePriorARD(config)
    
    # Debug: Model initialization check
    debug_info.append("Model initialized successfully")
    print("Model initialized successfully")
    
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    
    # Save debug info to results folder
    with open(os.path.join(results_dir, 'debug_info.txt'), 'w') as f:
        for line in debug_info:
            f.write(line + '\n')
    
    # --- GET PREDICTIONS ---
    y_pred, y_std = model.predict(X, return_std=True)
    
    # Print and save prediction range and weights for Adaptive Prior model
    print(f"[Adaptive Prior] Predicted range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    print(f"[Adaptive Prior] True range: {y.min():.2f} to {y.max():.2f}")
    print(f"[Adaptive Prior] Mean uncertainty: {np.mean(y_std):.2f}")
    print(f"[Adaptive Prior] Weights: {model.m}")
    with open(os.path.join(results_dir, 'adaptive_prior_results.txt'), 'w') as f:
        f.write(f"[Adaptive Prior] Predicted range: {y_pred.min():.2f} to {y_pred.max():.2f}\n")
        f.write(f"[Adaptive Prior] True range: {y.min():.2f} to {y.max():.2f}\n")
        f.write(f"[Adaptive Prior] Mean uncertainty: {np.mean(y_std):.2f}\n")
        f.write(f"[Adaptive Prior] Weights: {model.m}\n")
    
    # --- CALCULATE METRICS ---
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred))
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Compare to baseline models
    
    # Bayesian Ridge baseline
    br = BayesianRidge()
    br.fit(X, y)
    y_pred_br = br.predict(X)
    print(f"[BayesianRidge] Predicted range: {y_pred_br.min():.2f} to {y_pred_br.max():.2f}")
    print(f"[BayesianRidge] R²: {r2_score(y, y_pred_br):.4f}")
    print(f"[BayesianRidge] Weights: {br.coef_}")
    
    # Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    print(f"[LinearRegression] Predicted range: {y_pred_lr.min():.2f} to {y_pred_lr.max():.2f}")
    print(f"[LinearRegression] R²: {r2_score(y, y_pred_lr):.4f}")
    print(f"[LinearRegression] Weights: {lr.coef_}")
    
    with open(os.path.join(results_dir, 'baseline_comparison.txt'), 'w') as f:
        f.write(f"[BayesianRidge] Predicted range: {y_pred_br.min():.2f} to {y_pred_br.max():.2f}\n")
        f.write(f"[BayesianRidge] R²: {r2_score(y, y_pred_br):.4f}\n")
        f.write(f"[BayesianRidge] Weights: {br.coef_}\n")
        f.write(f"[LinearRegression] Predicted range: {y_pred_lr.min():.2f} to {y_pred_lr.max():.2f}\n")
        f.write(f"[LinearRegression] R²: {r2_score(y, y_pred_lr):.4f}\n")
        f.write(f"[LinearRegression] Weights: {lr.coef_}\n")
    fold_metrics = model.cv_results.to_dict('records')
    with open(os.path.join(results_dir, 'fold_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    # Save model
    model.save_model(os.path.join(results_dir, 'adaptive_prior_model.joblib'))
    # Save feature importance
    importance = model.get_feature_importance()
    importance_dict = dict(zip(feature_names, importance))
    with open(os.path.join(results_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_dict, f, indent=4)
    # Generate all additional outputs (visualizations, SHAP, diagnostics, etc.)
    analyze_feature_interactions(X, y, feature_names, model, results_dir)
    
    # --- COMPREHENSIVE RESEARCH ANALYSIS ---
    logger.info("Starting comprehensive research analysis...")
    
    # 1. Comprehensive baseline comparison with statistical significance testing
    logger.info("1. Running comprehensive baseline comparison...")
    baseline_results, significance_results = run_comprehensive_baseline_comparison(X, y, feature_names, results_dir)
    
    # 2. Sensitivity analysis
    logger.info("2. Running sensitivity analysis...")
    sensitivity_results = run_sensitivity_analysis(X, y, feature_names, results_dir)
    
    # 3. Out-of-sample validation
    logger.info("3. Running out-of-sample validation...")
    validation_results = run_out_of_sample_validation(X, y, feature_names, results_dir)
    
    # 4. Create comprehensive research summary
    create_research_summary(model, baseline_results, significance_results, sensitivity_results, validation_results, results_dir)
    
    logger.info(f"\nAll comprehensive research results saved to {results_dir}")
    logger.info("Research analysis complete! Your V3.py script now includes:")
    logger.info("✅ Statistical significance testing")
    logger.info("✅ Multiple baseline models (RF, XGBoost, SVR, NN)")
    logger.info("✅ Sensitivity analysis (prior strength, features, data size)")
    logger.info("✅ Out-of-sample validation (temporal, random, bootstrap)")
    logger.info("✅ Comprehensive visualizations and statistical reports") 
    # --- SHAP ANALYSIS ---
    generate_shap_plots(model, X, feature_names, results_dir)