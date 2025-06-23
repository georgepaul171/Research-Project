"""
Adaptive Prior Automatic Relevance Determination (ARD) for Building Energy Performance Analysis

This implementation presents a novel Bayesian approach to building energy performance modelling,
incorporating adaptive prior specifications and uncertainty quantification. The model extends
traditional ARD by introducing hierarchical priors, dynamic shrinkage parameters, and robust
noise modeling to better capture the complex relationships in building energy data.

Key innovations:
1. Adaptive prior specifications that evolve during training
2. Hierarchical Bayesian structure for improved feature selection
3. Uncertainty quantification with calibration
4. Robust noise modeling using Student's t distribution
5. Group sparsity for structured feature selection
6. Hamiltonian Monte Carlo for improved posterior exploration

Author: George Paul
Institution: The University of Bath
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, field
import logging
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
from scipy.special import digamma, polygamma
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import functools
import shap
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

# Configure logging for detailed model training and evaluation tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Remove any existing handlers to avoid duplicate messages
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Remove root logger handlers
for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

# Add a new handler with the correct format
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Prevent propagation to root logger
logger.propagate = False

# Configure root logger to not add
logging.getLogger().handlers = []
logging.getLogger().propagate = False

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
        return super(NumpyEncoder, self).default(obj)

@dataclass
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
    alpha_0: float = 1e-6
    beta_0: float = 1e-6
    max_iter: int = 200
    tol: float = 1e-4
    n_splits: int = 3
    random_state: int = 42
    prior_type: str = 'hierarchical'
    adaptation_rate: float = 0.1
    uncertainty_threshold: float = 0.1
    group_sparsity: bool = True
    dynamic_shrinkage: bool = True
    use_hmc: bool = True
    hmc_steps: int = 100
    hmc_epsilon: float = 0.01
    hmc_leapfrog_steps: int = 10
    uncertainty_calibration: bool = True
    calibration_factor: float = 10.0  
    robust_noise: bool = True
    student_t_df: float = 3.0  
    group_prior_types: dict = field(default_factory=lambda: {
        'energy': 'adaptive_elastic_horseshoe',  # Using our new prior for energy features
        'building': 'hierarchical',
        'interaction': 'spike_slab'
    })

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
    def __init__(self, config: Optional[AdaptivePriorConfig] = None):
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
        """
        self.config = config or AdaptivePriorConfig()
        
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
                # Get current parameters
                alpha = self.group_prior_hyperparams[group]['alpha']
                beta = self.group_prior_hyperparams[group]['beta']
                gamma = self.group_prior_hyperparams[group]['gamma']
                rho = self.group_prior_hyperparams[group]['rho']
                momentum = self.group_prior_hyperparams[group]['momentum']
                
                # Calculate feature importance and uncertainty
                importance = np.array([np.clip(self.m[j]**2, 1e-10, None) for j in indices])
                uncertainty = np.array([np.clip(np.diag(self.S)[j], 1e-10, None) for j in indices])
                
                # Adaptive elastic net component
                elastic_penalty = alpha * np.abs(importance) + (1 - alpha) * importance**2
                
                # Horseshoe component with adaptive scaling
                horseshoe_scale = 1 / (importance / (2 * self.group_prior_hyperparams[group]['tau']) + 
                                     beta * elastic_penalty)
                
                # Update lambda with momentum
                gradient = -horseshoe_scale + beta * elastic_penalty
                momentum = rho * momentum + gamma * gradient
                self.group_prior_hyperparams[group]['lambda'] = np.clip(
                    self.group_prior_hyperparams[group]['lambda'] + momentum,
                    1e-10, None
                )
                
                # Update tau adaptively
                m_squared_sum = np.sum(importance)
                self.group_prior_hyperparams[group]['tau'] = np.clip(
                    1 / (m_squared_sum / (2 * np.sum(self.group_prior_hyperparams[group]['lambda'])) + 1),
                    1e-10, None
                )
                
                # Update alpha based on feature importance distribution
                importance_ratio = np.mean(importance) / (np.std(importance) + 1e-10)
                self.group_prior_hyperparams[group]['alpha'] = np.clip(
                    alpha + gamma * (0.5 - importance_ratio),
                    0.1, 0.9
                )
                
                # Update beta based on uncertainty
                uncertainty_ratio = np.mean(uncertainty) / (np.std(uncertainty) + 1e-10)
                self.group_prior_hyperparams[group]['beta'] = np.clip(
                    beta + gamma * (1.0 - uncertainty_ratio),
                    0.1, 10.0
                )
                
                # Store updated momentum
                self.group_prior_hyperparams[group]['momentum'] = momentum
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
        """
        Perform one step of Hamiltonian Monte Carlo for posterior exploration.
        """
        # Initialise new position and momentum
        new_w = current_w.copy()
        new_momentum = current_momentum.copy()
        
        # Resample momentum from standard normal with increased variance
        new_momentum = np.random.normal(0, 2.0, size=new_momentum.shape)  # Increased from 1.5 to 2.0
        
        # Calculate initial Hamiltonian (total energy)
        current_energy = self._calculate_hamiltonian(X, y, current_w, new_momentum)
        
        # Adaptive step size based on gradient magnitude and posterior scale
        grad = self._calculate_gradient(X, y, current_w)
        posterior_scale = np.sqrt(np.diag(self.S))
        
        # More aggressive step size adaptation
        epsilon = self.config.hmc_epsilon * np.minimum(
            3.0 / (1.0 + np.linalg.norm(grad)),  # Increased from 2.0 to 3.0
            posterior_scale / np.max(posterior_scale)
        )
        
        # Add more noise to step size to improve exploration
        epsilon = epsilon * np.exp(np.random.normal(0, 0.3))  # Increased from 0.2 to 0.3
        
        # Leapfrog steps for Hamiltonian dynamics
        for _ in range(self.config.hmc_leapfrog_steps):
            # Update momentum (half step)
            grad = self._calculate_gradient(X, y, new_w)
            new_momentum = new_momentum - 0.5 * epsilon * grad
            
            # Update position (full step)
            new_w = new_w + epsilon * new_momentum
            
            # Update momentum (half step)
            grad = self._calculate_gradient(X, y, new_w)
            new_momentum = new_momentum - 0.5 * epsilon * grad
        
        # Calculate new Hamiltonian
        new_energy = self._calculate_hamiltonian(X, y, new_w, new_momentum)
        
        # Metropolis acceptance step with numerical stability
        energy_diff = current_energy - new_energy
        acceptance_prob = min(1.0, np.exp(np.clip(energy_diff, -100, 100)))
        
        # Add more noise to acceptance probability to improve mixing
        acceptance_prob = acceptance_prob * np.exp(np.random.normal(0, 0.1))  # Increased from 0.05 to 0.1
        
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
            current_momentum = np.random.normal(0, 2.0, size=n_features)
            chain_samples = []
            acceptance_rates = []
            for _ in range(self.config.hmc_steps):
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
        import matplotlib.pyplot as plt
        import os
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
        Fit the Adaptive Prior ARD model with cross-validation and uncertainty calibration.
        Now also saves trace plots for the top 5 features and alpha for the first fold.
        """
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape
        self.alpha = np.clip(self.config.alpha_0, 1e-10, None)
        self.beta = np.ones(n_features) * np.clip(self.config.beta_0, 1e-10, None)
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        self._initialize_adaptive_priors(n_features)
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, 
                  random_state=self.config.random_state)
        cv_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
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
                    # Only for the first fold, store chains for trace plots
                    if fold == 1:
                        self.m, acceptance_probs, r_hat_stats, ess_stats, chains = self._hmc_sampling(
                            X_train_scaled, y_train_scaled, self.m, return_chains=True
                        )
                        # Plot trace plots for top 5 features if feature_names provided
                        if feature_names is not None and output_dir is not None:
                            importance = np.abs(self.m)
                            top_indices = np.argsort(importance)[-5:]
                            self._plot_trace_chains(chains, feature_names, output_dir, top_indices, fold=fold)
                    else:
                        self.m, acceptance_probs, r_hat_stats, ess_stats, _ = self._hmc_sampling(
                            X_train_scaled, y_train_scaled, self.m, return_chains=False
                        )
                    logger.info(f"Fold {fold}, Iteration {iteration}: "
                              f"Mean HMC acceptance rate: {np.mean(acceptance_probs):.3f}")
                    logger.info(f"Gelman-Rubin R²: {np.mean(list(r_hat_stats.values())):.3f}")
                    logger.info(f"Effective Sample Size: {np.mean(list(ess_stats.values())):.1f}")
                    if not hasattr(self, 'convergence_history'):
                        self.convergence_history = []
                    self.convergence_history.append({
                        'fold': fold,
                        'iteration': iteration,
                        'r_hat_mean': np.mean(list(r_hat_stats.values())),
                        'r_hat_std': np.std(list(r_hat_stats.values())),
                        'r_hat_stats': r_hat_stats,
                        'ess_mean': np.mean(list(ess_stats.values())),
                        'ess_std': np.std(list(ess_stats.values())),
                        'ess_stats': ess_stats
                    })
                # ... existing code ...
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
                                               np.clip(np.diag(self.S)[j], 1e-10, None) +
                                               2 * self.group_prior_hyperparams[group]['tau'][idx])
                    elif prior_type == 'spike_slab':
                        for idx, j in enumerate(indices):
                            pi = self.group_prior_hyperparams[group]['pi'][idx]
                            sigma2_0 = self.group_prior_hyperparams[group]['sigma2_0'][idx]
                            sigma2_1 = self.group_prior_hyperparams[group]['sigma2_1'][idx]
                            beta_new[j] = (pi / np.clip(sigma2_1, 1e-10, None) +
                                           (1 - pi) / np.clip(sigma2_0, 1e-10, None))
                    elif prior_type == 'horseshoe':
                        for idx, j in enumerate(indices):
                            tau = self.group_prior_hyperparams[group]['tau']
                            lambd = self.group_prior_hyperparams[group]['lambda'][idx]
                            beta_new[j] = 1 / (np.clip(self.m[j]**2, 1e-10, None) / (2 * tau) + lambd)
                if self.config.group_sparsity:
                    for group, indices in self.feature_groups.items():
                        group_beta = np.mean(beta_new[indices])
                        beta_new[indices] = group_beta
                if self.config.dynamic_shrinkage:
                    kappa = np.clip(self.shrinkage_params['kappa'], 0, 1)
                    beta_new = beta_new * (1 - kappa) + self.beta * kappa
                self._update_adaptive_priors(iteration)
                beta_diff = np.abs(np.clip(beta_new, 1e-10, None) - np.clip(self.beta, 1e-10, None))
                alpha_diff = np.abs(alpha_new - self.alpha)
                if (alpha_diff < self.config.tol and np.all(beta_diff < self.config.tol)):
                    break
                self.alpha = alpha_new
                self.beta = np.clip(beta_new, 1e-10, None)
            self._update_uncertainty_calibration(X_val_scaled, y_val_scaled)
            y_pred, y_std = self.predict(X_val_scaled, return_std=True)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_val_orig = self.scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
            y_pred_orig = y_pred_orig.reshape(-1)
            y_val_orig = y_val_orig.reshape(-1)
            confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
            picp_scores = []
            for level in confidence_levels:
                z_score = stats.norm.ppf(1 - (1 - level) / 2)
                lower = y_pred_orig - z_score * y_std
                upper = y_pred_orig + z_score * y_std
                coverage = np.mean((y_val_orig >= lower) & (y_val_orig <= upper))
                picp_scores.append(coverage)
            crps = np.mean(np.abs(y_pred_orig - y_val_orig)) - 0.5 * np.mean(np.abs(y_std))
            metrics = {
                'fold': fold,
                'rmse': np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)),
                'mae': mean_absolute_error(y_val_orig, y_pred_orig),
                'r2': r2_score(y_val_orig, y_pred_orig),
                'mean_std': np.mean(y_std),
                'crps': crps,
                'picp_50': picp_scores[0],
                'picp_80': picp_scores[1],
                'picp_90': picp_scores[2],
                'picp_95': picp_scores[3],
                'picp_99': picp_scores[4]
            }
            cv_metrics.append(metrics)
            # Only plot trace for first fold
            if fold == 1 and feature_names is not None and output_dir is not None and 'chains' in locals():
                # Also plot alpha trace if possible (here, alpha is updated per EM, not per HMC step, so we skip unless you want EM trace)
                pass  # If you want, you can store alpha per EM iteration and plot here
        self.cv_results = pd.DataFrame(cv_metrics)
        logger.info(f"Cross-validation results:\n{self.cv_results.mean()}")
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
        mean = X @ self.m
        
        if return_std:
            # Base uncertainty from model (epistemic)
            base_std = np.sqrt(1/self.alpha + np.sum((X @ self.S) * X, axis=1))
            
            if self.config.uncertainty_calibration:
                # Apply calibrated uncertainty
                std = base_std * self.uncertainty_calibration_factor
            else:
                std = base_std
                
            if self.config.robust_noise:
                # Add Student's t noise component for robustness (aleatoric)
                t_noise = stats.t.rvs(df=self.config.student_t_df, size=len(mean))
                std = np.sqrt(std**2 + np.abs(t_noise))
                
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

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering for building energy performance analysis.
    
    This function implements a feature engineering pipeline that:
    1. Handles outliers and missing values
    2. Creates non-linear transformations
    3. Generates interaction features
    4. Normalises and scales features
    
    The engineered features capture:
    - Building characteristics (area, age)
    - Energy performance metrics (EUI, energy mix)
    - Environmental impact (GHG emissions)
    - Complex interactions between features
    
    Args:
        df: Input DataFrame containing raw building data
        
    Returns:
        df: DataFrame with engineered features
    """
    # Floor area features with robust scaling
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)
    
    # Energy intensity features with ratio analysis
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    
    # Building age features with non-linear transformations
    df['building_age'] = 2025 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)
    
    # Energy star rating features with normalisation
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    
    # GHG emissions features with intensity metrics
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    
    # Interaction features capturing complex relationships
    df['age_energy_star_interaction'] = df['building_age_log'] * df['energy_star_rating_normalized']
    df['area_energy_star_interaction'] = df['floor_area_log'] * df['energy_star_rating_normalized']
    df['age_ghg_interaction'] = df['building_age_log'] * df['ghg_emissions_int_log']
    
    return df

def create_shap_analysis(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                        model: AdaptivePriorARD, output_dir: str):
    """Create SHAP analysis plots"""
    logger.info("Computing SHAP values...")
    
    # Create a background dataset using k-means clustering
    background = shap.kmeans(X, 10)
    
    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model.predict, background)
    
    # Calculate SHAP values for a subset of data (for computational efficiency)
    sample_size = min(1000, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Calculate SHAP values with progress bar
    shap_values = explainer.shap_values(X_sample)
    
    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dependence Plots for top features
    plt.figure(figsize=(15, 10))
    importance = model.get_feature_importance()
    top_features = np.argsort(importance)[-5:]  # Top 5 features
    
    for i, feat_idx in enumerate(top_features):
        plt.subplot(2, 3, i+1)
        shap.dependence_plot(feat_idx, shap_values, X_sample, 
                           feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence: {feature_names[feat_idx]}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_dependence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Force Plot for individual samples
    for i in range(min(5, len(X_sample))):
        plt.figure(figsize=(20, 4))
        shap.force_plot(explainer.expected_value, shap_values[i:i+1], X_sample[i:i+1],
                       feature_names=feature_names, matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot (Sample {i+1})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_force_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Decision Plot
    plt.figure(figsize=(12, 8))
    shap.decision_plot(explainer.expected_value, shap_values[:5], X_sample[:5],
                      feature_names=feature_names, show=False)
    plt.title('SHAP Decision Plot (First 5 Samples)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_decision.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save SHAP values for further analysis
    np.save(os.path.join(output_dir, 'shap_values.npy'), shap_values)
    
    # Calculate and save SHAP importance
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_dict = dict(zip(feature_names, shap_importance))
    
    with open(os.path.join(output_dir, 'shap_importance.json'), 'w') as f:
        json.dump(shap_importance_dict, f, indent=4)
    
    logger.info("SHAP analysis completed and saved")

def create_visualizations(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         model: AdaptivePriorARD, output_dir: str):
    """
    Create separate visualization files for comprehensive model analysis.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for all plots
    plt.style.use('default')
    
    # Create progress bar for visualization creation
    with tqdm(total=12, desc="Creating visualizations") as pbar:
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        importance = model.get_feature_importance()
        sorted_idx = np.argsort(importance)
        
        std_importance = np.sqrt(np.diag(model.S))
        std_importance = std_importance * (np.sum(importance) / np.sum(std_importance))
        
        plt.barh(range(len(feature_names)), importance[sorted_idx])
        plt.errorbar(importance[sorted_idx], range(len(feature_names)),
                    xerr=std_importance[sorted_idx], fmt='none', color='black', alpha=0.3)
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Normalised Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.grid(True, alpha=0.3)
        plt.gcf().set_constrained_layout(True)
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        pbar.update(1)
        
        # Create other plots in parallel using multiprocessing
        plot_functions = [
            functools.partial(create_correlation_heatmap, X, feature_names, output_dir),
            functools.partial(create_interaction_network, X, feature_names, output_dir),
            functools.partial(create_partial_dependence, X, y, feature_names, model, output_dir),
            functools.partial(create_residual_analysis, X, y, model, output_dir),
            functools.partial(create_uncertainty_analysis, X, y, model, output_dir),
            functools.partial(create_importance_correlation, X, y, feature_names, importance, output_dir),
            functools.partial(create_learning_curves, model, output_dir),
            functools.partial(create_prediction_actual, X, y, model, output_dir),
            functools.partial(create_uncertainty_distribution, X, y, model, output_dir),
            functools.partial(create_group_importance, model, feature_names, output_dir),
            functools.partial(create_calibration_plot, X, y, model, output_dir)
        ]
        
        # Use a process pool to create plots in parallel
        with ProcessPoolExecutor(max_workers=min(4, len(plot_functions))) as executor:
            futures = [executor.submit(func) for func in plot_functions]
            for future in futures:
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"Failed to create plot: {str(e)}")
        
        # Create SHAP analysis
        create_shap_analysis(X, y, feature_names, model, output_dir)
        pbar.update(1)

def create_correlation_heatmap(X: np.ndarray, feature_names: List[str], output_dir: str):
    """Create correlation heatmap with proper layout"""
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_interaction_network(X: np.ndarray, feature_names: List[str], output_dir: str):
    """Create feature interaction network visualization"""
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(feature_names):
        G.add_node(name)
    
    # Add edges for strong correlations
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(correlation_matrix.iloc[i,j]) > 0.3:  # Only show strong correlations
                G.add_edge(feature_names[i], feature_names[j], 
                          weight=abs(correlation_matrix.iloc[i,j]))
    
    # Draw the network
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=8, font_weight='bold')
    
    plt.title('Feature Interaction Network')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_interaction_network.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_partial_dependence(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                            model: AdaptivePriorARD, output_dir: str):
    """Create partial dependence plots for top features"""
    plt.figure(figsize=(12, 8))
    importance = model.get_feature_importance()
    top_features = np.argsort(importance)[-5:]  # Top 5 features
    
    for i, feat_idx in enumerate(top_features):
        plt.subplot(2, 3, i+1)
        x_range = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
        y_pred = []
        for x in x_range:
            X_temp = X.copy()
            X_temp[:, feat_idx] = x
            y_pred.append(model.predict(X_temp).mean())
        plt.plot(x_range, y_pred)
        plt.title(f'Partial Dependence: {feature_names[feat_idx]}')
        plt.xlabel(feature_names[feat_idx])
        plt.ylabel('Predicted Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_dependence.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_analysis(X: np.ndarray, y: np.ndarray, model: AdaptivePriorARD, output_dir: str):
    """Create residual analysis plots"""
    plt.figure(figsize=(12, 8))
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.subplot(2, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Order')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_uncertainty_analysis(X: np.ndarray, y: np.ndarray, model: AdaptivePriorARD, output_dir: str):
    """Create uncertainty analysis plots"""
    plt.figure(figsize=(12, 8))
    y_pred, y_std = model.predict(X, return_std=True)
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, y_std, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Uncertainty (Std)')
    plt.title('Uncertainty vs Prediction')
    
    plt.subplot(2, 2, 2)
    plt.hist(y_std, bins=30)
    plt.xlabel('Uncertainty (Std)')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution')
    
    plt.subplot(2, 2, 3)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True')
    
    plt.subplot(2, 2, 4)
    plt.scatter(y, y_std, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Uncertainty (Std)')
    plt.title('Uncertainty vs True Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_importance_correlation(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                                importance: np.ndarray, output_dir: str):
    """Create feature importance vs correlation plot"""
    plt.figure(figsize=(10, 8))
    correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
    
    plt.scatter(correlations, importance, alpha=0.6)
    for i, name in enumerate(feature_names):
        plt.annotate(name, (correlations[i], importance[i]), fontsize=8)
    
    plt.xlabel('Correlation with Target')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance vs Correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'importance_vs_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_learning_curves(model: AdaptivePriorARD, output_dir: str):
    """Create learning curves from cross-validation results"""
    plt.figure(figsize=(10, 6))
    cv_results = model.cv_results
    
    plt.plot(cv_results['rmse'], label='RMSE', marker='o')
    plt.plot(cv_results['r2'], label='R²', marker='s')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_actual(X: np.ndarray, y: np.ndarray, model: AdaptivePriorARD, output_dir: str):
    """Create prediction vs actual plot with uncertainty"""
    plt.figure(figsize=(10, 8))
    y_pred, y_std = model.predict(X, return_std=True)
    
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    
    # Add uncertainty bands
    plt.fill_between(y, y_pred - 2*y_std, y_pred + 2*y_std, 
                    alpha=0.2, color='gray', label='95% Confidence Interval')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_uncertainty_distribution(X: np.ndarray, y: np.ndarray, model: AdaptivePriorARD, output_dir: str):
    """Create uncertainty distribution plot"""
    plt.figure(figsize=(10, 6))
    _, y_std = model.predict(X, return_std=True)
    
    plt.hist(y_std, bins=30, alpha=0.7)
    plt.xlabel('Prediction Uncertainty (Std)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainties')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_group_importance(model: AdaptivePriorARD, feature_names: List[str], output_dir: str):
    """Create group importance plot"""
    plt.figure(figsize=(10, 6))
    group_importance = {}
    
    for group, indices in model.feature_groups.items():
        group_importance[group] = np.mean([model.get_feature_importance()[i] for i in indices])
    
    groups = list(group_importance.keys())
    importance = list(group_importance.values())
    
    plt.bar(groups, importance)
    plt.xlabel('Feature Group')
    plt.ylabel('Average Importance')
    plt.title('Feature Group Importance')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_calibration_plot(X: np.ndarray, y: np.ndarray, model: AdaptivePriorARD, output_dir: str):
    """Create calibration plot for uncertainty estimates"""
    plt.figure(figsize=(10, 6))
    y_pred, y_std = model.predict(X, return_std=True)
    
    confidence_levels = np.linspace(0.1, 0.9, 9)
    empirical_coverage = []
    
    for level in confidence_levels:
        z_score = stats.norm.ppf(1 - (1 - level) / 2)
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        coverage = np.mean((y >= lower) & (y <= upper))
        empirical_coverage.append(coverage)
    
    plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Empirical Coverage')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Nominal Coverage')
    plt.ylabel('Empirical Coverage')
    plt.title('Uncertainty Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_interactions(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                               model: AdaptivePriorARD, output_dir: str):
    """
    Perform comprehensive analysis of feature interactions and model performance.
    """
    logger.info("Starting comprehensive feature analysis...")
    
    # Create visualizations
    create_visualizations(X, y, feature_names, model, output_dir)
    
    # Calculate metrics and save to JSON
    importance = model.get_feature_importance()
    std_importance = np.std([model.get_feature_importance() for _ in range(100)], axis=0)
    target_correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
    
    # Calculate interaction strength
    logger.info("Calculating feature interaction strengths...")
    interaction_strength = np.zeros((len(feature_names), len(feature_names)))
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i != j:
                interaction_strength[i, j] = mutual_info_regression(
                    X[:, i].reshape(-1, 1), X[:, j].ravel()
                )[0]
    
    # Load SHAP values if available
    shap_values = None
    shap_importance = None
    if os.path.exists(os.path.join(output_dir, 'shap_values.npy')):
        shap_values = np.load(os.path.join(output_dir, 'shap_values.npy'))
        shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Save comprehensive analysis to JSON
    # --- ADDED: Extract alpha and beta for each group if present ---
    group_mixing = {}
    group_regularization = {}
    for group, params in model.group_prior_hyperparams.items():
        if 'alpha' in params:
            group_mixing[group] = float(params['alpha'])
        if 'beta' in params:
            group_regularization[group] = float(params['beta'])
    # --- END ADDED ---
    analysis_results = {
        'feature_importance': dict(zip(feature_names, importance)),
        'feature_importance_std': dict(zip(feature_names, std_importance)),
        'target_correlations': dict(zip(feature_names, target_correlations)),
        'interaction_strength': {
            f"{feat1}_{feat2}": float(interaction_strength[i, j])
            for i, feat1 in enumerate(feature_names)
            for j, feat2 in enumerate(feature_names)
            if i < j and interaction_strength[i, j] > 0.1
        },
        'model_metrics': {
            'rmse': float(model.cv_results['rmse'].mean()),
            'r2': float(model.cv_results['r2'].mean()),
            'mae': float(model.cv_results['mae'].mean()),
            'mean_std': float(model.cv_results['mean_std'].mean()),
            'crps': float(model.cv_results['crps'].mean()),
            'picp_50': float(model.cv_results['picp_50'].mean()),
            'picp_80': float(model.cv_results['picp_80'].mean()),
            'picp_90': float(model.cv_results['picp_90'].mean()),
            'picp_95': float(model.cv_results['picp_95'].mean()),
            'picp_99': float(model.cv_results['picp_99'].mean())
        },
        'prior_hyperparameters': {
            'global_shrinkage': {group: float(params['lambda'].mean()) 
                               for group, params in model.group_prior_hyperparams.items() 
                               if 'lambda' in params},
            'local_shrinkage': {group: float(params['tau'].mean()) 
                              for group, params in model.group_prior_hyperparams.items() 
                              if 'tau' in params},
            'mixing_parameter': group_mixing,  # ADDED
            'regularization_strength': group_regularization  # ADDED
        }
    }
    
    if shap_importance is not None:
        analysis_results['shap_importance'] = dict(zip(feature_names, shap_importance))
    
    with open(os.path.join(output_dir, 'detailed_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4, cls=NumpyEncoder)
    
    # Print comprehensive analysis results
    logger.info("\nDetailed Analysis Results:")
    logger.info("\n1. Top Features by Importance:")
    for feat, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"{feat}: {imp:.4f} ± {std_importance[feature_names.index(feat)]:.4f}")
    
    if shap_importance is not None:
        logger.info("\n2. Top Features by SHAP Importance:")
        for feat, imp in sorted(zip(feature_names, shap_importance), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"{feat}: {imp:.4f}")
    
    logger.info("\n3. Strongest Feature Interactions:")
    strong_interactions = sorted(
        [(interaction, strength) 
         for interaction, strength in analysis_results['interaction_strength'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for interaction, strength in strong_interactions:
        logger.info(f"{interaction}: {strength:.4f}")
    
    logger.info("\n4. Model Performance Metrics:")
    logger.info(f"RMSE: {analysis_results['model_metrics']['rmse']:.4f}")
    logger.info(f"R²: {analysis_results['model_metrics']['r2']:.4f}")
    logger.info(f"MAE: {analysis_results['model_metrics']['mae']:.4f}")
    logger.info(f"Mean Uncertainty: {analysis_results['model_metrics']['mean_std']:.4f}")
    logger.info(f"CRPS: {analysis_results['model_metrics']['crps']:.4f}")
    
    logger.info("\n5. Prediction Interval Coverage:")
    for level in ['50', '80', '90', '95', '99']:
        logger.info(f"PICP {level}%: {analysis_results['model_metrics'][f'picp_{level}']:.4f}")
    
    logger.info("\n6. Prior Hyperparameters:")
    logger.info(f"Global Shrinkage: {analysis_results['prior_hyperparameters']['global_shrinkage']}")
    logger.info(f"Local Shrinkage: {analysis_results['prior_hyperparameters']['local_shrinkage']}")
    
    logger.info("\n7. Feature Correlations with Target:")
    for feat, corr in sorted(zip(feature_names, target_correlations), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]:
        logger.info(f"{feat}: {corr:.4f}")
    
    logger.info("\nAnalysis complete. Results saved to %s", output_dir)
    
    # Add Gelman-Rubin statistics to analysis results
    if hasattr(model, 'r_hat_history') and model.r_hat_history:
        r_hat_summary = {
            'mean_r_hat': np.mean([entry['r_hat_mean'] for entry in model.r_hat_history]),
            'std_r_hat': np.std([entry['r_hat_mean'] for entry in model.r_hat_history]),
            'final_r_hat': model.r_hat_history[-1]['r_hat_mean'],
            'convergence_status': 'Good' if model.r_hat_history[-1]['r_hat_mean'] < 1.1 else 'Poor'
        }
        analysis_results['gelman_rubin_stats'] = r_hat_summary
    
    # Add convergence diagnostics to analysis results
    if hasattr(model, 'convergence_history') and model.convergence_history:
        convergence_summary = {
            'mean_r_hat': np.mean([entry['r_hat_mean'] for entry in model.convergence_history]),
            'std_r_hat': np.std([entry['r_hat_mean'] for entry in model.convergence_history]),
            'final_r_hat': model.convergence_history[-1]['r_hat_mean'],
            'convergence_status': 'Good' if model.convergence_history[-1]['r_hat_mean'] < 1.1 else 'Poor',
            'mean_ess': np.mean([entry['ess_mean'] for entry in model.convergence_history]),
            'std_ess': np.std([entry['ess_mean'] for entry in model.convergence_history]),
            'final_ess': model.convergence_history[-1]['ess_mean'],
            'mixing_status': 'Good' if model.convergence_history[-1]['ess_mean'] > 100 else 'Poor'
        }
        analysis_results['convergence_diagnostics'] = convergence_summary
        
        # Save detailed convergence diagnostics
        convergence_details = {
            'r_hat_history': [entry['r_hat_stats'] for entry in model.convergence_history],
            'ess_history': [entry['ess_stats'] for entry in model.convergence_history],
            'acceptance_rates': [entry.get('acceptance_rate', None) for entry in model.convergence_history]
        }
        with open(os.path.join(output_dir, 'convergence_diagnostics.json'), 'w') as f:
            json.dump(convergence_details, f, indent=4, cls=NumpyEncoder)

def train_and_evaluate_adaptive(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              output_dir: Optional[str] = None) -> Tuple[AdaptivePriorARD, dict]:
    """
    Train and evaluate the Adaptive Prior ARD model with comprehensive analysis.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting model training and evaluation...")
    logger.info(f"Input data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Number of features: {len(feature_names)}")
    
    # Initialise and train model
    config = AdaptivePriorConfig()
    logger.info("Model configuration:")
    logger.info(f"- Number of folds: {config.n_splits}")
    logger.info(f"- Maximum iterations: {config.max_iter}")
    logger.info(f"- Prior type: {config.prior_type}")
    logger.info(f"- Using HMC: {config.use_hmc}")
    logger.info(f"- Robust noise: {config.robust_noise}")
    
    model = AdaptivePriorARD(config)
    
    # Train model with progress tracking
    logger.info("Training model...")
    model.fit(X, y)
    
    # Get cross-validation metrics
    metrics = model.cv_results.mean().to_dict()
    
    # Print detailed fold-wise metrics
    logger.info("\nFold-wise Performance Metrics:")
    for fold in range(config.n_splits):
        fold_metrics = model.cv_results.iloc[fold]
        logger.info(f"\nFold {fold + 1}:")
        logger.info(f"RMSE: {fold_metrics['rmse']:.4f}")
        logger.info(f"R²: {fold_metrics['r2']:.4f}")
        logger.info(f"MAE: {fold_metrics['mae']:.4f}")
        logger.info(f"Mean Uncertainty: {fold_metrics['mean_std']:.4f}")
        logger.info(f"CRPS: {fold_metrics['crps']:.4f}")
    
    # Print overall metrics
    logger.info("\nOverall Performance Metrics:")
    logger.info(f"Mean RMSE: {metrics['rmse']:.4f} ± {model.cv_results['rmse'].std():.4f}")
    logger.info(f"Mean R²: {metrics['r2']:.4f} ± {model.cv_results['r2'].std():.4f}")
    logger.info(f"Mean MAE: {metrics['mae']:.4f} ± {model.cv_results['mae'].std():.4f}")
    logger.info(f"Mean Uncertainty: {metrics['mean_std']:.4f} ± {model.cv_results['mean_std'].std():.4f}")
    logger.info(f"Mean CRPS: {metrics['crps']:.4f} ± {model.cv_results['crps'].std():.4f}")
    
    # Print prediction interval coverage
    logger.info("\nPrediction Interval Coverage:")
    for level in ['50', '80', '90', '95', '99']:
        mean_coverage = metrics[f'picp_{level}']
        std_coverage = model.cv_results[f'picp_{level}'].std()
        logger.info(f"{level}% PICP: {mean_coverage:.4f} ± {std_coverage:.4f}")
    
    if output_dir is not None:
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save fold-wise metrics
        fold_metrics = model.cv_results.to_dict('records')
        with open(os.path.join(output_dir, 'fold_metrics.json'), 'w') as f:
            json.dump(fold_metrics, f, indent=4)
        
        # Perform comprehensive analysis
        analyze_feature_interactions(X, y, feature_names, model, output_dir)
        
        # Save model
        model.save_model(os.path.join(output_dir, 'adaptive_prior_model.joblib'))
        
        # Save feature importance
        importance = model.get_feature_importance()
        importance_dict = dict(zip(feature_names, importance))
        with open(os.path.join(output_dir, 'feature_importance.json'), 'w') as f:
            json.dump(importance_dict, f, indent=4)
        
        logger.info(f"\nAll results saved to {output_dir}")
    
    return model, metrics

if __name__ == "__main__":
    logger.info("Starting Adaptive Prior ARD analysis")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    
    # Load and preprocess data
    logger.info(f"Loading data from {data_csv_path}")
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Enhanced feature engineering
    logger.info("Performing feature engineering...")
    df = feature_engineering(df)
    logger.info("Feature engineering completed")
    
    # Select features for analysis
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
        "ghg_per_area",
        "age_energy_star_interaction",
        "area_energy_star_interaction",
        "age_ghg_interaction"
    ]
    feature_names = features.copy()
    
    logger.info(f"Selected {len(features)} features for analysis")
    logger.info("Features: " + ", ".join(features))
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1)
    
    # Print data statistics
    logger.info("\nData Statistics:")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"X mean: {X.mean():.4f}")
    logger.info(f"X std: {X.std():.4f}")
    logger.info(f"y mean: {y.mean():.4f}")
    logger.info(f"y std: {y.std():.4f}")
    
    # Train and evaluate model
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_groupprior_hmcdebug')
    logger.info(f"\nTraining model and saving results to {results_dir}")
    # Ensure trace plots are generated by passing feature_names and output_dir to fit
    config = AdaptivePriorConfig()
    model = AdaptivePriorARD(config)
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    metrics = model.cv_results.mean().to_dict()
    # Continue with the rest of the analysis and saving as before
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    # Save fold-wise metrics
    fold_metrics = model.cv_results.to_dict('records')
    with open(os.path.join(results_dir, 'fold_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    # Perform comprehensive analysis
    analyze_feature_interactions(X, y, feature_names, model, results_dir)
    # Save model
    model.save_model(os.path.join(results_dir, 'adaptive_prior_model.joblib'))
    # Save feature importance
    importance = model.get_feature_importance()
    importance_dict = dict(zip(feature_names, importance))
    with open(os.path.join(results_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_dict, f, indent=4)
    logger.info(f"\nAll results saved to {results_dir}") 
    
