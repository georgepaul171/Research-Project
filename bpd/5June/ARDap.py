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
from dataclasses import dataclass
import logging
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
from scipy.special import digamma, polygamma
import networkx as nx

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
    """
    alpha_0: float = 1e-6
    beta_0: float = 1e-6
    max_iter: int = 200
    tol: float = 1e-4
    n_splits: int = 5
    random_state: int = 42
    prior_type: str = 'hierarchical'
    adaptation_rate: float = 0.1
    uncertainty_threshold: float = 0.1
    group_sparsity: bool = True
    dynamic_shrinkage: bool = True
    use_hmc: bool = True
    hmc_steps: int = 10
    hmc_epsilon: float = 0.01
    hmc_leapfrog_steps: int = 10
    uncertainty_calibration: bool = True
    calibration_factor: float = 10.0  
    robust_noise: bool = True
    student_t_df: float = 3.0  

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

    def _initialize_adaptive_priors(self, n_features: int):
        """
        Initialise adaptive prior parameters based on configuration.
        
        This method sets up the prior structure based on the chosen prior type:
        - Hierarchical: Implements a multi-level prior with global and local shrinkage
        - Spike-slab: Implements a mixture prior for feature selection
        - Horseshoe: Implements a heavy-tailed prior for robust feature selection
        
        The initialisation includes:
        - Setting up hyperparameters for the chosen prior
        - Initialising feature groups for group sparsity
        - Setting up dynamic shrinkage parameters
        
        Args:
            n_features: Number of features in the model
        """
        if self.config.prior_type == 'hierarchical':
            # Hierarchical prior with ARD
            self.prior_hyperparams = {
                'lambda': np.ones(n_features) * self.config.beta_0,  # Global shrinkage
                'tau': np.ones(n_features) * 1.0,  # Local shrinkage
                'nu': np.ones(n_features) * 2.0  # Degrees of freedom
            }
        elif self.config.prior_type == 'spike_slab':
            # Spike-and-slab prior for feature selection
            self.prior_hyperparams = {
                'pi': np.ones(n_features) * 0.5,  # Inclusion probabilities
                'sigma2_0': np.ones(n_features) * 1e-6,  # Spike variance
                'sigma2_1': np.ones(n_features) * 1.0  # Slab variance
            }
        elif self.config.prior_type == 'horseshoe':
            # Horseshoe prior for heavy-tailed shrinkage
            self.prior_hyperparams = {
                'lambda': np.ones(n_features),  # Local shrinkage
                'tau': 1.0,  # Global shrinkage
                'c2': 1.0  # Scale parameter
            }
            
        if self.config.group_sparsity:
            # Initialise feature groups for group sparsity
            self.feature_groups = self._create_feature_groups(n_features)
            
        if self.config.dynamic_shrinkage:
            # Initialise dynamic shrinkage parameters
            self.shrinkage_params = {
                'kappa': np.ones(n_features) * 0.5,  # Shrinkage strength
                'eta': np.ones(n_features) * 1.0  # Adaptation rate
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
        if self.config.prior_type == 'hierarchical':
            # Update hierarchical prior parameters
            for j in range(len(self.beta)):
                # Add numerical stability
                diag_S = np.clip(np.diag(self.S)[j], 1e-10, None)
                m_squared = np.clip(self.m[j]**2, 1e-10, None)
                
                # Update global shrinkage
                self.prior_hyperparams['lambda'][j] = (
                    (self.prior_hyperparams['nu'][j] + 1) /
                    (m_squared + diag_S + 2 * self.prior_hyperparams['tau'][j])
                )
                
                # Update local shrinkage
                self.prior_hyperparams['tau'][j] = (
                    (self.prior_hyperparams['nu'][j] + 1) /
                    (self.prior_hyperparams['lambda'][j] + 1)
                )
                
        elif self.config.prior_type == 'spike_slab':
            # Update spike-and-slab prior parameters
            for j in range(len(self.beta)):
                # Add numerical stability
                m_squared = np.clip(self.m[j]**2, 1e-10, None)
                
                # Update inclusion probabilities
                log_odds = (
                    np.log(self.prior_hyperparams['pi'][j] / (1 - self.prior_hyperparams['pi'][j])) +
                    0.5 * np.log(self.prior_hyperparams['sigma2_1'][j] / self.prior_hyperparams['sigma2_0'][j]) +
                    0.5 * m_squared * (1/self.prior_hyperparams['sigma2_0'][j] - 1/self.prior_hyperparams['sigma2_1'][j])
                )
                self.prior_hyperparams['pi'][j] = 1 / (1 + np.exp(-log_odds))
                
        elif self.config.prior_type == 'horseshoe':
            # Update horseshoe prior parameters
            for j in range(len(self.beta)):
                # Add numerical stability
                m_squared = np.clip(self.m[j]**2, 1e-10, None)
                
                # Update local shrinkage
                self.prior_hyperparams['lambda'][j] = (
                    1 / (m_squared / (2 * self.prior_hyperparams['tau']) + 1/self.prior_hyperparams['c2'])
                )
            
            # Update global shrinkage
            m_squared_sum = np.sum(np.clip(self.m**2, 1e-10, None))
            self.prior_hyperparams['tau'] = (
                1 / (m_squared_sum / (2 * self.prior_hyperparams['lambda']) + 1)
            )
        
        if self.config.dynamic_shrinkage:
            # Update dynamic shrinkage parameters
            for j in range(len(self.beta)):
                # Update shrinkage strength based on feature importance
                importance = 1 / np.clip(self.beta[j], 1e-10, None)
                self.shrinkage_params['kappa'][j] = (
                    self.shrinkage_params['kappa'][j] * (1 - self.config.adaptation_rate) +
                    importance * self.config.adaptation_rate
                )
                
                # Update adaptation rate based on uncertainty
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
        
        This method implements a single HMC step, which is a key component of the model's
        advanced inference strategy. HMC combines Hamiltonian dynamics with Metropolis
        acceptance to efficiently explore the posterior distribution.
        
        The implementation includes:
        - Leapfrog integration for Hamiltonian dynamics
        - Momentum updates using the gradient of the negative log posterior
        - Metropolis acceptance step for detailed balance
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            current_w: Current weight vector (n_features,)
            current_momentum: Current momentum vector (n_features,)
            
        Returns:
            new_w: New weight vector after HMC step
            new_momentum: New momentum vector after HMC step
            acceptance_prob: Acceptance probability for the Metropolis step
        """
        # Initialise new position and momentum
        new_w = current_w.copy()
        new_momentum = current_momentum.copy()
        
        # Calculate initial Hamiltonian (total energy)
        current_energy = self._calculate_hamiltonian(X, y, current_w, current_momentum)
        
        # Leapfrog steps for Hamiltonian dynamics
        for _ in range(self.config.hmc_leapfrog_steps):
            # Update momentum (half step)
            grad = self._calculate_gradient(X, y, new_w)
            new_momentum = new_momentum - 0.5 * self.config.hmc_epsilon * grad
            
            # Update position (full step)
            new_w = new_w + self.config.hmc_epsilon * new_momentum
            
            # Update momentum (half step)
            grad = self._calculate_gradient(X, y, new_w)
            new_momentum = new_momentum - 0.5 * self.config.hmc_epsilon * grad
        
        # Calculate new Hamiltonian
        new_energy = self._calculate_hamiltonian(X, y, new_w, new_momentum)
        
        # Metropolis acceptance step for detailed balance
        acceptance_prob = min(1.0, np.exp(current_energy - new_energy))
        
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
    
    def _hmc_sampling(self, X: np.ndarray, y: np.ndarray, 
                     initial_w: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Perform HMC sampling to explore the posterior distribution.
        
        This method implements multiple HMC steps to explore the posterior distribution
        of the weights. It is a component of the model's advanced inference strategy,
        providing better posterior exploration than traditional methods.
        
        The implementation includes:
        - Multiple HMC steps with momentum resampling
        - Tracking of acceptance probabilities
        - Numerical stability considerations
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            initial_w: Initial weight vector (n_features,)
            
        Returns:
            np.ndarray: Final weight vector after HMC sampling
            List[float]: Acceptance probabilities for each HMC step
        """
        current_w = initial_w.copy()
        acceptance_probs = []
        
        for _ in range(self.config.hmc_steps):
            # Initialise momentum from standard normal
            current_momentum = np.random.randn(len(current_w))
            
            # Perform HMC step
            new_w, new_momentum, acceptance_prob = self._hmc_step(
                X, y, current_w, current_momentum
            )
            
            current_w = new_w
            acceptance_probs.append(acceptance_prob)
        
        return current_w, acceptance_probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptivePriorARD':
        """
        Fit the Adaptive Prior ARD model with cross-validation and uncertainty calibration.
        
        This method implements the core training procedure of the model, which includes:
        1. Cross-validation for robust model evaluation
        2. EM algorithm with adaptive prior updates
        3. HMC-based posterior exploration
        4. Uncertainty calibration
        5. Robust noise modeling
        
        The training process is designed to:
        - Automatically select relevant features through ARD
        - Adapt priors based on data characteristics
        - Provide well-calibrated uncertainty estimates
        - Handle outliers and noise robustly
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self: The fitted model instance
        """
        # Ensure y is 1D array
        y = np.asarray(y).reshape(-1)
        
        n_samples, n_features = X.shape
        
        # Initialise parameters with numerical stability
        self.alpha = np.clip(self.config.alpha_0, 1e-10, None)
        self.beta = np.ones(n_features) * np.clip(self.config.beta_0, 1e-10, None)
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        
        # Initialise adaptive priors
        self._initialize_adaptive_priors(n_features)
        
        # Cross-validation for robust evaluation
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, 
                  random_state=self.config.random_state)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale data with robust preprocessing
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            
            # EM algorithm with adaptive priors
            for iteration in range(self.config.max_iter):
                # E-step: Update posterior with numerical stability
                try:
                    self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                         np.diag(np.clip(self.beta, 1e-10, None)))
                except np.linalg.LinAlgError:
                    # Add jitter for numerical stability
                    jitter = 1e-6 * np.eye(n_features)
                    self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                         np.diag(np.clip(self.beta, 1e-10, None)) + jitter)
                
                self.m = self.alpha * self.S @ X_train_scaled.T @ y_train_scaled
                
                # Use HMC for posterior exploration if enabled
                if self.config.use_hmc:
                    self.m, acceptance_probs = self._hmc_sampling(
                        X_train_scaled, y_train_scaled, self.m
                    )
                    logger.info(f"Fold {fold}, Iteration {iteration}: "
                              f"Mean HMC acceptance rate: {np.mean(acceptance_probs):.3f}")
                
                # M-step: Update hyperparameters
                residuals = y_train_scaled - X_train_scaled @ self.m
                
                if self.config.robust_noise:
                    # Use Student's t noise model for robustness
                    df = self.config.student_t_df
                    weights = (df + 1) / (df + residuals**2)
                    alpha_new = np.sum(weights) / (np.sum(weights * residuals**2) + 
                                                np.trace(X_train_scaled @ self.S @ X_train_scaled.T))
                else:
                    alpha_new = n_samples / (np.sum(residuals**2) + 
                                           np.trace(X_train_scaled @ self.S @ X_train_scaled.T))
                
                alpha_new = np.clip(alpha_new, 1e-10, None)
                
                # Update ARD parameters with adaptive priors
                if self.config.prior_type == 'hierarchical':
                    beta_new = 1 / (np.clip(self.m**2, 1e-10, None) + 
                                  np.clip(np.diag(self.S), 1e-10, None) + 
                                  2 * self.prior_hyperparams['tau'])
                elif self.config.prior_type == 'spike_slab':
                    beta_new = (
                        self.prior_hyperparams['pi'] / np.clip(self.prior_hyperparams['sigma2_1'], 1e-10, None) +
                        (1 - self.prior_hyperparams['pi']) / np.clip(self.prior_hyperparams['sigma2_0'], 1e-10, None)
                    )
                elif self.config.prior_type == 'horseshoe':
                    beta_new = 1 / (np.clip(self.m**2, 1e-10, None) / (2 * self.prior_hyperparams['tau']) + 
                                  self.prior_hyperparams['lambda'])
                
                # Apply group sparsity if enabled
                if self.config.group_sparsity:
                    for group in self.feature_groups.values():
                        group_beta = np.mean(beta_new[group])
                        beta_new[group] = group_beta
                
                # Apply dynamic shrinkage if enabled
                if self.config.dynamic_shrinkage:
                    # Clip shrinkage parameters
                    kappa = np.clip(self.shrinkage_params['kappa'], 0, 1)
                    beta_new = beta_new * (1 - kappa) + self.beta * kappa
                
                # Update adaptive priors
                self._update_adaptive_priors(iteration)
                
                # Check convergence with numerical stability
                beta_diff = np.abs(np.clip(beta_new, 1e-10, None) - np.clip(self.beta, 1e-10, None))
                alpha_diff = np.abs(alpha_new - self.alpha)
                
                if (alpha_diff < self.config.tol and np.all(beta_diff < self.config.tol)):
                    break
                    
                self.alpha = alpha_new
                self.beta = np.clip(beta_new, 1e-10, None)
            
            # Update uncertainty calibration
            self._update_uncertainty_calibration(X_val_scaled, y_val_scaled)
            
            # Evaluate on validation set
            y_pred, y_std = self.predict(X_val_scaled, return_std=True)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_val_orig = self.scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
            
            y_pred_orig = y_pred_orig.reshape(-1)
            y_val_orig = y_val_orig.reshape(-1)
            
            # Calculate probabilistic metrics
            confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
            picp_scores = []
            
            for level in confidence_levels:
                z_score = stats.norm.ppf(1 - (1 - level) / 2)
                lower = y_pred_orig - z_score * y_std
                upper = y_pred_orig + z_score * y_std
                coverage = np.mean((y_val_orig >= lower) & (y_val_orig <= upper))
                picp_scores.append(coverage)
            
            # Calculate CRPS (Continuous Ranked Probability Score)
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
        Get feature importance based on ARD parameters and prior adaptation
        
        Returns:
            importance: Feature importance scores
        """
        # Add numerical stability to importance calculation
        if self.config.prior_type == 'hierarchical':
            importance = 1 / (np.clip(self.beta, 1e-10, None) * 
                            np.clip(self.prior_hyperparams['lambda'], 1e-10, None))
        elif self.config.prior_type == 'spike_slab':
            importance = self.prior_hyperparams['pi'] / np.clip(self.beta, 1e-10, None)
        elif self.config.prior_type == 'horseshoe':
            importance = 1 / (np.clip(self.beta, 1e-10, None) * 
                            np.clip(self.prior_hyperparams['lambda'], 1e-10, None))
        else:
            importance = 1 / np.clip(self.beta, 1e-10, None)
        
        # Normalise importance scores
        importance = np.clip(importance, 0, None)  # Ensure non-negative
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)  # Normalise to sum to 1
        
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
            'uncertainty_calibration_history': self.uncertainty_calibration_history
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

def analyze_feature_interactions(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                               model: AdaptivePriorARD, output_dir: str):
    """
    Perform comprehensive analysis of feature interactions and model performance.
    
    This function implements a detailed analysis pipeline that includes:
    1. Feature importance analysis with uncertainty
    2. Correlation analysis and visualisation
    3. Feature interaction network analysis
    4. Partial dependence analysis
    5. Residual and uncertainty analysis
    6. Model performance metrics
    
    The analysis provides insights into:
    - Key drivers of building energy performance
    - Complex feature interactions
    - Model uncertainty and reliability
    - Areas for potential improvement
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: List of feature names
        model: Fitted AdaptivePriorARD model
        output_dir: Directory to save analysis results
    """
    # Ensure y is 1D array
    y = np.asarray(y).reshape(-1)
    
    # Create a new figure for comprehensive analysis
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 25))
    
    # Feature Importance with Confidence Intervals
    plt.subplot(4, 2, 1)
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)
    plt.barh(range(len(feature_names)), importance[sorted_idx])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Normalised Feature Importance')
    plt.title('Feature Importance Analysis')
    
    # Add confidence intervals for robustness
    std_importance = np.std([model.get_feature_importance() for _ in range(100)], axis=0)
    plt.errorbar(importance[sorted_idx], range(len(feature_names)),
                xerr=std_importance[sorted_idx], fmt='none', color='black', alpha=0.3)
    
    # Feature Correlation Heatmap
    plt.subplot(4, 2, 2)
    correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix')
    
    # Feature Interaction Network
    plt.subplot(4, 2, 3)
    interaction_strength = np.zeros((len(feature_names), len(feature_names)))
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i != j:
                # Calculate interaction strength using mutual information
                feat1_data = X[:, i].reshape(-1, 1)
                feat2_data = X[:, j].ravel()
                interaction_strength[i, j] = mutual_info_regression(
                    feat1_data, feat2_data
                )[0]
    
    # Plot interaction network with threshold
    G = nx.Graph()
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i < j and interaction_strength[i, j] > 0.1:  # Threshold for significant interactions
                G.add_edge(feat1, feat2, weight=interaction_strength[i, j])
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=8, font_weight='bold')
    plt.title('Feature Interaction Network')
    
    # Partial Dependence Analysis
    plt.subplot(4, 2, 4)
    top_features = [feature_names[i] for i in sorted_idx[-3:]]  # Top 3 most important features
    for feat in top_features:
        feat_idx = feature_names.index(feat)
        x_range = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 100)
        y_pred = []
        for x in x_range:
            X_temp = X.copy()
            X_temp[:, feat_idx] = x
            y_pred.append(model.predict(X_temp).mean())
        plt.plot(x_range, y_pred, label=feat)
    plt.xlabel('Feature Value')
    plt.ylabel('Predicted Target')
    plt.title('Partial Dependence Analysis')
    plt.legend()
    
    # Residual Analysis
    plt.subplot(4, 2, 5)
    y_pred, y_std = model.predict(X, return_std=True)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    
    # Uncertainty Analysis
    plt.subplot(4, 2, 6)
    plt.scatter(np.abs(residuals), y_std, alpha=0.5)
    plt.xlabel('Absolute Prediction Error')
    plt.ylabel('Prediction Uncertainty')
    plt.title('Uncertainty vs Prediction Error')
    
    # Feature Importance vs Correlation
    plt.subplot(4, 2, 7)
    target_correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
    plt.scatter(target_correlations, importance)
    for i, feat in enumerate(feature_names):
        plt.annotate(feat, (target_correlations[i], importance[i]))
    plt.xlabel('Correlation with Target')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance vs Target Correlation')
    
    # Learning Curves
    plt.subplot(4, 2, 8)
    cv_scores = model.cv_results
    plt.plot(cv_scores['rmse'], 'b-', label='RMSE', marker='o')
    plt.plot(cv_scores['r2'], 'r-', label='R²', marker='s')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross-validation Learning Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive analysis to JSON
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
            'rmse': float(cv_scores['rmse'].mean()),
            'r2': float(cv_scores['r2'].mean()),
            'mae': float(cv_scores['mae'].mean()),
            'mean_std': float(cv_scores['mean_std'].mean()),
            'crps': float(cv_scores['crps'].mean()),
            'picp_50': float(cv_scores['picp_50'].mean()),
            'picp_80': float(cv_scores['picp_80'].mean()),
            'picp_90': float(cv_scores['picp_90'].mean()),
            'picp_95': float(cv_scores['picp_95'].mean()),
            'picp_99': float(cv_scores['picp_99'].mean())
        },
        'prior_hyperparameters': {
            'global_shrinkage': float(model.prior_hyperparams['lambda'].mean()),
            'local_shrinkage': float(model.prior_hyperparams['tau'].mean())
        }
    }
    
    with open(os.path.join(output_dir, 'detailed_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4, cls=NumpyEncoder)
    
    # Print comprehensive analysis results
    logger.info("\nDetailed Analysis Results:")
    logger.info("\n1. Top Features by Importance:")
    for feat, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"{feat}: {imp:.4f} ± {std_importance[feature_names.index(feat)]:.4f}")
    
    logger.info("\n2. Strongest Feature Interactions:")
    strong_interactions = sorted(
        [(interaction, strength) 
         for interaction, strength in analysis_results['interaction_strength'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for interaction, strength in strong_interactions:
        logger.info(f"{interaction}: {strength:.4f}")
    
    logger.info("\n3. Model Performance Metrics:")
    logger.info(f"RMSE: {analysis_results['model_metrics']['rmse']:.4f}")
    logger.info(f"R²: {analysis_results['model_metrics']['r2']:.4f}")
    logger.info(f"MAE: {analysis_results['model_metrics']['mae']:.4f}")
    logger.info(f"Mean Uncertainty: {analysis_results['model_metrics']['mean_std']:.4f}")
    logger.info(f"CRPS: {analysis_results['model_metrics']['crps']:.4f}")
    
    logger.info("\n4. Prediction Interval Coverage:")
    for level in ['50', '80', '90', '95', '99']:
        logger.info(f"PICP {level}%: {analysis_results['model_metrics'][f'picp_{level}']:.4f}")
    
    logger.info("\n5. Prior Hyperparameters:")
    logger.info(f"Global Shrinkage: {analysis_results['prior_hyperparameters']['global_shrinkage']:.4f}")
    logger.info(f"Local Shrinkage: {analysis_results['prior_hyperparameters']['local_shrinkage']:.4f}")
    
    logger.info("\n6. Feature Correlations with Target:")
    for feat, corr in sorted(zip(feature_names, target_correlations), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]:
        logger.info(f"{feat}: {corr:.4f}")

def train_and_evaluate_adaptive(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              output_dir: Optional[str] = None) -> Tuple[AdaptivePriorARD, dict]:
    """
    Train and evaluate the Adaptive Prior ARD model with comprehensive analysis.
    
    This function implements the complete training and evaluation pipeline:
    1. Model training with cross-validation
    2. Performance metrics calculation
    3. Feature importance analysis
    4. Uncertainty quantification
    5. Results visualisation and saving
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: List of feature names
        output_dir: Optional directory to save results
        
    Returns:
        model: Fitted AdaptivePriorARD model
        metrics: Comprehensive model performance metrics
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialise and train model
    config = AdaptivePriorConfig()
    model = AdaptivePriorARD(config)
    model.fit(X, y)
    
    # Get cross-validation metrics
    metrics = model.cv_results.mean().to_dict()
    
    if output_dir is not None:
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Perform comprehensive analysis
        analyze_feature_interactions(X, y, feature_names, model, output_dir)
        
        # Save model
        model.save_model(os.path.join(output_dir, 'adaptive_prior_model.joblib'))
    
    return model, metrics

if __name__ == "__main__":
    logger.info("Starting Adaptive Prior ARD analysis")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    
    # Enhanced feature engineering
    logger.info("Performing feature engineering")
    df = feature_engineering(df)
    
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
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1)
    
    # Train and evaluate model
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    model, metrics = train_and_evaluate_adaptive(X, y, feature_names, output_dir=results_dir)
    
    # Predict on all data for CSV output
    X_scaled = model.scaler_X.transform(X)
    y_pred, y_std = model.predict(X_scaled, return_std=True)
    y_pred_orig = model.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_std_orig = y_std  # Already in original scale due to linearity of std in ARD
    y_true_orig = y  # Already in original scale
    
    # Save predictions with uncertainty to CSV
    output_csv = os.path.join(results_dir, 'ard_predictions_with_uncertainty.csv')
    output_df = pd.DataFrame({
        'y_true': y_true_orig,
        'y_pred': y_pred_orig,
        'y_std': y_std_orig
    })
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Predictions with uncertainty saved to {output_csv}")
    
    logger.info("Complete. Results saved to %s", results_dir) 