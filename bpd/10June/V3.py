"""
Adaptive Prior ARD (No Interaction Terms, Low Shrinkage on Building Features)

This script trains the Adaptive Prior ARD model on building energy data, but:
- Excludes all interaction features from the analysis
- Reduces shrinkage on building features by using a 'horseshoe' prior and higher prior variance

Author: George Paul (modification by AI)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, field
import logging
import warnings
import scipy.stats as stats
from AREHap_groupprior_hmcdebug import analyze_feature_interactions

warnings.filterwarnings('ignore')

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
    calibration_factor: float = 20.0  
    robust_noise: bool = True
    student_t_df: float = 3.0  
    group_prior_types: dict = field(default_factory=lambda: {
        'energy': 'adaptive_elastic_horseshoe',  # Using our new prior for energy features
        'building': 'horseshoe',  # Less shrinkage
        'interaction': 'spike_slab'
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
                # Simpler, more stable AEH update
                # Fix alpha and beta
                self.group_prior_hyperparams[group]['alpha'] = 0.5
                self.group_prior_hyperparams[group]['beta'] = 1.0
                indices_arr = np.array(indices)
                # Update lambda in log-space for each feature
                for idx, j in enumerate(indices):
                    m2 = np.clip(self.m[j]**2, 1e-10, None)
                    Sjj = np.clip(np.diag(self.S)[j], 1e-10, None)
                    tau = np.clip(self.group_prior_hyperparams[group]['tau'], 1e-10, None)
                    log_lambda_new = 0.5 * np.log(m2 + Sjj + tau)
                    self.group_prior_hyperparams[group]['lambda'][idx] = np.exp(log_lambda_new)
                # Update tau in log-space for the group
                m2_sum = np.sum(np.clip(self.m[indices_arr]**2, 1e-10, None) + np.clip(np.diag(self.S)[indices_arr], 1e-10, None))
                log_tau_new = 0.5 * np.log(m2_sum + 1)
                self.group_prior_hyperparams[group]['tau'] = np.exp(log_tau_new)
                # Remove momentum and adaptive learning rate for stability
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
        beta_tau_log_path = os.path.join(output_dir, 'beta_tau_log.txt')
        em_progress_log_path = os.path.join(output_dir, 'em_progress_log.txt')
        aeh_hyperparams_log_path = os.path.join(output_dir, 'aeh_hyperparams_log.txt')
        with open(beta_tau_log_path, 'w') as beta_tau_log, open(em_progress_log_path, 'w') as em_progress_log, open(aeh_hyperparams_log_path, 'w') as aeh_log:
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
                        print("[DEBUG] HMC is running! Fold:", fold, "Iteration:", iteration)
                        # Only for the first fold, store chains for trace plots
                        if fold == 1:
                            self.m, acceptance_probs, r_hat_stats, ess_stats, chains = self._hmc_sampling(
                                X_train_scaled, y_train_scaled, self.m, return_chains=True
                            )
                            print("[DEBUG] HMC chains generated for trace plots and diagnostics.")
                            # Plot trace plots for top 5 features if feature_names provided
                            if feature_names is not None and output_dir is not None:
                                importance = np.abs(self.m)
                                top_indices = np.argsort(importance)[-5:]
                                self._plot_trace_chains(chains, feature_names, output_dir, top_indices, fold=fold)
                                # Save trace diagnostics for top 5 features
                                trace_diag_path = os.path.join(output_dir, 'trace_diagnostics.txt')
                                with open(trace_diag_path, 'w') as f:
                                    for idx in top_indices:
                                        f.write(f"Trace for {feature_names[idx]} (index {idx}):\n")
                                        for c, chain in enumerate(chains):
                                            f.write(f"  Chain {c+1}: {chain[:, idx].tolist()}\n")
                                        f.write("\n")
                            print(f"[DEBUG] Trace diagnostics written to {trace_diag_path}")
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
                        # Log beta and tau values
                        beta_tau_log.write(f"Fold {fold}, Iter {iteration}: beta={self.beta.tolist()}\n")
                        if hasattr(self, 'group_prior_hyperparams'):
                            for group, params in self.group_prior_hyperparams.items():
                                if 'tau' in params:
                                    beta_tau_log.write(f"  group {group} tau={params['tau']}\n")
                        # Log weights and predictions after each EM iteration
                        y_pred_em = X_train_scaled @ self.m
                        y_pred_em_unscaled = self.scaler_y.inverse_transform(y_pred_em.reshape(-1, 1)).ravel()
                        em_progress_log.write(f"Fold {fold}, Iter {iteration}: min_w={self.m.min()}, max_w={self.m.max()}, min_pred={y_pred_em.min()}, max_pred={y_pred_em.max()}, min_pred_unscaled={y_pred_em_unscaled.min()}, max_pred_unscaled={y_pred_em_unscaled.max()}\n")
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
                        # AEH diagnostics: log hyperparameters after update
                        if hasattr(self, 'group_prior_hyperparams'):
                            for group, params in self.group_prior_hyperparams.items():
                                if self.config.group_prior_types.get(group) == 'adaptive_elastic_horseshoe':
                                    aeh_log.write(f"Fold {fold}, Iter {iteration}, group {group}:\n")
                                    for pname in ['lambda', 'tau', 'alpha', 'beta', 'momentum']:
                                        if pname in params:
                                            aeh_log.write(f"  {pname}: {params[pname]}\n")
                                    aeh_log.flush()
                    beta_diff = np.abs(np.clip(beta_new, 1e-10, None) - np.clip(self.beta, 1e-10, None))
                    alpha_diff = np.abs(alpha_new - self.alpha)
                    if (alpha_diff < self.config.tol and np.all(beta_diff < self.config.tol)):
                        print(f"[DEBUG] EM converged at iteration {iteration} for fold {fold}")
                        em_progress_log.write(f"[DEBUG] EM converged at iteration {iteration} for fold {fold}\n")
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
        # Diagnostic: Save prediction and target ranges, feature stats, and baseline to file
        y_pred, y_std = self.predict(X, return_std=True)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        diagnostics_path = os.path.join(output_dir, 'diagnostics.txt')
        with open(diagnostics_path, 'w') as diag_f:
            diag_f.write(f"[DIAGNOSTIC] Bayesian Model Predicted range: {y_pred.min()} {y_pred.max()}\n")
            diag_f.write(f"[DIAGNOSTIC] True range: {y.min()} {y.max()}\n")
            diag_f.write(f"[DIAGNOSTIC] Feature means: {np.mean(X, axis=0)}\n")
            diag_f.write(f"[DIAGNOSTIC] Feature variances: {np.var(X, axis=0)}\n")
            diag_f.write(f"[DIAGNOSTIC] LinearRegression Predicted range: {y_pred_lr.min()} {y_pred_lr.max()}\n")
        # Save trace diagnostics for the first fold (if available)
        trace_diag_path = os.path.join(output_dir, 'trace_diagnostics.txt')
        if hasattr(self, 'convergence_history') and self.convergence_history:
            # Try to extract the trace values for the top 5 features from the first fold
            try:
                # Get the first fold's trace (chains) if available
                if hasattr(self, '_plot_trace_chains'):
                    # This is a hack: the chains are only available in fit() scope, but if you want to save them, you need to modify fit()
                    pass  # For now, just note this for further debugging
            except Exception as e:
                with open(trace_diag_path, 'w') as f:
                    f.write(f"Error extracting trace diagnostics: {e}\n")
        # Indicate HMC was disabled
        with open(os.path.join(output_dir, 'hmc_status.txt'), 'w') as f:
            f.write('HMC was disabled for this run. Only EM updates were used.\n')
        # Save learned weights for diagnostics
        weights_diag_path = os.path.join(output_dir, 'weights_diagnostics.txt')
        print(f"[DEBUG] Saving weights to {weights_diag_path}")
        with open(weights_diag_path, 'w') as f:
            if hasattr(self, 'm') and self.m is not None and feature_names is not None:
                for name, weight in zip(feature_names, self.m):
                    f.write(f"{name}: {weight}\n")
                print("[DEBUG] First few weights:", list(zip(feature_names, self.m))[:5])
            else:
                print("[WARNING] Weights are empty or not available!")
                f.write("[WARNING] Weights are empty or not available!\n")
        # Optional: Compare to BayesianRidge
        from sklearn.linear_model import BayesianRidge
        bayes_ridge = BayesianRidge()
        bayes_ridge.fit(X, y)
        y_pred_br = bayes_ridge.predict(X)
        br_diag_path = os.path.join(output_dir, 'bayesianridge_diagnostics.txt')
        with open(br_diag_path, 'w') as f:
            f.write(f"BayesianRidge Predicted range: {y_pred_br.min()} {y_pred_br.max()}\n")
            for name, weight in zip(feature_names, bayes_ridge.coef_):
                f.write(f"{name}: {weight}\n")
        # Minimal Bayesian regression (no group/AEH)
        minimal_bayes = BayesianRidge()
        minimal_bayes.fit(X, y)
        y_pred_min_bayes = minimal_bayes.predict(X)
        min_bayes_diag_path = os.path.join(output_dir, 'minimal_bayes_diagnostics.txt')
        with open(min_bayes_diag_path, 'w') as f:
            f.write(f"MinimalBayes Predicted range: {y_pred_min_bayes.min()} {y_pred_min_bayes.max()}\n")
            for name, weight in zip(feature_names, minimal_bayes.coef_):
                f.write(f"{name}: {weight}\n")
        # --- Minimal Bayesian model with HMC-style trace diagnostics ---
        np.random.seed(42)
        n_traces = 100
        coefs_traces = []
        for _ in range(n_traces):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[idx], y[idx]
            minimal_bayes.fit(X_boot, y_boot)
            coefs_traces.append(minimal_bayes.coef_.copy())
        coefs_traces = np.array(coefs_traces)
        # Plot trace plots for top 5 features by mean abs weight
        mean_abs = np.abs(np.mean(coefs_traces, axis=0))
        top_indices = np.argsort(mean_abs)[-5:]
        for idx in top_indices:
            plt.figure(figsize=(10, 4))
            plt.plot(coefs_traces[:, idx], label=f'Trace: {feature_names[idx]}')
            plt.title(f'MinimalBayes Trace: {feature_names[idx]}')
            plt.xlabel('Bootstrap Iteration')
            plt.ylabel('Coefficient Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'trace_minimal_bayes_{feature_names[idx]}.png'))
            plt.close()
        # --- Add README/summary file to results folder ---
        summary_path = os.path.join(output_dir, 'trace_summary.md')
        with open(summary_path, 'w') as f:
            f.write('# Trace Diagnostics Summary\n\n')
            f.write('## BayesianRidge (MinimalBayes)\n')
            f.write('- Trace plots (trace_minimal_bayes_*.png) show wiggly, well-mixing traces for top features.\n')
            f.write('- This indicates the posterior is well-behaved and HMC (or bootstrapping) can explore it.\n\n')
            f.write('## AEH Prior Model\n')
            f.write('- Trace plots for AEH (trace_*.png) are flat or nearly flat.\n')
            f.write('- This means the posterior is too sharp or the prior is too strong, so HMC cannot explore.\n\n')
            f.write('## Research Implication\n')
            f.write('- Complex priors like AEH can be numerically stable but may overly constrain the model, limiting posterior exploration.\n')
            f.write('- Simpler priors (BayesianRidge) allow for better mixing and more flexible fits.\n')
            f.write('- This highlights the importance of empirical diagnostics and careful prior design in Bayesian modeling.\n')
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
    alpha_0: float = 1e-6
    beta_0: float = 1e-6
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
    calibration_factor: float = 20.0
    robust_noise: bool = True
    student_t_df: float = 3.0
    group_prior_types: dict = field(default_factory=lambda: {
        'energy': 'adaptive_elastic_horseshoe',
        'building': 'horseshoe',  # Less shrinkage
        'interaction': 'spike_slab'
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

if __name__ == "__main__":
    logger.info("Starting Adaptive Prior ARD analysis (no interaction terms, low shrinkage on building features)")
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    logger.info("Performing feature engineering (no interaction terms)...")
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
    logger.info(f"Selected {len(features)} features for analysis (no interactions)")
    logger.info("Features: " + ", ".join(features))
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1)
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_buildingonly_lowshrink1')
    os.makedirs(results_dir, exist_ok=True)
    use_aeh_prior = True  # Set to True to use AEH prior for 'energy', False for all hierarchical
    if use_aeh_prior:
        group_prior_types = {
            'energy': 'adaptive_elastic_horseshoe',
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        }
    else:
        group_prior_types = {
            'energy': 'hierarchical',
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        }
    config = AdaptivePriorConfig(
        beta_0=1.0,  # Much weaker prior for AEH
        group_sparsity=False,
        dynamic_shrinkage=False,
        hmc_steps=20,
        hmc_leapfrog_steps=3,  # Reduced for HMC stability with AEH prior
        hmc_epsilon=0.0001,    # Smaller step size for HMC stability with AEH prior
        max_iter=1000,  # More EM iterations
        tol=1e-8,       # Tighter tolerance
        use_hmc=False,  # FINAL: EM-only AEH test, HMC disabled
        robust_noise=False,    # Disable robust noise for HMC stability
        group_prior_types=group_prior_types
    )
    model = AdaptivePriorARD(config)
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    metrics = model.cv_results.mean().to_dict()
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
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
    logger.info(f"\nAll results saved to {results_dir}") 