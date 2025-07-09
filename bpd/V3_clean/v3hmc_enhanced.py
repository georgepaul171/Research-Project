import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import joblib
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AdaptivePriorConfig:
    group_prior_types: dict
    max_iter: int = 50
    use_hmc: bool = True
    hmc_epsilon: float = 0.001  # Initial step size
    hmc_steps: int = 100  # Increased from 50
    hmc_leapfrog_steps: int = 8  # Increased from 5
    n_chains: int = 8  # Increased from 4
    warmup_steps: int = 1000  # New: warmup phase
    target_acceptance: float = 0.65  # Target acceptance rate
    hmc_adaptation_window: int = 100  # Steps for adaptive tuning

class AdaptivePriorARD:
    def __init__(self, config: AdaptivePriorConfig):
        self.config = config
        self.group_prior_hyperparams = {}
        self.feature_importance = {}
        self.convergence_history = []
        
    def _initialize_adaptive_priors(self, X, y, feature_names):
        """Initialize adaptive prior hyperparameters for each group."""
        n_features = X.shape[1]
        
        # Define feature groups
        energy_features = [i for i, name in enumerate(feature_names) 
                          if any(keyword in name.lower() for keyword in 
                                ['eui', 'energy', 'ghg', 'fuel', 'electric'])]
        building_features = [i for i, name in enumerate(feature_names) 
                           if any(keyword in name.lower() for keyword in 
                                 ['area', 'age', 'floor', 'building'])]
        interaction_features = [i for i, name in enumerate(feature_names) 
                              if any(keyword in name.lower() for keyword in 
                                    ['squared', 'ratio', 'mix'])]
        
        self.feature_groups = {
            'energy': energy_features,
            'building': building_features,
            'interaction': interaction_features
        }
        
        # Initialize hyperparameters for each group
        for group_name, group_prior_type in self.config.group_prior_types.items():
            if group_prior_type == 'adaptive_elastic_horseshoe':
                self.group_prior_hyperparams[group_name] = {
                    'alpha': 0.5,  # L1 regularization
                    'beta': 0.5,   # L2 regularization
                    'gamma': 0.1,  # Adaptation rate
                    'rho': 0.1,    # Momentum
                    'tau': 1.0,    # Global shrinkage
                    'lambda_local': np.ones(len(self.feature_groups[group_name])) * 0.1
                }
            elif group_prior_type == 'horseshoe':
                self.group_prior_hyperparams[group_name] = {
                    'tau': 1.0,
                    'lambda_local': np.ones(len(self.feature_groups[group_name])) * 0.1
                }
            elif group_prior_type == 'hierarchical':
                self.group_prior_hyperparams[group_name] = {
                    'mu': 0.0,
                    'sigma': 1.0
                }
    
    def _calculate_gradient(self, X, y, w):
        """Calculate gradient of negative log posterior."""
        n_samples = X.shape[0]
        
        # Likelihood gradient
        residual = y - X @ w
        likelihood_grad = -2 * X.T @ residual / n_samples
        
        # Prior gradient
        prior_grad = np.zeros_like(w)
        
        for group_name, group_indices in self.feature_groups.items():
            group_prior_type = self.config.group_prior_types[group_name]
            group_w = w[group_indices]
            
            if group_prior_type == 'adaptive_elastic_horseshoe':
                params = self.group_prior_hyperparams[group_name]
                alpha, beta, tau, lambda_local = params['alpha'], params['beta'], params['tau'], params['lambda_local']
                
                # Elastic net + horseshoe gradient
                elastic_grad = alpha * np.sign(group_w) + 2 * beta * group_w
                horseshoe_grad = group_w / (tau**2 * lambda_local**2)
                prior_grad[group_indices] = elastic_grad + horseshoe_grad
                
            elif group_prior_type == 'horseshoe':
                params = self.group_prior_hyperparams[group_name]
                tau, lambda_local = params['tau'], params['lambda_local']
                prior_grad[group_indices] = group_w / (tau**2 * lambda_local**2)
                
            elif group_prior_type == 'hierarchical':
                params = self.group_prior_hyperparams[group_name]
                mu, sigma = params['mu'], params['sigma']
                prior_grad[group_indices] = (group_w - mu) / sigma**2
        
        return likelihood_grad + prior_grad
    
    def _calculate_hamiltonian(self, X, y, w, momentum):
        """Calculate Hamiltonian (energy) for HMC."""
        n_samples = X.shape[0]
        
        # Kinetic energy
        kinetic_energy = 0.5 * np.sum(momentum**2)
        
        # Potential energy (negative log posterior)
        residual = y - X @ w
        likelihood_energy = np.sum(residual**2) / n_samples
        
        prior_energy = 0.0
        for group_name, group_indices in self.feature_groups.items():
            group_prior_type = self.config.group_prior_types[group_name]
            group_w = w[group_indices]
            
            if group_prior_type == 'adaptive_elastic_horseshoe':
                params = self.group_prior_hyperparams[group_name]
                alpha, beta, tau, lambda_local = params['alpha'], params['beta'], params['tau'], params['lambda_local']
                elastic_energy = alpha * np.sum(np.abs(group_w)) + beta * np.sum(group_w**2)
                horseshoe_energy = 0.5 * np.sum((group_w / (tau * lambda_local))**2)
                prior_energy += elastic_energy + horseshoe_energy
                
            elif group_prior_type == 'horseshoe':
                params = self.group_prior_hyperparams[group_name]
                tau, lambda_local = params['tau'], params['lambda_local']
                prior_energy += 0.5 * np.sum((group_w / (tau * lambda_local))**2)
                
            elif group_prior_type == 'hierarchical':
                params = self.group_prior_hyperparams[group_name]
                mu, sigma = params['mu'], params['sigma']
                prior_energy += 0.5 * np.sum(((group_w - mu) / sigma)**2)
        
        return kinetic_energy + likelihood_energy + prior_energy
    
    def _hmc_step(self, X, y, current_w, current_momentum, epsilon):
        """Single HMC step with leapfrog integration."""
        w = current_w.copy()
        momentum = current_momentum.copy()
        
        # Half step for momentum
        gradient = self._calculate_gradient(X, y, w)
        momentum = momentum - 0.5 * epsilon * gradient
        
        # Full steps for position and momentum
        for _ in range(self.config.hmc_leapfrog_steps - 1):
            w = w + epsilon * momentum
            gradient = self._calculate_gradient(X, y, w)
            momentum = momentum - epsilon * gradient
        
        # Final half step for momentum
        w = w + epsilon * momentum
        gradient = self._calculate_gradient(X, y, w)
        momentum = momentum - 0.5 * epsilon * gradient
        
        # Calculate acceptance probability
        current_hamiltonian = self._calculate_hamiltonian(X, y, current_w, current_momentum)
        proposed_hamiltonian = self._calculate_hamiltonian(X, y, w, momentum)
        
        acceptance_prob = min(1.0, np.exp(current_hamiltonian - proposed_hamiltonian))
        
        return w, momentum, acceptance_prob
    
    def _adaptive_hmc_warmup(self, X, y, initial_w):
        """Adaptive warmup phase to tune step size."""
        print("Starting adaptive warmup phase...")
        w = initial_w.copy()
        epsilon = self.config.hmc_epsilon
        acceptance_history = []
        
        for step in range(self.config.warmup_steps):
            momentum = np.random.randn(len(w))
            
            # HMC step
            w_new, _, acceptance_prob = self._hmc_step(X, y, w, momentum, epsilon)
            
            if np.random.random() < acceptance_prob:
                w = w_new
                acceptance_history.append(1)
            else:
                acceptance_history.append(0)
            
            # Adaptive step size tuning
            if step > 0 and step % self.config.hmc_adaptation_window == 0:
                recent_acceptance = np.mean(acceptance_history[-self.config.hmc_adaptation_window:])
                
                if recent_acceptance > self.config.target_acceptance + 0.1:
                    epsilon *= 1.1  # Increase step size
                elif recent_acceptance < self.config.target_acceptance - 0.1:
                    epsilon *= 0.9  # Decrease step size
                
                print(f"Warmup step {step}: acceptance rate = {recent_acceptance:.3f}, epsilon = {epsilon:.6f}")
        
        final_acceptance = np.mean(acceptance_history[-100:])
        print(f"Warmup complete: final acceptance rate = {final_acceptance:.3f}, final epsilon = {epsilon:.6f}")
        
        return w, epsilon
    
    def _hmc_sampling(self, X, y, initial_w, n_chains=None, return_chains=True):
        """Enhanced HMC sampling with warmup and better initialization."""
        if n_chains is None:
            n_chains = self.config.n_chains
            
        print(f"Starting HMC sampling with {n_chains} chains...")
        
        # Warmup phase for each chain
        warmup_results = []
        for chain in range(n_chains):
            print(f"Warming up chain {chain + 1}/{n_chains}...")
            # Different starting points for each chain
            chain_initial_w = initial_w + 0.2 * np.random.randn(len(initial_w))
            warmup_w, tuned_epsilon = self._adaptive_hmc_warmup(X, y, chain_initial_w)
            warmup_results.append((warmup_w, tuned_epsilon))
        
        # Main sampling phase
        chains = []
        acceptance_probs = []
        
        for chain in range(n_chains):
            print(f"Sampling chain {chain + 1}/{n_chains}...")
            w, epsilon = warmup_results[chain]
            chain_samples = []
            chain_acceptances = []
            
            for step in range(self.config.hmc_steps):
                momentum = np.random.randn(len(w))
                
                # HMC step
                w_new, _, acceptance_prob = self._hmc_step(X, y, w, momentum, epsilon)
                
                if np.random.random() < acceptance_prob:
                    w = w_new
                    chain_acceptances.append(1)
                else:
                    chain_acceptances.append(0)
                
                chain_samples.append(w.copy())
            
            chains.append(np.array(chain_samples))
            acceptance_probs.append(np.mean(chain_acceptances))
        
        # Calculate diagnostics
        chains_array = np.array(chains)
        mean_w = np.mean(chains_array, axis=(0, 1))
        
        # R-hat and ESS calculations
        r_hat_stats = self._calculate_gelman_rubin(chains_array)
        ess_stats = self._calculate_effective_sample_size(chains_array)
        
        print(f"HMC sampling complete:")
        print(f"  Acceptance rates: {[f'{acc:.3f}' for acc in acceptance_probs]}")
        print(f"  Mean acceptance: {np.mean(acceptance_probs):.3f}")
        print(f"  R-hat range: {np.min(r_hat_stats):.2f} - {np.max(r_hat_stats):.2f}")
        print(f"  ESS range: {np.min(ess_stats):.1f} - {np.max(ess_stats):.1f}")
        
        if return_chains:
            return mean_w, acceptance_probs, r_hat_stats, ess_stats, chains
        else:
            return mean_w, acceptance_probs, r_hat_stats, ess_stats
    
    def _calculate_gelman_rubin(self, chains):
        """Calculate Gelman-Rubin R-hat statistic."""
        n_chains, n_samples, n_params = chains.shape
        
        # Within-chain variance
        within_var = np.var(chains, axis=1, ddof=1)
        within_var_mean = np.mean(within_var, axis=0)
        
        # Between-chain variance
        chain_means = np.mean(chains, axis=1)
        between_var = n_samples * np.var(chain_means, axis=0, ddof=1)
        
        # R-hat calculation
        r_hat = np.sqrt((between_var / within_var_mean + 1) / n_chains)
        
        return r_hat
    
    def _calculate_effective_sample_size(self, chains):
        """Calculate effective sample size."""
        n_chains, n_samples, n_params = chains.shape
        ess = np.zeros(n_params)
        
        for param in range(n_params):
            # Combine all chains for this parameter
            combined_samples = chains[:, :, param].flatten()
            
            # Calculate autocorrelation
            autocorr = np.correlate(combined_samples, combined_samples, mode='full')
            autocorr = autocorr[len(combined_samples)-1:] / len(combined_samples)
            
            # ESS calculation (using first 100 lags)
            lag = min(100, len(autocorr) - 1)
            ess[param] = len(combined_samples) / (1 + 2 * np.sum(autocorr[1:lag+1]))
        
        return ess
    
    def _update_adaptive_priors(self, w):
        """Update adaptive prior hyperparameters based on current weights."""
        for group_name, group_indices in self.feature_groups.items():
            group_prior_type = self.config.group_prior_types[group_name]
            group_w = w[group_indices]
            
            if group_prior_type == 'adaptive_elastic_horseshoe':
                params = self.group_prior_hyperparams[group_name]
                
                # Update local shrinkage parameters
                params['lambda_local'] = 1.0 / (np.abs(group_w) + 1e-6)
                
                # Update global shrinkage
                params['tau'] = np.median(np.abs(group_w)) / 0.6745  # MAD estimator
                
                # Adaptive regularization
                params['alpha'] = np.clip(params['alpha'] + params['gamma'] * (0.5 - np.mean(np.abs(group_w))), 0.1, 0.9)
                params['beta'] = np.clip(params['beta'] + params['gamma'] * (0.5 - np.mean(group_w**2)), 0.1, 0.9)
    
    def fit(self, X, y, feature_names=None, output_dir=None):
        """Fit the Adaptive Prior ARD model with enhanced HMC."""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        self._initialize_adaptive_priors(X, y, feature_names)
        
        # Initialize weights
        w = np.zeros(X.shape[1])
        
        print(f"Starting Adaptive Prior ARD with enhanced HMC...")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Max iterations: {self.config.max_iter}")
        print(f"  HMC enabled: {self.config.use_hmc}")
        
        if self.config.use_hmc:
            print(f"  HMC chains: {self.config.n_chains}")
            print(f"  HMC steps: {self.config.hmc_steps}")
            print(f"  Warmup steps: {self.config.warmup_steps}")
        
        # EM iterations
        for iteration in range(self.config.max_iter):
            print(f"\nIteration {iteration + 1}/{self.config.max_iter}")
            
            if self.config.use_hmc:
                # HMC sampling for posterior exploration
                w, acceptance_probs, r_hat_stats, ess_stats, chains = self._hmc_sampling(
                    X, y, w, return_chains=True
                )
                
                # Save diagnostics
                if output_dir:
                    self._save_hmc_diagnostics(
                        iteration, acceptance_probs, r_hat_stats, ess_stats, 
                        chains, output_dir
                    )
            else:
                # Simple gradient descent (fallback)
                gradient = self._calculate_gradient(X, y, w)
                w = w - 0.01 * gradient
            
            # Update adaptive priors
            self._update_adaptive_priors(w)
            
            # Calculate convergence metrics
            y_pred = X @ w
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.convergence_history.append({
                'iteration': iteration + 1,
                'mse': mse,
                'r2': r2,
                'w_norm': np.linalg.norm(w)
            })
            
            print(f"  MSE: {mse:.4f}, R²: {r2:.4f}, ||w||: {np.linalg.norm(w):.4f}")
            
            # Early stopping if converged
            if iteration > 5:
                recent_r2 = [h['r2'] for h in self.convergence_history[-5:]]
                if max(recent_r2) - min(recent_r2) < 0.001:
                    print("Convergence detected, stopping early.")
                    break
        
        # Final feature importance
        self.feature_importance = dict(zip(feature_names, np.abs(w)))
        
        # Save final results
        if output_dir:
            self._save_results(output_dir)
        
        print(f"\nTraining complete!")
        print(f"Final R²: {r2:.4f}")
        print(f"Final MSE: {mse:.4f}")
        
        return self
    
    def _save_hmc_diagnostics(self, iteration, acceptance_probs, r_hat_stats, ess_stats, chains, output_dir):
        """Save comprehensive HMC diagnostics."""
        # Save acceptance rates
        acceptance_data = {
            'iteration': iteration + 1,
            'acceptance_rates': acceptance_probs,
            'mean_acceptance': float(np.mean(acceptance_probs)),
            'std_acceptance': float(np.std(acceptance_probs))
        }
        
        with open(os.path.join(output_dir, f'hmc_acceptance_iter_{iteration+1:02d}.json'), 'w') as f:
            json.dump(acceptance_data, f, indent=2)
        
        # Save R-hat and ESS
        diagnostics_data = {
            'iteration': iteration + 1,
            'r_hat_stats': r_hat_stats.tolist(),
            'ess_stats': ess_stats.tolist(),
            'mean_r_hat': float(np.mean(r_hat_stats)),
            'max_r_hat': float(np.max(r_hat_stats)),
            'mean_ess': float(np.mean(ess_stats)),
            'min_ess': float(np.min(ess_stats))
        }
        
        with open(os.path.join(output_dir, f'hmc_rhat_ess_iter_{iteration+1:02d}.json'), 'w') as f:
            json.dump(diagnostics_data, f, indent=2)
        
        # Generate trace plots for each feature
        n_features = chains[0].shape[1]
        n_chains = len(chains)
        
        for feature_idx in range(n_features):
            # Create subplot grid for 8 chains (2x4 or 4x2)
            if n_chains <= 4:
                rows, cols = 2, 2
            elif n_chains <= 6:
                rows, cols = 2, 3
            else:  # 8 chains
                rows, cols = 2, 4
            
            plt.figure(figsize=(16, 8))
            
            # Trace plots for each chain
            for chain_idx in range(n_chains):
                plt.subplot(rows, cols, chain_idx + 1)
                plt.plot(chains[chain_idx][:, feature_idx], alpha=0.7)
                plt.title(f'Chain {chain_idx + 1} - {self.feature_names[feature_idx]}')
                plt.xlabel('Step')
                plt.ylabel('Weight')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'trace_{feature_idx:02d}_{self.feature_names[feature_idx]}_iter_{iteration+1:02d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save convergence history
        with open(os.path.join(output_dir, 'convergence_history.json'), 'w') as f:
            json.dump(self.convergence_history, f, indent=2)
    
    def _save_results(self, output_dir):
        """Save final model results."""
        # Save metrics
        final_metrics = {
            'final_r2': self.convergence_history[-1]['r2'],
            'final_mse': self.convergence_history[-1]['mse'],
            'iterations': len(self.convergence_history),
            'convergence_history': self.convergence_history
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save feature importance
        with open(os.path.join(output_dir, 'feature_importance.json'), 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        # Save model
        joblib.dump(self, os.path.join(output_dir, 'adaptive_prior_model.joblib'))
        
        # Plot convergence
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        iterations = [h['iteration'] for h in self.convergence_history]
        r2_values = [h['r2'] for h in self.convergence_history]
        plt.plot(iterations, r2_values, 'b-', linewidth=2)
        plt.title('R² Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('R²')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        mse_values = [h['mse'] for h in self.convergence_history]
        plt.plot(iterations, mse_values, 'r-', linewidth=2)
        plt.title('MSE Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        w_norms = [h['w_norm'] for h in self.convergence_history]
        plt.plot(iterations, w_norms, 'g-', linewidth=2)
        plt.title('Weight Norm Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('||w||')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_plots.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def predict(self, X, return_std=False):
        """Make predictions with uncertainty quantification."""
        if not hasattr(self, 'feature_importance'):
            raise ValueError("Model must be fitted before making predictions")
        
        # Get the final weights from convergence history
        if hasattr(self, 'convergence_history') and self.convergence_history:
            # Use the last iteration's weights (approximation)
            # In a full implementation, you'd sample from the posterior
            w = np.array(list(self.feature_importance.values()))
        else:
            w = np.zeros(X.shape[1])
        
        y_pred = X @ w
        
        if return_std:
            # Simple uncertainty estimation based on residuals
            if hasattr(self, 'convergence_history') and self.convergence_history:
                final_mse = self.convergence_history[-1]['mse']
                y_std = np.sqrt(final_mse) * np.ones(len(y_pred))
            else:
                y_std = np.ones(len(y_pred))
            return y_pred, y_std
        
        return y_pred

def feature_engineering_no_interactions(df):
    """Feature engineering without interaction terms."""
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
    # Set up paths and parameters
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultshmc_enhanced')
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

    # Configure and run enhanced model
    config = AdaptivePriorConfig(
        group_prior_types={
            'energy': 'adaptive_elastic_horseshoe',
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        },
        max_iter=30,  # Reduced due to longer HMC runs
        use_hmc=True,
        hmc_epsilon=0.001,
        hmc_steps=100,
        hmc_leapfrog_steps=8,
        n_chains=8,
        warmup_steps=1000,
        target_acceptance=0.65,
        hmc_adaptation_window=100
    )

    model = AdaptivePriorARD(config)
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)

    # Final evaluation
    y_pred, y_std = model.predict(X, return_std=True)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\nFinal Results:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save final analysis
    analysis = {
        'model_type': 'Enhanced HMC Adaptive Prior ARD',
        'final_metrics': {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        },
        'hmc_config': {
            'n_chains': config.n_chains,
            'hmc_steps': config.hmc_steps,
            'warmup_steps': config.warmup_steps,
            'target_acceptance': config.target_acceptance
        },
        'convergence': model.convergence_history[-1] if model.convergence_history else None
    }
    
    with open(os.path.join(results_dir, 'enhanced_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nEnhanced HMC analysis complete! Results saved to {results_dir}") 