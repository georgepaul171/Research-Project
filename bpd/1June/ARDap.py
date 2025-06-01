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

# Set up logging configuration
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
    Custom JSON encoder for numpy types to handle serialization of numpy arrays and scalars
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
    Configuration class for the Adaptive Prior ARD model parameters
    
    Attributes:
        alpha_0: Initial noise precision
        beta_0: Initial weight precision
        max_iter: Maximum number of EM iterations
        tol: Convergence tolerance
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility
        prior_type: Type of prior ('hierarchical', 'spike_slab', or 'horseshoe')
        adaptation_rate: Rate at which priors adapt
        uncertainty_threshold: Threshold for uncertainty-based adaptation
        group_sparsity: Whether to enable group-wise sparsity
        dynamic_shrinkage: Whether to enable dynamic shrinkage parameters
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

class AdaptivePriorARD:
    """
    Adaptive Prior ARD model with hierarchical priors
    
    This implements a Bayesian linear regression model with:
    - Automatic relevance determination for feature selection
    - Adaptive prior specifications
    - Uncertainty quantification
    - Cross-validation for robust evaluation
    """
    def __init__(self, config: Optional[AdaptivePriorConfig] = None):
        """
        Initialise model with configuration parameters
        
        Args:
            config: Optional configuration object. If None, uses default settings.
        """
        self.config = config or AdaptivePriorConfig()
        
        # Model parameters
        self.alpha = None  # Noise precision
        self.beta = None  # Weight precisions (ARD parameters)
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

    def _initialize_adaptive_priors(self, n_features: int):
        """
        Initialise adaptive prior parameters based on configuration
        
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
            # InitialiSe feature groups for group sparsity
            self.feature_groups = self._create_feature_groups(n_features)
            
        if self.config.dynamic_shrinkage:
            # InitialiSe dynamic shrinkage parameters
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
        # Group features by type (I'll consifer changing based on feature structure)
        groups = {
            'energy': list(range(0, 4)),  # Energy-related features
            'building': list(range(4, 8)),  # Building characteristics
            'interaction': list(range(8, n_features))  # Interaction terms
        }
        return groups
    
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
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptivePriorARD':
        """
        Fit the model with adaptive priors and cross-validation
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self: The fitted model
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
        
        # Cross-validation
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, 
                  random_state=self.config.random_state)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale data
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
                    # Add to diagonal for stability
                    jitter = 1e-6 * np.eye(n_features)
                    self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                         np.diag(np.clip(self.beta, 1e-10, None)) + jitter)
                
                self.m = self.alpha * self.S @ X_train_scaled.T @ y_train_scaled
                
                # M-step: Update hyperparameters
                residuals = y_train_scaled - X_train_scaled @ self.m
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
            
            # Evaluate on validation set
            y_pred, y_std = self.predict(X_val_scaled, return_std=True)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_val_orig = self.scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
            
            y_pred_orig = y_pred_orig.reshape(-1)
            y_val_orig = y_val_orig.reshape(-1)
            
            metrics = {
                'fold': fold,
                'rmse': np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)),
                'mae': mean_absolute_error(y_val_orig, y_pred_orig),
                'r2': r2_score(y_val_orig, y_pred_orig),
                'mean_std': np.mean(y_std)
            }
            cv_metrics.append(metrics)
            
        self.cv_results = pd.DataFrame(cv_metrics)
        logger.info(f"Cross-validation results:\n{self.cv_results.mean()}")
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty estimates
        
        Args:
            X: Input features
            return_std: Whether to return standard deviation of predictions
            
        Returns:
            mean: Mean predictions
            std: Standard deviation of predictions (if return_std=True)
        """
        mean = X @ self.m
        if return_std:
            std = np.sqrt(1/self.alpha + np.sum((X @ self.S) * X, axis=1))
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
            path: Path to save the model - results/models/ARDap.joblib
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
            'config': self.config
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'AdaptivePriorARD':
        """
        Load model
        
        Args:
            path: Path to the saved model - results/models/ARDap.joblib
            
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
        return model

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        df: DataFrame with engineered features
    """
    # Floor area features with scaling
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)
    
    # Energy intensity features
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    
    # Building age features
    df['building_age'] = 2025 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)
    
    # Energy star rating features
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    
    # GHG emissions features
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    
    # Interaction features
    df['age_energy_star_interaction'] = df['building_age_log'] * df['energy_star_rating_normalized']
    df['area_energy_star_interaction'] = df['floor_area_log'] * df['energy_star_rating_normalized']
    df['age_ghg_interaction'] = df['building_age_log'] * df['ghg_emissions_int_log']
    
    return df

def analyze_feature_interactions(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                               model: AdaptivePriorARD, output_dir: str):
    """
    Perform detailed analysis of feature interactions and model results
    
    Args:
        X: Input features
        y: Target values
        feature_names: List of feature names
        model: Fitted model
        output_dir: Directory to save analysis results
    """
    # Ensure y is 1D array
    y = np.asarray(y).reshape(-1)
    
    # Create a new figure for analysis
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 25))
    
    # Feature Importance with Confidence Intervals
    plt.subplot(4, 2, 1)
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)
    plt.barh(range(len(feature_names)), importance[sorted_idx])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Normalized Feature Importance')
    plt.title('Feature Importance Analysis')
    
    # Add confidence intervals
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
                # First feature as input (2D array)
                feat1_data = X[:, i].reshape(-1, 1)  # Shape: (n_samples, 1)
                # Second feature as target (1D array)
                feat2_data = X[:, j].ravel()  # Shape: (n_samples,)
                interaction_strength[i, j] = mutual_info_regression(
                    feat1_data, feat2_data
                )[0]
    
    # Plot interaction network
    G = nx.Graph()
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i < j and interaction_strength[i, j] > 0.1:  # Threshold
                G.add_edge(feat1, feat2, weight=interaction_strength[i, j])
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=8, font_weight='bold')
    plt.title('Feature Interaction Network')
    
    # Partial Dependence Analysis
    plt.subplot(4, 2, 4)
    top_features = [feature_names[i] for i in sorted_idx[-3:]]  # Top 3 features, but extend this out if needed
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
    
    # Save analysis to JSON
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
            'mean_std': float(cv_scores['mean_std'].mean())
        },
        'prior_hyperparameters': {
            'global_shrinkage': float(model.prior_hyperparams['lambda'].mean()),
            'local_shrinkage': float(model.prior_hyperparams['tau'].mean())
        }
    }
    
    with open(os.path.join(output_dir, 'detailed_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4, cls=NumpyEncoder)
    
    # Print analysis
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
    
    logger.info("\n4. Prior Hyperparameters:")
    logger.info(f"Global Shrinkage: {analysis_results['prior_hyperparameters']['global_shrinkage']:.4f}")
    logger.info(f"Local Shrinkage: {analysis_results['prior_hyperparameters']['local_shrinkage']:.4f}")
    
    logger.info("\n5. Feature Correlations with Target:")
    for feat, corr in sorted(zip(feature_names, target_correlations), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]:
        logger.info(f"{feat}: {corr:.4f}")

def train_and_evaluate_adaptive(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              output_dir: Optional[str] = None) -> Tuple[AdaptivePriorARD, dict]:
    """
    Train and evaluate the model
    
    Args:
        X: Input features
        y: Target values
        feature_names: List of feature names
        output_dir: Optional directory to save results
        
    Returns:
        model: Fitted model
        metrics: Model performance metrics
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
        
        # Perform analysis
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
    
    # Select features
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
    
    logger.info("Complete. Results saved to %s", results_dir) 