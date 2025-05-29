"""
EnhancedBayesianARD.py: Advanced Bayesian ARD Model with Enhanced Features
-------------------------------------------------------------------------------
- Implements ARD for feature selection using standard Bayesian techniques
- Includes cross-validation and hyperparameter tuning
- Enhanced feature engineering and analysis capabilities
- Model persistence and loading functionality
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@dataclass
class ModelConfig:
    """Configuration for the Enhanced Bayesian ARD model"""
    alpha_0: float = 1e-6
    beta_0: float = 1e-6
    max_iter: int = 200
    tol: float = 1e-4
    n_splits: int = 5
    random_state: int = 42

class EnhancedBayesianARD:
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Enhanced Bayesian ARD model
        
        Parameters:
        -----------
        config : Optional[ModelConfig]
            Model configuration parameters
        """
        self.config = config or ModelConfig()
        
        # Model parameters
        self.alpha = None  # Noise precision
        self.beta = None   # Weight precisions (ARD parameters)
        self.m = None      # Mean of weights
        self.S = None      # Covariance of weights
        
        # Feature engineering components
        self.scaler_X = RobustScaler()
        self.scaler_y = StandardScaler()
        
        # Cross-validation results
        self.cv_results = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnhancedBayesianARD':
        """
        Fit the Enhanced Bayesian ARD model with cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.alpha = self.config.alpha_0
        self.beta = np.ones(n_features) * self.config.beta_0
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        
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
            
            # EM algorithm
            for _ in range(self.config.max_iter):
                # E-step: Update posterior
                self.S = np.linalg.inv(self.alpha * X_train_scaled.T @ X_train_scaled + 
                                     np.diag(self.beta))
                self.m = self.alpha * self.S @ X_train_scaled.T @ y_train_scaled
                
                # M-step: Update hyperparameters
                alpha_new = n_samples / (np.sum((y_train_scaled - X_train_scaled @ self.m)**2) + 
                                       np.trace(X_train_scaled @ self.S @ X_train_scaled.T))
                beta_new = 1 / (self.m**2 + np.diag(self.S))
                
                # Check convergence
                if (np.abs(alpha_new - self.alpha) < self.config.tol and 
                    np.all(np.abs(beta_new - self.beta) < self.config.tol)):
                    break
                    
                self.alpha = alpha_new
                self.beta = beta_new
            
            # Evaluate on validation set
            y_pred, y_std = self.predict(X_val_scaled, return_std=True)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_val_orig = self.scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
            
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
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        return_std : bool
            Whether to return standard deviation of predictions
            
        Returns:
        --------
        mean : np.ndarray
            Mean predictions
        std : np.ndarray, optional
            Standard deviation of predictions
        """
        mean = X @ self.m
        if return_std:
            std = np.sqrt(1/self.alpha + np.sum((X @ self.S) * X, axis=1))
            return mean, std
        return mean
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on ARD parameters
        
        Returns:
        --------
        importance : np.ndarray
            Feature importance scores (1/beta)
        """
        return 1 / self.beta
    
    def save_model(self, path: str):
        """Save model and scalers to disk"""
        model_data = {
            'alpha': self.alpha,
            'beta': self.beta,
            'm': self.m,
            'S': self.S,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': self.config
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'EnhancedBayesianARD':
        """Load model from disk"""
        model_data = joblib.load(path)
        model = cls(config=model_data['config'])
        model.alpha = model_data['alpha']
        model.beta = model_data['beta']
        model.m = model_data['m']
        model.S = model_data['S']
        model.scaler_X = model_data['scaler_X']
        model.scaler_y = model_data['scaler_y']
        return model

def enhanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform enhanced feature engineering on the dataset with improved scaling
    """
    # 1. Enhanced floor area features with robust scaling
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)  # Log transform squared term
    
    # 2. Enhanced energy intensity features
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    
    # 3. Enhanced building age features
    df['building_age'] = 2024 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)  # Log transform squared term
    
    # 4. Enhanced energy star rating features
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    
    # 5. Enhanced GHG emissions features
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    
    # 6. Interaction features with proper scaling
    df['age_energy_star_interaction'] = df['building_age_log'] * df['energy_star_rating_normalized']
    df['area_energy_star_interaction'] = df['floor_area_log'] * df['energy_star_rating_normalized']
    df['age_ghg_interaction'] = df['building_age_log'] * df['ghg_emissions_int_log']
    
    return df

def analyze_feature_interactions(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                               output_dir: str):
    """
    Perform detailed analysis of feature interactions
    """
    # Create a new figure for interaction analysis
    plt.figure(figsize=(20, 15))
    
    # 1. Pairwise Feature Interactions
    plt.subplot(2, 2, 1)
    top_indices = np.argsort([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])[-5:]
    top_features = [feature_names[i] for i in top_indices]
    
    interaction_matrix = np.zeros((len(top_features), len(top_features)))
    for i, feat1 in enumerate(top_features):
        for j, feat2 in enumerate(top_features):
            if i != j:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)
                interaction_matrix[i, j] = np.corrcoef(X[:, idx1], X[:, idx2])[0, 1]
    
    sns.heatmap(interaction_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                xticklabels=top_features, yticklabels=top_features)
    plt.title('Top Feature Interactions')
    
    # 2. Feature Importance vs Correlation
    plt.subplot(2, 2, 2)
    target_correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
    feature_variances = np.var(X, axis=0)
    
    plt.scatter(target_correlations, feature_variances, alpha=0.5)
    for i, feat in enumerate(feature_names):
        plt.annotate(feat, (target_correlations[i], feature_variances[i]))
    plt.xlabel('Correlation with Target')
    plt.ylabel('Feature Variance')
    plt.title('Feature Variance vs Target Correlation')
    
    # 3. Feature Distribution Analysis
    plt.subplot(2, 2, 3)
    feature_skewness = [stats.skew(X[:, i]) for i in range(X.shape[1])]
    feature_kurtosis = [stats.kurtosis(X[:, i]) for i in range(X.shape[1])]
    
    plt.scatter(feature_skewness, feature_kurtosis, alpha=0.5)
    for i, feat in enumerate(feature_names):
        plt.annotate(feat, (feature_skewness[i], feature_kurtosis[i]))
    plt.xlabel('Skewness')
    plt.ylabel('Kurtosis')
    plt.title('Feature Distribution Analysis')
    
    # 4. Feature Importance vs Information Value
    plt.subplot(2, 2, 4)
    mutual_info = mutual_info_regression(X, y)
    plt.scatter(mutual_info, feature_variances, alpha=0.5)
    for i, feat in enumerate(feature_names):
        plt.annotate(feat, (mutual_info[i], feature_variances[i]))
    plt.xlabel('Mutual Information with Target')
    plt.ylabel('Feature Variance')
    plt.title('Feature Information Value vs Variance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_feature_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed feature analysis to JSON
    feature_analysis = {
        'correlations': dict(zip(feature_names, target_correlations)),
        'variances': dict(zip(feature_names, feature_variances)),
        'skewness': dict(zip(feature_names, feature_skewness)),
        'kurtosis': dict(zip(feature_names, feature_kurtosis)),
        'mutual_info': dict(zip(feature_names, mutual_info))
    }
    
    with open(os.path.join(output_dir, 'feature_analysis.json'), 'w') as f:
        json.dump(feature_analysis, f, indent=4, cls=NumpyEncoder)

def train_and_evaluate_enhanced(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              output_dir: Optional[str] = None) -> Tuple[EnhancedBayesianARD, dict]:
    """
    Train and evaluate the Enhanced Bayesian ARD model with detailed analysis
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform detailed feature analysis
        analyze_feature_interactions(X, y, feature_names, output_dir)
    
    # Initialize and train model
    config = ModelConfig()
    model = EnhancedBayesianARD(config)
    model.fit(X, y)
    
    # Get cross-validation metrics
    metrics = model.cv_results.mean().to_dict()
    
    if output_dir is not None:
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create detailed visualizations
        plt.style.use('default')
        
        # 1. Main Analysis Figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1.1 Feature Importance Plot
        plt.subplot(2, 2, 1)
        importance = model.get_feature_importance()
        sorted_idx = np.argsort(importance)
        plt.barh(range(len(feature_names)), importance[sorted_idx])
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Feature Importance (1/β)')
        plt.title('Enhanced Feature Importance Analysis')
        
        # 1.2 Cross-validation Results
        plt.subplot(2, 2, 2)
        cv_data = model.cv_results.drop('fold', axis=1)
        sns.boxplot(data=cv_data)
        plt.xticks(rotation=45)
        plt.title('Cross-validation Metrics Distribution')
        
        # 1.3 Correlation Heatmap
        plt.subplot(2, 2, 3)
        correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        
        # 1.4 Learning Curves
        plt.subplot(2, 2, 4)
        plt.plot(model.cv_results['rmse'], 'b-', label='RMSE', marker='o')
        plt.plot(model.cv_results['r2'], 'r-', label='R²', marker='s')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Learning Curves Across Folds')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enhanced_ard_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Performance Analysis
        fig = plt.figure(figsize=(20, 15))
        
        # 2.1 Prediction vs Actual Plot
        plt.subplot(2, 2, 1)
        y_pred, y_std = model.predict(X, return_std=True)
        y_pred_orig = model.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_orig = model.scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        
        plt.scatter(y_orig, y_pred_orig, alpha=0.5)
        plt.plot([y_orig.min(), y_orig.max()], [y_orig.min(), y_orig.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs Actual Values')
        
        # 2.2 Residuals Analysis
        plt.subplot(2, 2, 2)
        residuals = y_orig - y_pred_orig
        plt.scatter(y_pred_orig, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        
        # 2.3 Uncertainty Analysis
        plt.subplot(2, 2, 3)
        plt.scatter(np.abs(residuals), y_std, alpha=0.5)
        plt.xlabel('Absolute Prediction Error')
        plt.ylabel('Prediction Uncertainty')
        plt.title('Uncertainty vs Prediction Error')
        
        # 2.4 Feature Importance Distribution
        plt.subplot(2, 2, 4)
        sns.histplot(importance, bins=20)
        plt.xlabel('Feature Importance')
        plt.ylabel('Count')
        plt.title('Distribution of Feature Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Interaction Analysis
        fig = plt.figure(figsize=(20, 15))
        
        # 3.1 Top Feature Interactions
        top_indices = np.argsort(importance)[-5:]
        top_features = [feature_names[i] for i in top_indices]
        interaction_matrix = np.zeros((len(top_features), len(top_features)))
        
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features):
                if i != j:
                    idx1 = feature_names.index(feat1)
                    idx2 = feature_names.index(feat2)
                    interaction_matrix[i, j] = np.corrcoef(X[:, idx1], X[:, idx2])[0, 1]
        
        plt.subplot(2, 2, 1)
        sns.heatmap(interaction_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                   xticklabels=top_features, yticklabels=top_features)
        plt.title('Top Feature Interactions')
        
        # 3.2 Feature Importance vs Correlation with Target
        plt.subplot(2, 2, 2)
        target_correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        plt.scatter(target_correlations, importance)
        for i, feat in enumerate(feature_names):
            plt.annotate(feat, (target_correlations[i], importance[i]))
        plt.xlabel('Correlation with Target')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance vs Target Correlation')
        
        # 3.3 Feature Importance vs Variance
        plt.subplot(2, 2, 3)
        feature_variances = np.var(X, axis=0)
        plt.scatter(feature_variances, importance)
        for i, feat in enumerate(feature_names):
            plt.annotate(feat, (feature_variances[i], importance[i]))
        plt.xlabel('Feature Variance')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance vs Variance')
        
        # 3.4 Cumulative Feature Importance
        plt.subplot(2, 2, 4)
        sorted_importance = np.sort(importance)[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        plt.plot(range(1, len(importance) + 1), cumulative_importance, 'b-', marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_interaction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model
        model.save_model(os.path.join(output_dir, 'enhanced_ard_model.joblib'))
        
        # Print detailed analysis
        logger.info("\nDetailed Analysis:")
        logger.info("\n1. Feature Importance:")
        for feat, imp in zip(feature_names, importance):
            logger.info(f"{feat}: {imp:.4f}")
        
        logger.info("\n2. Cross-validation Results:")
        logger.info(f"Mean RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Mean R²: {metrics['r2']:.4f}")
        logger.info(f"Mean MAE: {metrics['mae']:.4f}")
        logger.info(f"Mean Uncertainty: {metrics['mean_std']:.4f}")
        
        logger.info("\n3. Feature Correlations with Target:")
        for feat, corr in zip(feature_names, target_correlations):
            logger.info(f"{feat}: {corr:.4f}")
        
        logger.info("\n4. Feature Variances:")
        for feat, var in zip(feature_names, feature_variances):
            logger.info(f"{feat}: {var:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    logger.info("[INFO] Starting Enhanced Bayesian ARD analysis...")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    
    # Enhanced feature engineering
    logger.info("[INFO] Performing enhanced feature engineering...")
    df = enhanced_feature_engineering(df)
    
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
    y = df[target].values.astype(np.float32)
    
    # Train and evaluate model
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    model, metrics = train_and_evaluate_enhanced(X, y, feature_names, output_dir=results_dir)
    
    logger.info("[INFO] Analysis complete. Results saved to %s", results_dir) 