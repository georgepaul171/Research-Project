"""
BayesianARD.py: Standard Bayesian ARD (Automatic Relevance Determination) Model
-------------------------------------------------------------------------------
- Implements ARD for feature selection using standard Bayesian techniques
- Uses linear regression with ARD priors
- Provides uncertainty estimates and feature importance
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
from typing import Tuple, List, Optional

class BayesianARD:
    def __init__(self, alpha_0: float = 1e-6, beta_0: float = 1e-6, 
                 max_iter: int = 100, tol: float = 1e-3):
        """
        Initialize Bayesian ARD model
        
        Parameters:
        -----------
        alpha_0 : float
            Initial precision of the noise
        beta_0 : float
            Initial precision of the weights
        max_iter : int
            Maximum number of iterations for optimization
        tol : float
            Convergence tolerance
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.max_iter = max_iter
        self.tol = tol
        
        # Model parameters
        self.alpha = None  # Noise precision
        self.beta = None   # Weight precisions (ARD parameters)
        self.m = None      # Mean of weights
        self.S = None      # Covariance of weights
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianARD':
        """
        Fit the Bayesian ARD model
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.alpha = self.alpha_0
        self.beta = np.ones(n_features) * self.beta_0
        self.m = np.zeros(n_features)
        self.S = np.eye(n_features)
        
        # EM algorithm
        for _ in range(self.max_iter):
            # E-step: Update posterior
            self.S = np.linalg.inv(self.alpha * X.T @ X + np.diag(self.beta))
            self.m = self.alpha * self.S @ X.T @ y
            
            # M-step: Update hyperparameters
            alpha_new = n_samples / (np.sum((y - X @ self.m)**2) + 
                                   np.trace(X @ self.S @ X.T))
            beta_new = 1 / (self.m**2 + np.diag(self.S))
            
            # Check convergence
            if (np.abs(alpha_new - self.alpha) < self.tol and 
                np.all(np.abs(beta_new - self.beta) < self.tol)):
                break
                
            self.alpha = alpha_new
            self.beta = beta_new
            
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions
        
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

def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                      output_dir: Optional[str] = None) -> Tuple[BayesianARD, dict]:
    """
    Train and evaluate the Bayesian ARD model
    
    Parameters:
    -----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target values
    feature_names : List[str]
        Names of features
    output_dir : Optional[str]
        Directory to save results
        
    Returns:
    --------
    model : BayesianARD
        Trained model
    metrics : dict
        Evaluation metrics
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    # Train model
    model = BayesianARD(alpha_0=1e-6, beta_0=1e-6, max_iter=100, tol=1e-3)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred, y_std = model.predict(X_val, return_std=True)
    
    # Calculate metrics
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_std_orig = y_std * scaler_y.scale_
    
    metrics = {
        'rmse': float(np.sqrt(np.mean((y_val_orig - y_pred_orig) ** 2))),
        'mae': float(np.mean(np.abs(y_val_orig - y_pred_orig))),
        'r2': float(1 - np.sum((y_val_orig - y_pred_orig) ** 2) / 
                    np.sum((y_val_orig - np.mean(y_val_orig)) ** 2)),
        'mean_std': float(np.mean(y_std_orig))
    }
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. Predictions vs Actual Values with Uncertainty
        plt.subplot(2, 2, 1)
        plt.scatter(y_val_orig, y_pred_orig, alpha=0.5, label='Predictions')
        plt.errorbar(y_val_orig, y_pred_orig, yerr=y_std_orig, fmt='none', 
                    ecolor='gray', alpha=0.2, label='Uncertainty')
        
        min_val = min(y_val_orig.min(), y_pred_orig.min())
        max_val = max(y_val_orig.max(), y_pred_orig.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.legend()
        
        # 2. Feature Importance
        plt.subplot(2, 2, 2)
        importance = model.get_feature_importance()
        sorted_idx = np.argsort(importance)
        plt.barh(range(len(feature_names)), importance[sorted_idx])
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Feature Importance (1/β)')
        plt.title('Feature Importance from ARD')
        
        # 3. Uncertainty Analysis
        plt.subplot(2, 2, 3)
        abs_error = np.abs(y_val_orig - y_pred_orig)
        plt.scatter(abs_error, y_std_orig, alpha=0.5)
        plt.xlabel('Absolute Prediction Error')
        plt.ylabel('Prediction Uncertainty')
        plt.title('Uncertainty vs Prediction Error')
        
        z = np.polyfit(abs_error, y_std_orig, 1)
        p = np.poly1d(z)
        plt.plot(sorted(abs_error), p(sorted(abs_error)), "r--")
        
        # 4. Residuals Plot
        plt.subplot(2, 2, 4)
        residuals = y_val_orig - y_pred_orig
        plt.scatter(y_pred_orig, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ard_analysis.png'))
        plt.close()
        
        # Print analysis
        print("\nDetailed Analysis:")
        print("\n1. Feature Importance:")
        for feat, imp in zip(feature_names, importance):
            print(f"{feat}: {imp:.4f}")
        
        print("\n2. Uncertainty Analysis:")
        print(f"Mean Uncertainty: {metrics['mean_std']:.4f}")
        print(f"Correlation with Error: {np.corrcoef(abs_error, y_std_orig)[0,1]:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    print("[INFO] Starting Bayesian ARD analysis...")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    
    # Feature Engineering
    print("[INFO] Performing feature engineering...")
    
    # 1. Enhanced floor_area transformation
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    
    # 2. Enhanced energy intensity features
    df['total_eui'] = df['electric_eui'] + df['fuel_eui']
    df['electric_ratio'] = df['electric_eui'] / df['total_eui']
    df['fuel_ratio'] = df['fuel_eui'] / df['total_eui']
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    
    # 3. Enhanced building age features
    df['building_age'] = 2024 - df['year_built']
    df['building_age_log'] = np.log1p(df['building_age'])
    
    # 4. Enhanced energy star rating features
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    
    # 5. Handle missing values in ghg_emissions_int
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    
    # Select features
    features = [
        "ghg_emissions_int_log",
        "total_eui",
        "floor_area_log",
        "electric_eui",
        "fuel_eui",
        "energy_star_rating_normalized",
        "energy_mix",
        "building_age_log"
    ]
    feature_names = features.copy()
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    
    # Train and evaluate model
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultsBayesianARD')
    model, metrics = train_and_evaluate(X, y, feature_names, output_dir=results_dir)
    
    print("[INFO] Analysis complete. Results saved to", results_dir) 