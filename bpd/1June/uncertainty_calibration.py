import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Optional, Dict, List
import joblib
import os
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_output_directory() -> str:
    """
    Create a new timestamped directory for results.
    
    Returns:
        str: Path to the created directory
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(current_dir, 'calibration_results')
    
    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, f'calibration_{timestamp}')
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    return output_dir

class UncertaintyCalibrator(BaseEstimator, RegressorMixin):
    """
    A calibrator for improving uncertainty estimates from Bayesian models.
    Implements several calibration techniques:
    1. Isotonic Regression Calibration
    2. Temperature Scaling
    3. Ensemble-based Calibration
    4. Quantile Regression
    """
    
    def __init__(self, 
                 method: str = 'isotonic',
                 n_bins: int = 10,
                 ensemble_size: int = 5,
                 temperature: float = 1.0,
                 random_state: Optional[int] = None,
                 min_std: float = 1e-6):  # Added minimum std threshold
        """
        Initialize the uncertainty calibrator.
        
        Args:
            method: Calibration method ('isotonic', 'temperature', 'ensemble', 'quantile')
            n_bins: Number of bins for isotonic regression
            ensemble_size: Number of models for ensemble calibration
            temperature: Initial temperature for temperature scaling
            random_state: Random seed for reproducibility
            min_std: Minimum standard deviation to prevent numerical instability
        """
        self.method = method
        self.n_bins = n_bins
        self.ensemble_size = ensemble_size
        self.temperature = temperature
        self.random_state = random_state
        self.min_std = min_std
        self.calibration_params = {}
        self.is_fitted = False
        
    def _clip_std(self, std: np.ndarray) -> np.ndarray:
        """Clip standard deviations to prevent numerical instability."""
        return np.clip(std, self.min_std, None)
    
    def _isotonic_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_std: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calibrate uncertainty using isotonic regression.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            
        Returns:
            Tuple of (calibrated_std, calibration_params)
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        
        # Calculate empirical coverage
        z_scores = np.abs(y_true - y_pred) / y_std
        empirical_coverage = np.zeros(self.n_bins)
        expected_coverage = np.linspace(0, 1, self.n_bins)
        
        for i in range(self.n_bins):
            threshold = stats.norm.ppf((1 + expected_coverage[i]) / 2)
            empirical_coverage[i] = np.mean(z_scores <= threshold)
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(expected_coverage, empirical_coverage)
        
        # Calibrate standard deviations
        calibrated_std = y_std * iso_reg.predict(np.abs(y_true - y_pred) / y_std)
        calibrated_std = self._clip_std(calibrated_std)
        
        return calibrated_std, {'iso_reg': iso_reg}
    
    def _temperature_scaling(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_std: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calibrate uncertainty using temperature scaling.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            
        Returns:
            Tuple of (calibrated_std, calibration_params)
        """
        from scipy.optimize import minimize
        
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        
        def objective(temp):
            scaled_std = self._clip_std(y_std * temp)
            z_scores = np.abs(y_true - y_pred) / scaled_std
            empirical_coverage = np.mean(z_scores <= 1.96)  # 95% coverage
            return (empirical_coverage - 0.95) ** 2
        
        # Optimize temperature parameter
        result = minimize(objective, x0=self.temperature, method='Nelder-Mead')
        optimal_temp = max(result.x[0], self.min_std)  # Ensure positive temperature
        
        # Apply temperature scaling
        calibrated_std = self._clip_std(y_std * optimal_temp)
        
        return calibrated_std, {'temperature': optimal_temp}
    
    def _ensemble_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_std: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calibrate uncertainty using ensemble methods.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            
        Returns:
            Tuple of (calibrated_std, calibration_params)
        """
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        
        # Generate ensemble predictions using bootstrap
        n_samples = len(y_true)
        ensemble_preds = np.zeros((self.ensemble_size, n_samples))
        ensemble_stds = np.zeros((self.ensemble_size, n_samples))
        
        for i in range(self.ensemble_size):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            ensemble_preds[i] = y_pred[indices]
            ensemble_stds[i] = y_std[indices]
        
        # Calculate ensemble statistics
        mean_pred = np.mean(ensemble_preds, axis=0)
        mean_std = np.mean(ensemble_stds, axis=0)
        pred_std = np.std(ensemble_preds, axis=0)
        
        # Combine uncertainties with numerical stability
        calibrated_std = self._clip_std(np.sqrt(mean_std**2 + pred_std**2))
        
        return calibrated_std, {'ensemble_size': self.ensemble_size}
    
    def _quantile_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_std: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calibrate uncertainty using quantile regression.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            
        Returns:
            Tuple of (calibrated_std, calibration_params)
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Calculate empirical quantiles
        errors = np.abs(y_true - y_pred)
        quantiles = np.percentile(errors, [25, 50, 75])
        
        # Train quantile regressor
        q_reg = GradientBoostingRegressor(
            loss='quantile',
            alpha=0.5,
            n_estimators=100,
            random_state=self.random_state
        )
        q_reg.fit(y_pred.reshape(-1, 1), errors)
        
        # Predict calibrated uncertainties
        calibrated_std = self._clip_std(q_reg.predict(y_pred.reshape(-1, 1)))
        
        return calibrated_std, {'quantile_regressor': q_reg}
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> 'UncertaintyCalibrator':
        """
        Fit the calibrator to the data.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            
        Returns:
            self: The fitted calibrator
        """
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        
        if self.method == 'isotonic':
            _, self.calibration_params = self._isotonic_calibration(y_true, y_pred, y_std)
        elif self.method == 'temperature':
            _, self.calibration_params = self._temperature_scaling(y_true, y_pred, y_std)
        elif self.method == 'ensemble':
            _, self.calibration_params = self._ensemble_calibration(y_true, y_pred, y_std)
        elif self.method == 'quantile':
            _, self.calibration_params = self._quantile_calibration(y_true, y_pred, y_std)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def predict(self, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
        """
        Predict calibrated uncertainties.
        
        Args:
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
            
        Returns:
            np.ndarray: Calibrated standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        
        if self.method == 'isotonic':
            calibrated_std = y_std * self.calibration_params['iso_reg'].predict(
                np.abs(y_pred - y_pred) / y_std
            )
        elif self.method == 'temperature':
            calibrated_std = y_std * self.calibration_params['temperature']
        elif self.method == 'ensemble':
            # For ensemble method, we need to generate new ensemble predictions
            n_samples = len(y_pred)
            ensemble_preds = np.zeros((self.ensemble_size, n_samples))
            ensemble_stds = np.zeros((self.ensemble_size, n_samples))
            
            for i in range(self.ensemble_size):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                ensemble_preds[i] = y_pred[indices]
                ensemble_stds[i] = y_std[indices]
            
            mean_std = np.mean(ensemble_stds, axis=0)
            pred_std = np.std(ensemble_preds, axis=0)
            calibrated_std = np.sqrt(mean_std**2 + pred_std**2)
        elif self.method == 'quantile':
            calibrated_std = self.calibration_params['quantile_regressor'].predict(
                y_pred.reshape(-1, 1)
            )
        
        return self._clip_std(calibrated_std)
    
    def evaluate_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_std: np.ndarray, calibrated_std: np.ndarray) -> Dict:
        """
        Evaluate the calibration performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Original standard deviations
            calibrated_std: Calibrated standard deviations
            
        Returns:
            Dict: Calibration metrics
        """
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        calibrated_std = self._clip_std(calibrated_std)
        
        # Calculate PICP for different confidence levels
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        original_picp = {}
        calibrated_picp = {}
        
        for alpha in confidence_levels:
            z_score = stats.norm.ppf((1 + alpha) / 2)
            
            # Original PICP
            lower_orig = y_pred - z_score * y_std
            upper_orig = y_pred + z_score * y_std
            original_picp[alpha] = np.mean((y_true >= lower_orig) & (y_true <= upper_orig))
            
            # Calibrated PICP
            lower_cal = y_pred - z_score * calibrated_std
            upper_cal = y_pred + z_score * calibrated_std
            calibrated_picp[alpha] = np.mean((y_true >= lower_cal) & (y_true <= upper_cal))
        
        # Calculate CRPS with numerical stability
        def calculate_crps(y_true, y_pred, y_std):
            sigma = self._clip_std(y_std)
            x = y_true
            term1 = sigma * (1/np.sqrt(np.pi) - 2 * np.exp(-(x - y_pred)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi))
            term2 = (x - y_pred) * (2 * stats.norm.cdf((x - y_pred) / sigma) - 1)
            return np.mean(term1 + term2)
        
        original_crps = calculate_crps(y_true, y_pred, y_std)
        calibrated_crps = calculate_crps(y_true, y_pred, calibrated_std)
        
        return {
            'original_picp': original_picp,
            'calibrated_picp': calibrated_picp,
            'original_crps': original_crps,
            'calibrated_crps': calibrated_crps
        }
    
    def plot_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_std: np.ndarray, calibrated_std: np.ndarray,
                        save_path: Optional[str] = None):
        """
        Plot calibration diagnostics.
        
        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Original standard deviations
            calibrated_std: Calibrated standard deviations
            save_path: Optional path to save the plots
        """
        # Ensure non-zero standard deviations
        y_std = self._clip_std(y_std)
        calibrated_std = self._clip_std(calibrated_std)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reliability Diagram
        confidence_levels = np.linspace(0, 1, 100)
        expected_coverage = confidence_levels
        original_coverage = []
        calibrated_coverage = []
        
        for alpha in confidence_levels:
            z_score = stats.norm.ppf((1 + alpha) / 2)
            
            # Original coverage
            lower_orig = y_pred - z_score * y_std
            upper_orig = y_pred + z_score * y_std
            original_coverage.append(np.mean((y_true >= lower_orig) & (y_true <= upper_orig)))
            
            # Calibrated coverage
            lower_cal = y_pred - z_score * calibrated_std
            upper_cal = y_pred + z_score * calibrated_std
            calibrated_coverage.append(np.mean((y_true >= lower_cal) & (y_true <= upper_cal)))
        
        axes[0, 0].plot(expected_coverage, expected_coverage, 'k--', label='Perfect Calibration')
        axes[0, 0].plot(expected_coverage, original_coverage, 'b-', label='Original')
        axes[0, 0].plot(expected_coverage, calibrated_coverage, 'r-', label='Calibrated')
        axes[0, 0].set_xlabel('Expected Coverage')
        axes[0, 0].set_ylabel('Actual Coverage')
        axes[0, 0].set_title('Reliability Diagram')
        axes[0, 0].legend()
        
        # 2. Uncertainty vs Error
        errors = np.abs(y_true - y_pred)
        axes[0, 1].scatter(errors, y_std, alpha=0.5, label='Original')
        axes[0, 1].scatter(errors, calibrated_std, alpha=0.5, label='Calibrated')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Uncertainty')
        axes[0, 1].set_title('Uncertainty vs Error')
        axes[0, 1].legend()
        
        # 3. Histogram of Standardized Errors
        z_scores_orig = (y_true - y_pred) / y_std
        z_scores_cal = (y_true - y_pred) / calibrated_std
        
        # Remove infinite values
        z_scores_orig = z_scores_orig[np.isfinite(z_scores_orig)]
        z_scores_cal = z_scores_cal[np.isfinite(z_scores_cal)]
        
        axes[1, 0].hist(z_scores_orig, bins=50, alpha=0.5, label='Original')
        axes[1, 0].hist(z_scores_cal, bins=50, alpha=0.5, label='Calibrated')
        axes[1, 0].set_xlabel('Standardized Error')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Standardized Errors')
        axes[1, 0].legend()
        
        # 4. QQ Plot
        stats.probplot(z_scores_cal, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Standardized Errors')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration plots saved to {save_path}")
        
        plt.close()

def main():
    """
    Main function to demonstrate uncertainty calibration.
    """
    # Create output directory
    output_dir = create_output_directory()
    
    # Load the original predictions and uncertainties
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    csv_path = os.path.join(results_dir, 'ard_predictions_with_uncertainty.csv')
    df = pd.read_csv(csv_path)
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    y_std = df['y_std'].values
    
    # Split data for calibration
    y_true_train, y_true_test, y_pred_train, y_pred_test, y_std_train, y_std_test = train_test_split(
        y_true, y_pred, y_std, test_size=0.2, random_state=42
    )
    
    # Try different calibration methods
    methods = ['isotonic', 'temperature', 'ensemble', 'quantile']
    best_crps = float('inf')
    best_method = None
    best_calibrated_std = None
    method_results = {}
    
    for method in methods:
        logger.info(f"\nTrying {method} calibration...")
        calibrator = UncertaintyCalibrator(method=method, min_std=1e-6)
        calibrator.fit(y_true_train, y_pred_train, y_std_train)
        calibrated_std = calibrator.predict(y_pred_test, y_std_test)
        metrics = calibrator.evaluate_calibration(y_true_test, y_pred_test, y_std_test, calibrated_std)
        
        # Save method-specific results
        method_results[method] = {
            'metrics': metrics,
            'calibrated_std': calibrated_std
        }
        
        # Save method-specific plots
        plot_path = os.path.join(output_dir, f'calibration_plots_{method}.png')
        calibrator.plot_calibration(y_true_test, y_pred_test, y_std_test, calibrated_std, save_path=plot_path)
        
        if metrics['calibrated_crps'] < best_crps:
            best_crps = metrics['calibrated_crps']
            best_method = method
            best_calibrated_std = calibrated_std
    
    logger.info(f"\nBest calibration method: {best_method}")
    logger.info(f"Best CRPS: {best_crps:.4f}")
    
    # Save summary of all methods
    summary = {
        'best_method': best_method,
        'best_crps': float(best_crps),
        'method_results': {
            method: {
                'metrics': {
                    'original_picp': {str(k): float(v) for k, v in method_results[method]['metrics']['original_picp'].items()},
                    'calibrated_picp': {str(k): float(v) for k, v in method_results[method]['metrics']['calibrated_picp'].items()},
                    'original_crps': float(method_results[method]['metrics']['original_crps']),
                    'calibrated_crps': float(method_results[method]['metrics']['calibrated_crps'])
                }
            }
            for method in method_results
        }
    }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, 'calibration_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")
    
    # Use the best method for final calibration
    calibrator = UncertaintyCalibrator(method=best_method, min_std=1e-6)
    calibrator.fit(y_true_train, y_pred_train, y_std_train)
    calibrated_std = calibrator.predict(y_pred_test, y_std_test)
    metrics = calibrator.evaluate_calibration(y_true_test, y_pred_test, y_std_test, calibrated_std)
    
    # Print results
    logger.info("\nCalibration Results:")
    logger.info("\nOriginal PICP:")
    for alpha, picp in metrics['original_picp'].items():
        logger.info(f"{alpha*100}% confidence: {picp:.4f}")
    
    logger.info("\nCalibrated PICP:")
    for alpha, picp in metrics['calibrated_picp'].items():
        logger.info(f"{alpha*100}% confidence: {picp:.4f}")
    
    logger.info(f"\nOriginal CRPS: {metrics['original_crps']:.4f}")
    logger.info(f"Calibrated CRPS: {metrics['calibrated_crps']:.4f}")
    
    # Save final calibrated predictions
    output_df = pd.DataFrame({
        'y_true': y_true_test,
        'y_pred': y_pred_test,
        'y_std_original': y_std_test,
        'y_std_calibrated': calibrated_std
    })
    output_path = os.path.join(output_dir, 'calibrated_predictions.csv')
    output_df.to_csv(output_path, index=False)
    logger.info(f"\nCalibrated predictions saved to {output_path}")
    
    # Save calibration parameters
    params_path = os.path.join(output_dir, 'calibration_parameters.json')
    with open(params_path, 'w') as f:
        json.dump({
            'method': best_method,
            'min_std': 1e-6,
            'calibration_params': calibrator.calibration_params
        }, f, indent=4)
    logger.info(f"Calibration parameters saved to {params_path}")

if __name__ == "__main__":
    main() 