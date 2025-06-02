import numpy as np
from typing import Tuple, List, Optional
import logging
import scipy.stats as stats
import os
import json
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_results_directory() -> str:
    """
    Ensure the results directory exists and return its path.
    
    Returns:
        str: Path to the results directory
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'probabilistic_metrics')
    
    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")
    
    return results_dir

def calculate_picp(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, 
                  confidence_levels: Optional[List[float]] = None) -> Tuple[dict, dict]:
    """
    Calculate Prediction Interval Coverage Probability (PICP) for different confidence levels.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        confidence_levels: List of confidence levels (default: [0.5, 0.8, 0.9, 0.95, 0.99])
        
    Returns:
        Tuple containing:
        - Dictionary of PICP values for each confidence level
        - Dictionary of prediction intervals for each confidence level
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    picp_values = {}
    prediction_intervals = {}
    
    for alpha in confidence_levels:
        # Calculate z-score for the confidence level using normal distribution
        z_score = stats.norm.ppf((1 + alpha) / 2)
        
        # Calculate prediction intervals
        lower_bound = y_pred - z_score * y_std
        upper_bound = y_pred + z_score * y_std
        
        # Calculate coverage
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        picp_values[alpha] = coverage
        prediction_intervals[alpha] = {
            'lower': lower_bound,
            'upper': upper_bound
        }
        
        logger.info(f"PICP for {alpha*100}% confidence level: {coverage:.4f}")
    
    return picp_values, prediction_intervals

def calculate_crps(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS) for probabilistic predictions.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        
    Returns:
        float: CRPS value
    """
    # CRPS for normal distribution
    sigma = y_std
    x = y_true
    
    # Calculate CRPS components
    term1 = sigma * (1/np.sqrt(np.pi) - 2 * np.exp(-(x - y_pred)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi))
    term2 = (x - y_pred) * (2 * stats.norm.cdf((x - y_pred) / sigma) - 1)
    
    crps = np.mean(term1 + term2)
    
    logger.info(f"CRPS: {crps:.4f}")
    return crps

def evaluate_probabilistic_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray,
                                     confidence_levels: Optional[List[float]] = None,
                                     save_results: bool = True) -> dict:
    """
    Evaluate probabilistic predictions using PICP and CRPS metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        confidence_levels: List of confidence levels (default: [0.5, 0.8, 0.9, 0.95, 0.99])
        save_results: Whether to save results to file
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    # Calculate PICP
    picp_values, prediction_intervals = calculate_picp(y_true, y_pred, y_std, confidence_levels)
    
    # Calculate CRPS
    crps = calculate_crps(y_true, y_pred, y_std)
    
    # Calculate additional metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Calculate reliability diagram
    reliability = {}
    for alpha in confidence_levels:
        expected_coverage = alpha
        actual_coverage = picp_values[alpha]
        reliability[alpha] = {
            'expected': expected_coverage,
            'actual': actual_coverage
        }
    
    results = {
        'picp': picp_values,
        'crps': crps,
        'mae': mae,
        'rmse': rmse,
        'reliability': reliability,
        'prediction_intervals': prediction_intervals
    }
    
    if save_results:
        results_dir = ensure_results_directory()
        results_file = os.path.join(results_dir, 'probabilistic_metrics_results.json')
        
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {
            'picp': {str(k): float(v) for k, v in results['picp'].items()},
            'crps': float(results['crps']),
            'mae': float(results['mae']),
            'rmse': float(results['rmse']),
            'reliability': {
                str(k): {
                    'expected': float(v['expected']),
                    'actual': float(v['actual'])
                } for k, v in results['reliability'].items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Results saved to {results_file}")
    
    return results

def plot_probabilistic_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray,
                             confidence_levels: Optional[List[float]] = None,
                             save_path: Optional[str] = None):
    """
    Plot probabilistic evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        confidence_levels: List of confidence levels
        save_path: Optional path to save the plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    # Calculate metrics
    results = evaluate_probabilistic_predictions(y_true, y_pred, y_std, confidence_levels)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Reliability Diagram
    plt.subplot(2, 2, 1)
    expected = [results['reliability'][alpha]['expected'] for alpha in confidence_levels]
    actual = [results['reliability'][alpha]['actual'] for alpha in confidence_levels]
    plt.plot(expected, actual, 'bo-', label='Actual Coverage')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Expected Coverage')
    plt.ylabel('Actual Coverage')
    plt.title('Reliability Diagram')
    plt.legend()
    
    # 2. Prediction Intervals
    plt.subplot(2, 2, 2)
    alpha = 0.95  # Show 95% prediction interval
    z_score = abs(np.percentile(np.random.standard_normal(10000), (1 - alpha) * 100))
    lower = y_pred - z_score * y_std
    upper = y_pred + z_score * y_std
    
    plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions')
    plt.fill_between(y_true, lower, upper, alpha=0.2, label=f'{alpha*100}% Prediction Interval')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Intervals')
    plt.legend()
    
    # 3. Error Distribution
    plt.subplot(2, 2, 3)
    errors = y_true - y_pred
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    # 4. Uncertainty vs Error
    plt.subplot(2, 2, 4)
    plt.scatter(np.abs(errors), y_std, alpha=0.5)
    plt.xlabel('Absolute Error')
    plt.ylabel('Predicted Uncertainty')
    plt.title('Uncertainty vs Error')
    
    plt.tight_layout()
    
    if save_path is None:
        results_dir = ensure_results_directory()
        save_path = os.path.join(results_dir, 'probabilistic_metrics_plots.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plots saved to {save_path}")
    
    plt.close()

if __name__ == "__main__":
    # Load predictions and uncertainties from ARD model output
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    csv_path = os.path.join(results_dir, 'ard_predictions_with_uncertainty.csv')
    df = pd.read_csv(csv_path)
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    y_std = df['y_std'].values
    
    # Evaluate predictions (original)
    results = evaluate_probabilistic_predictions(y_true, y_pred, y_std)
    
    # Plot results (original)
    plot_probabilistic_metrics(y_true, y_pred, y_std)
    
    # --- Diagnostic: Compare uncertainty vs. error ---
    import matplotlib.pyplot as plt
    abs_error = np.abs(y_true - y_pred)
    plt.figure(figsize=(8, 5))
    plt.hist(abs_error, bins=50, alpha=0.6, label='|y_true - y_pred| (Absolute Error)')
    plt.hist(y_std, bins=50, alpha=0.6, label='y_std (Predicted Uncertainty)')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution of Absolute Error vs. Predicted Uncertainty')
    plt.legend()
    plt.tight_layout()
    diag_path = os.path.join(results_dir, 'uncertainty_vs_error_hist.png')
    plt.savefig(diag_path, dpi=200)
    plt.close()
    logger.info(f"Diagnostic plot saved to {diag_path}")
    
    # --- Quick test: Scale up uncertainty and re-evaluate ---
    scale_factor = 10.0
    y_std_scaled = y_std * scale_factor
    logger.info(f"\n--- Scaling y_std by {scale_factor} for diagnostic ---")
    results_scaled = evaluate_probabilistic_predictions(y_true, y_pred, y_std_scaled, save_results=False)
    logger.info(f"PICP after scaling uncertainty: {results_scaled['picp']}")
    logger.info(f"CRPS after scaling uncertainty: {results_scaled['crps']}") 