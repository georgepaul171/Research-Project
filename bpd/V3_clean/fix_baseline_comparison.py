
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Add the current directory to the path to import V3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from V3 import (
    AdaptivePriorARD, 
    AdaptivePriorConfig, 
    run_comprehensive_baseline_comparison,
    run_sensitivity_analysis,
    run_out_of_sample_validation,
    create_research_summary,
    feature_engineering_no_interactions
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the corrected baseline comparison including AEH model."""
    
    logger.info("Starting corrected baseline comparison with AEH model...")
    
    data_path = "../cleaned_office_buildings.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_path, na_values=na_vals, low_memory=False)
    
    # Feature engineering using the original function
    logger.info("Performing feature engineering...")
    df_processed = feature_engineering_no_interactions(df)
    
    # Prepare features and target
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
    target = "site_eui"
    
    X = df_processed[features].values.astype(np.float32)
    y = df_processed[target].values.astype(np.float32).reshape(-1)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {feature_names}")
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Configure AEH model
    config = AdaptivePriorConfig(
        beta_0=0.1,  # Moderate regularisation
        max_iter=50,
        use_hmc=False,  
        calibration_factor=0.03,  # reduced from 0.05 to improve PICP calibration
        group_prior_types={
            'energy': 'adaptive_elastic_horseshoe',
            'building': 'hierarchical',
            'interaction': 'hierarchical'
        }
    )
    
    # Train main AEH model
    logger.info("Training main AEH model...")
    model = AdaptivePriorARD(config=config)
    model.fit(X, y, feature_names=feature_names, output_dir=results_dir)
    
    # Run comprehensive baseline comparison (includes AEH model)
    logger.info("Running comprehensive baseline comparison...")
    baseline_results, significance_results = run_comprehensive_baseline_comparison(
        X, y, feature_names, results_dir
    )
    
    # Run sensitivity analysis
    logger.info("Running sensitivity analysis...")
    sensitivity_results = run_sensitivity_analysis(X, y, feature_names, results_dir)
    
    # Run out-of-sample validation
    logger.info("Running out-of-sample validation...")
    validation_results = run_out_of_sample_validation(X, y, feature_names, results_dir)
    
    # Create comprehensive research summary
    logger.info("Creating comprehensive research summary...")
    summary = create_research_summary(
        model, baseline_results, significance_results, 
        sensitivity_results, validation_results, results_dir
    )
    
    # Print key results
    logger.info("=== KEY RESULTS ===")
    logger.info(f"AEH Model R²: {baseline_results['AdaptivePriorARD (AEH)']['r2_mean']:.3f}")
    logger.info(f"Best Baseline R²: {max([v['r2_mean'] for k, v in baseline_results.items() if k != 'AdaptivePriorARD (AEH)']):.3f}")
    
    # Print performance ranking
    logger.info("=== PERFORMANCE RANKING ===")
    performance_ranking = sorted(baseline_results.keys(), key=lambda x: baseline_results[x]['r2_mean'], reverse=True)
    for i, model_name in enumerate(performance_ranking, 1):
        r2 = baseline_results[model_name]['r2_mean']
        if model_name == 'AdaptivePriorARD (AEH)':
            logger.info(f"{i}. {model_name} (R² = {r2:.3f}) - OUR MODEL")
        else:
            logger.info(f"{i}. {model_name} (R² = {r2:.3f})")
    
    # Print statistical significance results
    logger.info("=== STATISTICAL SIGNIFICANCE ===")
    ae_model_name = "AdaptivePriorARD (AEH)"
    baseline_models_only = [m for m in baseline_results.keys() if m != ae_model_name]
    best_baseline = max(baseline_models_only, key=lambda x: baseline_results[x]['r2_mean'])
    
    ae_vs_best = f"{ae_model_name}_vs_{best_baseline}"
    if ae_vs_best in significance_results:
        sig_test = significance_results[ae_vs_best]
        logger.info(f"AEH vs {best_baseline}:")
        logger.info(f"  p-value: {sig_test['t_test']['p_value']:.4f}")
        logger.info(f"  Significant: {sig_test['t_test']['significant']}")
        logger.info(f"  Effect size (Cohen's d): {sig_test['effect_size']['cohens_d']:.3f}")
    
    logger.info("=== COMPLETED ===")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("Check the following files:")
    logger.info(f"  - {results_dir}/comprehensive_baseline_results.json")
    logger.info(f"  - {results_dir}/comprehensive_research_summary.json")
    logger.info(f"  - {results_dir}/EXECUTIVE_SUMMARY.md")

if __name__ == "__main__":
    main() 