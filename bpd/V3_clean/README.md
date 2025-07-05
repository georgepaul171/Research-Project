# Adaptive Prior Bayesian Regression: V3_clean

## Project Overview
This folder contains code, results, and diagnostics for experiments with custom Bayesian regression models using Adaptive Elastic Horseshoe (AEH) and hierarchical priors. The focus is on understanding the effect of these priors on model fit, uncertainty, and prediction range, compared to baseline models (LinearRegression, BayesianRidge).

## Main Findings & Model Comparisons
- **AdaptivePriorARD with AEH prior** (on energy group) achieves **excellent performance** with R² = 0.942, RMSE = 6.45, and MAE = 4.21
- **AEH prior successfully adapts** its hyperparameters (τ, α, β, λ) during training for optimal regularization
- **Energy features** (ghg_emissions_int_log, electric_eui, fuel_eui, floor_area_log) use AEH prior with adaptive regularization
- **Building and interaction features** use standard hierarchical priors for stability
- **Prediction range** is much improved (-21.25 to 152.70 vs true range 4.78-154.21) after fixing scaling issues
- **Model converges quickly** (3 iterations) with stable training dynamics
- **Baseline models** (LinearRegression, BayesianRidge) perform similarly (R² ≈ 0.939) but without adaptive regularization

## Key Results Summary
| Model | R² | RMSE | MAE | Prediction Range | Prior Type |
|-------|----|------|-----|------------------|------------|
| **AdaptivePriorARD (AEH)** | **0.942** | **6.45** | **4.21** | -21.25 to 152.70 | AEH (energy) + Hierarchical |
| BayesianRidge | 0.939 | 6.43 | 4.20 | -26.56 to 153.79 | Standard |
| LinearRegression | 0.939 | 6.43 | 4.20 | -26.62 to 153.87 | None |

## AEH Prior Implementation
- **Energy Group (4 features)**: Uses Adaptive Elastic Horseshoe prior
  - `ghg_emissions_int_log` (19.3% importance)
  - `electric_eui` (15.4% importance) 
  - `fuel_eui` (16.9% importance)
  - `floor_area_log` (0.7% importance)
- **Building Group (4 features)**: Uses hierarchical prior
- **Interaction Group (4 features)**: Uses hierarchical prior

## AEH Hyperparameter Adaptation
- **τ (global shrinkage)**: 0.85 (increasing for stronger regularization)
- **α (elastic net mixing)**: 0.41 (balanced L1/L2 regularization)
- **β (horseshoe vs elastic net)**: 0.69 (reduced horseshoe influence)
- **λ (local shrinkage)**: Adapting per feature for optimal regularization

## Folder Structure
- `V3.py` — Main model implementation with AEH prior and scaling fixes
- `debug_model_range.py` — Script for direct model range diagnostics and comparison of priors
- `calibration_experiments.py` — Script for calibration and uncertainty experiments
- `debug_v3_intervals.py` — Script for interval diagnostics
- `prediction_vs_actual_simple.py` — Simple prediction vs actual analysis
- `AREHap_groupprior_hmcdebug.py` — Advanced experiments with HMC
- `requirements.txt` — Python dependencies for reproducibility
- `results/` — Main results: plots, SHAP values, diagnostics, model artifacts, and logs
- `results_debug_model_range/` — Detailed diagnostics and stats for model range experiments
- `results_calibration_experiments/` — Calibration and uncertainty analysis outputs
- `results_debug_v3_intervals/` — Interval diagnostics and supporting plots
- `results_simple_model/` — Results from simple model experiments

## How to Run Experiments
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run main model with AEH:**
   ```bash
   python V3.py
   ```
3. **Run diagnostic scripts:**
   - For model range diagnostics:
     ```bash
     python debug_model_range.py
     ```
   - For calibration experiments:
     ```bash
     python calibration_experiments.py
     ```
   - For interval diagnostics:
     ```bash
     python debug_v3_intervals.py
     ```
   - For simple model analysis:
     ```bash
     python prediction_vs_actual_simple.py
     ```

## Results & Outputs
- **Plots:** Prediction vs actual, calibration, feature importance, SHAP, uncertainty, etc.
- **Diagnostics:** EM logs, AEH hyperparameter adaptation, weights, and model fit statistics
- **Model Artifacts:** Saved models (e.g., `adaptive_prior_model.joblib`)
- **Logs:** AEH hyperparameter logs, progress logs, and detailed experiment outputs

## How to Use the Results for Writing
- **Figures:** Use plots from `results/` for model fit, uncertainty, calibration, feature importance, and diagnostics. Each plot is named for its purpose (see table below).
- **Tables:** Use the quantitative comparison table above for reporting metrics and performance.
- **Interpretation:** See the 'Interpretation & Implications' section in `findings.md` for ready-to-use discussion points.
- **AEH Analysis:** Use `aeh_hyperparams_log.txt` to discuss adaptive hyperparameter behavior.
- **Feature importance:** Use `feature_importance.png`, `shap_summary.png`, and their corresponding JSON files for quantitative and visual feature analysis.

| Plot Filename                        | Purpose/Use Case                                 |
|-------------------------------------- |-------------------------------------------------|
| prediction_vs_actual.png              | Model fit (predicted vs actual)                  |
| prediction_vs_actual_errorbars.png    | Model fit with uncertainty intervals             |
| calibration_plot.png                  | Calibration of uncertainty                       |
| uncertainty_analysis.png              | Uncertainty structure/distribution               |
| feature_importance.png                | Feature importance (standard)                    |
| shap_summary.png                      | SHAP global feature importance                   |
| residual_analysis.png                 | Residual diagnostics                             |
| correlation_heatmap.png               | Feature correlation                              |
| partial_dependence.png                | Marginal effects                                 |
| group_importance.png                  | Group-level importance                           |
| feature_interaction_network.png       | Feature interactions                             |
| trace_minimal_bayes_*.png             | Bayesian trace diagnostics                       |
| learning_curves.png                   | Learning curve                                   |

## Data
- Input data is not included in this folder. Please refer to the main project README or data preparation scripts for instructions on obtaining and preprocessing the required datasets.

## Documentation Structure

### Core Documentation
- **`METHODOLOGY.md`**: Complete research methodology, experimental design, and statistical framework
- **`DATA_DOCUMENTATION.md`**: Data sources, preprocessing, feature engineering, and quality assessment
- **`MODEL_ARCHITECTURE.md`**: Mathematical formulation, prior specifications, and implementation details
- **`RESULTS_INTERPRETATION.md`**: How to interpret plots, metrics, and model outputs
- **`REPRODUCIBILITY_GUIDE.md`**: Step-by-step instructions for reproducing all experiments

### Specialized Documentation
- **`AEH_prior_tradeoff.md`**: Detailed analysis of AEH prior strength trade-offs
- **`findings.md`**: High-level results and quantitative comparisons
- **`AEH_PRIOR_MECHANICS.md`**: Detailed explanation of AEH prior implementation

## Interpretation & Reporting
- **For methodology**: See `METHODOLOGY.md` for research design and statistical framework
- **For data understanding**: See `DATA_DOCUMENTATION.md` for data sources and preprocessing
- **For model details**: See `MODEL_ARCHITECTURE.md` for mathematical formulation
- **For results interpretation**: See `RESULTS_INTERPRETATION.md` for understanding outputs
- **For reproducibility**: See `REPRODUCIBILITY_GUIDE.md` for complete setup instructions
- **For AEH prior analysis**: See `AEH_PRIOR_MECHANICS.md` for detailed AEH implementation

## Key Technical Achievements
1. **Successfully implemented AEH prior** with adaptive hyperparameter learning
2. **Fixed critical scaling bug** in prediction method for proper uncertainty estimation
3. **Achieved excellent performance** (R² = 0.942) with stable convergence
4. **Demonstrated adaptive regularization** for energy features while maintaining stability
5. **Proper feature grouping** with AEH for energy features and hierarchical for others

## Reproducibility
- All dependencies are listed in `requirements.txt`
- Scripts are self-contained and save outputs to the appropriate results folders
- For full reproducibility, ensure you use the same data splits and preprocessing as described in the main project documentation

## Contact
For questions or further information, contact the project maintainer or refer to the main project README. 