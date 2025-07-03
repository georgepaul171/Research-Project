# Adaptive Prior Bayesian Regression: V3_clean

## Project Overview
This folder contains code, results, and diagnostics for experiments with custom Bayesian regression models using Adaptive Elastic Horseshoe (AEH) and hierarchical priors. The focus is on understanding the effect of these priors on model fit, uncertainty, and prediction range, compared to baseline models (LinearRegression, BayesianRidge).

## Main Findings & Model Comparisons
- **Baseline models (LinearRegression, BayesianRidge)** fit the full range of the target variable and provide interpretable weights and well-calibrated uncertainty.
- **AdaptivePriorARD with AEH prior** (on all or energy group) leads to over-regularization: weights are shrunk, predictions do not match the true range, and uncertainty is underestimated.
- **Hierarchical priors (non-AEH)** perform similarly to baselines and do not suffer from over-regularization.
- **Trace diagnostics** show that AEH priors can overly constrain the posterior, while baselines and hierarchical priors allow for better exploration.

## Folder Structure
- `debug_model_range.py` — Script for direct model range diagnostics and comparison of priors.
- `calibration_experiments.py` — Script for calibration and uncertainty experiments.
- `debug_v3_intervals.py` — Script for interval diagnostics.
- `prediction_vs_actual_simple.py` — Simple prediction vs actual analysis.
- `AREHap_groupprior_hmcdebug.py`, `V3.py` — Main model implementations and advanced experiments.
- `requirements.txt` — Python dependencies for reproducibility.
- `results/` — Main results: plots, SHAP values, diagnostics, model artifacts, and logs.
- `results_debug_model_range/` — Detailed diagnostics and stats for model range experiments.
- `results_calibration_experiments/` — Calibration and uncertainty analysis outputs.
- `results_debug_v3_intervals/` — Interval diagnostics and supporting plots.
- `results_simple_model/` — Results from simple model experiments.

## How to Run Experiments
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run main scripts:**
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
- **Diagnostics:** EM logs, scaling, weights, and model fit statistics.
- **Model Artifacts:** Saved models (e.g., `adaptive_prior_model.joblib`).
- **Logs:** Hyperparameter logs, progress logs, and detailed experiment outputs.

## How to Use the Results for Writing
- **Figures:** Use plots from `results/` for model fit, uncertainty, calibration, feature importance, and diagnostics. Each plot is named for its purpose (see table below).
- **Tables:** Use the quantitative comparison table in `findings.md` for reporting min/max, weights, and metrics. You can copy this directly into your report.
- **Interpretation:** See the 'Interpretation & Implications' section in `findings.md` for ready-to-use discussion points.
- **Trace and diagnostics:** Use trace plots and `trace_summary.md` to discuss model convergence and posterior exploration.
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

## Interpretation & Reporting
- See the summary file (`findings.md`) for high-level results, quantitative comparisons, and conclusions.
- Each results folder contains detailed outputs for the corresponding experiment.
- For questions about the AEH prior, see `AEH_prior_tradeoff.md`.

## Reproducibility
- All dependencies are listed in `requirements.txt`.
- Scripts are self-contained and save outputs to the appropriate results folders.
- For full reproducibility, ensure you use the same data splits and preprocessing as described in the main project documentation.

## Contact
For questions or further information, contact the project maintainer or refer to the main project README. 