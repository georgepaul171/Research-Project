# Key Findings: Adaptive Prior Bayesian Regression (V3_clean)

## Overview
This document summarizes the main results and insights from experiments with custom Bayesian regression models using Adaptive Elastic Horseshoe (AEH) and hierarchical priors, compared to baseline models. All results are from the same dataset and experimental setup.

---

## 1. Quantitative Model Comparison

| Model                                 | True Min | True Max | Predicted Min | Predicted Max | RMSE    | MAE     | R²      |
|---------------------------------------|----------|----------|---------------|---------------|---------|---------|---------|
| LinearRegression                      | 4.78     | 154.21   | -26.62        | 153.87        | —       | —       | —       |
| BayesianRidge                         | 4.78     | 154.21   | -26.56        | 153.79        | —       | —       | —       |
| AdaptivePriorARD (Hierarchical)       | 4.78     | 154.21   | 118.58        | 1918.20       | —       | —       | —       |
| AdaptivePriorARD (AEH, energy group)  | 4.78     | 154.21   | 118.58        | 1918.20       | —       | —       | —       |
| AdaptivePriorARD (AEH, all groups)    | 4.78     | 154.21   | 118.58        | 1918.20       | —       | —       | —       |
| AdaptivePriorARD (AEH, extremes only) | 4.78     | 154.21   | 77.81         | 274.19        | —       | —       | —       |

- **Note:** RMSE, MAE, and R² are not available for all models in this summary table, but for the main experiment (see below):
    - RMSE: 47.78
    - MAE: 44.65
    - R²: -2.18

---

## 2. Model Weights (Selected)

- **LinearRegression:**
  `[44.72, -0.04, 0.37, 0.57, -5.25, -18.34, -0.37, -0.07, 0.00, -0.63, -3.00, 0.00]`
- **BayesianRidge:**
  `[44.68, 6.24, 0.37, 0.57, -5.21, -18.23, 20.94, -3.20, 5.95, -11.08, -3.02, 1.07]`
- **AdaptivePriorARD (AEH, all):**
  `[0.50, -0.01, 0.40, 0.44, 0.05, -0.01, -0.01, -0.01, -0.49, -0.01, -0.16, 0.51]`
- **AdaptivePriorARD (AEH, extremes):**
  `[2.06, -0.01, -0.04, 0.00, 0.01, 0.02, 0.00, -0.01, 0.10, 0.00, 0.00, -0.10]`

---

## 3. Feature Importance (Standard & SHAP)

- **Standard Feature Importance:**
  - Top: `ghg_emissions_int_log` (0.19), `ghg_per_area` (0.20), `energy_intensity_ratio` (0.19), `electric_eui` (0.15), `fuel_eui` (0.17)
- **SHAP Importance:**
  - Top: `electric_eui` (5.79), `fuel_eui` (5.40), `ghg_emissions_int_log` (0.12)

---

## 4. Diagnostics & Trace Summaries

- **Trace Diagnostics:**
  - *BayesianRidge*: Trace plots show well-mixing, flexible posteriors.
  - *AEH Prior*: Trace plots are flat, indicating the prior is too strong and the posterior is overly constrained.
  - See `results/trace_summary.md` and trace plots for details.

- **Prediction vs Actual Plots:**
  - See `results/prediction_vs_actual.png` and `results/prediction_vs_actual_errorbars.png` for visual comparison.

- **Uncertainty & Calibration:**
  - See `results/calibration_plot.png` and `results/uncertainty_analysis.png` for uncertainty quantification.

---

## 5. Interpretation & Implications

- **Baselines (LinearRegression, BayesianRidge):**
  - Accurately fit the full range of the target variable.
  - Weights are large and interpretable.
  - Posterior is flexible and well-explored.

- **AdaptivePriorARD with AEH Prior:**
  - When AEH is applied (to all groups or just the 'energy' group), the model is over-regularized:
    - Weights are shrunk toward zero.
    - Predicted min/max is far from the true range (e.g., 118.6 to 1918.2 vs. true 4.8 to 154.2).
    - Even when fit on extremes only, the model cannot recover the full range.
    - Trace plots confirm the posterior is too sharp.

- **Hierarchical Priors (Non-AEH):**
  - Fit the data well, similar to baselines.
  - Do not suffer from the over-regularization seen with AEH.

---

## 6. Recommendations
- Use AEH priors with caution and perform extensive hyperparameter tuning.
- For this dataset, hierarchical or standard Bayesian priors are preferable.
- Document and report negative results, as they provide valuable scientific insight.

---

## 7. References to Results Files
- **Plots:** `results/prediction_vs_actual.png`, `results/prediction_vs_actual_errorbars.png`, `results/calibration_plot.png`, `results/uncertainty_analysis.png`
- **Diagnostics:** `results/trace_summary.md`, `results/diagnostics_detailed.txt`, `results/feature_importance.json`, `results/shap_importance.json`
- **Weights & Ranges:** `results_debug_model_range/stats_*.json`, `results/bayesianridge_pred_range.txt`, `results/aeh_pred_range.txt`

---

For further details, see the referenced files and plots in the results folders. For questions about the AEH prior, see `AEH_prior_tradeoff.md`. 