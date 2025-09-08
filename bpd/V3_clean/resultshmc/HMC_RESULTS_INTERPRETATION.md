# Interpretation of HMC Run Results

This document summarizes and interprets the results from the Adaptive Prior ARD model run with Hamiltonian Monte Carlo (HMC) enabled. All referenced files are from the `resultshmc` folder.

## Overview

- **Inference method:** Hamiltonian Monte Carlo (HMC)
- **Model:** Adaptive Prior ARD with AEH prior for energy features, hierarchical for others
- **Dataset:** Cleaned office buildings
- **Features used:** 12 engineered features (see `debug_info.txt`)

## Main Performance Metrics

| Metric | Value |
|--------|-------|
| R²     | (see `metrics.json`) |
| RMSE   | (see `metrics.json`) |
| MAE    | (see `metrics.json`) |

> **Note:** Please refer to `metrics.json` for the exact values. Typically, HMC should yield similar or slightly improved uncertainty calibration compared to EM.

## Uncertainty & Calibration
- **Uncertainty estimates:**
  - See `adaptive_prior_results.txt` for prediction and uncertainty ranges.
  - See `fold_metrics.json` for calibration across folds.
- **Prediction intervals:**
  - HMC is expected to provide more robust posterior samples, potentially improving interval calibration.
- **PICP (coverage):**
  - Check `out_of_sample_validation.json` for coverage and calibration on held-out data.

## Log Files (Diagnostics)
- **`aeh_hyperparams_log.txt`:**
  - Contains the evolution of AEH hyperparameters (lambda, tau, alpha, beta, momentum) for each group and iteration.
  - Useful for diagnosing adaptation and convergence of the prior.
- **`beta_tau_log.txt`:**
  - Logs beta and tau values for each iteration, including group-specific shrinkage parameters.
  - Useful for monitoring parameter stability and HMC mixing.
- **`em_progress_log.txt`:**
  - Tracks EM progress, weights, predictions, and convergence diagnostics (even with HMC, some EM steps may be used for initialization or hybrid inference).

## Visualizations & Feature Importance
- **Feature importance:**
  - See `feature_importance.json` and `feature_importance_simple.json` for normalized importance scores.
  - Plots: `feature_importance_simple.png`, `shap_summary.png`, and force plots for individual samples.
- **Calibration and validation:**
  - `sensitivity_analysis.png`, `out_of_sample_validation.png`, and `baseline_comparison_comprehensive.png` provide visual diagnostics of model performance and robustness.

## Comparison to EM-only Results
- **Log files are now populated:**
  - Unlike the EM-only run, the HMC run provides detailed logs for hyperparameters and parameter trajectories.
- **Uncertainty calibration:**
  - HMC may yield more accurate or robust uncertainty estimates, especially for complex posteriors.
- **Performance metrics:**
  - Compare R², RMSE, and MAE to previous EM-only results to assess any gains or trade-offs.
- **Computation:**
  - HMC is typically slower but provides richer posterior information.

## Notable Findings
- The HMC run successfully generated detailed diagnostics and uncertainty estimates.
- Hyperparameter adaptation and parameter mixing can be reviewed in the log files for deeper model understanding.
- If PICP and calibration metrics are improved, this supports the use of HMC for uncertainty quantification.

## Recommendations
- Use HMC results for applications where robust uncertainty quantification is critical.
- For faster experimentation, EM may suffice if performance and calibration are similar.
- Review log files for any signs of poor mixing or convergence issues (e.g., large jumps or non-stationary traces).

---
*Generated automatically based on the contents of the resultshmc folder.* 