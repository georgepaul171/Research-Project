# Adaptive Prior Bayesian Regression: V3_clean

## Project Overview
This folder contains code, results, and diagnostics for experiments with custom Bayesian regression models using Adaptive Elastic Horseshoe (AEH) and hierarchical priors. The focus is on understanding the effect of these priors on model fit, uncertainty, and prediction range, compared to baseline models (LinearRegression, BayesianRidge).

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

## Data
- Input data is not included in this folder. Please refer to the main project README or data preparation scripts for instructions on obtaining and preprocessing the required datasets.

## Interpretation & Reporting
- See the summary file (`findings.md`) for high-level results and conclusions.
- Each results folder contains detailed outputs for the corresponding experiment.
- For questions about the AEH prior, see `AEH_prior_tradeoff.md`.

## Reproducibility
- All dependencies are listed in `requirements.txt`.
- Scripts are self-contained and save outputs to the appropriate results folders.
- For full reproducibility, ensure you use the same data splits and preprocessing as described in the main project documentation.

## Contact
For questions or further information, contact the project maintainer or refer to the main project README. 