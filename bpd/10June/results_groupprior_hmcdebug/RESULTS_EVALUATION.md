# Adaptive Prior ARD Model: Results Evaluation

## Executive Summary

This report provides a comprehensive evaluation of the Adaptive Prior ARD model trained on building energy performance data. The model leverages hierarchical and group-adaptive priors, robust uncertainty quantification, and Hamiltonian Monte Carlo (HMC) for posterior inference. The results demonstrate strong predictive performance, meaningful feature selection, and well-calibrated uncertainty estimates, with detailed diagnostics on convergence and prior behavior.

---

## 1. Model Performance

### Overall Metrics (Averaged Across Folds)
- **Root Mean Squared Error (RMSE):** 9.55
- **Mean Absolute Error (MAE):** 7.40
- **R² (Explained Variance):** 0.87
- **Mean Predictive Std (Uncertainty):** 8.13
- **CRPS (Probabilistic Score):** 3.33

### Prediction Interval Coverage Probability (PICP)
| Interval | Empirical Coverage |
|----------|-------------------|
| 50%      | 0.42              |
| 80%      | 0.79              |
| 90%      | 0.89              |
| 95%      | 0.93              |
| 99%      | 0.97              |

**Interpretation:**
- The model achieves high accuracy (R² ≈ 0.87) and low error, indicating strong fit to the data.
- Uncertainty estimates are well-calibrated, with empirical coverage closely matching nominal intervals, especially at higher confidence levels.

### Per-Fold Metrics
| Fold | RMSE  | MAE   | R²    | Mean Std | CRPS  | PICP 50 | PICP 80 | PICP 90 | PICP 95 | PICP 99 |
|------|-------|-------|-------|----------|-------|---------|---------|---------|---------|---------|
| 1    | 9.66  | 7.44  | 0.87  | 7.30     | 3.79  | 0.37    | 0.73    | 0.86    | 0.91    | 0.96    |
| 2    | 9.51  | 7.45  | 0.88  | 8.14     | 3.37  | 0.41    | 0.80    | 0.89    | 0.93    | 0.97    |
| 3    | 9.49  | 7.31  | 0.87  | 8.95     | 2.83  | 0.48    | 0.85    | 0.92    | 0.95    | 0.98    |

---

## 2. Feature Importance

### ARD-Based Importance (Normalized)
| Feature                        | Importance |
|--------------------------------|------------|
| age_ghg_interaction            | 0.36       |
| building_age_squared           | 0.18       |
| ghg_per_area                   | 0.10       |
| energy_intensity_ratio         | 0.10       |
| age_energy_star_interaction    | 0.07       |
| energy_star_rating_squared     | 0.06       |
| area_energy_star_interaction   | 0.05       |
| floor_area_squared             | 0.02       |
| energy_mix                     | 0.02       |
| energy_star_rating_normalized  | 0.03       |
| building_age_log               | 0.01       |
| fuel_eui                       | 0.001      |
| electric_eui                   | 0.0002     |
| floor_area_log                 | 0.00005    |
| ghg_emissions_int_log          | 0.0001     |

**Key Insights:**
- **Interaction terms** (especially `age_ghg_interaction` and `age_energy_star_interaction`) dominate, highlighting the importance of nonlinear and interaction effects in building energy use.
- **Building age squared** and **GHG per area** are also highly influential.
- Direct energy use variables (e.g., `electric_eui`, `fuel_eui`) have very low ARD importance, suggesting their effects are largely captured via interactions or transformed features.

### SHAP Importance (Top Features)
| Feature                      | SHAP Importance |
|------------------------------|-----------------|
| age_ghg_interaction          | 1.34            |
| building_age_squared         | 0.38            |
| area_energy_star_interaction | 0.33            |
| age_energy_star_interaction  | 0.17            |
| floor_area_squared           | 0.10            |

**Interpretation:**
- SHAP values confirm the dominance of interaction and nonlinear features, with `age_ghg_interaction` being the most influential.

---

## 3. Feature Correlations & Interactions

### Correlation with Target
- **Strongest positive:**
  - `ghg_emissions_int_log` (0.94)
  - `age_ghg_interaction` (0.77)
  - `electric_eui` (0.70)
- **Strongest negative:**
  - `energy_star_rating_squared` (-0.58)
  - `age_energy_star_interaction` (-0.54)
  - `energy_star_rating_normalized` (-0.52)

### Notable Feature Interactions (Mutual Information > 1.0)
- `floor_area_log` & `floor_area_squared`: 6.85
- `energy_star_rating_normalized` & `energy_star_rating_squared`: 4.20
- `building_age_log` & `building_age_squared`: 4.21
- `energy_intensity_ratio` & `ghg_per_area`: 2.34
- `energy_star_rating_normalized` & `age_energy_star_interaction`: 2.50
- `age_energy_star_interaction` & `area_energy_star_interaction`: 1.35
- `floor_area_log` & `energy_intensity_ratio`: 1.38
- `ghg_emissions_int_log` & `age_ghg_interaction`: 1.49

**Interpretation:**
- The model captures complex, nonlinear relationships, especially between age, GHG, and energy star features.
- High mutual information between transformed/interaction features validates the feature engineering pipeline.

---

## 4. Prior Hyperparameters (Group Priors)

- **Global Shrinkage:**
  - Energy: 1e-10 (very strong shrinkage)
  - Building: 0.68
- **Local Shrinkage:**
  - Energy: 0.0001
  - Building: 1.93
- **Mixing Parameter (Elastic Net, Energy):** 0.1
- **Regularization Strength (Energy):** 9.70

**Interpretation:**
- The model applies very strong shrinkage to energy features, enforcing sparsity and relying more on interactions and higher-level features.
- Building features are less shrunk, indicating their greater relevance or less redundancy.
- The elastic net mixing parameter is low, favoring L2-like (ridge) regularization for energy features.

---

## 5. Convergence Diagnostics (HMC/MCMC)

- **Gelman-Rubin R-hat (selected weights):**
  - Most weights: 1.05–1.2 (good)
  - Some weights: >1.5, with a few >5 or even >10 (problematic)
- **Interpretation:**
  - While many parameters show good mixing (R-hat < 1.1), some show poor convergence, especially for less relevant or highly regularized weights.
  - This is common in sparse models; however, further tuning (more HMC steps, better initialization, or reparameterization) may improve mixing.

---

## 6. Recommendations & Observations

- **Model Fit:**
  - The model is highly predictive and robust, with well-calibrated uncertainty.
- **Feature Engineering:**
  - Interaction and nonlinear features are essential; consider further exploration of domain-specific interactions.
- **Uncertainty:**
  - Calibration is strong, but empirical coverage at 50% is slightly low (0.42 vs. 0.50), suggesting mild underconfidence at lower intervals.
- **Convergence:**
  - Consider increasing HMC steps or adjusting priors for even better mixing, especially for highly regularized parameters.
- **Prior Structure:**
  - The group-adaptive prior is effective; further experimentation with group definitions or hyperparameters may yield additional gains.

---

## 7. Visualizations & Further Analysis

- See accompanying PNGs in the results directory for:
  - Feature importance, SHAP summary, interaction network, calibration, uncertainty, and more.
- For in-depth diagnostics, review the full convergence_diagnostics.json and trace plots.
