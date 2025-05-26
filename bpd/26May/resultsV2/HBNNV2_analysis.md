# HBNNV2: Hierarchical Bayesian Neural Network Experiments – Results Summary

## Overview
This report summarizes the results of automated experiments with the Hierarchical Bayesian Neural Network (HBNN) using different prior distributions: **Normal**, **Laplace**, **Student-t**, and **Mixture Gaussian**. The experiments evaluate model calibration and feature importance for each prior, enabling systematic comparison.

## Summary Table

| Prior      | ECE    | floor_area | ghg_emissions_int | fuel_eui | electric_eui |
|------------|--------|------------|-------------------|----------|--------------|
| Normal     | 0.4273 | 0.0074     | 0.0295            | 0.0138   | 0.0138       |
| Laplace    | 0.3124 | 0.0066     | 0.0579            | 0.0199   | 0.0200       |
| Student-t  | 0.1873 | 0.0003     | 0.0015            | 0.0009   | 0.0008       |
| Mixture    | 0.4411 | 0.0100     | 0.0340            | 0.0166   | 0.0160       |

- **ECE**: Expected Calibration Error (lower is better)
- Feature importance values are relative and indicate the influence of each feature on predictions

## Key Observations

- **Calibration**: The Student-t prior achieved the best calibration (lowest ECE: 0.1873), while Mixture and Normal priors had the highest ECE values (0.4411 and 0.4273).
- **Feature Importance**:
  - For all priors, **ghg_emissions_int** is the most important feature, especially pronounced for Laplace and Mixture priors.
  - **floor_area** consistently has the lowest importance across all priors.
  - Laplace and Mixture priors yield higher overall feature importance values, suggesting more confident attributions.
- **Prior Sensitivity**:
  - The choice of prior significantly affects both calibration and feature attribution.
  - Student-t prior leads to more conservative (lower) feature importance values and best calibration.

## Per-Prior Details

### Normal Prior
- **ECE**: 0.4273 (highest)
- **Feature Importance**: ghg_emissions_int > fuel_eui ≈ electric_eui > floor_area
- **Comment**: Tends to overestimate uncertainty, leading to poorer calibration.

### Laplace Prior
- **ECE**: 0.3124
- **Feature Importance**: ghg_emissions_int >> fuel_eui ≈ electric_eui > floor_area
- **Comment**: Strongest attribution to ghg_emissions_int, moderate calibration.

### Student-t Prior
- **ECE**: 0.1873 (best)
- **Feature Importance**: All features have low, similar importance; ghg_emissions_int is still highest.
- **Comment**: Most conservative, best-calibrated model.

### Mixture Prior
- **ECE**: 0.4411
- **Feature Importance**: ghg_emissions_int > fuel_eui ≈ electric_eui > floor_area
- **Comment**: Similar to Normal, but with slightly higher feature attributions.

## Recommendations
- For best calibration, use the **Student-t prior**.
- For interpretability and strong feature attributions, **Laplace** or **Mixture** priors may be preferred.
- The model consistently finds **ghg_emissions_int** to be the most predictive feature.
- Consider further tuning of priors and model architecture for improved calibration and interpretability.

## Next Steps
- Explore hybrid or hierarchical priors for further improvement.
- Investigate why floor_area is consistently less important.
- Use SHAP and reliability diagrams (see respective PNGs in each prior's folder) for deeper visual analysis. 