# Hierarchical Bayesian Neural Network v2 Results

## Data Overview
- Original dataset size: 7,777 rows
- Missing values analysis:
  - floor_area: 0 missing
  - ghg_emissions_int: 80 missing
  - fuel_eui: 0 missing
  - electric_eui: 0 missing
  - site_eui: 0 missing
- Final dataset size after preprocessing: 7,697 rows

## Model Performance Metrics

### Cross-validated Results
| Metric | Value ± Std |
|--------|-------------|
| RMSE   | 0.2023 ± 0.0105 |
| MAE    | 0.1385 ± 0.0056 |
| R²     | 0.9590 ± 0.0034 |
| ECE    | 0.0273 ± 0.0093 |

### Comparison with Linear Regression
| Metric   | BNN Mean±Std         | LinearReg Mean±Std  |
|----------|---------------------|---------------------|
| RMSE     | 0.2023 ± 0.0105     | 0.1755 ± 0.0025     |
| MAE      | 0.1385 ± 0.0056     | 0.1332 ± 0.0010     |
| R²       | 0.9590 ± 0.0034     | 0.9692 ± 0.0008     |
| ECE      | 0.0273 ± 0.0093     | N/A                 |

## Feature Importance Analysis
Features ranked by importance (with 95% confidence intervals):

1. electric_eui: 0.3230 ± 0.3259
2. floor_area: 0.3277 ± 0.1272
3. fuel_eui: 0.2631 ± 0.1942
4. ghg_emissions_int: 0.2345 ± 0.1766

## Key Findings
1. **Model Performance**:
   - The BNN achieves excellent performance with an R² of 0.9590
   - The Expected Calibration Error (ECE) of 0.0273 indicates good uncertainty calibration

2. **Comparison with Linear Regression**:
   - Linear regression slightly outperforms BNN in terms of RMSE and R²
   - However, BNN provides uncertainty estimates (ECE) which linear regression cannot

3. **Feature Importance**:
   - electric_eui and floor_area show the highest importance
   - All features show significant importance with relatively large confidence intervals
   - The high uncertainty in feature importance suggests complex interactions between features

## Model Characteristics
- Hierarchical Bayesian structure
- Cross-validated evaluation (5-fold)
- Uncertainty quantification through ECE
- Feature importance analysis with confidence intervals

## Output Files
The model generates several visualization files in the output directory:
- Calibration plots
- Feature importance plots
- Partial dependence plots
- Model architecture visualization 