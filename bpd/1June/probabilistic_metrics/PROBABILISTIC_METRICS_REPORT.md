# Probabilistic Metrics Report

## Why Include Probabilistic Metrics?
Traditional regression metrics (like MAE or RMSE) only measure the average error of point predictions. Probabilistic metrics provide:

- **Uncertainty Quantification:** How much trust to place in each prediction.
- **Risk Assessment:** Ability to make decisions under uncertainty.
- **Model Calibration:** Whether the predicted confidence intervals match the actual observed frequencies.

## Metrics Used
- **PICP (Prediction Interval Coverage Probability):** Measures the proportion of true values that fall within the predicted confidence intervals at various levels (e.g., 50%, 80%, 95%).
- **CRPS (Continuous Ranked Probability Score):** A proper scoring rule for probabilistic forecasts; lower values indicate better probabilistic predictions.
- **MAE (Mean Absolute Error) & RMSE (Root Mean Squared Error):** Standard regression metrics for point prediction accuracy.

Note: PICP and CRPS are explicitly mentioned in my literature review.

## Findings
### Numerical Results
| Confidence Level | PICP (Coverage) | Target |
|------------------|-----------------|--------|
| 50%              | 3.1%            | 50%    |
| 80%              | 6.4%            | 80%    |
| 90%              | 7.9%            | 90%    |
| 95%              | 9.3%            | 95%    |
| 99%              | 12.2%           | 99%    |

- **CRPS:** 3.9959
- The PICP values are much lower than their targets, indicating severe underestimation of uncertainty.

### Model Limitation: Underestimated Uncertainty
- The ARD model's predicted uncertainties (`y_std`) are much smaller than the actual errors (`|y_true - y_pred|`).
- This is a common limitation of Bayesian linear models like ARD on real-world data: they only capture model and data noise as far as their assumptions (linearity, Gaussian noise) are correct.
- Real-world data often has outliers, non-Gaussian noise, and nonlinearities, leading to overconfident (too narrow) prediction intervals.

### Uncertainty Calibration
- Diagnostic plots and scaling tests show that even multiplying the predicted uncertainty by 10x does not fully calibrate the intervals.
- For better-calibrated uncertainty, consider a more flexible method.

## Conclusion
The ARD model's uncertainty estimates may not be reliable for this real-world dataset. This highlights the importance of both model choice and post-hoc calibration when applying probabilistic metrics in practice.