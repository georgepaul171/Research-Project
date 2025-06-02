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

### CRPS Analysis
- **CRPS Value:** 3.9959
- The CRPS score of 3.9959 indicates the overall quality of the probabilistic forecasts
- A lower CRPS value would indicate better probabilistic predictions
- This relatively high CRPS value aligns with the poor PICP results, suggesting that the model's uncertainty estimates need improvement
- The CRPS takes into account both the accuracy of the mean predictions and the calibration of the uncertainty estimates

### Visual Analysis
- The `probabilistic_metrics_plots.png` shows the calibration of prediction intervals across different confidence levels
- The `uncertainty_vs_error_hist.png` shows the relationship between predicted uncertainties and actual errors, highlighting the systematic underestimation of uncertainty
- These plots provide visual confirmation of the numerical findings from PICP and CRPS metrics

### Model Limitation: Underestimated Uncertainty
- The ARD model's predicted uncertainties (`y_std`) are much smaller than the actual errors (`|y_true - y_pred|`).
- This is a common limitation of Bayesian linear models like ARD on real-world data: they only capture model and data noise as far as their assumptions (linearity, Gaussian noise) are correct.
- Real-world data often has outliers, non-Gaussian noise, and nonlinearities, leading to overconfident (too narrow) prediction intervals.

### Uncertainty Calibration
- Diagnostic plots and scaling tests show that even multiplying the predicted uncertainty by 10x does not fully calibrate the intervals.
- For better-calibrated uncertainty, consider a more flexible method.

## Conclusion
The ARD model's uncertainty estimates may not be reliable for this real-world dataset. This highlights the importance of both model choice and post-hoc calibration when applying probabilistic metrics in practice.