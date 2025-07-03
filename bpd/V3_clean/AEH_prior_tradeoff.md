# Trade-off Between AEH Prior Strength and Model Flexibility

## Background
The Adaptive Elastic Horseshoe (AEH) prior is designed to provide strong regularization and automatic feature selection in Bayesian regression models. The strength of this prior is controlled by the parameter `beta_0`, which sets the prior variance for the model weights.

## Experimental Findings
- **Small `beta_0` (e.g., 1.0):**
  - Strong regularization.
  - Model predictions are highly restricted and cannot fit high target values.
  - Uncertainty intervals are reasonable, but the model underfits the data, especially at the high end.
- **Moderate `beta_0` (e.g., 10.0):**
  - Weaker regularization.
  - Model predictions improve and cover a wider range of target values, but still do not reach the highest values in the data.
- **Large `beta_0` (e.g., 100.0+):**
  - Very weak prior; model behaves more like standard linear regression.
  - Model can fit the full range of target values, but the benefits of Bayesian regularization and sparsity are lost.
  - Risk of overfitting increases, and the model may become numerically unstable.

## Trade-off
- **Low `beta_0`:** Strong Bayesian regularization, but poor fit to high values.
- **High `beta_0`:** Good fit to high values, but little regularization and loss of AEH benefits.

| `beta_0` Value | Regularization | Fit to High Values | Overfitting Risk |
|---------------|---------------|-------------------|------------------|
| 1.0           | Strong        | Poor              | Low              |
| 10.0          | Moderate      | Better            | Moderate         |
| 100.0+        | Weak/None     | Best              | Higher           |

## Recommendation
For this project, a **moderate value of `beta_0` (e.g., 10.0)** is recommended as a practical trade-off:
- It allows the model to fit a wider range of target values than a very strong prior.
- It retains some of the regularization and feature selection benefits of the AEH prior.
- It avoids the risk of overfitting and numerical instability associated with very large `beta_0`.

If the primary goal is interpretability and feature selection, use a lower `beta_0`. If the goal is predictive accuracy across the full range, use a higher value, but document the trade-off.

---

*This document summarizes the effect of AEH prior strength on model performance and provides guidance for future experiments and reporting.* 