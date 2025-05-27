# Hierarchical Bayesian Neural Network (HBNN) Analysis

## Overview
The HBNNV2 implementation provides a sophisticated framework for Bayesian Neural Networks with multiple prior distributions. This document provides a detailed analysis of the implementation, its components, and the Bayesian approximations used.

## 1. Core Components

### 1.1 Prior Distributions
The implementation includes several prior distribution classes:

- **PriorDistribution**: Base class for all priors
- **NormalPrior**: Standard normal distribution
- **LaplacePrior**: Laplace distribution
- **StudentTPrior**: Student's t-distribution
- **MixtureGaussianPrior**: Mixture of Gaussian distributions

Each prior implements a `kl_divergence` method to compute the Kullback-Leibler divergence between the prior and posterior distributions.

### 1.2 Neural Network Architecture
The network architecture consists of:

- **ResidualBlock**: Implements residual connections with batch normalization and dropout
- **HierarchicalBayesianLinear**: Bayesian linear layer that samples weights from a distribution
- **EnhancedHBNN**: Main model class combining:
  - Input layer
  - Hidden layers with residual connections
  - Output layer
  - Batch normalization layers

## 2. Bayesian Neural Network Implementation

### 2.1 Weight Parameterization
Weights are parameterized using the reparameterization trick:

```python
weight_sigma = torch.log1p(torch.exp(self.weight_rho))  # Softplus transformation
bias_sigma = torch.log1p(torch.exp(self.bias_rho))
weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
```

This uses:
- `mu`: Mean of the weight distribution
- `rho`: Raw parameter for standard deviation
- Softplus transformation to ensure positive standard deviation
- Reparameterization trick for sampling: `weight = mu + sigma * epsilon` where `epsilon ~ N(0,1)`

### 2.2 Prior Distributions and KL Divergence

#### Normal Prior
```python
def kl_divergence(self, mu, sigma):
    prior_var = torch.exp(self.logvar)
    prior_mu = self.mu
    kl = torch.log(prior_var.sqrt() / sigma) + (sigma ** 2 + (mu - prior_mu) ** 2) / (2 * prior_var) - 0.5
    return kl.sum()
```

#### Laplace Prior
```python
def kl_divergence(self, mu, sigma):
    b = torch.exp(0.5 * self.logvar)
    kl = torch.log(2 * b * torch.sqrt(torch.tensor(np.e))) - torch.log(sigma) + (sigma + torch.abs(mu - self.mu)) / b - 1
    return kl.sum()
```

#### Student's t Prior
```python
def kl_divergence(self, mu, sigma):
    prior_var = torch.exp(self.logvar)
    kl = 0.5 * torch.log(prior_var / sigma**2) + \
         (self.df + 1) * torch.log(1 + (mu - self.mu)**2 / (self.df * prior_var)) - \
         (self.df + 1) * torch.log(1 + (mu - self.mu)**2 / (self.df * sigma**2))
    return kl.sum()
```

#### Mixture Gaussian Prior
```python
def kl_divergence(self, mu, sigma):
    kl = torch.zeros_like(mu)
    for m, v, w in zip(self.mus, self.logvars, self.weights):
        prior_var = torch.exp(v)
        kl += w * (torch.log(prior_var.sqrt() / sigma) +
                  (sigma ** 2 + (mu - m) ** 2) / (2 * prior_var) - 0.5)
    return kl.sum()
```

### 2.3 Loss Function
The total loss combines:
- Mean Squared Error (MSE) for prediction accuracy
- KL divergence for Bayesian regularization
```python
mse_loss = F.mse_loss(output, batch_y)
kl = torch.abs(kl)  # Ensure KL is positive
loss = mse_loss + kl_weight * kl
```

### 2.4 Uncertainty Estimation
The model estimates uncertainty through:
```python
def predict(self, x, num_samples=100):
    predictions = []
    for _ in range(num_samples):
        pred, _ = self.forward(x)
        predictions.append(pred)
    predictions = torch.stack(predictions)
    return predictions.mean(dim=0), predictions.std(dim=0)
```

## 3. Training and Evaluation

### 3.1 Training Process
The training process includes:
- KL weight scaling to balance accuracy and uncertainty
- Early stopping to prevent overfitting
- Learning rate scheduling for better convergence
- Batch normalization for training stability

### 3.2 Calibration Metrics
The implementation includes calibration metrics to evaluate uncertainty estimates:
```python
def expected_calibration_error(y_true, y_pred, y_std, n_bins=10):
    confidences = 1.96 * y_std  # 95% confidence interval
    accuracies = np.abs(y_true - y_pred) <= confidences
```

## 4. Approximations and Trade-offs

The implementation makes several approximations:
- Uses variational inference instead of full Bayesian inference
- Approximates posterior with factorized normal distributions
- Uses Monte Carlo sampling for predictions
- Approximates KL divergences for non-Gaussian priors

## 5. Output and Visualization

The code generates various outputs for each prior:
- Training curves
- Reliability diagrams
- Feature importance plots
- SHAP summary plots
- Calibration metrics
- Comparison summary CSV

## 6. Key Features and Benefits

- Bayesian uncertainty estimation
- Multiple prior distributions
- Residual connections for better gradient flow
- Batch normalization for training stability
- Dropout for regularization
- Comprehensive evaluation metrics
- Interpretability tools
- Automated experiment pipeline

## 7. Software Engineering Practices

The implementation follows good software engineering practices:
- Clear class hierarchy
- Modular design
- Comprehensive documentation
- Error handling
- Organized output structure

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