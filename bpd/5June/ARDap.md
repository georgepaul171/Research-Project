# Adaptive Prior Automatic Relevance Determination (ARD) Model

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Model Architecture](#model-architecture)
4. [Prior Specifications](#prior-specifications)
5. [Inference and Learning](#inference-and-learning)
6. [Uncertainty Quantification](#uncertainty-quantification)
7. [Implementation Details](#implementation-details)
8. [Feature Engineering](#feature-engineering)
9. [Model Analysis](#model-analysis)

## Introduction

The Adaptive Prior ARD model is a Bayesian linear regression model that combines ARDwith adaptive prior specifications.

### Key Features
- Hierarchical Bayesian modeling with ARD
- Adaptive prior specifications
- Uncertainty quantification with calibration
- Cross-validation for robust evaluation
- Feature interaction analysis
- Dynamic shrinkage parameters

## Key Differences from Previous Models

### 1. Adaptive Prior Structure
Unlike traditional ARD models that use fixed prior specifications, this model introduces an adaptive prior structure that adjusts to the data characteristics. This adaptation allows for:
- More flexible feature selection
- Better handling of varying feature scales
- Improved robustness to outliers

### 2. Enhanced Uncertainty Quantification
Previous models often focused primarily on point predictions, while this model provides:
- Calibrated uncertainty estimates
- Robust noise modeling using Student's t distribution
- Empirical coverage tracking
- Dynamic calibration factors

### 3. Dynamic Shrinkage Mechanism
Traditional ARD models use static shrinkage parameters, whereas this model implements:
- Dynamic shrinkage strength ($\kappa$)
- Adaptive learning rates ($\eta$)
- Feature-specific adaptation

### 4. Comprehensive Feature Analysis
While earlier models focused on basic feature selection, this model offers:
- Detailed feature interaction analysis
- Network-based feature relationship visualisation
- Partial dependence analysis
- Confidence intervals for feature importance

### 5. Robust Implementation
The model improves upon previous implementations by including:
- Advanced numerical stability measures
- Comprehensive cross-validation framework
- Sophisticated feature engineering capabilities
- Extensive diagnostic tools

## Mathematical Foundation

### 1. Basic Model Structure

The model assumes a linear relationship between features and target:

$$y = Xw + \epsilon$$

where:
- $y \in \mathbb{R}^n$ is the target vector
- $X \in \mathbb{R}^{n \times p}$ is the feature matrix
- $w \in \mathbb{R}^p$ is the weight vector
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is the noise term

### 2. Hierarchical Prior Structure

The model employs a hierarchical prior structure:

$$p(w|\alpha, \beta) = \prod_{j=1}^p \mathcal{N}(w_j|0, \beta_j^{-1})$$

where $\beta_j$ are precision parameters with their own prior:

$$p(\beta_j|\lambda_j, \tau_j) = \text{Gamma}(\beta_j|\lambda_j, \tau_j)$$

### 3. ARD Mechanism

The ARD mechanism is implemented through the precision parameters $\beta_j$. For each feature $j$:

$$\beta_j = \frac{1}{\sigma_j^2}$$

where $\sigma_j^2$ is the variance of the weight $w_j$. This allows the model to:
- Learn feature relevance automatically
- Shrink irrelevant features to zero
- Maintain uncertainty in relevant features

## Model Architecture

### 1. Core Components

The model consists of several key components:

1. **Prior Hyperparameters**
   - Global shrinkage parameters ($\lambda$)
   - Local shrinkage parameters ($\tau$)
   - Degrees of freedom ($\nu$)

2. **Dynamic Shrinkage**
   - Shrinkage strength ($\kappa$)
   - Adaptation rate ($\eta$)

3. **Uncertainty Calibration**
   - Calibration factor
   - Empirical coverage tracking

### 2. Prior Types

The model supports three types of priors:

**Hierarchical Prior**
```python
p_wj_given_beta = "Normal(0, 1 / beta_j)"
p_betaj = "Gamma(lambda_j, tau_j)"
```


**Spike-and-Slab Prior**
```python
p_wj = pi_j * Normal(0, sigma1_j) + (1 - pi_j) * Normal(0, sigma0_j)
```


**Horseshoe Prior**
```python
p_wj = Normal(0, (lambda_j**2) * (tau**2))
p_lambda_j = "Half-Cauchy(0, 1)"
p_tau = "Half-Cauchy(0, 1)"
```

### 3. Prior Selection Rationale for Energy Data

The choice of priors in this model is specifically tailored to address the unique characteristics of building energy data:

1. **Hierarchical Prior for Building Characteristics**
   - Building features (age, size, type) often exhibit strong hierarchical relationships
   - The hierarchical structure allows for:
     - Group-wise shrinkage of related building parameters
     - Automatic handling of varying scales between different building types
     - Robust handling of missing or uncertain building metadata
   - Particularly suitable for building energy data due to:
     - Natural grouping of building characteristics
     - Varying reliability of different building parameters
     - Need for adaptive shrinkage based on data quality

2. **Horseshoe Prior for Energy Features**
   - Energy consumption patterns often follow heavy-tailed distributions
   - The horseshoe prior is ideal because:
     - It handles the sparsity in energy-related features effectively
     - Provides strong shrinkage for irrelevant energy parameters
     - Maintains uncertainty for important energy drivers
   - Well-suited for energy data due to:
     - Presence of extreme energy consumption values
     - Need for robust feature selection in energy parameters
     - Handling of complex energy-related interactions

3. **Spike-Slab Prior for Interaction Terms**
   - Building-energy interactions are often sparse but important
   - The spike-slab prior is chosen because:
     - It explicitly models feature inclusion/exclusion
     - Handles the binary nature of many building-energy interactions
     - Provides clear feature selection decisions
   - Particularly valuable for energy data because:
     - Many potential interactions are irrelevant
     - Some interactions are crucial for energy performance
     - Need for interpretable feature selection in energy models

4. **Robust Noise Modeling**
   - Energy data often contains outliers and non-Gaussian noise
   - Student's t distribution is used because:
     - It handles heavy-tailed noise in energy measurements
     - Provides robustness against outliers in energy consumption data
     - Better models the uncertainty in energy measurements
   - Essential for energy data due to:
     - Measurement errors in energy meters
     - Extreme weather events affecting consumption
     - Irregular building operations

5. **Dynamic Adaptation**
   - Energy patterns change over time and across different building types
   - Dynamic shrinkage is implemented because:
     - It allows priors to adapt to local energy patterns
     - Handles varying reliability of energy data
     - Provides flexible feature selection
   - Critical for energy data because:
     - Energy consumption patterns vary by building type
     - Data quality varies across different energy parameters
     - Need for adaptive modeling of energy relationships

## Inference and Learning

### 1. Expectation-Maximization (EM) Algorithm

The model uses an EM algorithm for inference:

#### E-step
Compute posterior moments:
```python
S = np.linalg.inv(alpha * X.T @ X + np.diag(beta))
m = alpha * S @ X.T @ y
```

#### M-step
Update hyperparameters:
```python
alpha_new = n / (np.linalg.norm(y - X @ m)**2 + np.trace(X @ S @ X.T))
beta_j_new = 1 / (m[j]**2 + S[j, j] + 2 * tau_j)
```


### 2. Hamiltonian Monte Carlo (HMC)

For better posterior exploration, the model implements HMC:

1. **Hamiltonian Dynamics**
```python
def hamiltonian(w, p, U, K):
    return U(w) + K(p)
```
   where:
   - $U(w)$ is the potential energy (negative log posterior)
   - $K(p)$ is the kinetic energy

2. **Leapfrog Integration**
```python
p -= 0.5 * epsilon * grad_U(w)
w += epsilon * p
p -= 0.5 * epsilon * grad_U(w)
```

## Uncertainty Quantification

### 1. Predictive Distribution

The model provides a full predictive distribution:

```python
mean_pred = x_star.T @ m
var_pred = x_star.T @ S @ x_star + 1 / alpha
```

### 2. Uncertainty Calibration

The model implements uncertainty calibration:

1. **Empirical Coverage**
```python
coverage = np.mean((y >= y_hat - z_alpha_half * sigma_hat) & 
                   (y <= y_hat + z_alpha_half * sigma_hat))
```

2. **Calibration Factor**
```python
from scipy.optimize import minimize_scalar

def objective(c):
    return abs(empirical_coverage(c) - target_coverage)

calibration_factor = minimize_scalar(objective).x
```

### 3. Robust Noise Modeling

For robust noise modeling, the model uses Student's t distribution:


$$
p(\epsilon|\nu) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi\sigma^2}}(1 + \frac{\epsilon^2}{\nu\sigma^2})^{-(\nu+1)/2}
$$


## Implementation Details

### 1. Numerical Stability

The implementation includes several numerical stability measures:

**Clipping**
```python
beta_j = np.clip(beta_j, epsilon, np.inf)
```

**Jitter**
```python
S = np.linalg.inv(X.T @ X + np.diag(beta) + epsilon * np.eye(X.shape[1]))
```

### 2. Cross-Validation

The model uses k-fold cross-validation:

```python
cv_score = np.mean([score(y_val, model.predict(X_val)) 
                    for train_idx, val_idx in kf.split(X)])
```

### 3. Feature Selection

Feature importance is computed as:

```python
importance_j = 1 / (beta_j * lambda_j)
```

## Feature Engineering

The model includes comprehensive feature engineering:

```python
# Log Transformations
x_log = np.log1p(x)

# Interaction Terms
x_interaction = x1 * x2

# Polynomial Features
x_squared = x ** 2
```


## Model Analysis

### 1. Performance Metrics

The model tracks multiple performance metrics:

1. **Point Predictions**
   - RMSE: $\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$
   - MAE: $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
   - R²: $1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$

2. **Uncertainty Metrics**
   - CRPS: $\frac{1}{n}\sum_{i=1}^n \int_{-\infty}^{\infty} (F_i(x) - \mathbb{I}(x \geq y_i))^2 dx$
   - PICP: $\frac{1}{n}\sum_{i=1}^n \mathbb{I}(y_i \in [\hat{y}_i \pm z_{\alpha/2}\hat{\sigma}_i])$

### 2. Feature Analysis

The model provides detailed feature analysis:

1. **Importance Analysis**
   - Feature importance scores
   - Confidence intervals
   - Correlation analysis

2. **Interaction Analysis**
   - Mutual information
   - Network visualization
   - Partial dependence plots

### 3. Model Diagnostics

The model includes comprehensive diagnostics:

1. **Residual Analysis**
   - Residual plots
   - QQ plots
   - Heteroscedasticity tests

2. **Uncertainty Analysis**
   - Calibration plots
   - Reliability diagrams
   - Sharpness assessment

## Conclusion

The Adaptive Prior ARD model provides a robust framework for Bayesian regression with automatic feature selection and uncertainty quantification. Its hierarchical structure and adaptive priors make it particularly suitable for complex regression problems where feature relevance and uncertainty are important considerations.

The model's implementation includes careful attention to numerical stability, comprehensive feature engineering, and detailed analysis capabilities, making it a powerful tool for practical applications. 