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