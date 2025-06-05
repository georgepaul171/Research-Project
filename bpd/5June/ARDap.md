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

The Adaptive Prior ARD model is a sophisticated Bayesian linear regression model that combines Automatic Relevance Determination (ARD) with adaptive prior specifications. This model is particularly well-suited for problems requiring robust feature selection, uncertainty quantification, and handling of complex data structures.

### Key Features
- Hierarchical Bayesian modeling with ARD
- Adaptive prior specifications
- Uncertainty quantification with calibration
- Robust noise modeling
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

1. **Hierarchical Prior**
   ```python
   p(w_j|\beta_j) = \mathcal{N}(w_j|0, \beta_j^{-1})
   p(\beta_j|\lambda_j, \tau_j) = \text{Gamma}(\beta_j|\lambda_j, \tau_j)
   ```

2. **Spike-and-Slab Prior**
   ```python
   p(w_j|\pi_j, \sigma^2_{0j}, \sigma^2_{1j}) = \pi_j\mathcal{N}(w_j|0, \sigma^2_{1j}) + (1-\pi_j)\mathcal{N}(w_j|0, \sigma^2_{0j})
   ```

3. **Horseshoe Prior**
   ```python
   p(w_j|\lambda_j, \tau) = \mathcal{N}(w_j|0, \lambda_j^2\tau^2)
   p(\lambda_j) = \text{Half-Cauchy}(0,1)
   p(\tau) = \text{Half-Cauchy}(0,1)
   ```

## Inference and Learning

### 1. Expectation-Maximization (EM) Algorithm

The model uses an EM algorithm for inference:

#### E-step
Compute posterior moments:
```python
S = (\alpha X^T X + \text{diag}(\beta))^{-1}
m = \alpha S X^T y
```

#### M-step
Update hyperparameters:
```python
\alpha_{new} = \frac{n}{\|y - Xm\|^2 + \text{tr}(XSX^T)}
\beta_{j,new} = \frac{1}{m_j^2 + S_{jj} + 2\tau_j}
```

### 2. Hamiltonian Monte Carlo (HMC)

For better posterior exploration, the model implements HMC:

1. **Hamiltonian Dynamics**
   ```python
   H(w,p) = U(w) + K(p)
   ```
   where:
   - $U(w)$ is the potential energy (negative log posterior)
   - $K(p)$ is the kinetic energy

2. **Leapfrog Integration**
   ```python
   p_{t+\epsilon/2} = p_t - \frac{\epsilon}{2}\nabla U(w_t)
   w_{t+\epsilon} = w_t + \epsilon p_{t+\epsilon/2}
   p_{t+\epsilon} = p_{t+\epsilon/2} - \frac{\epsilon}{2}\nabla U(w_{t+\epsilon})
   ```

## Uncertainty Quantification

### 1. Predictive Distribution

The model provides a full predictive distribution:

```python
p(y_*|x_*, X, y) = \mathcal{N}(y_*|x_*^T m, x_*^T S x_* + \alpha^{-1})
```

### 2. Uncertainty Calibration

The model implements uncertainty calibration:

1. **Empirical Coverage**
   ```python
   \text{coverage} = \frac{1}{n}\sum_{i=1}^n \mathbb{I}(y_i \in [\hat{y}_i \pm z_{\alpha/2}\hat{\sigma}_i])
   ```

2. **Calibration Factor**
   ```python
   \text{calibration\_factor} = \text{argmin}_c \|\text{coverage} - \text{target\_coverage}\|
   ```

### 3. Robust Noise Modeling

For robust noise modeling, the model uses Student's t distribution:

```python
p(\epsilon|\nu) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi\sigma^2}}(1 + \frac{\epsilon^2}{\nu\sigma^2})^{-(\nu+1)/2}
```

## Implementation Details

### 1. Numerical Stability

The implementation includes several numerical stability measures:

1. **Clipping**
   ```python
   \beta_j = \text{clip}(\beta_j, \epsilon, \infty)
   ```

2. **Jitter**
   ```python
   S = (X^T X + \text{diag}(\beta) + \epsilon I)^{-1}
   ```

### 2. Cross-Validation

The model uses k-fold cross-validation:

```python
\text{CV score} = \frac{1}{k}\sum_{i=1}^k \text{score}(y_i, \hat{y}_i)
```

### 3. Feature Selection

Feature importance is computed as:

```python
\text{importance}_j = \frac{1}{\beta_j \lambda_j}
```

## Feature Engineering

The model includes comprehensive feature engineering:

1. **Log Transformations**
   ```python
   x_{log} = \log(1 + x)
   ```

2. **Interaction Terms**
   ```python
   x_{interaction} = x_1 \cdot x_2
   ```

3. **Polynomial Features**
   ```python
   x_{squared} = x^2
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