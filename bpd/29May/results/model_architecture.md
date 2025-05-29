# ARD Model Architecture

## Core Components

### 1. Model Parameters
- **Noise Precision (α)**: Controls the model's uncertainty about the data
- **Weight Precisions (β)**: ARD parameters that determine feature importance
- **Weight Mean (m)**: Model coefficients representing feature contributions
- **Weight Covariance (S)**: Uncertainty in the weight estimates

### 2. Data Preprocessing
- **Input Features**: RobustScaler for handling outliers
- **Target Variable**: StandardScaler for normalisation
- **Feature Engineering**: 15 engineered features including:

## Distributional Assumptions

### 1. Likelihood
- Data (y) follows a normal distribution:
  ```python
  y ~ N(X @ m, 1/α * I)
  ```
- Mean: Linear predictor (X @ m)
- Variance: 1/α (noise precision)

### 2. Prior Distributions
- Weights (w) have a normal prior:
  ```python
  w ~ N(0, diag(1/β))
  ```
- Mean: Zero vector (symmetric prior)
- Variance: Diagonal matrix with elements 1/β

### 3. Posterior Distribution
- Weights posterior is also normal:
  ```python
  w|y ~ N(m, S)
  ```
- Updated through EM algorithm
- Captures uncertainty in weight estimates

## Priors and Initialisation

### 1. Current Prior Settings
- **Noise Precision Prior (α)**:
  - Initial value: 1e-6
  - Non-informative prior for noise precision
  - Updated during EM algorithm

- **Weight Precision Prior (β)**:
  - Initial value: 1e-6 for each feature
  - Non-informative prior for ARD parameters
  - Controls feature sparsity

- **Weight Mean Prior (m)**:
  - Initial value: Zero vector
  - Symmetric prior centered at zero
  - Represents no a priori preference for weight direction

- **Weight Covariance Prior (S)**:
  - Initial value: Identity matrix
  - Independent prior variances of 1 for each weight
  - Represents equal initial uncertainty across features

### 2. Prior Updates
- Updated through Expectation-Maximization (EM) algorithm
- E-step: Updates posterior distribution of weights
- M-step: Updates hyperparameters (&alpha; and &beta;)

## Training Process

### 1. Expectation-Maximization (EM) Algorithm
- **E-step**: Updates posterior distribution of weights
  ```python
  S = inv(α * X.T @ X + diag(β))
  m = α * S @ X.T @ y
  ```
- **M-step**: Updates hyperparameters
  ```python
  α_new = n_samples / (sum((y - X @ m)²) + trace(X @ S @ X.T))
  β_new = 1 / (m² + diag(S))
  ```

### 2. Cross-Validation
- 5-fold cross-validation by default
- Convergence tolerance: 1e-4
- Maximum iterations: 200

## Feature Engineering

### 1. Building Age Features
- `building_age_log`: Logarithmic transformation
- `building_age_squared`: Squared term with log transform

### 2. Energy Metrics
- `ghg_emissions_int_log`: Log-transformed GHG emissions
- `electric_eui`: Electric Energy Use Intensity
- `fuel_eui`: Fuel Energy Use Intensity

### 3. Energy Star Features
- `energy_star_rating_normalized`: Normalised rating
- `energy_star_rating_squared`: Squared term

### 4. Interaction Terms
- `age_energy_star_interaction`
- `area_energy_star_interaction`
- `age_ghg_interaction`

## Model Performance

### 1. Metrics
- **R² Score**: 0.9455
- **RMSE**: 6.2411
- **MAE**: 3.9196
- **Mean Uncertainty**: 0.2080

### 2. Feature Importance
Top 5 most important features:
1. `building_age_log` (7.0314)
2. `building_age_squared` (5.3851)
3. `ghg_emissions_int_log` (0.7400)
4. `energy_star_rating_normalized` (0.4659)
5. `ghg_per_area` (0.4293)

## Prediction and Uncertainty

### 1. Point Predictions
```python
mean = X @ m
```

### 2. Uncertainty Estimation
```python
std = sqrt(1/α + sum((X @ S) * X, axis=1))
```

## Model Persistence

### 1. Saved Components
- Model parameters (α, β, m, S)
- Feature scalers
- Configuration settings
- Cross-validation results

### 2. Loading Process
- Restores all model components
- Maintains prediction capabilities
- Preserves uncertainty estimates

## Future Improvements

### 1. Prior Selection
- Investigate different prior distributions:
  - Gamma priors 
  - Student's t 
  - Horseshoe
- Optimise hyperparameter initialisation
- Explore hierarchical priors:

### 2. Feature Engineering
- Improve floor area feature performance
- Develop additional interaction terms

### 3. Model Enhancements
- Implement better uncertainty quantification
- Add support for non-linear relationships
- Enhance cross-validation strategies 