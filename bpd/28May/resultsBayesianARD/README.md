# Standard Bayesian ARD for Building Energy Prediction

## Overview
This project implements a standard Bayesian Automatic Relevance Determination (ARD) model for predicting building energy usage intensity (EUI). The model uses a linear regression framework with ARD priors to provide both accurate predictions and feature importance analysis, while maintaining interpretability through its linear structure.

## Model Architecture

### Core Components

1. **Bayesian ARD Model**
   - Implements standard Bayesian ARD with linear regression
   - Uses precision parameters (α for noise, β for weights)
   - Provides full posterior distributions for weights
   - Computes feature importance through ARD parameters

2. **Model Structure**
   - Linear regression with ARD priors
   - Full Bayesian treatment of parameters
   - Analytical posterior computation
   - Uncertainty quantification through posterior variance

### Key Features

- **Feature Selection**: ARD automatically determines feature relevance
- **Uncertainty Quantification**: Provides both predictions and uncertainty estimates
- **Interpretability**: Linear model structure allows direct interpretation of coefficients
- **Analytical Solution**: Uses Expectation-Maximization (EM) algorithm for optimization

## Training Process

### Hyperparameters
- Initial noise precision (α₀): 1e-6
- Initial weight precision (β₀): 1e-6
- Maximum iterations: 100
- Convergence tolerance: 1e-3

### Feature Engineering
The model uses the following engineered features:
- Log-transformed floor area
- Total EUI (electric + fuel)
- Energy mix ratios
- Log-transformed building age
- Normalized Energy Star rating
- Log-transformed GHG emissions intensity

## Results

### Model Performance
- RMSE: 7.100
- MAE: 4.293
- R²: 0.929
- Mean Uncertainty (std): 6.486

### Visualization
The model generates comprehensive visualizations including:
1. Predictions vs Actual Values with uncertainty bands
2. Feature importance analysis based on ARD parameters
3. Uncertainty vs Prediction Error analysis
4. Residuals plot for model diagnostics

## Implementation Details

### ARD Mechanism
The ARD implementation uses precision parameters for each feature:
- Feature importance is determined by 1/β (inverse of precision)
- Higher values indicate more important features
- The precision parameters are updated through the EM algorithm

### Uncertainty Estimation
- Uses analytical posterior variance for uncertainty estimates
- Provides both mean predictions and standard deviations
- Uncertainty estimates include both model and data uncertainty

## Usage

The model can be used for:
1. Building energy consumption prediction
2. Feature importance analysis
3. Uncertainty-aware decision making
4. Building energy efficiency assessment
5. Interpretable coefficient analysis

## Comparison with ARDBNN
While both models use ARD for feature selection, this standard Bayesian ARD model:
1. Uses a simpler linear structure
2. Provides analytical solutions
3. Offers better interpretability
4. Has faster training time
5. May capture simpler relationships more effectively

## Future Improvements
1. Implement cross-validation for more robust evaluation
2. Add more sophisticated feature engineering
3. Experiment with different prior specifications
4. Incorporate temporal aspects of building energy usage
5. Add model selection criteria (e.g., BIC, AIC) 