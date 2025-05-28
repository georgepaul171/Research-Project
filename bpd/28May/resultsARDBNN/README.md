# ARD Bayesian Neural Network for Building Energy Prediction

## Overview
This project implements an Automatic Relevance Determination (ARD) Bayesian Neural Network for predicting building energy usage intensity (EUI). The model combines the uncertainty quantification capabilities of Bayesian Neural Networks with ARD's feature selection capabilities to provide both accurate predictions and insights into feature importance.

## Model Architecture

### Core Components

1. **ARD Bayesian Linear Layer**
   - Implements non-centered parameterization for better training stability
   - Uses ARD parameters (α and β) to determine feature relevance
   - Includes dropout for regularization
   - Computes KL divergence for variational inference

2. **Neural Network Structure**
   - Input layer with ARD feature scaling
   - Hidden layers: [256, 128, 64] neurons
   - Output layer for EUI prediction
   - ReLU activation functions between layers
   - Dropout rate: 0.2

### Key Features

- **Uncertainty Quantification**: Provides both point predictions and uncertainty estimates
- **Feature Selection**: ARD automatically determines feature importance
- **Regularization**: Combines dropout and KL divergence for robust training
- **Adaptive Learning**: Uses ReduceLROnPlateau scheduler for learning rate adjustment

## Training Process

### Hyperparameters
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 150
- KL weight: 5e-4
- Early stopping patience: 10 epochs

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
- RMSE: 0.547
- MAE: 0.408
- R²: 0.698
- Mean Uncertainty (std): 0.341

### Visualization
The model generates comprehensive visualizations including:
1. Predictions vs Actual Values with uncertainty bands
2. Feature importance analysis
3. ARD parameter distributions
4. Uncertainty vs Prediction Error analysis

## Implementation Details

### ARD Mechanism
The ARD implementation uses two parameters (α and β) for each feature:
- Feature importance is determined by the product of softplus(α) and softplus(β)
- Higher values indicate more important features
- The scaling is applied at the input layer

### Uncertainty Estimation
- Uses Monte Carlo sampling (100 samples) for predictions
- Provides both mean predictions and standard deviations
- Uncertainty estimates correlate with prediction errors

## Usage

The model can be used for:
1. Building energy consumption prediction
2. Feature importance analysis
3. Uncertainty-aware decision making
4. Building energy efficiency assessment

## Future Improvements
1. Implement cross-validation for more robust evaluation
2. Add more sophisticated feature engineering
3. Experiment with different network architectures
4. Incorporate temporal aspects of building energy usage 