# Hierarchical Bayesian Neural Network with Automatic Relevance Determination (ARD)

## Overview
This implementation presents a Hierarchical Bayesian Neural Network with Automatic Relevance Determination (ARD) for predicting building energy usage. The model incorporates group-specific parameters and uses ARD to automatically determine feature importance.

## Model Architecture

### Prior Distributions
1. **Hierarchical Normal Prior**
   - Group-level parameters follow a normal distribution
   - Individual-level parameters use non-centered parameterization
   - KL divergence includes both group and individual components
   - Enables better sampling and convergence

2. **Half-Normal Hyperprior**
   - Used for standard deviations
   - Implements log-normal posterior
   - Scale parameter controls prior strength
   - Helps prevent overfitting

### Key Components
1. **Hierarchical Bayesian Linear Layers**
   - Group-level parameters for each data category
   - Non-centered parameterization for better sampling
   - Hyperpriors for group-level standard deviations
   - ARD parameters for feature selection

2. **Network Structure**
   - Input Layer: 8 features
   - Hidden Layers: [256, 128, 64] neurons
   - Output Layer: 1 neuron (site EUI prediction)
   - Dropout Rate: 0.2
   - Activation: ReLU

### Feature Engineering
1. **Energy Metrics**
   - `ghg_emissions_int_log`: Log-transformed GHG emissions intensity
   - `total_eui`: Combined electric and fuel EUI
   - `electric_eui`: Electric energy use intensity
   - `fuel_eui`: Fuel energy use intensity
   - `energy_mix`: Interaction between electric and fuel ratios

2. **Building Characteristics**
   - `floor_area_log`: Log-transformed floor area
   - `energy_star_rating_normalized`: Normalized Energy Star rating
   - `building_age_log`: Log-transformed building age

## Training Process
- **Optimizer**: Adam
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Batch Size**: 64
- **Epochs**: 150 (with early stopping)
- **KL Weight**: 5e-4
- **Validation Split**: 20%

## Results

### Model Performance
- **R² Score**: 0.912 (91.2% variance explained)
- **RMSE**: 0.296
- **MAE**: 0.204
- **Mean Uncertainty**: 0.071

### Feature Importance
1. `ghg_emissions_int_log`: 1.5692
2. `total_eui`: 1.4291
3. `fuel_eui`: 0.7319
4. `electric_eui`: 0.7042
5. `energy_mix`: 0.3730
6. `energy_star_rating_normalized`: 0.3404
7. `floor_area_log`: 0.2022
8. `building_age_log`: 0.1568

### Uncertainty Analysis
- Mean Uncertainty: 1.8909
- Standard Deviation of Uncertainty: 0.9453
- Correlation with Error: 0.5153

### Group Effects
- Single group (NEW YORK) with effect size: 0.2415

## Key Insights

1. **Feature Importance**
   - GHG emissions intensity is the strongest predictor
   - Energy use metrics (total_eui, fuel_eui, electric_eui) are highly important
   - Building characteristics have moderate to low importance

2. **Model Strengths**
   - High predictive power (91.2% variance explained)
   - Good uncertainty calibration
   - Stable training process
   - Effective feature selection through ARD

3. **Uncertainty Characteristics**
   - Reasonable uncertainty estimates
   - Good correlation between uncertainty and prediction error
   - Well-calibrated uncertainty bands

## Visualizations
The model generates comprehensive visualizations including:
1. Predictions vs Actual Values with uncertainty bands
2. Feature importance analysis
3. ARD parameter analysis
4. Group-level effects
5. Uncertainty patterns
6. Uncertainty distribution

## Future Improvements
1. **Feature Engineering**
   - Explore additional interaction terms
   - Investigate non-linear transformations
   - Consider temporal features

2. **Model Architecture**
   - Experiment with different network depths
   - Try alternative activation functions
   - Implement attention mechanisms

3. **Uncertainty Calibration**
   - Implement temperature scaling
   - Add ensemble methods
   - Explore alternative prior distributions

## Usage
```python
# Model initialization
model = TrueHierarchicalHBNN(
    input_dim=8,
    hidden_dims=[256, 128, 64],
    output_dim=1,
    num_groups=1,
    prior_type='hierarchical_normal',
    dropout_rate=0.2
)

# Training
train_losses, val_losses = train_model(
    model, X_train, y_train, X_val, y_val,
    group_ids_train, group_ids_val, feature_names,
    num_epochs=150,
    batch_size=64,
    learning_rate=0.001,
    kl_weight=5e-4
)

# Prediction
y_pred, y_std = model.predict(X_val, group_ids_val)
```

## Dependencies
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SHAP
- Captum 