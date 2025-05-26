# Hierarchical Bayesian Neural Network Further Directions (HBNNFD) Results

## Model Architecture
The model implements a Hierarchical Bayesian Neural Network with the following components:

### HierarchicalBayesianLinear Layer
- Implements a Bayesian linear layer with learnable hyperpriors
- Parameters:
  - Prior mean and log-variance (hyperpriors)
  - Weight and bias means and variances (posterior parameters)
- Uses KL divergence for regularization

### DeepHierarchicalBayesianNeuralNetwork
- Architecture:
  - Input dimension: 4 features
  - Hidden layers: [128, 64, 32] neurons
  - Output dimension: 1 (site_eui prediction)
- Activation: ReLU between layers
- Predictive uncertainty through Monte Carlo sampling

## Data Processing
- Features used:
  - floor_area
  - ghg_emissions_int
  - fuel_eui
  - electric_eui
- Target: site_eui
- Preprocessing:
  - Standardization of features and target
  - Train/validation split (80/20)
  - Missing value handling

## Training Process
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Training epochs: 50
- KL weight: 1e-3
- Loss function: MSE + KL divergence

## Model Evaluation
The model is evaluated on multiple aspects:

### 1. Predictive Performance
- RMSE (Root Mean Square Error)
- R² Score
- Validation set performance metrics

### 2. Uncertainty Quantification
- Predictive uncertainty through standard deviation of predictions
- Calibration analysis
- Out-of-Distribution (OOD) detection

### 3. Interpretability
- SHAP (SHapley Additive exPlanations) analysis
  - Uses KernelExplainer for model-agnostic interpretation
  - Background set size: 100 samples
  - Validation set subset: 200 samples
- Feature importance visualization

## Actual Results
### Training Progress
```
Epoch 10: Loss=50.1866
Epoch 20: Loss=49.5647
Epoch 30: Loss=49.2160
Epoch 40: Loss=49.0290
Epoch 50: Loss=48.8475
```

### Validation Metrics
- Validation RMSE: 0.1930
- Validation R$^2$: 0.9615

### SHAP Analysis Details
- Background set size: 100 samples
- Validation subset size: 200 samples
- Features analyzed: ['floor_area', 'ghg_emissions_int', 'fuel_eui', 'electric_eui']

### Output Files Location
All plots and results are saved in: `/Users/georgepaul/Desktop/Research-Project/bpd/further-directions`

## Output Files
The following files are generated in the output directory:

1. `ood_uncertainty.png`
   - Visualization of predictive uncertainty
   - Comparison between training and OOD data

2. `shap_summary_kernel.png`
   - SHAP summary plot
   - Feature importance visualization

## Model Characteristics
- Bayesian approach provides uncertainty estimates
- Hierarchical structure allows for better generalization
- Deep architecture captures complex relationships
- Interpretable through SHAP analysis

## Usage Notes
- The model requires PyTorch and SHAP for full functionality
- SHAP analysis may be computationally intensive
- OOD detection helps identify potential model limitations