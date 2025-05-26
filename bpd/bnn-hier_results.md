# Hierarchical Bayesian Neural Network Results

## Data Overview
- Original dataset size: 7,777 rows
- Missing values analysis:
  - floor_area: 0 missing
  - ghg_emissions_int: 80 missing
  - fuel_eui: 0 missing
  - electric_eui: 0 missing
  - site_eui: 0 missing
- Final dataset size after preprocessing: 7,697 rows

## Training Progress
The model was trained for 100 epochs with the following key milestones:
- Initial loss (Epoch 1): Train=10.7263, Val=10.2491
- Mid training (Epoch 50): Train=0.2719, Val=0.2824
- Final loss (Epoch 100): Train=0.2615, Val=0.2649
- Best validation loss: 0.2263 (achieved at Epoch 96)

## Model Performance Metrics
- RMSE: 0.3169
- MAE: 0.2388
- R² Score: 0.9049

## Feature Importance Analysis
Features ranked by importance:
1. electric_eui: 0.2823
2. fuel_eui: 0.2201
3. floor_area: 0.2049
4. ghg_emissions_int: 0.1298

## Example Predictions with Uncertainty
Sample predictions from the validation set:
1. -0.7762 ± 0.2078
2. -0.4271 ± 0.2475
3. -0.4506 ± 0.2329
4. -0.8684 ± 0.2279
5. 1.3206 ± 0.3512

## Model Characteristics
- Hierarchical Bayesian structure
- Architecture: 4 input features → [64, 32] hidden layers → 1 output
- Training parameters:
  - Learning rate: 0.001
  - Weight decay: 1e-5
  - KL weight: 1e-3
  - Batch size: 32
  - Optimizer: Adam
  - Learning rate scheduler: ReduceLROnPlateau

## Output Files
The model generates several visualization files in the output directory (bpd/bnn-hier):
1. training_losses_hier.png
   - Training and validation loss curves
2. predictions_vs_actual_hier.png
   - Scatter plot of predictions vs actual values
3. uncertainty_distribution_hier.png
   - Distribution of prediction uncertainties
4. residual_plot.png
   - Residual analysis plot
5. feature_importance.png
   - Bar plot of feature importance scores
6. calibration_plot.png
   - Model calibration analysis

## Key Findings
1. **Model Performance**:
   - The model achieves good performance with an R² of 0.9049
   - The RMSE of 0.3169 indicates reasonable prediction accuracy
   - The MAE of 0.2388 shows good average prediction error

2. **Training Behavior**:
   - Rapid initial improvement in first 20 epochs
   - Stable convergence after epoch 50
   - No significant overfitting observed

3. **Feature Importance**:
   - electric_eui is the most influential feature
   - All features show significant importance
   - ghg_emissions_int has the lowest impact

4. **Uncertainty Quantification**:
   - Model provides reasonable uncertainty estimates
   - Uncertainty varies across predictions
   - Higher uncertainty for more extreme predictions 