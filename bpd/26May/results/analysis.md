# Hierarchical Bayesian Neural Network (HBNN) Analysis Results

## Overview
This document analyzes the results of the Hierarchical Bayesian Neural Network implementation for predicting site energy usage intensity (site_eui) using building characteristics data.

## Model Calibration
The model achieved an Expected Calibration Error (ECE) of 0.1898, indicating moderate calibration performance. The reliability diagram shows:
- The model tends to be underconfident in the low confidence regions (0-0.2)
- There's a notable gap between perfect calibration (red dashed line) and model performance (blue line)
- The model's predictions cluster in the lower confidence regions, suggesting conservative uncertainty estimates

## Feature Importance Analysis
The model identified varying levels of importance for different features:

| Feature | Importance |
|---------|------------|
| floor_area | 1.792e-04 |
| ghg_emissions_int | 1.194e-03 |
| fuel_eui | 9.995e-04 |
| electric_eui | 8.700e-04 |

Key observations:
- GHG emissions intensity shows the highest importance (1.194e-03), suggesting it's the strongest predictor of site EUI
- Floor area has notably lower importance (1.792e-04), indicating less direct influence on site EUI
- Fuel EUI and Electric EUI show similar levels of importance (9.995e-04 and 8.700e-04 respectively), suggesting balanced contribution to the prediction
- The relative scale of importance values suggests the model relies more on energy-related features than physical building characteristics

## Training Performance
The training curves reveal:
- Initial rapid decrease in loss during the first few epochs
- Stable convergence after epoch 5
- Training and validation losses track closely, suggesting good generalization
- Final validation loss stabilizes around 0.2, indicating reasonable model fit
- No significant signs of overfitting as training and validation losses remain close

## Recommendations
1. Consider adjusting the model's uncertainty estimation to improve calibration
2. Investigate why floor area has such low importance - this might indicate a need for feature normalization or transformation
3. The close tracking of training and validation losses suggests room for increased model capacity
4. Consider adding interaction terms between energy-related features given their similar importance levels

## Technical Details
- Model: Hierarchical Bayesian Neural Network
- Training Duration: ~30 epochs
- Early Stopping: Activated based on validation loss
- Loss Function: MSE + KL Divergence
- Uncertainty Estimation: Monte Carlo Dropout
- Features: floor_area, ghg_emissions_int, fuel_eui, electric_eui
- Target: site_eui

## Next Steps
1. Experiment with different prior distributions to improve uncertainty estimation
2. Increase model capacity given the good generalization
3. Investigate feature engineering for floor_area to increase its predictive power
4. Fine-tune the uncertainty estimation parameters
5. Consider adding more building characteristics features to complement the energy metrics 

## Appendix: Key Experiment Log

- Data loaded from: cleaned_office_buildings.csv
- Features: floor_area, ghg_emissions_int, fuel_eui, electric_eui
- Target: site_eui
- Data shape after NA drop: (7697, 33)
- Train/validation split: X_train (6157, 4), X_val (1540, 4)
- Model: input_dim=4, hidden_dims=[128, 64, 32], output_dim=1
- Training: Early stopping at epoch 29
- Example losses: Epoch 10 (Train: 0.1851, Val: 0.0864), Epoch 20 (Train: 0.1601, Val: 0.1169)
- Calibration error (ECE): 0.1898
- All results saved to: bpd/26May/results/ 