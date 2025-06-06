This .md file presents the findings from the ARD analysis.

## Model Performance
The model demonstrated strong predictive capabilities:
- **R² Score**: 0.9455
- **RMSE**: 6.2411
- **MAE**: 3.9196
- **Mean Uncertainty**: 0.2080

These indicate:
- High accuracy in predictions
- Low prediction error
- Good model confidence
- Consistent performance across cross-validation folds

## Feature Importance Analysis

### Top 5 Most Important Features
1. `building_age_log` (7.0314)
2. `building_age_squared` (5.3851)
3. `ghg_emissions_int_log` (0.7400)
4. `energy_star_rating_normalized` (0.4659)
5. `ghg_per_area` (0.4293)

### Features with Low Importance
- `energy_mix` (0.0003)
- `energy_star_rating_squared` (0.0046)
- `floor_area_squared` (0.0164) 

## Feature Correlations with Target

### Strong Positive Correlations
- `ghg_emissions_int_log`: 0.9389
- `electric_eui`: 0.6983
- `age_ghg_interaction`: 0.7676
- `fuel_eui`: 0.6256

### Strong Negative Correlations
- `energy_star_rating_squared`: -0.5779
- `age_energy_star_interaction`: -0.5352
- `energy_star_rating_normalized`: -0.5151
- `area_energy_star_interaction`: -0.4350

## Feature Variance Analysis

### High Variance Features
- `electric_eui`: 352.7712
- `fuel_eui`: 287.6976
- `area_energy_star_interaction`: 8.0131
- `age_ghg_interaction`: 2.3206
- `floor_area_squared`: 4.5792

### Low Variance Features
- `energy_intensity_ratio`: 0.0000
- `ghg_per_area`: 0.0000
- `energy_mix`: 0.0032
- `energy_star_rating_normalized`: 0.0506

## Insights

1. **Building Age Impact**
   - Building age features (both log and squared) are the strongest predictors. I would say this is expected.

2. **GHG Emissions**
   - Strong correlation with target variable
   - Important predictor of energy performance
   - Log transformation improves predictive power

3. **Energy Star Rating**
   - Negative correlations suggest inverse relationship with energy consumption. I would say this is expected.
   - Normalised rating more important than squared term

4. **Feature Interactions**
   - Age-GHG interaction shows strong positive correlation
   - Age-Energy Star interaction shows strong negative correlation
   - Area-Energy Star interaction shows moderate negative correlation

## Recommendations & Thoughts

1. **Feature Selection**
   - Consider removing low-importance features:
     - `energy_mix`
     - `energy_star_rating_squared`
     - `floor_area_squared`, which I need to work on.

2. **Feature Engineering**
   - Focus on building age transformations
   - Maintain GHG emissions log transformation
   - Consider additional interaction terms

3. **Model Improvements**
   - Address high variance in energy metrics
   - Consider feature scaling for high-variance features
   - Explore non-linear relationships further

## Conclusion

The analysis includes several visualisation files. The model provides good predictive performance while offering  nsights into feature importance and interactions. The strong performance of building age features and GHG emissions suggests these are key factors in building energy performance. The model's high R² score and low error metrics indicate its reliability for energy performance prediction. 

## Further Work

- I want to improve the predictive performance of floor_area feature.
- Another area to investigate is the prior selection - talk to Mike