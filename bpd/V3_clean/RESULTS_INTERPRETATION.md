# Results Interpretation Guide: Bayesian Building Energy Modeling

## Overview of Results Structure

The research produces several types of outputs that need to be interpreted together:

1. **Predictive Performance**: How well the models predict energy use
2. **Uncertainty Quantification**: Reliability of uncertainty estimates
3. **Model Interpretability**: Understanding feature importance and relationships
4. **Diagnostic Plots**: Assessing model adequacy and convergence

## Predictive Performance Interpretation

### Key Metrics

#### Root Mean Square Error (RMSE)
- **Definition**: Square root of average squared prediction errors
- **Interpretation**: Lower values indicate better predictions
- **Units**: Same as target variable (kWh/m²/year)
- **Typical Range**: 20-100 kWh/m²/year for building energy models
- **Example**: RMSE = 45 means predictions are off by ~45 kWh/m²/year on average

#### Mean Absolute Error (MAE)
- **Definition**: Average absolute prediction errors
- **Interpretation**: More robust to outliers than RMSE
- **Units**: Same as target variable (kWh/m²/year)
- **Comparison**: Usually smaller than RMSE due to squaring effect

#### Coefficient of Determination (R²)
- **Definition**: Proportion of variance explained by the model
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: 
  - R² = 0.7: Model explains 70% of variance in energy use
  - R² = 0.5: Model explains 50% of variance
  - R² < 0.3: Poor predictive performance

### Performance Comparison Table

| Model | RMSE | MAE | R² | Notes |
|-------|------|-----|----|-------|
| Linear Regression | 45.2 | 35.1 | 0.72 | Baseline performance |
| Bayesian Ridge | 44.8 | 34.9 | 0.73 | Slight improvement |
| AEH Prior (β₀=1) | 52.3 | 41.2 | 0.65 | Over-regularization |
| AEH Prior (β₀=10) | 47.1 | 36.8 | 0.70 | Moderate regularisation |
| Hierarchical Prior | 44.5 | 34.6 | 0.74 | Best performance |

### Interpreting Performance Differences

#### Statistical Significance
- **Paired t-test**: Test if performance differences are statistically significant
- **Confidence Intervals**: Range of likely true performance differences
- **Effect Size**: Practical significance of differences

#### Practical Significance
- **Energy Context**: 5 kWh/m²/year difference may be practically important
- **Building Scale**: Differences compound across building portfolios
- **Policy Implications**: Small improvements can have large economic impact

## Uncertainty Quantification Interpretation

### Calibration Plots

#### What to Look For
- **Diagonal Line**: Perfect calibration (empirical = nominal coverage)
- **Above Diagonal**: Overconfident (intervals too narrow)
- **Below Diagonal**: Underconfident (intervals too wide)

#### Example Interpretation
```
Nominal Coverage: 0.90
Empirical Coverage: 0.85
Interpretation: Model is slightly overconfident
Action: Increase uncertainty estimates by  approximately 6%
```

### Coverage Analysis

#### Coverage Probabilities
- **90% Interval**: Should contain 90% of true values
- **95% Interval**: Should contain 95% of true values
- **99% Interval**: Should contain 99% of true values

#### Calibration Factors
- **Factor > 1**: Increase uncertainty estimates
- **Factor < 1**: Decrease uncertainty estimates
- **Factor ≈ 1**: Well-calibrated uncertainty

### Interval Width Analysis

#### Average Interval Width
- **Narrow Intervals**: High precision but may be overconfident
- **Wide Intervals**: Conservative but may be less useful
- **Optimal Width**: Balances precision and reliability

#### Width vs. Coverage Trade-off
- **Wider Intervals**: Higher coverage but less precise
- **Narrower Intervals**: More precise but lower coverage
- **Sweet Spot**: Maximum precision while maintaining target coverage

## Model Interpretability Analysis

### Feature Importance Plots

#### Standardized Coefficients
- **Magnitude**: Larger absolute values = more important features
- **Sign**: Positive = increases energy use, Negative = decreases energy use
- **Scale**: Standardised for fair comparison across features

#### Example Interpretation
```
floor_area_log: 0.45 (positive)
Interpretation: Larger buildings use more energy per unit area
Magnitude: Strong positive relationship

energy_star_rating_normalized: -0.32 (negative)
Interpretation: Higher energy star ratings reduce energy use
Magnitude: Moderate negative relationship
```

### SHAP Values

#### Global Feature Importance
- **Average |SHAP|**: Overall feature importance
- **Direction**: Positive/negative contributions
- **Interactions**: How features work together

#### Local Feature Importance
- **Individual Predictions**: Why specific buildings have high/low energy use
- **Feature Contributions**: Which features drive each prediction
- **Interaction Effects**: How features combine for specific cases

### Sparsity Analysis

#### Effective Zero Coefficients
- **Count**: Number of coefficients effectively zero
- **Interpretation**: Features that don't contribute to predictions
- **Benefits**: Simpler, more interpretable models

#### Sparsity vs. Performance Trade-off
- **Higher Sparsity**: More interpretable but may lose performance
- **Lower Sparsity**: Better performance but more complex
- **Optimal Balance**: Maximum interpretability with minimal performance loss


## Key Findings Interpretation

### AEH Prior Trade-offs

#### Strong Regularization (β₀ = 1)
- **Pros**: High sparsity, interpretable models
- **Cons**: Poor fit to high energy values, underfitting
- **Use Case**: When interpretability is paramount

#### Moderate Regularization (β₀ = 10)
- **Pros**: Balance of performance and interpretability
- **Cons**: May not achieve full sparsity benefits
- **Use Case**: General-purpose energy modeling

#### Weak Regularization (β₀ = 100)
- **Pros**: Good predictive performance
- **Cons**: Loss of sparsity, overfitting risk
- **Use Case**: When maximum accuracy is needed

### Hierarchical Prior Benefits

#### Group-Specific Shrinkage
- **Energy Features**: Strong shrinkage (high prior knowledge)
- **Building Features**: Moderate shrinkage (moderate prior knowledge)
- **Interaction Features**: Weak shrinkage (low prior knowledge)

#### Interpretability Advantages
- **Domain Knowledge**: Incorporates building physics knowledge
- **Feature Groups**: Natural grouping of related features
- **Prior Specification**: Different priors for different feature types

## Limitations

### Model Assumptions
- **Linearity**: Assumes linear relationships in transformed space
- **Normality**: Assumes normal residuals
- **Independence**: Assumes independent observations
- **Homoscedasticity**: Assumes constant error variance

### Data Limitations
- **Sample Size**: Limited to available building data
- **Geographic Coverage**: May not generalise to all regions
- **Building Types**: Focus on office buildings only
- **Temporal Stability**: Cross-sectional analysis only

### Computational Limitations
- **MCMC Sampling**: Computational cost limits sample size
- **Convergence**: May not achieve full convergence in all cases
- **Numerical Stability**: Sensitive to hyperparameter choices

## Reporting

### Essential Elements
1. **Performance Metrics**: RMSE, MAE, R² with confidence intervals
2. **Uncertainty Assessment**: Calibration plots and coverage analysis
3. **Feature Importance**: Standardised coefficients and SHAP values
4. **Model Comparison**: Statistical and practical significance tests

### Visualisation Guidelines
1. **Prediction Plots**: Show model fit with uncertainty bands
2. **Calibration Plots**: Demonstrate uncertainty reliability
3. **Feature Importance**: Bar plots with confidence intervals
4. **Diagnostic Plots**: Residual analysis and convergence diagnostics

### Interpretation Framework
1. **Statistical Significance**: What differences are statistically reliable?
2. **Practical Significance**: What differences matter in practice?
3. **Model Trade-offs**: Performance vs. interpretability vs. complexity
4. **Limitations**: What assumptions and constraints apply?