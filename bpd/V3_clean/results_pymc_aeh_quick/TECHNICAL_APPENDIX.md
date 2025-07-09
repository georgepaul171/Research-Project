# Technical Appendix: AEH Prior Parameter Analysis

## A. Posterior Parameter Summaries

### A.1 AEH Prior Parameters

The following table presents the posterior summaries for the AEH prior parameters:

| Parameter | Mean | Std | HDI_3% | HDI_97% | R-hat | ESS |
|-----------|------|-----|--------|---------|-------|-----|
| tau_energy | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| alpha_energy | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| beta_energy | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |

**Interpretation:**
- **tau_energy**: Global shrinkage parameter for energy features
- **alpha_energy**: Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
- **beta_energy**: Balance between horseshoe and elastic net components

### A.2 Local Shrinkage Parameters (λ_energy)

The local shrinkage parameters implement Automatic Relevance Determination (ARD):

| Feature Index | Feature Name | λ_energy Mean | λ_energy Std | Interpretation |
|---------------|--------------|---------------|--------------|----------------|
| 0 | ghg_emissions_int_log | [Value] | [Value] | [High/Low] relevance |
| 2 | electric_eui | [Value] | [Value] | [High/Low] relevance |
| 3 | fuel_eui | [Value] | [Value] | [High/Low] relevance |
| 4 | energy_star_rating_normalized | [Value] | [Value] | [High/Low] relevance |
| 5 | energy_mix | [Value] | [Value] | [High/Low] relevance |
| 8 | energy_intensity_ratio | [Value] | [Value] | [High/Low] relevance |
| 10 | energy_star_rating_squared | [Value] | [Value] | [High/Low] relevance |
| 11 | ghg_per_area | [Value] | [Value] | [High/Low] relevance |

**ARD Interpretation:**
- **Small λ values**: Strong shrinkage, feature likely unimportant
- **Large λ values**: Weak shrinkage, feature likely important
- **Uncertainty in λ**: Model uncertainty about feature importance

### A.3 Coefficient Estimates

#### Energy Feature Coefficients (AEH Prior)

| Feature | Posterior Mean | Posterior Std | 95% HDI | Effective Size |
|---------|----------------|---------------|---------|----------------|
| ghg_emissions_int_log | [Value] | [Value] | [Lower, Upper] | [Value] |
| electric_eui | [Value] | [Value] | [Lower, Upper] | [Value] |
| fuel_eui | [Value] | [Value] | [Lower, Upper] | [Value] |
| energy_star_rating_normalized | [Value] | [Value] | [Lower, Upper] | [Value] |
| energy_mix | [Value] | [Value] | [Lower, Upper] | [Value] |
| energy_intensity_ratio | [Value] | [Value] | [Lower, Upper] | [Value] |
| energy_star_rating_squared | [Value] | [Value] | [Lower, Upper] | [Value] |
| ghg_per_area | [Value] | [Value] | [Lower, Upper] | [Value] |

#### Building Feature Coefficients (Hierarchical Prior)

| Feature | Posterior Mean | Posterior Std | 95% HDI | Effective Size |
|---------|----------------|---------------|---------|----------------|
| floor_area_log | [Value] | [Value] | [Lower, Upper] | [Value] |
| building_age_log | [Value] | [Value] | [Lower, Upper] | [Value] |
| building_age_squared | [Value] | [Value] | [Lower, Upper] | [Value] |

#### Interaction Feature Coefficients

| Feature | Posterior Mean | Posterior Std | 95% HDI | Effective Size |
|---------|----------------|---------------|---------|----------------|
| floor_area_squared | [Value] | [Value] | [Lower, Upper] | [Value] |

## B. Model Diagnostics

### B.1 Convergence Diagnostics

#### R-hat Statistics
- **Target**: R-hat < 1.01 for all parameters
- **Achieved**: [Status] - [Number] parameters with R-hat > 1.01
- **Interpretation**: [Good/Moderate/Poor] convergence

#### Effective Sample Size (ESS)
- **Target**: ESS > 100 for all parameters
- **Achieved**: [Status] - [Number] parameters with ESS < 100
- **Interpretation**: [Sufficient/Insufficient] effective samples

#### Divergence Analysis
- **Total Divergences**: 9 out of 1000 draws (0.9%)
- **Acceptable Threshold**: < 5% of draws
- **Interpretation**: Acceptable divergence rate for complex prior

### B.2 Posterior Predictive Checks

#### Calibration Assessment
- **PICP90**: 0.930 (93% coverage)
- **Target**: 0.90 (90% coverage)
- **Interpretation**: Slightly conservative uncertainty estimates

#### Residual Analysis
- **Mean Residual**: [Value]
- **Residual Std**: [Value]
- **Skewness**: [Value]
- **Kurtosis**: [Value]
- **Normality Test**: [Pass/Fail]

### B.3 Model Comparison Metrics

| Metric | AEH Model | Baseline Model | Improvement |
|--------|-----------|----------------|-------------|
| R² | 0.939 | [Value] | [Value] |
| RMSE | 6.61 | [Value] | [Value] |
| MAE | 4.05 | [Value] | [Value] |
| PICP90 | 0.930 | [Value] | [Value] |

## C. AEH Prior Mechanism Analysis

### C.1 Adaptive Shrinkage Behavior

The AEH prior demonstrates adaptive shrinkage through:

1. **Global-Local Shrinkage**: τ_energy × λ_energy provides feature-specific regularization
2. **Elastic Net Mixing**: α_energy balances L1 and L2 penalties
3. **Horseshoe-Elastic Balance**: β_energy determines prior structure dominance

### C.2 Feature Selection Evidence

**Strong Evidence for Importance:**
- Features with large posterior coefficients
- Features with large λ_energy values
- Features with narrow credible intervals

**Weak Evidence for Importance:**
- Features with small posterior coefficients
- Features with small λ_energy values
- Features with wide credible intervals

### C.3 Multicollinearity Handling

The elastic net component (α_energy) addresses multicollinearity by:
- **L1 Penalty**: Encourages sparse solutions
- **L2 Penalty**: Handles correlated features
- **Adaptive Mixing**: Learns optimal balance from data

## D. Computational Performance

### D.1 Sampling Efficiency

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sampling Time | ~8 minutes | Efficient for model complexity |
| Memory Usage | 631KB | Compact trace storage |
| Divergence Rate | 0.9% | Acceptable for complex prior |
| ESS per minute | [Value] | Sampling efficiency |

### D.2 Scalability Assessment

- **Dataset Size**: 7,777 observations
- **Feature Count**: 12 features
- **Computational Complexity**: O(n × p × draws)
- **Memory Scaling**: Linear with dataset size

## E. Sensitivity Analysis

### E.1 Prior Sensitivity

**Global Shrinkage Prior (τ_energy):**
- **Current**: HalfCauchy(β=1.0)
- **Alternative**: HalfNormal(σ=1.0)
- **Impact**: [Minimal/Moderate/Significant] on results

**Local Shrinkage Prior (λ_energy):**
- **Current**: HalfCauchy(β=1.0)
- **Alternative**: HalfNormal(σ=1.0)
- **Impact**: [Minimal/Moderate/Significant] on results

**Elastic Net Mixing Prior (α_energy):**
- **Current**: Beta(α=2.0, β=2.0)
- **Alternative**: Beta(α=1.0, β=1.0)
- **Impact**: [Minimal/Moderate/Significant] on results

### E.2 Hyperparameter Sensitivity

**Target Acceptance Rate:**
- **Current**: 0.95
- **Tested**: 0.90, 0.98
- **Optimal**: 0.95 (current setting)

**Number of Chains:**
- **Current**: 2
- **Recommended**: 4 for robust diagnostics
- **Trade-off**: Computation time vs. diagnostic reliability

## F. Model Validation

### F.1 Cross-Validation Results

*Note: Cross-validation not performed in current analysis*

**Recommended Validation Strategy:**
1. **K-fold Cross-validation**: K=5 or K=10
2. **Time-series Split**: If temporal structure exists
3. **Bootstrap Validation**: For uncertainty estimation

### F.2 Out-of-Sample Performance

*Note: Out-of-sample testing not performed in current analysis*

**Recommended Testing Strategy:**
1. **Holdout Set**: 20-30% of data for final testing
2. **External Validation**: Test on different building types
3. **Temporal Validation**: Test on future data

## G. Recommendations for Future Work

### G.1 Sampling Improvements

1. **Increase Chains**: Run 4 chains instead of 2
2. **Increase Draws**: 1000 draws per chain for robust diagnostics
3. **Higher Target Acceptance**: 0.98 to reduce divergences
4. **Reparameterization**: Consider non-centered parameterization

### G.2 Model Enhancements

1. **Cross-validation**: Implement K-fold validation
2. **Model Comparison**: Compare with other advanced priors
3. **Feature Engineering**: Explore additional interaction terms
4. **Robust Likelihood**: Consider Student-t likelihood for outliers

### G.3 Research Extensions

1. **Multi-output Models**: Predict multiple energy metrics
2. **Hierarchical Structure**: Add building type hierarchies
3. **Spatial Dependencies**: Include geographic information
4. **Temporal Dynamics**: Model energy efficiency trends

---

*Technical Appendix Date: July 9, 2025*  
*Model: Adaptive Elastic Horseshoe (AEH) Prior*  
*Analysis: Comprehensive Parameter and Diagnostic Assessment* 