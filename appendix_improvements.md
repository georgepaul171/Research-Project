# Appendix Improvements for Research

## **Critical Corrections Needed:**

### **1. Fix PICP Values Throughout**
**Current (incorrect):**
- 50% PICP: 0.35
- 80% PICP: 0.61  
- 90% PICP: 0.71
- 95% PICP: 0.78
- 99% PICP: 0.86

**Should be (from actual results):**
- 50% PICP: 0.93
- 80% PICP: 0.98
- 90% PICP: 0.99
- 95% PICP: 0.99
- 99% PICP: 0.99

### **2. Update Figure Paths**
Replace placeholder paths with actual generated figures:
- `correlation_heatmap.png` → `bpd/10June/results_groupprior_hmcdebug/correlation_heatmap.png`
- `calibration_plot.png` → `bpd/10June/results_groupprior_hmcdebug/calibration_plot.png`
- `residual_analysis.png` → `bpd/10June/results_groupprior_hmcdebug/residual_analysis.png`
- `shap_summary.png` → `bpd/10June/results_groupprior_hmcdebug/shap_summary.png`
- `feature_importance.png` → `bpd/10June/results_groupprior_hmcdebug/feature_importance.png`
- `prediction_vs_actual.png` → `bpd/10June/results_groupprior_hmcdebug/prediction_vs_actual.png`

### **3. Revise Trace Plot Section (C3)**
**Current issues:**
- Claims "good mixing" when traces are actually flat
- Doesn't acknowledge that flat traces are expected with AEH priors
- Misleading interpretation

**Should be revised to:**
- Acknowledge flat traces indicate poor HMC mixing
- Explain this is expected with strong priors like AEH
- Focus on convergence diagnostics (R-hat, ESS) instead
- Remove or minimize trace plot discussion

### **4. Update Performance Metrics**
**Current table shows:**
- RMSE: 7.48
- R²: 0.922

**Should verify against actual results and update if needed**

### **5. Add Missing Technical Details**

#### **A. Calibration Factor Discussion**
Add explanation of the calibration factor adjustment process:
- Initial value: 20.0 (over-conservative)
- Final value: 0.03 (well-calibrated)
- Impact on PICP values

#### **B. Model Configuration Details**
Add actual hyperparameters used:
- Calibration factor: 0.03
- HMC parameters: step_size=0.01, n_steps=10, n_chains=4
- Convergence tolerance: 1e-4
- Maximum iterations: 200

#### **C. Cross-Validation Results**
Add actual fold-by-fold results from your JSON files

### **6. Improve Theoretical Foundation**

#### **A. AEH Prior Mathematical Formulation**
The current formulation is good but could be clearer about:
- How the elastic penalty combines L1 and L2
- The momentum update mechanism
- The adaptive learning rates

#### **B. Uncertainty Calibration Section**
Add mathematical details about:
- How the calibration factor is updated
- The relationship between nominal and empirical coverage
- The impact on prediction intervals

### **7. Add Missing Analysis Sections**

#### **A. Baseline Comparison**
Add comparison with:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Standard Bayesian Ridge

#### **B. Feature Group Analysis**
Add detailed analysis of how different priors perform on different feature groups:
- Energy features (AEH prior)
- Building features (Hierarchical ARD)
- Interaction features (Spike-and-Slab)

#### **C. Computational Efficiency**
Add timing analysis:
- Training time per fold
- Prediction time
- Memory usage

### **8. Improve Data Section**

#### **A. Data Quality Metrics**
Add:
- Missing value percentages
- Outlier detection results
- Data distribution plots

#### **B. Feature Engineering Validation**
Add:
- Correlation analysis results
- Multicollinearity diagnostics
- Feature importance stability

### **9. Add Reproducibility Section**

#### **A. Random Seed Information**
- All random seeds used
- Reproducibility guarantees

#### **B. Environment Details**
- Exact package versions
- System specifications
- Installation instructions

### **10. Improve Results Presentation**

#### **A. Uncertainty Analysis**
Add:
- Uncertainty distribution plots
- Calibration curve analysis
- Reliability diagrams

#### **B. Model Comparison**
Add:
- Statistical significance tests
- Effect size measures
- Practical significance discussion

## **Priority Order:**
1. **Fix PICP values** (critical for credibility)
2. **Update figure paths** (needed for completeness)
3. **Revise trace plot section** (avoid misleading claims)
4. **Add calibration factor discussion** (important for methodology)
5. **Update performance metrics** (ensure accuracy)
6. **Add baseline comparisons** (needed for context)
7. **Improve theoretical foundation** (strengthen methodology)
8. **Add missing analysis sections** (completeness)
9. **Improve data section** (transparency)
10. **Add reproducibility section** (scientific rigor)

## **Files to Update:**
- Main appendix LaTeX file
- Figure references
- Performance metrics tables
- Theoretical formulations
- Results interpretation 