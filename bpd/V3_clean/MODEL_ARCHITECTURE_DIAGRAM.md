# Group-wise Bayesian Model Architecture with Tailored Prior Assignments

## Overview
This diagram illustrates the hierarchical structure of the Bayesian linear regression framework, showing how features are partitioned into groups and assigned specialized priors (Adaptive Elastic Horseshoe, Hierarchical ARD, Spike-Slab).

## Model Architecture Diagram

```mermaid
graph TB
    %% Input Layer
    subgraph Input[Input Features]
        X1[floor_area_log]
        X2[electric_eui]
        X3[fuel_eui]
        X4[energy_star_rating_normalized]
        X5[building_age_log]
        X6[ghg_emissions_int_log]
        X7[energy_mix]
        X8[energy_intensity_ratio]
        X9[floor_area_squared]
        X10[building_age_squared]
        X11[energy_star_rating_squared]
        X12[ghg_per_area]
    end
    
    %% Feature Grouping
    subgraph Groups[Feature Groups]
        G1[Energy Features]
        G2[Building Features]
        G3[Environmental Features]
        G4[Interaction Features]
    end
    
    %% Prior Assignments
    subgraph Priors[Prior Specifications]
        P1[Adaptive Elastic Horseshoe<br/>AEH Prior]
        P2[Hierarchical ARD<br/>Normal-InverseGamma]
        P3[Spike-Slab<br/>Mixture Prior]
    end
    
    %% Model Components
    subgraph Model[Bayesian Model Components]
        M1[Likelihood<br/>y_i ~ Normal(μ_i, σ²)]
        M2[Linear Predictor<br/>μ_i = X_i^T β]
        M3[Posterior Sampling<br/>HMC]
        M4[EM Algorithm<br/>Parameter Updates]
    end
    
    %% Output
    subgraph Output[Model Outputs]
        O1[Predictions<br/>ŷ_i]
        O2[Uncertainty<br/>Prediction Intervals]
        O3[Feature Importance<br/>|β_j|]
        O4[Model Diagnostics<br/>Trace Plots]
    end
    
    %% Connections from Input to Groups
    X2 --> G1
    X3 --> G1
    X7 --> G1
    X8 --> G1
    
    X1 --> G2
    X5 --> G2
    X4 --> G2
    
    X6 --> G3
    X12 --> G3
    
    X9 --> G4
    X10 --> G4
    X11 --> G4
    
    %% Connections from Groups to Priors
    G1 --> P1
    G2 --> P2
    G3 --> P2
    G4 --> P3
    
    %% Connections from Priors to Model
    P1 --> M1
    P2 --> M1
    P3 --> M1
    
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> M3
    
    %% Connections from Model to Output
    M3 --> O1
    M3 --> O2
    M3 --> O3
    M3 --> O4
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef group fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef prior fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef model fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12 input
    class G1,G2,G3,G4 group
    class P1,P2,P3 prior
    class M1,M2,M3,M4 model
    class O1,O2,O3,O4 output
```

## Prior Specifications

### 1. Adaptive Elastic Horseshoe (AEH) Prior
**Applied to: Energy Features**
- **Features**: electric_eui, fuel_eui, energy_mix, energy_intensity_ratio
- **Mathematical Form**:
  ```
  β_j ~ Normal(0, τ²λ_j²)
  λ_j ~ Half-Cauchy(0, 1)
  τ ~ Half-Cauchy(0, τ_0)
  τ_0 ~ Gamma(a, b)
  ```
- **Rationale**: Heavy-tailed distribution suitable for energy consumption patterns with extreme values

### 2. Hierarchical ARD Prior
**Applied to: Building & Environmental Features**
- **Building Features**: floor_area_log, building_age_log, energy_star_rating_normalized
- **Environmental Features**: ghg_emissions_int_log, ghg_per_area
- **Mathematical Form**:
  ```
  β_j ~ Normal(0, σ_j²)
  σ_j² ~ InverseGamma(α_j, β_j)
  ```
- **Rationale**: Captures hierarchical structure while maintaining interpretability

### 3. Spike-Slab Prior
**Applied to: Interaction Features**
- **Features**: floor_area_squared, building_age_squared, energy_star_rating_squared
- **Mathematical Form**:
  ```
  β_j ~ (1 - π_j) δ_0 + π_j Normal(0, σ_j²)
  π_j ~ Beta(a_j, b_j)
  ```
- **Rationale**: Automatic feature selection for interaction terms

## Model Components

### Likelihood Function
```
y_i ~ Normal(μ_i, σ²)
```
Where y_i is the site energy use intensity for building i.

### Linear Predictor
```
μ_i = X_i^T β
```
Where X_i is the feature vector and β is the coefficient vector.

### Posterior Sampling
- **Method**: Hamiltonian Monte Carlo (HMC)
- **Purpose**: Efficient exploration of posterior distribution
- **Benefits**: Handles complex posterior geometries

### EM Algorithm
- **E-Step**: Posterior sampling via HMC
- **M-Step**: Parameter updates based on posterior samples
- **Convergence**: Monitored through trace diagnostics

## Feature Group Details

### Energy Features (G₁)
- **electric_eui**: Electricity Energy Use Intensity
- **fuel_eui**: Fuel Energy Use Intensity  
- **energy_mix**: Energy source complexity metric
- **energy_intensity_ratio**: Efficiency metric
- **Prior**: Adaptive Elastic Horseshoe
- **Rationale**: Energy consumption patterns often show heavy tails and extreme values

### Building Features (G₂)
- **floor_area_log**: Log-transformed floor area
- **building_age_log**: Log-transformed building age
- **energy_star_rating_normalized**: Normalized Energy Star rating
- **Prior**: Hierarchical ARD
- **Rationale**: Building characteristics have natural hierarchical structure

### Environmental Features (G₃)
- **ghg_emissions_int_log**: Log-transformed GHG emissions intensity
- **ghg_per_area**: Area-normalized emissions
- **Prior**: Hierarchical ARD
- **Rationale**: Environmental metrics related to building characteristics

### Interaction Features (G₄)
- **floor_area_squared**: Quadratic floor area effects
- **building_age_squared**: Quadratic age effects
- **energy_star_rating_squared**: Quadratic rating effects
- **Prior**: Spike-Slab
- **Rationale**: Automatic selection of relevant interaction terms

## Model Outputs

### Predictions
- **Point Predictions**: ŷ_i = E[y_i | X_i, data]
- **Uncertainty Quantification**: Prediction intervals

### Feature Analysis
- **Feature Importance**: |β_j| for each feature
- **Uncertainty**: Posterior standard deviations
- **Selection**: Effective feature inclusion probabilities

### Diagnostics
- **Trace Plots**: MCMC convergence monitoring
- **Calibration**: Uncertainty calibration assessment
- **Model Fit**: Residual analysis and goodness-of-fit

---

*This diagram provides a comprehensive view of the group-wise Bayesian model architecture, showing how different feature groups are assigned specialized priors based on their characteristics and modeling requirements.* 