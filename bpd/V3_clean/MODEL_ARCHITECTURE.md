# Model Architecture: Adaptive Bayesian Regression with AEH Prior

## Overview
This document provides a detailed visual representation of the model architecture for the Adaptive Bayesian Regression project, including the hierarchical structure, prior specifications, and inference mechanisms.

## Model Architecture Diagram

```mermaid
graph TD
    %% Data Layer
    A[Input Features] --> B[Feature Scaling]
    B --> C[Feature Groups]
    
    %% Feature Groups
    C --> D[Energy Features<br/>4 features]
    C --> E[Building Features<br/>4 features]
    C --> F[Interaction Features<br/>4 features]
    
    %% Prior Specifications
    D --> G[AEH Prior<br/>Adaptive Elastic Horseshoe]
    E --> H[Hierarchical Prior<br/>Normal Distribution]
    F --> I[Hierarchical Prior<br/>Normal Distribution]
    
    %% AEH Prior Components
    G --> G1[Global Shrinkage τ]
    G --> G2[Local Shrinkage λj]
    G --> G3[Elastic Net α]
    G --> G4[Adaptive Parameters<br/>β, γ, δ]
    
    %% Hierarchical Prior Components
    H --> H1[Group Mean μb]
    H --> H2[Group Variance σb²]
    I --> I1[Group Mean μi]
    I --> I2[Group Variance σi²]
    
    %% Model Parameters
    G1 --> J[Regression Coefficients β]
    G2 --> J
    G3 --> J
    G4 --> J
    H1 --> J
    H2 --> J
    I1 --> J
    I2 --> J
    
    %% Likelihood
    J --> K[Linear Predictor]
    K --> L[Gaussian Likelihood<br/>y ~ N(Xβ, σ²)]
    
    %% Inference
    L --> M[EM Algorithm]
    M --> N[Parameter Updates]
    N --> O[Convergence Check]
    O --> P{Converged?}
    P -->|No| M
    P -->|Yes| Q[Final Model]
    
    %% Output
    Q --> R[Predictions]
    Q --> S[Uncertainty Estimates]
    Q --> T[Feature Importance]
    
    %% Styling
    classDef dataLayer fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef featureGroup fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef aePrior fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef hierPrior fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef model fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef inference fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    
    class A,B,C dataLayer
    class D,E,F featureGroup
    class G,G1,G2,G3,G4 aePrior
    class H,H1,H2,I,I1,I2 hierPrior
    class J,K,L model
    class M,N,O,P inference
    class Q,R,S,T output
```

## Model Components

### 1. Feature Groups and Prior Assignment

#### Energy Features (4 features) - AEH Prior
- **Features**: `ghg_emissions_int_log`, `floor_area_log`, `electric_eui`, `fuel_eui`
- **Prior**: Adaptive Elastic Horseshoe (AEH)
- **Rationale**: Energy-related features benefit from adaptive regularization that can handle varying sparsity patterns

#### Building Features (4 features) - Hierarchical Prior
- **Features**: `energy_star_rating_normalized`, `energy_mix`, `building_age_log`, `floor_area_squared`
- **Prior**: Hierarchical Normal
- **Rationale**: Building characteristics typically have moderate, stable effects

#### Interaction Features (4 features) - Hierarchical Prior
- **Features**: `energy_intensity_ratio`, `building_age_squared`, `energy_star_rating_squared`, `ghg_per_area`
- **Prior**: Hierarchical Normal
- **Rationale**: Interaction terms benefit from group-level regularization

### 2. AEH Prior Specification

The Adaptive Elastic Horseshoe prior for energy features combines multiple regularization techniques:

#### Global Shrinkage Parameter (τ)
- **Prior**: τ ~ Half-Cauchy(0, 1)
- **Role**: Controls overall sparsity level across all energy features
- **Update**: τ² = (p-1)/(βᵀβ + 2b₀)

#### Local Shrinkage Parameters (λⱼ)
- **Prior**: λⱼ ~ Half-Cauchy(0, 1) for each feature j
- **Role**: Allows individual features to escape shrinkage
- **Update**: λⱼ² = 1/(βⱼ²/τ² + 1)

#### Elastic Net Parameter (α)
- **Prior**: α ~ Beta(2, 2)
- **Role**: Balances L1 (Lasso) and L2 (Ridge) regularization
- **Update**: α = (Σ|βⱼ|)/(Σ|βⱼ| + Σβⱼ²)

#### Adaptive Parameters (β, γ, δ)
- **Role**: Fine-tune the balance between different regularization components
- **Update**: Via EM algorithm with gradient-based optimization

### 3. Hierarchical Prior Specification

For building and interaction features, standard hierarchical priors are used:

#### Group-Level Parameters
- **Group Mean (μ)**: μ ~ N(0, 10²)
- **Group Variance (σ²)**: σ² ~ Inverse-Gamma(1, 1)

#### Feature-Level Parameters
- **Coefficients (β)**: β ~ N(μ, σ²) within each group

### 4. Likelihood Model

The response variable follows a Gaussian distribution:
- **Model**: y ~ N(Xβ, σ²)
- **Noise Variance**: σ² ~ Inverse-Gamma(1, 1)

### 5. Inference Algorithm

#### Expectation-Maximization (EM) Algorithm
1. **E-Step**: Compute expected values of latent variables
2. **M-Step**: Update model parameters using closed-form solutions
3. **Convergence**: Check relative change in log-likelihood

#### Parameter Updates
- **Regression Coefficients**: β = (XᵀX + Σ⁻¹)⁻¹Xᵀy
- **AEH Parameters**: Updated via gradient-based optimization
- **Hierarchical Parameters**: Closed-form updates for means and variances

### 6. Model Outputs

#### Predictions
- **Point Predictions**: ŷ = Xβ
- **Prediction Intervals**: Based on posterior uncertainty

#### Uncertainty Quantification
- **Parameter Uncertainty**: Posterior variances of coefficients
- **Prediction Uncertainty**: Combined parameter and noise uncertainty
- **Calibration**: Empirical coverage of prediction intervals

#### Feature Analysis
- **Importance**: Magnitude of regression coefficients
- **Stability**: Consistency across EM iterations
- **Sparsity**: Number of effectively non-zero coefficients

## Key Features

### Adaptive Regularization
- **AEH Prior**: Automatically adapts regularization strength based on data
- **Feature-Specific**: Different shrinkage for different feature groups
- **Data-Driven**: Hyperparameters learned from the data

### Uncertainty Quantification
- **Parameter Uncertainty**: Full posterior distributions for all parameters
- **Prediction Uncertainty**: Probabilistic predictions with intervals
- **Model Uncertainty**: Accounted for in final predictions

### Computational Efficiency
- **EM Algorithm**: Fast convergence with closed-form updates
- **Scalable**: Handles moderate-sized datasets efficiently
- **Stable**: Robust to initialization and numerical issues