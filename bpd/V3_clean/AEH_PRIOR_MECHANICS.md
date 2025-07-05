# Conceptual Illustration of Adaptive Elastic Horseshoe (AEH) Prior Mechanics

## Overview
This diagram visualizes the adaptive shrinkage behavior of the AEH prior, demonstrating how it dynamically balances L1 (sparsity-inducing) and L2 (magnitude-controlling) regularization based on feature characteristics.

## AEH Prior Mechanics Diagram

```mermaid
graph TD
    %% Input
    F[Feature j]
    
    %% Feature Analysis
    A1[Feature Importance<br/>|β_j|]
    A2[Uncertainty<br/>σ_j]
    A3[Data Support<br/>Evidence]
    
    %% Adaptive Parameters
    P1[Local Shrinkage<br/>λ_j]
    P2[Global Shrinkage<br/>τ]
    P3[Elastic Mix<br/>α]
    P4[Strength<br/>β_0]
    
    %% Shrinkage Components
    S1[L1 Regularization<br/>Sparsity]
    S2[L2 Regularization<br/>Magnitude]
    S3[Horseshoe<br/>Heavy Tails]
    
    %% Combined Effect
    C[Combined Shrinkage<br/>τ²λ_j²]
    
    %% Output
    O[Final Coefficient<br/>β_j]
    
    %% Connections
    F --> A1
    F --> A2
    F --> A3
    
    A1 --> P1
    A2 --> P2
    A3 --> P3
    A3 --> P4
    
    P1 --> S1
    P2 --> S2
    P3 --> S1
    P3 --> S2
    P4 --> S3
    
    S1 --> C
    S2 --> C
    S3 --> C
    
    C --> O
    
    %% Feedback Loop
    O -.-> A1
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef analysis fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef params fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef shrinkage fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef combined fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    
    class F input
    class A1,A2,A3 analysis
    class P1,P2,P3,P4 params
    class S1,S2,S3 shrinkage
    class C combined
    class O output
```

## Adaptive Shrinkage Mechanism

### 1. Feature Analysis
- **Feature Importance**: Magnitude of coefficient |β_j|
- **Uncertainty**: Posterior standard deviation σ_j
- **Data Support**: Evidence from likelihood

### 2. Adaptive Parameters
- **Local Shrinkage (λ_j)**: Feature-specific shrinkage parameter
- **Global Shrinkage (τ)**: Overall regularization strength
- **Elastic Mix (α)**: Balance between L1 and L2 (0 ≤ α ≤ 1)
- **Strength (β_0)**: Prior strength hyperparameter

### 3. Shrinkage Components

#### L1 Regularization (Sparsity)
- **Purpose**: Induces exact zeros for irrelevant features
- **Effect**: Strong shrinkage for small coefficients
- **Control**: Elastic mix parameter α

#### L2 Regularization (Magnitude)
- **Purpose**: Controls coefficient magnitudes
- **Effect**: Prevents overfitting
- **Control**: Global shrinkage τ

#### Horseshoe Component (Heavy Tails)
- **Purpose**: Allows large coefficients when supported by data
- **Effect**: Heavy-tailed prior distribution
- **Control**: Local shrinkage λ_j

### 4. Combined Effect
The final shrinkage is determined by:
```
Shrinkage = τ² × λ_j²
```

Where:
- **τ**: Global shrinkage (controls overall regularization)
- **λ_j**: Local shrinkage (adapts to each feature)

## Mathematical Formulation

### AEH Prior Structure
```
β_j ~ Normal(0, τ²λ_j²)
λ_j ~ Half-Cauchy(0, 1)
τ ~ Half-Cauchy(0, τ_0)
τ_0 ~ Gamma(a, b)
```

### Adaptive Updates
The parameters are updated based on:
1. **Feature importance**: Higher importance → less shrinkage
2. **Uncertainty**: Higher uncertainty → more shrinkage
3. **Data support**: Strong evidence → less shrinkage

### Elastic Net Integration
The AEH prior combines with elastic net regularization:
```
Penalty = α × L1 + (1-α) × L2
```

Where α is adaptively determined based on feature characteristics.

## Key Properties

### 1. Adaptivity
- **Strong features**: Minimal shrinkage, large coefficients allowed
- **Weak features**: Strong shrinkage, coefficients near zero
- **Irrelevant features**: Exact zero coefficients (sparsity)

### 2. Heavy Tails
- Allows large coefficients when data strongly supports them
- Prevents over-regularization of important features
- Maintains model flexibility

### 3. Automatic Feature Selection
- Irrelevant features are automatically excluded
- Relevant features are retained with appropriate magnitudes
- No manual feature selection required

### 4. Uncertainty Quantification
- Provides uncertainty estimates for each coefficient
- Accounts for both data uncertainty and prior uncertainty
- Enables reliable prediction intervals

## Practical Implications

### For Energy Modeling
- **Energy features**: Often show heavy tails, AEH handles extreme values
- **Building features**: Hierarchical structure, moderate regularization
- **Interaction features**: Automatic selection of relevant interactions

### Model Performance
- **Sparsity**: Reduces model complexity
- **Flexibility**: Captures non-linear relationships
- **Robustness**: Handles outliers and extreme values
- **Interpretability**: Clear feature importance ranking

---

*This diagram illustrates how the AEH prior dynamically adapts its shrinkage behavior based on feature characteristics, providing an optimal balance between sparsity, flexibility, and interpretability for building energy performance modeling.* 