# Adaptive Elastic Horseshoe (AEH) Prior Architecture

## High-Level Architecture

```mermaid
graph TB
    subgraph Input
        X[Feature Matrix]
        y[Target Vector]
    end

    subgraph AEH_Prior[AEH Prior Components]
        EN[Elastic Net Component]
        HS[Horseshoe Component]
        ARD[ARD Mechanism]
    end

    subgraph Update_Mechanism[Adaptive Update System]
        M[Momentum Updates]
        P[Parameter Updates]
        C[Convergence Check]
    end

    subgraph Output
        W[Weight Vector]
        U[Uncertainty Estimates]
    end

    X --> EN
    X --> HS
    y --> ARD
    
    EN --> Update_Mechanism
    HS --> Update_Mechanism
    ARD --> Update_Mechanism
    
    Update_Mechanism --> W
    Update_Mechanism --> U
```

## Detailed Component Architecture

```mermaid
graph TB
    subgraph Elastic_Net[Elastic Net Component]
        L1[L1 Regularization]
        L2[L2 Regularization]
        Mix[Adaptive Mixing]
    end

    subgraph Horseshoe[Horseshoe Component]
        Global[Global Shrinkage]
        Local[Local Shrinkage]
        Scale[Scale Parameter]
    end

    subgraph ARD[ARD Mechanism]
        Lambda[Local Parameters]
        Update[Parameter Updates]
        Check[Relevance Check]
    end

    subgraph Updates[Update System]
        Momentum[Momentum Calculation]
        Grad[Gradient Computation]
        Clip[Clipping Operations]
    end

    L1 --> Mix
    L2 --> Mix
    Mix --> Scale
    
    Global --> Scale
    Local --> Scale
    Scale --> Updates
    
    Lambda --> Update
    Update --> Check
    Check --> Updates
    
    Momentum --> Grad
    Grad --> Clip
    Clip --> Update
```

## Parameter Flow Diagram

```mermaid
flowchart LR
    subgraph Parameters[Key Parameters]
        alpha[α: Mixing Parameter]
        beta[β: Regularization Strength]
        tau[τ: Global Shrinkage]
        lambda[λ: Local Shrinkage]
    end

    subgraph Updates[Update Rules]
        alpha_update[α_{t+1} = clip(α_t + γ * (0.5 - importance_ratio), 0.1, 0.9)]
        beta_update[β_{t+1} = clip(β_t + γ * (1.0 - uncertainty_ratio), 0.1, 10.0)]
        lambda_update[λ_{t+1} = λ_t + momentum_{t+1}]
    end

    subgraph Momentum[Momentum System]
        momentum[momentum_{t+1} = ρ * momentum_t + γ * grad_w log p(w_t)]
    end

    Parameters --> Updates
    Updates --> Momentum
    Momentum --> Parameters
```

## Implementation Flow

```mermaid
sequenceDiagram
    participant Data
    participant Model
    participant EN as Elastic Net
    participant HS as Horseshoe
    participant ARD
    participant Update

    Data->>Model: Input Features & Target
    Model->>EN: Compute Elastic Penalty
    Model->>HS: Compute Horseshoe Scale
    Model->>ARD: Initialize Parameters
    
    loop Until Convergence
        EN->>Update: Send Penalty
        HS->>Update: Send Scale
        ARD->>Update: Send Parameters
        Update->>Model: Update Weights
        Model->>EN: Update Mixing
        Model->>HS: Update Shrinkage
        Model->>ARD: Update Relevance
    end
    
    Model->>Data: Return Predictions & Uncertainty
```

## Notes on Architecture

1. **Elastic Net Component**
   - Combines L1 and L2 regularization
   - Adaptive mixing parameter (α) balances sparsity and density
   - Handles feature selection and regularization

2. **Horseshoe Component**
   - Provides heavy-tailed shrinkage
   - Global (τ) and local (λ) shrinkage parameters
   - Enables adaptive feature selection

3. **ARD Mechanism**
   - Automatic Relevance Determination
   - Feature-specific shrinkage parameters
   - Dynamic feature importance assessment

4. **Update System**
   - Momentum-based updates for stability
   - Gradient computation for optimization
   - Clipping operations for numerical stability

5. **Convergence Criteria**
   - Bounded parameter updates
   - Momentum-based convergence
   - Multiple convergence thresholds 