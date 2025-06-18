# Adaptive Elastic Horseshoe (AEH) Prior: Key Diagrams

## 1. Group-wise Prior Allocation Diagram

This diagram illustrates how different feature groups are assigned specific priors based on their characteristics and modeling requirements.

```mermaid
graph TB
    subgraph Raw_Features[Raw Features]
        RF[Input Features]
    end

    subgraph Feature_Groups[Feature Groups]
        G1[G₁: Energy Features]
        G2[G₂: Building Features]
        G3[G₃: Interaction Features]
    end

    subgraph Priors[Prior Assignments]
        AEH[Adaptive Elastic Horseshoe Prior]
        HARD[Hierarchical ARD Prior]
        SS[Spike-and-Slab Prior]
    end

    RF --> G1
    RF --> G2
    RF --> G3

    G1 --> AEH
    G2 --> HARD
    G3 --> SS

    classDef group fill:#f9f,stroke:#333,stroke-width:2px
    classDef prior fill:#bbf,stroke:#333,stroke-width:2px
    class G1,G2,G3 group
    class AEH,HARD,SS prior
```

### Rationale for Prior Assignments

1. **Energy Features (G₁)**: Adaptive Elastic Horseshoe Prior
   - Handles heavy-tailed distributions
   - Provides adaptive shrinkage
   - Ideal for energy consumption patterns with extreme values
   - Maintains sparsity while allowing for strong signals

2. **Building Features (G₂)**: Hierarchical ARD Prior
   - Captures hierarchical structure of building characteristics
   - Maintains interpretability
   - Provides automatic relevance determination
   - Balances model complexity with performance

3. **Interaction Features (G₃)**: Spike-and-Slab Prior
   - Effectively models sparse interactions
   - Helps identify truly significant feature interactions
   - Provides clear feature selection mechanism
   - Reduces overfitting in interaction terms

## 2. EM and HMC Integration Flowchart

This diagram represents the dual-stage inference approach combining Expectation-Maximization (EM) and Hamiltonian Monte Carlo (HMC) for robust parameter estimation and uncertainty quantification.

```mermaid
graph TB
    subgraph Initial[Initial Parameters]
        IP[Initial Parameters]
    end

    subgraph EM[EM Algorithm Loop]
        E[E-step]
        M[M-step]
        WS[Warm Start]
        APU[Adaptive Prior Updates]
    end

    subgraph HMC[Hamiltonian Monte Carlo]
        LI[Leapfrog Integration]
        MR[Momentum Resampling]
        MA[Metropolis Acceptance]
    end

    subgraph Output[Output]
        PD[Full Posterior Distributions]
        UQ[Uncertainty Quantified Predictions]
    end

    IP --> EM
    EM --> WS
    WS --> HMC
    HMC --> PD
    PD --> UQ

    E --> M
    M --> APU
    APU --> E

    LI --> MR
    MR --> MA
    MA --> LI

    classDef process fill:#f9f,stroke:#333,stroke-width:2px
    classDef output fill:#bbf,stroke:#333,stroke-width:2px
    class E,M,LI,MR,MA process
    class PD,UQ output
```

### Inference Process Description

1. **EM Algorithm Stage**:
   - **E-step**: Computes expected values of latent variables
   - **M-step**: Updates model parameters
   - **Adaptive Prior Updates**: Refines prior parameters based on data evidence
   - **Warm Start**: Provides initial estimates for HMC

2. **Hamiltonian Monte Carlo Stage**:
   - **Leapfrog Integration**: Simulates Hamiltonian dynamics for efficient exploration
   - **Momentum Resampling**: Ensures proper exploration of the parameter space
   - **Metropolis Acceptance**: Maintains detailed balance for correct sampling
   - **Posterior Exploration**: Generates full posterior distributions

3. **Output Generation**:
   - **Full Posterior Distributions**: Complete uncertainty quantification for all parameters
   - **Uncertainty Quantified Predictions**: Predictions with associated confidence intervals
   - **Model Diagnostics**: Convergence metrics and sampling statistics 