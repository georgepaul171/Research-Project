# Model Architecture Diagram: Adaptive Prior ARD with AEH Prior

Below is a high-level architecture diagram of the Adaptive Prior ARD model as implemented in the current research. This diagram illustrates the flow from input data through feature grouping, group-specific priors, Bayesian regression, EM-based inference, diagnostics, and final outputs.

```mermaid
flowchart TD
    A["Input Data: Engineered Features (X), Target (y)"] --> B["Feature Grouping"]
    B --> C1["Energy Features (G1): AEH Prior"]
    B --> C2["Building Features (G2): Hierarchical ARD Prior"]
    C1 --> D1["Group-specific Prior Hyperparameters (lambda, tau, alpha, beta)"]
    C2 --> D2["Group-specific Prior Hyperparameters (beta_j)"]
    D1 & D2 --> E["Bayesian Linear Regression Core"]
    E --> F["EM Algorithm: Posterior Mean (m), Covariance (S), Adaptive Prior Updates"]
    F --> G["Cross-Validation & Diagnostics"]
    G --> H["Prediction & Uncertainty Quantification"]
    H --> I["Outputs: Predictions, Uncertainty, Diagnostics, Feature Importance"]
    style C1 fill:#e6f7ff,stroke:#1890ff
    style C2 fill:#f6ffed,stroke:#52c41a
    style D1 fill:#bae7ff,stroke:#1890ff
    style D2 fill:#d9f7be,stroke:#52c41a
    style E fill:#fffbe6,stroke:#faad14
    style F fill:#fff1b8,stroke:#faad14
    style G fill:#f9f0ff,stroke:#722ed1
    style H fill:#f0f5ff,stroke:#2f54eb
    style I fill:#ffffff,stroke:#000000
```

**Figure:** *Model architecture for the Adaptive Prior ARD with AEH prior. The pipeline begins with input data, proceeds through feature grouping and assignment of group-specific priors, then Bayesian linear regression with EM-based inference and adaptive prior updates. Diagnostics and cross-validation are integrated, and the model outputs both predictions and uncertainty estimates, along with detailed diagnostics and feature importance.* 