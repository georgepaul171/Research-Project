# Conceptual Diagram: AEH Prior Adaptive Shrinkage

Below is a conceptual diagram illustrating the adaptive shrinkage mechanism of the AEH prior.

```mermaid
flowchart TD
    A["Input Weights (w_g)"] --> B["Compute Importance Ratio"]
    A --> C["Compute Uncertainty Ratio"]
    B --> D["Update alpha (L1/L2 Mix)"]
    C --> E["Update beta (Regularization Strength)"]
    D --> F["Compute Prior Scale"]
    E --> F
    F --> G["Shrinkage Applied to Weights"]
    G -.-> A
    style F fill:#e6f7ff,stroke:#1890ff
    style G fill:#fffbe6,stroke:#faad14
```

**Figure:** *Conceptual diagram of the AEH prior's adaptive shrinkage mechanism. Feature importance and uncertainty ratios drive updates to the L1/L2 mix (alpha) and regularization strength (beta), which together determine the prior scale and the amount of shrinkage applied to each weight. The process is adaptive and iterative.* 