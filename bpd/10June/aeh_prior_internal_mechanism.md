# Internal Mechanism of the Adaptive Elastic Horseshoe (AEH) Prior

Below is a detailed diagram of the internal mechanism of the AEH prior as implemented in the Adaptive Prior ARD model. This diagram illustrates the flow from feature group assignment, through AEH prior initialization, prior scale computation, EM-based posterior update, and the full adaptive update mechanism (importance/uncertainty ratios, elastic net mixing, momentum-based lambda update), looping until convergence.

```mermaid
flowchart TD
    A["Input: Feature Group (e.g., Energy Features)"] --> B["Assign AEH Prior"]
    B --> C["Initialize Hyperparameters (lambda, tau, alpha, beta, momentum)"]
    C --> D["Compute Prior Scale: (w^2)/(2*tau) + beta * (alpha*|w| + (1-alpha)*w^2)"]
    D --> E["Posterior Update (EM Step): Compute m, S"]
    E --> F["Adaptive Updates"]
    F --> F1["1. Compute Importance Ratio"]
    F --> F2["2. Compute Uncertainty Ratio"]
    F --> F3["3. Update alpha, beta (Elastic Net Mix/Strength)"]
    F --> F4["4. Momentum-based Update for lambda"]
    F1 --> G["Update alpha: clip(alpha + importance_ratio * lr * (1-alpha), 0.1, 0.9)"]
    F2 --> H["Update beta: clip(beta + gamma * (1-uncertainty_ratio) * beta, 0.1, 10.0)"]
    F4 --> I["momentum = rho * momentum + gamma * grad_w log p(w)"]
    I --> J["lambda = lambda + momentum"]
    F3 --> G
    F3 --> H
    G --> K["Update Prior Scale for Next Iteration"]
    H --> K
    J --> K
    K --> L["Check Convergence / Continue EM"]
    style B fill:#e6f7ff,stroke:#1890ff
    style C fill:#bae7ff,stroke:#1890ff
    style D fill:#fffbe6,stroke:#faad14
    style E fill:#fff1b8,stroke:#faad14
    style F fill:#f9f0ff,stroke:#722ed1
    style F1 fill:#f9f0ff,stroke:#722ed1
    style F2 fill:#f9f0ff,stroke:#722ed1
    style F3 fill:#f9f0ff,stroke:#722ed1
    style F4 fill:#f9f0ff,stroke:#722ed1
    style G fill:#f0f5ff,stroke:#2f54eb
    style H fill:#f0f5ff,stroke:#2f54eb
    style I fill:#f0f5ff,stroke:#2f54eb
    style J fill:#f0f5ff,stroke:#2f54eb
    style K fill:#ffffff,stroke:#000000
    style L fill:#ffffff,stroke:#000000
```

**Figure:** *Internal mechanism of the Adaptive Elastic Horseshoe (AEH) prior. The process begins with feature group assignment and AEH prior initialization, proceeds through prior scale computation and EM-based posterior update, and includes adaptive updates for hyperparameters (importance/uncertainty ratios, elastic net mixing, and momentum-based lambda update). The loop continues until convergence is achieved.* 