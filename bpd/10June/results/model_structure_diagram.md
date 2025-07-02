# Hierarchical Model Structure with Prior Assignments

Below is a diagram illustrating the hierarchical model structure and group-wise prior assignments.

```mermaid
flowchart TD
    Y["Observed Data (y)"]
    X["Observed Features (X)"]
    X --> X1["G1: Energy Features"]
    X --> X2["G2: Building Features"]
    X --> X3["G3: Interaction Features"]
    X1 --> W1["w (Energy)"]
    X2 --> W2["w (Building)"]
    X3 --> W3["w (Interaction)"]
    W1 --> P1["AEH Prior"]
    W2 --> P2["Hierarchical ARD Prior"]
    W3 --> P3["Spike-and-Slab Prior"]
    Y -.-> W1
    Y -.-> W2
    Y -.-> W3
    style P1 fill:#e6f7ff,stroke:#1890ff
    style P2 fill:#f6ffed,stroke:#52c41a
    style P3 fill:#fffbe6,stroke:#faad14
```

**Figure:** *Hierarchical model structure with group-wise prior assignments. Each feature group is assigned a specific prior: AEH for energy features, hierarchical ARD for building features, and spike-and-slab for interaction features.* 