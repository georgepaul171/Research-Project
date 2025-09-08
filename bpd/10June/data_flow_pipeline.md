# Data Flow Pipeline: Adaptive Prior ARD Model

Below is a high-level data flow pipeline for the Adaptive Prior ARD model, arranged in a true zigzag (snake) layout for clarity and compactness.

```mermaid
flowchart LR
    %% Top row (odd steps)
    A["Raw Data Acquisition (BPD CSV)"] --> B["Data Cleaning & Outlier Treatment"]
    B --> C["Data Type Conversion & Imputation"]
    C --> D["Feature Engineering (Log, Ratios, GHG Normalisation)"]
    D --> E["Feature Selection (No Interaction Terms)"]
    E --> F["Train/Test Split or K-Fold Cross-Validation"]

    %% Bottom row (even steps, staggered below)
    F --> G["Feature Scaling (Robust/Standard Scaler)"]
    G --> H["Model Training: Adaptive Prior ARD (EM Algorithm)"]
    H --> I["Adaptive Prior Updates (AEH/ARD)"]
    I --> J["Diagnostics & Logging (Prediction Range, Hyperparameters, Trace Plots)"]
    J --> K["Model Evaluation (CV Metrics, Baselines)"]
    K --> L["Outputs: Predictions, Uncertainty, Diagnostics, Feature Importance"]

    %% Invisible links to force zigzag layout
    A -.-> G
    B -.-> H
    C -.-> I
    D -.-> J
    E -.-> K
    F -.-> L

    %% Styling
    style D fill:#e6f7ff,stroke:#1890ff
    style H fill:#fffbe6,stroke:#faad14
    style I fill:#bae7ff,stroke:#1890ff
    style J fill:#f9f0ff,stroke:#722ed1
    style K fill:#f0f5ff,stroke:#2f54eb
    style L fill:#ffffff,stroke:#000000
```

**Figure:** *Data flow pipeline for the Adaptive Prior ARD model, arranged in a true zigzag (snake) layout. The process begins with raw data acquisition and proceeds through cleaning, imputation, feature engineering, selection, scaling, model training, adaptive prior updates, diagnostics, evaluation, and outputs.* 