# Data Flow Pipeline: Adaptive Prior ARD Model

Below is a high-level data flow pipeline for the Adaptive Prior ARD model as implemented in this research. The diagram illustrates the end-to-end process from raw data acquisition to final model outputs, including all major preprocessing, feature engineering, and diagnostic steps. The pipeline is arranged in a true 'snake' (zig-zag) layout for improved readability in wide documents.

```mermaid
flowchart LR
    A["Raw Data Acquisition (BPD CSV)"]
    B["Data Cleaning & Outlier Treatment"]
    C["Data Type Conversion & Imputation"]
    D["Feature Engineering (Log, Ratios, GHG Normalisation)"]
    E["Feature Selection (No Interaction Terms)"]
    F["Train/Test Split or K-Fold Cross-Validation"]
    G["Feature Scaling (Robust/Standard Scaler)"]
    H["Model Training: Adaptive Prior ARD (EM Algorithm)"]
    I["Adaptive Prior Updates (AEH/ARD)"]
    J["Diagnostics & Logging (Prediction Range, Hyperparameters, Trace Plots)"]
    K["Model Evaluation (CV Metrics, Baselines)"]
    L["Outputs: Predictions, Uncertainty, Diagnostics, Feature Importance"]

    %% Zigzag layout: alternate up and down
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L

    %% Positioning for zigzag effect
    B -->| | C
    C -.->| | D
    D -->| | E
    E -.->| | F
    F -->| | G
    G -.->| | H
    H -->| | I
    I -.->| | J
    J -->| | K
    K -.->| | L

    %% Manually set subgraphs for vertical alternation
    subgraph row1[ ]
        direction LR
        A B D F H J L
    end
    subgraph row2[ ]
        direction LR
        C E G I K
    end

    %% Styling
    style D fill:#e6f7ff,stroke:#1890ff
    style H fill:#fffbe6,stroke:#faad14
    style I fill:#bae7ff,stroke:#1890ff
    style J fill:#f9f0ff,stroke:#722ed1
    style K fill:#f0f5ff,stroke:#2f54eb
    style L fill:#ffffff,stroke:#000000
```

**Figure:** *Data flow pipeline for the Adaptive Prior ARD model, arranged in a true 'snake' (zig-zag) layout for improved readability. The process begins with raw data acquisition and proceeds through cleaning, imputation, and advanced feature engineering. After feature selection and scaling, the model is trained using the EM algorithm with adaptive prior updates. Comprehensive diagnostics and evaluation are performed, and the pipeline outputs predictions, uncertainty estimates, and detailed diagnostics for research and decision support.* 