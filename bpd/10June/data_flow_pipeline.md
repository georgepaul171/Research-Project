# Data Flow Pipeline: Adaptive Prior ARD Model

Below is a high-level data flow pipeline for the Adaptive Prior ARD model as implemented in this research. The diagram illustrates the end-to-end process from raw data acquisition to final model outputs, including all major preprocessing, feature engineering, and diagnostic steps.

```mermaid
flowchart TD
    A["Raw Data Acquisition (BPD CSV)"] --> B["Data Cleaning & Outlier Treatment"]
    B --> C["Data Type Conversion & Imputation"]
    C --> D["Feature Engineering (Log, Ratios, GHG Normalisation)"]
    D --> E["Feature Selection (No Interaction Terms)"]
    E --> F["Train/Test Split or K-Fold Cross-Validation"]
    F --> G["Feature Scaling (Robust/Standard Scaler)"]
    G --> H["Model Training: Adaptive Prior ARD (EM Algorithm)"]
    H --> I["Adaptive Prior Updates (AEH/ARD)"]
    I --> J["Diagnostics & Logging (Prediction Range, Hyperparameters, Trace Plots)"]
    J --> K["Model Evaluation (CV Metrics, Baselines)"]
    K --> L["Outputs: Predictions, Uncertainty, Diagnostics, Feature Importance"]
    style D fill:#e6f7ff,stroke:#1890ff
    style H fill:#fffbe6,stroke:#faad14
    style I fill:#bae7ff,stroke:#1890ff
    style J fill:#f9f0ff,stroke:#722ed1
    style K fill:#f0f5ff,stroke:#2f54eb
    style L fill:#ffffff,stroke:#000000
```

**Figure:** *Data flow pipeline for the Adaptive Prior ARD model. The process begins with raw data acquisition and proceeds through cleaning, imputation, and advanced feature engineering. After feature selection and scaling, the model is trained using the EM algorithm with adaptive prior updates. Comprehensive diagnostics and evaluation are performed, and the pipeline outputs predictions, uncertainty estimates, and detailed diagnostics for research and decision support.* 