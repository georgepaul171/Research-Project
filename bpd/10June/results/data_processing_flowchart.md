# Data Processing Flowchart

Below is a flowchart illustrating the data processing pipeline from raw data acquisition to the final engineered dataset used for modeling.

```mermaid
flowchart TD
    A["Raw BPD Data"] --> B["Filtering (NYC Office Buildings)"]
    B --> C["Outlier Treatment"]
    C --> D["Missing Value Imputation"]
    D --> E["Non-linear Transformations"]
    E --> F["Ratio Features"]
    F --> G["Feature Scaling"]
    G --> H["Cleaned & Engineered Dataset for Modeling"]
    style H fill:#e6f7ff,stroke:#1890ff
```

**Figure:** *Data processing pipeline from raw BPD data to the final cleaned and engineered dataset for modeling.* 