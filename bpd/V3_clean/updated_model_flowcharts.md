# Updated Model Flowcharts for AEH Prior Comparative Study

## Data Acquisition and Preparation Pipeline

```mermaid
flowchart TD
    A1["BPD Dataset"]
    A2["Data Loading"]
    A3["Data Cleaning"]
    A4["Feature Engineering"]
    A5["Feature Selection"]
    A6["Model Training"]
    A7a["Baseline Models"]
    A7b["Adaptive Prior Models"]
    A7c["Hierarchical Models"]
    A8a["Linear Regression"]
    A8b["Bayesian Ridge"]
    A8c["AEH Prior"]
    A8d["Horseshoe Prior"]
    A8e["Elastic Net Prior"]
    A8f["Spike-Slab Prior"]
    A8g["Group Priors"]
    A8h["Hierarchical Mixture"]
    A9["Prior Comparison & Selection"]
    A10["Model Evaluation"]
    A11a["Performance Metrics"]
    A11b["Uncertainty Assessment"]
    A11c["Model Diagnostics"]
    A11d["Feature Analysis"]
    A12["Results Generation"]
    A13["Visualizations & Model Artifacts"]

    A1 --> A2 --> A3 --> A4 --> A5 --> A6
    A6 --> A7a
    A6 --> A7b
    A6 --> A7c
    A7a --> A8a
    A7a --> A8b
    A7b --> A8c
    A7b --> A8d
    A7b --> A8e
    A7b --> A8f
    A7c --> A8g
    A7c --> A8h
    A8c --> A9
    A8d --> A9
    A8e --> A9
    A8f --> A9
    A8g --> A9
    A8h --> A9
    A8a --> A10
    A8b --> A10
    A9 --> A10
    A10 --> A11a
    A10 --> A11b
    A10 --> A11c
    A10 --> A11d
    A11a --> A12
    A11b --> A12
    A11c --> A12
    A11d --> A12
    A12 --> A13
```

*Fig. 1: Updated pipeline for EUI modelling, including explicit prior comparison and selection for the new model.*

---

## Group-wise Bayesian Model Architecture with Prior Choice

```mermaid
flowchart TD
    B1["Input Features"]
    B2a["Energy Features"]
    B2b["Building Features"]
    B2c["Interaction Features"]
    B3a["AEH Prior"]
    B3b["Horseshoe Prior"]
    B3c["Elastic Net Prior"]
    B3d["Hierarchical Normal Prior"]
    B4["Likelihood"]
    B5["Linear Predictor"]
    B6["HMC/EM Inference"]
    B7a["Predictions"]
    B7b["Uncertainty"]
    B7c["Feature Importance"]
    B7d["Diagnostics"]

    B1 --> B2a
    B1 --> B2b
    B1 --> B2c
    B2a --> B3a
    B2a --> B3b
    B2a --> B3c
    B2b --> B3d
    B2c --> B3d
    B3a --> B4
    B3b --> B4
    B3c --> B4
    B3d --> B4
    B4 --> B5 --> B6
    B6 --> B7a
    B6 --> B7b
    B6 --> B7c
    B6 --> B7d
```

*Fig. 2: Updated group-wise Bayesian model architecture, showing explicit prior choices for energy features in the new model.* 