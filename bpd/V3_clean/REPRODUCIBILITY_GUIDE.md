# Reproducibility Guide: Bayesian Building Energy Modeling

## Overview

This guide provides complete instructions for reproducing all experiments, results, and analyses from the Bayesian building energy modeling research project. Follow these steps to ensure full reproducibility of the research findings.

## Environment Setup

### 1. System Requirements

#### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for data and results
- **OS**: Linux, macOS, or Windows with Python support

#### Software Requirements
- **Python**: 3.8 or higher
- **Package Manager**: pip or conda
- **Git**: For version control (optional)

### 2. Python Environment Setup

#### Option A: Using pip (Recommended)
```bash
# Create virtual environment
python -m venv bpd_env

# Activate environment
# On Linux/macOS:
source bpd_env/bin/activate
# On Windows:
bpd_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n bpd_env python=3.9

# Activate environment
conda activate bpd_env

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```python
# Test script to verify all packages are installed correctly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sklearn
import shap
import networkx as nx
import joblib
import tqdm

print("All packages installed successfully!")
```

## Data Preparation

### 1. Data Source

The research uses the cleaned office buildings dataset:
- **File**: `cleaned_office_buildings.csv`
- **Location**: `/Users/georgepaul/Desktop/Research-Project/bpd/`
- **Format**: CSV with building energy performance data

### 2. Data Structure Verification

```python
import pandas as pd

# Load and verify data
data_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
df = pd.read_csv(data_path)

# Check data structure
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target variable 'site_eui' range: {df['site_eui'].min():.1f} - {df['site_eui'].max():.1f}")
```

### 3. Required Data Fields

Ensure your dataset contains these essential columns:
- `site_eui`: Target variable (Site Energy Use Intensity)
- `floor_area`: Building floor area
- `electric_eui`: Electricity energy use intensity
- `fuel_eui`: Fuel energy use intensity
- `energy_star_rating`: Energy Star rating
- `ghg_emissions_int`: Greenhouse gas emissions intensity
- `year_built`: Building construction year

## Running Experiments

### 1. Main Model Training

#### Run the primary model (V3.py)
```bash
cd /Users/georgepaul/Desktop/Research-Project/bpd/V3_clean
python V3.py
```

**Expected Outputs:**
- `results/` directory with model outputs
- `adaptive_prior_model.joblib`: Saved model
- Various plots and diagnostic files

#### Run advanced model with HMC debugging
```bash
python AREHap_groupprior_hmcdebug.py
```

**Expected Outputs:**
- `results_groupprior_hmcdebug/` directory
- Comprehensive model diagnostics
- Advanced uncertainty analysis

### 2. Diagnostic Experiments

#### Model Range Diagnostics
```bash
python debug_model_range.py
```

**Purpose**: Compare model prediction ranges across different priors
**Outputs**: `results_debug_model_range/` with comparison plots

#### Calibration Experiments
```bash
python calibration_experiments.py
```

**Purpose**: Test uncertainty calibration across different models
**Outputs**: `results_calibration_experiments/` with calibration analysis

#### Interval Diagnostics
```bash
python debug_v3_intervals.py
```

**Purpose**: Analyze prediction interval properties
**Outputs**: `results_debug_v3_intervals/` with interval analysis

#### Simple Model Analysis
```bash
python prediction_vs_actual_simple.py
```

**Purpose**: Baseline model comparison
**Outputs**: `results_simple_model/` with baseline results

### 3. Expected Runtime

#### Time Estimates
- **V3.py**: 10-30 minutes (depending on hardware)
- **AREHap_groupprior_hmcdebug.py**: 30-60 minutes
- **Diagnostic scripts**: 5-15 minutes each
- **Total runtime**: 1-2 hours for complete reproduction

#### Progress Monitoring
- Scripts display progress bars and logging information
- Check console output for convergence messages
- Monitor memory usage during HMC sampling

## Results Verification

### 1. Expected File Structure

After running all experiments, you should have:

```
V3_clean/
├── results/
│   ├── adaptive_prior_model.joblib
│   ├── prediction_vs_actual.png
│   ├── calibration_plot.png
│   ├── feature_importance.png
│   └── [other output files]
├── results_groupprior_hmcdebug/
│   ├── [advanced model outputs]
├── results_debug_model_range/
│   ├── [range comparison outputs]
├── results_calibration_experiments/
│   ├── [calibration analysis]
├── results_debug_v3_intervals/
│   ├── [interval diagnostics]
└── results_simple_model/
    ├── [baseline comparisons]
```

### 2. Key Results to Verify

#### Performance Metrics
Check that your results are within expected ranges:
- **RMSE**: 40-60 kWh/m²/year
- **MAE**: 30-45 kWh/m²/year
- **R²**: 0.65-0.80

#### Model Rankings
Verify the expected performance order:
1. Hierarchical Prior (best)
2. Bayesian Ridge
3. Linear Regression
4. AEH Prior (β₀=10)
5. AEH Prior (β₀=1) (worst)

#### Uncertainty Calibration
- Calibration factors should be between 0.8-1.2
- Coverage probabilities should be close to nominal values
- Interval widths should be reasonable (20-100 kWh/m²/year)

### 3. Result Comparison Script

```python
# Script to verify your results match expected patterns
import json
import numpy as np

def verify_results():
    """Verify that results are within expected ranges"""
    
    # Load results
    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Check performance metrics
    assert 40 <= metrics['rmse'] <= 60, f"RMSE {metrics['rmse']} outside expected range"
    assert 30 <= metrics['mae'] <= 45, f"MAE {metrics['mae']} outside expected range"
    assert 0.65 <= metrics['r2'] <= 0.80, f"R² {metrics['r2']} outside expected range"
    
    print("All metrics within expected ranges!")
    
    # Check model rankings
    models = ['hierarchical', 'bayesian_ridge', 'linear_regression', 'aeh_10', 'aeh_1']
    rmse_values = [metrics[f'{model}_rmse'] for model in models]
    
    # Verify ranking (lower RMSE is better)
    assert rmse_values[0] <= rmse_values[1] <= rmse_values[2], "Model ranking incorrect"
    
    print("Model rankings verified!")

if __name__ == "__main__":
    verify_results()
```

## Troubleshooting

### 1. Common Issues

#### Import Errors
```bash
# If you get import errors, reinstall packages
pip install --upgrade -r requirements.txt
```

#### Memory Issues
```python
# Reduce HMC steps for lower memory usage
config.hmc_steps = 100  # Instead of 200
config.hmc_leapfrog_steps = 5  # Instead of 10
```

#### Convergence Issues
```python
# Increase iterations for better convergence
config.max_iter = 200  # Instead of 100
config.tol = 1e-5  # Tighter tolerance
```

#### Numerical Instability
```python
# Use more conservative hyperparameters
config.beta_0 = 1e-5  # Instead of 1e-6
config.alpha_0 = 1e-5  # Instead of 1e-6
```

### 2. Platform-Specific Issues

#### Windows
- Use forward slashes in file paths
- Ensure Python is in PATH
- Use Windows-compatible package versions

#### macOS
- May need to install Xcode command line tools
- Use Homebrew for system dependencies
- Check Python version compatibility

#### Linux
- Install system dependencies: `sudo apt-get install python3-dev`
- Use virtual environment for isolation
- Check file permissions

### 3. Performance Optimization

#### Faster Execution
```python
# Reduce computational cost
config.n_splits = 2  # Fewer CV folds
config.hmc_steps = 100  # Fewer HMC steps
config.max_iter = 50  # Fewer EM iterations
```

#### Memory Optimization
```python
# Use smaller data sample for testing
sample_size = 1000  # Instead of full dataset
X = X[:sample_size]
y = y[:sample_size]
```

## Customization

### 1. Modifying Hyperparameters

```python
# Edit configuration in scripts
config = AdaptivePriorConfig(
    beta_0=10.0,  # Adjust prior strength
    max_iter=150,  # More iterations
    hmc_steps=300,  # More HMC steps
    prior_type='hierarchical'  # Different prior
)
```

### 2. Adding New Features

```python
# Add new features to feature engineering
def feature_engineering_custom(df):
    # Existing features...
    
    # New features
    df['new_feature'] = df['existing_feature'] ** 2
    df['interaction_feature'] = df['feature1'] * df['feature2']
    
    return df
```

### 3. Testing Different Priors

```python
# Test different prior specifications
prior_configs = [
    {'prior_type': 'hierarchical', 'beta_0': 1e-6},
    {'prior_type': 'spike_slab', 'beta_0': 1e-6},
    {'prior_type': 'horseshoe', 'beta_0': 1e-6}
]

for config in prior_configs:
    # Run experiment with this configuration
    pass
```

## Documentation and Reporting

### 1. Results Documentation

After running experiments, document your results:

```python
# Generate results summary
def generate_summary():
    """Generate summary of all results"""
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'system_info': platform.platform(),
        'python_version': sys.version,
        'results_summary': {}
    }
    
    # Add results from each experiment
    # Save to results_summary.json
    
    return summary
```

### 2. Reproducibility Checklist

- [ ] Environment setup completed
- [ ] All dependencies installed
- [ ] Data loaded and verified
- [ ] All experiments run successfully
- [ ] Results match expected patterns
- [ ] Output files generated correctly
- [ ] Performance metrics within expected ranges
- [ ] Model rankings verified
- [ ] Uncertainty calibration checked

### 3. Sharing Results

#### Archive Creation
```bash
# Create reproducible archive
tar -czf bpd_reproduction_$(date +%Y%m%d).tar.gz \
    V3_clean/ \
    requirements.txt \
    README.md
```

#### Results Package
Include in your results package:
- All Python scripts
- Requirements file
- Data preprocessing instructions
- Results summary
- Troubleshooting guide

## Support and Contact

### Getting Help
1. Check this reproducibility guide first
2. Review error messages carefully
3. Verify environment setup
4. Check data format and structure
5. Consult the main README.md

### Reporting Issues
When reporting issues, include:
- Operating system and version
- Python version
- Error messages
- Steps to reproduce
- Expected vs. actual behavior

---

*This reproducibility guide ensures that all research findings can be independently verified and reproduced. Follow these instructions carefully to achieve consistent results.* 