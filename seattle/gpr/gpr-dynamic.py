import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from scipy import stats
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

def print_data_stats(X, y, name=""):
    print(f"\n=== {name} Data Statistics ===")
    print("\nFeature Statistics:")
    for i, col in enumerate(numeric_columns):
        print(f"\n{col}:")
        print(f"  Mean: {X[:, i].mean():.2f}")
        print(f"  Std: {X[:, i].std():.2f}")
        print(f"  Min: {X[:, i].min():.2f}")
        print(f"  Max: {X[:, i].max():.2f}")
        print(f"  Correlation with target: {stats.pearsonr(X[:, i], y)[0]:.3f}")
    
    print(f"\nTarget Statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")

# Define columns to match gpr.py
numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "GHGEmissionsIntensity", "SiteEUI(kBtu/sf)"
]

target_col = "SiteEUI(kBtu/sf)"

# Load and clean
X_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_train_imputed.csv")[numeric_columns[:-1]]
X_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_test_imputed.csv")[numeric_columns[:-1]]
y_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_train_imputed.csv")[target_col]
y_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_test_imputed.csv")[target_col]

# Take a moderate subset for better training
subset_size = 1000
X_train = X_train.iloc[:subset_size]
X_test = X_test.iloc[:subset_size]
y_train = y_train.iloc[:subset_size]
y_test = y_test.iloc[:subset_size]

# Convert to numpy arrays and ensure correct types and shapes
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32).reshape(-1)
y_test_np = y_test.values.astype(np.float32).reshape(-1)

print("\n=== Feature-Target Correlations (Training Set) ===")
for i, col in enumerate(numeric_columns[:-1]):
    corr, pval = stats.pearsonr(X_train_np[:, i], y_train_np)
    print(f"{col:20s}: correlation = {corr: .3f} (p={pval:.2g})")

print("\n=== Feature-Target Correlations (Test Set) ===")
for i, col in enumerate(numeric_columns[:-1]):
    corr, pval = stats.pearsonr(X_test_np[:, i], y_test_np)
    print(f"{col:20s}: correlation = {corr: .3f} (p={pval:.2g})")

# Exit after printing correlations
sys.exit(0)

def create_gp_model(X, y_data, input_dim):
    print(f"\nInside create_gp_model:")
    print(f"X shape: {X.shape}")
    print(f"y_data shape: {y_data.shape}")
    print(f"input_dim: {input_dim}")
    
    with pm.Model() as model:
        # Data containers
        X = pm.Data("X_data", X)
        y = pm.Data("y_data", y_data)

        # More informative priors based on domain knowledge
        # Shorter lengthscales for energy-related features
        lengthscale_energy = pm.Gamma("lengthscale_energy", alpha=2, beta=0.5, shape=3)  # For energy features
        lengthscale_building = pm.Gamma("lengthscale_building", alpha=2, beta=0.2, shape=3)  # For building features
        
        # Combine lengthscales
        lengthscale = pt.concatenate([lengthscale_energy, lengthscale_building])
        
        # Variance with more informative prior
        variance = pm.HalfNormal("variance", sigma=20)
        
        # Noise with more informative prior
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Kernel structure
        # 1. RBF kernel for general smoothness
        cov_rbf = variance**2 * pm.gp.cov.ExpQuad(
            input_dim=input_dim, 
            ls=lengthscale
        )
        
        # 2. Linear kernel for linear trends
        cov_linear = variance**2 * pm.gp.cov.Linear(
            input_dim=input_dim,
            c=0.1
        )
        
        # Combine kernels
        cov = cov_rbf + 0.3 * cov_linear

        # Gaussian Process
        gp = pm.gp.Marginal(cov_func=cov)
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

    return model, gp

# Build the GP model
print("\nBuilding GP model...")
gp_model, gp = create_gp_model(X_train_np, y_train_np, X_train.shape[1])

# Use variational inference with better settings
with gp_model:
    print("\nStarting variational inference...")
    approx = pm.fit(
        n=2000,  # Increased iterations for better convergence
        method='advi',
        random_seed=42,
        obj_optimizer=pm.adam(learning_rate=0.05)  # Moderate learning rate
    )
    print("Variational inference completed.")
    print(f"Final ELBO: {approx.hist[-1]:.2f}")

# Get the trace with more samples
trace = approx.sample(200)  # Increased samples

# Print trace summary
print("\nTrace Summary:")
print(az.summary(trace))

# Save the trace
try:
    az.to_netcdf(trace, "gp_trace.nc")
    print("Trace saved successfully")
except Exception as e:
    print(f"Error saving trace: {e}")

# Get predictions
print("\nGenerating predictions...")
try:
    with gp_model:
        pm.set_data({"X_data": X_test_np})
        f_pred = gp.conditional("f_pred", X_test_np)
        posterior_pred = pm.sample_posterior_predictive(
            trace, var_names=["f_pred"], random_seed=42
        )
    
    # Extract predictions
    f_samples = posterior_pred.posterior_predictive["f_pred"].values
    f_samples = f_samples.reshape(-1, f_samples.shape[-1])
    mu_pred = f_samples.mean(axis=0)
    std_pred = f_samples.std(axis=0)

    # Ensure predictions are non-negative
    mu_pred = np.maximum(mu_pred, 0)

    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(f"Mean prediction: {mu_pred.mean():.2f}")
    print(f"Std of predictions: {mu_pred.std():.2f}")
    print(f"Min prediction: {mu_pred.min():.2f}")
    print(f"Max prediction: {mu_pred.max():.2f}")
    print(f"Mean uncertainty (std): {std_pred.mean():.2f}")

    # Evaluate
    r2 = r2_score(y_test_np, mu_pred)
    rmse = np.sqrt(mean_squared_error(y_test_np, mu_pred))

    print(f"\nModel Performance:")
    print(f"GP R² Score: {r2:.3f}")
    print(f"GP RMSE: {rmse:.2f}")

    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_np, mu_pred, alpha=0.5)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Values")
    plt.grid(True)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = y_test_np - mu_pred
    plt.scatter(mu_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    
    # Uncertainty
    plt.subplot(2, 2, 3)
    plt.scatter(mu_pred, std_pred, alpha=0.5)
    plt.xlabel("Predicted Value")
    plt.ylabel("Uncertainty (std)")
    plt.title("Prediction Uncertainty")
    plt.grid(True)
    
    # Residual Distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.5)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Basic diagnostics
    print("\nModel Summary:")
    print(az.summary(trace))
    
except Exception as e:
    print(f"Error during prediction or plotting: {e}")
    print("Full error details:", str(e))
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
