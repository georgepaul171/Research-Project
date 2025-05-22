import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Define columns to match gpr.py
numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "GHGEmissionsIntensity", "SiteEUI(kBtu/sf)"
]

target_col = "SiteEUI(kBtu/sf)"

def create_gp_model(X, y_data, input_dim):
    with pm.Model() as model:
        # Data containers
        X = pm.Data("X_data", X)
        y = pm.Data("y_data", y_data)

        # Simpler priors
        lengthscale = pm.Gamma("lengthscale", alpha=2, beta=0.5, shape=input_dim)
        variance = pm.HalfNormal("variance", sigma=20)
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Much larger jitter for numerical stability
        jitter = 1e-2

        # Simpler kernel structure - just RBF with white noise
        cov_rbf = variance**2 * pm.gp.cov.ExpQuad(
            input_dim=input_dim, 
            ls=lengthscale
        )
        
        # Add white noise kernel
        cov_white = pm.gp.cov.WhiteNoise(jitter)
        
        # Combine kernels
        cov = cov_rbf + cov_white

        # Gaussian Process
        gp = pm.gp.Marginal(cov_func=cov)
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

    return model, gp

def main():
    # Load data
    print("Loading data...")
    X_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_train_imputed.csv")[numeric_columns[:-1]]
    X_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_test_imputed.csv")[numeric_columns[:-1]]
    y_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_train_imputed.csv")[target_col]
    y_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_test_imputed.csv")[target_col]

    # Take a smaller subset for better stability
    subset_size = 500  # Reduced from 1000
    X_train = X_train.iloc[:subset_size]
    X_test = X_test.iloc[:subset_size]
    y_train = y_train.iloc[:subset_size]
    y_test = y_test.iloc[:subset_size]

    # Convert to numpy arrays and standardize
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32).reshape(-1)
    y_test_np = y_test.values.astype(np.float32).reshape(-1)

    # Standardize features
    X_mean = X_train_np.mean(axis=0)
    X_std = X_train_np.std(axis=0)
    X_train_np = (X_train_np - X_mean) / X_std
    X_test_np = (X_test_np - X_mean) / X_std

    # Create model
    print("Creating GP model...")
    gp_model, gp = create_gp_model(X_train_np, y_train_np, X_train.shape[1])

    # Load trace
    print("Loading saved trace...")
    try:
        trace = az.from_netcdf("gp_trace.nc")
        print("Trace loaded successfully")
    except Exception as e:
        print(f"Error loading trace: {e}")
        return

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

    except Exception as e:
        print(f"Error during prediction or plotting: {e}")
        print("Full error details:", str(e))
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 