import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def debug_print(msg, obj=None):
    print(f"\n=== {msg} ===")
    if obj is not None:
        print(f"Type: {type(obj)}")
        if hasattr(obj, 'shape'):
            print(f"Shape: {obj.shape}")
        if hasattr(obj, 'keys'):
            print(f"Keys: {obj.keys()}")
        print(f"Value: {obj}")

# Load the saved trace
print("Loading saved trace...")
trace = az.from_netcdf("gp_trace.nc")
debug_print("Loaded trace", trace)
print("Trace loaded successfully")

# Load training data (needed for predictions)
print("\nLoading training data...")
X_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_train_imputed.csv")
y_train = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_train_imputed.csv")

# Load test data
print("\nLoading test data...")
X_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_test_imputed.csv")
y_test = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_test_imputed.csv")

# Take the same subset size as training
subset_size = 1000
X_train = X_train.iloc[:subset_size]
y_train = y_train.iloc[:subset_size]
X_test = X_test.iloc[:subset_size]
y_test = y_test.iloc[:subset_size]

# Convert to numpy arrays
X_train_np = X_train.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32).reshape(-1)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32).reshape(-1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)
X_test_scaled = scaler.transform(X_test_np).astype(np.float32)

debug_print("Data shapes", {
    "X_train_scaled": X_train_scaled.shape,
    "y_train_np": y_train_np.shape,
    "X_test_scaled": X_test_scaled.shape,
    "y_test_np": y_test_np.shape
})

# Create a simple GP model for predictions
def create_prediction_model(X_train, y_train, X_test, input_dim):
    debug_print("Creating prediction model", {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "input_dim": input_dim
    })
    
    with pm.Model() as model:
        # Data containers
        X_train_data = pm.Data("X_train", X_train)
        y_train_data = pm.Data("y_train", y_train)
        X_test_data = pm.Data("X_test", X_test)
        
        debug_print("Created data containers", {
            "X_train": X_train_data,
            "y_train": y_train_data,
            "X_test": X_test_data
        })
        
        # Hyperparameters
        lengthscale = pm.Gamma("lengthscale", alpha=2, beta=1, shape=input_dim)
        variance = pm.HalfNormal("variance", sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        
        debug_print("Created hyperparameters", {
            "lengthscale": lengthscale,
            "variance": variance,
            "sigma": sigma
        })

        # Covariance function
        cov = variance**2 * pm.gp.cov.ExpQuad(
            input_dim=input_dim, 
            ls=lengthscale
        )
        debug_print("Created covariance function", cov)

        # Gaussian Process
        gp = pm.gp.Marginal(cov_func=cov)
        debug_print("Created GP", gp)
        
        # Add marginal likelihood
        y_ = gp.marginal_likelihood("y", X=X_train_data, y=y_train_data, noise=sigma)
        
        # Add conditional for predictions
        f_pred = gp.conditional("f_pred", X_test_data)
        
    return model, gp

# Create model for predictions
print("\nCreating prediction model...")
pred_model, gp = create_prediction_model(X_train_scaled, y_train_np, X_test_scaled, X_train.shape[1])
debug_print("Created prediction model", pred_model)

# Make predictions
print("\nGenerating predictions...")
try:
    with pred_model:
        debug_print("Setting test data")
        pm.set_data({
            "X_train": X_train_scaled,
            "y_train": y_train_np,
            "X_test": X_test_scaled
        })
        
        # Sample from the posterior predictive
        print("\nSampling from posterior predictive...")
        debug_print("Trace before sampling", trace)
        ppc = pm.sample_posterior_predictive(
            trace, 
            var_names=["f_pred"],
            random_seed=42
        )
        debug_print("Posterior predictive samples", ppc)
        
        # Extract predictions
        print("\nExtracting predictions...")
        debug_print("Available keys in ppc", ppc.keys())
        
        # Get mean and standard deviation of predictions
        f_samples = ppc.posterior_predictive["f_pred"]
        debug_print("Prediction samples", f_samples)
        
        # Reshape samples to (n_samples, n_test_points)
        f_samples = f_samples.values.reshape(-1, f_samples.shape[-1])
        mu_pred = f_samples.mean(axis=0)
        std_pred = f_samples.std(axis=0)
        
        debug_print("Final predictions", {
            "mean_shape": mu_pred.shape,
            "std_shape": std_pred.shape,
            "mean": mu_pred,
            "std": std_pred
        })
        
        # Evaluate
        r2 = r2_score(y_test_np, mu_pred)
        rmse = np.sqrt(mean_squared_error(y_test_np, mu_pred))
        
        print(f"\nModel Performance:")
        print(f"GP R² Score: {r2:.3f}")
        print(f"GP RMSE: {rmse:.2f}")
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test_np, mu_pred, alpha=0.5)
        plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Values")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"Error during prediction: {e}")
    print("Full error details:", str(e))
    import traceback
    print("\nFull traceback:")
    traceback.print_exc() 