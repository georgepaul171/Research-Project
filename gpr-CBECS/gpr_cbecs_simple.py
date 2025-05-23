import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
from scipy import stats
import shap
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# Create plots directory if it doesn't exist
PLOTS_DIR = "gpr-CBECS/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def select_features(X, y, k=8):
    """Select top k features using f-regression."""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_selected, selected_indices

def create_simple_gp_model(X, y_data, input_dim):
    """Create a simplified GP model with only RBF kernel."""
    with pm.Model() as model:
        # Data containers
        X = pm.Data("X_data", X)
        y = pm.Data("y_data", y_data)

        # Simplified priors
        lengthscale = pm.Gamma("lengthscale", alpha=2, beta=1, shape=input_dim)
        variance = pm.HalfNormal("variance", sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=1)
        jitter = 1e-4

        # Single RBF kernel
        cov = variance**2 * pm.gp.cov.ExpQuad(
            input_dim=input_dim, 
            ls=lengthscale
        ) + pm.gp.cov.WhiteNoise(jitter)

        # Gaussian Process
        gp = pm.gp.Marginal(cov_func=cov)
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

    return model, gp

def load_and_prepare_data():
    """Load and prepare data with improved preprocessing."""
    print("Loading data...")
    df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data-CBECS/Data_Ready_to_Model.csv')
    
    # Define features
    features = [
        'EUI_kWh_per_sqmt', 'SQMT', 'NFLOOR', 'FLCEILHT',
        'MONUSE', 'OCCUPYP', 'WKHRS', 'NWKER',
        'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65',
        'YRCONC', 'PUBCLIM'
    ]
    
    # Prepare data
    df_model = df[features].dropna()
    
    # Log transform target variable
    y = np.log1p(df_model['EUI_kWh_per_sqmt'])
    
    # Prepare features
    X = df_model.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select top features
    X_selected, selected_indices = select_features(X_scaled, y)
    selected_features = X.columns[selected_indices]
    
    print("\nSelected features:")
    for idx, feature in enumerate(selected_features):
        print(f"{idx + 1}. {feature}")
    
    # Take a balanced subset
    subset_size = 300  # Slightly increased
    indices = np.random.choice(len(X_selected), subset_size, replace=False)
    X_subset = X_selected[indices]
    y_subset = y[indices]
    
    return X_subset, y_subset, selected_features, X.iloc[indices]

def cross_validate_model(X, y, n_splits=5):
    """Perform k-fold cross-validation with error handling."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {'r2': [], 'rmse': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        try:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and fit model
            model, gp = create_simple_gp_model(X_train, y_train, X.shape[1])
            
            with model:
                # Sampling with error handling
                try:
                    trace = pm.sample(
                        draws=800,
                        tune=800,
                        target_accept=0.95,
                        return_inferencedata=True,
                        cores=1,
                        chains=4
                    )
                except Exception as e:
                    print(f"Sampling error in fold {fold + 1}: {str(e)}")
                    continue
                
                # Generate predictions
                f_pred = gp.conditional("f_pred", X_val)
                posterior_pred = pm.sample_posterior_predictive(
                    trace, var_names=["f_pred"], random_seed=42
                )
            
            # Extract predictions
            f_samples = posterior_pred.posterior_predictive["f_pred"].values
            f_samples = f_samples.reshape(-1, f_samples.shape[-1])
            mu_pred = f_samples.mean(axis=0)
            
            # Transform predictions back to original scale
            mu_pred = np.expm1(mu_pred)
            y_val_orig = np.expm1(y_val)
            
            # Calculate metrics
            r2 = r2_score(y_val_orig, mu_pred)
            rmse = np.sqrt(mean_squared_error(y_val_orig, mu_pred))
            
            cv_scores['r2'].append(r2)
            cv_scores['rmse'].append(rmse)
            
            print(f"\nFold {fold + 1}/{n_splits}:")
            print(f"R² Score: {r2:.3f}")
            print(f"RMSE: {rmse:.2f}")
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")
            continue
    
    return cv_scores

def plot_results(y_true, mu_pred, std_pred, X_df, feature_names):
    """Plot results with improved visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Transform back to original scale
    y_true = np.expm1(y_true)
    mu_pred = np.expm1(mu_pred)
    
    # Create main results plot
    plt.figure(figsize=(15, 10))
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, mu_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual EUI (kWh/m²)")
    plt.ylabel("Predicted EUI (kWh/m²)")
    plt.title("Actual vs Predicted Values")
    plt.grid(True)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = y_true - mu_pred
    plt.scatter(mu_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted EUI (kWh/m²)")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    
    # Uncertainty
    plt.subplot(2, 2, 3)
    plt.scatter(mu_pred, std_pred, alpha=0.5)
    plt.xlabel("Predicted EUI (kWh/m²)")
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
    plt.savefig(f"{PLOTS_DIR}/model_results_{timestamp}.png")
    plt.close()
    
    # SHAP Analysis
    print("\nGenerating SHAP values...")
    from sklearn.ensemble import RandomForestRegressor
    surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
    surrogate.fit(X_df, mu_pred)
    
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_df)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary_{timestamp}.png")
    plt.close()

def main():
    # Load and prepare data
    X_np, y_np, feature_names, X_df = load_and_prepare_data()
    
    # Perform cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_validate_model(X_np, y_np)
    
    # Print CV summary
    print("\nCross-Validation Summary:")
    print(f"Mean R² Score: {np.mean(cv_scores['r2']):.3f} ± {np.std(cv_scores['r2']):.3f}")
    print(f"Mean RMSE: {np.mean(cv_scores['rmse']):.2f} ± {np.std(cv_scores['rmse']):.2f}")
    
    # Train final model
    print("\nTraining final model...")
    model, gp = create_simple_gp_model(X_np, y_np, X_np.shape[1])
    
    with model:
        trace = pm.sample(
            draws=800,
            tune=800,
            target_accept=0.95,
            return_inferencedata=True,
            cores=1,
            chains=4
        )
    
    # Generate predictions
    print("\nGenerating predictions...")
    with model:
        f_pred = gp.conditional("f_pred", X_np)
        posterior_pred = pm.sample_posterior_predictive(
            trace, var_names=["f_pred"], random_seed=42
        )
    
    # Extract predictions
    f_samples = posterior_pred.posterior_predictive["f_pred"].values
    f_samples = f_samples.reshape(-1, f_samples.shape[-1])
    mu_pred = f_samples.mean(axis=0)
    std_pred = f_samples.std(axis=0)
    
    # Plot results
    plot_results(y_np, mu_pred, std_pred, X_df, feature_names)

if __name__ == "__main__":
    main() 