import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import warnings
from scipy import stats
import shap
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# Create plots directory if it doesn't exist
PLOTS_DIR = "gpr-CBECS/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def create_gp_model(X, y_data, input_dim):
    with pm.Model() as model:
        # Data containers
        X = pm.Data("X_data", X)
        y = pm.Data("y_data", y_data)

        # More informative priors with regularization
        # RBF kernel with stronger regularization
        lengthscale_rbf = pm.Gamma("lengthscale_rbf", alpha=2, beta=1, shape=input_dim)
        variance_rbf = pm.HalfNormal("variance_rbf", sigma=10)  # Reduced variance

        # Linear kernel with stronger regularization
        variance_linear = pm.HalfNormal("variance_linear", sigma=5)
        offset = pm.HalfNormal("offset", sigma=5)

        # Noise parameters with stronger regularization
        sigma = pm.HalfNormal("sigma", sigma=2)
        jitter = 1e-3  # Reduced jitter

        # Kernel components
        cov_rbf = variance_rbf**2 * pm.gp.cov.ExpQuad(
            input_dim=input_dim, 
            ls=lengthscale_rbf
        )
        
        cov_linear = variance_linear**2 * pm.gp.cov.Linear(
            input_dim=input_dim,
            c=offset
        )
        
        cov_white = pm.gp.cov.WhiteNoise(jitter)
        
        # Combine kernels with regularization
        cov = cov_rbf + 0.5 * cov_linear + cov_white  # Reduced linear component

        # Gaussian Process
        gp = pm.gp.Marginal(cov_func=cov)
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

    return model, gp

def load_and_prepare_data():
    # Load data
    print("Loading data...")
    df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data-CBECS/Data_Ready_to_Model.csv')
    features = [
        'EUI_kWh_per_sqmt', 'SQMT', 'NFLOOR', 'FLCEILHT',
        'MONUSE', 'OCCUPYP', 'WKHRS', 'NWKER',
        'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65',
        'YRCONC', 'PUBCLIM'
    ]
    df_model = df[features].dropna()

    # Scale numeric features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df_model.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])),
        columns=[col for col in df_model.columns if col not in ['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']]
    )
    df_ready = pd.concat([X_scaled, df_model[['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']].reset_index(drop=True)], axis=1)

    # Take a balanced subset
    subset_size = 250  # Slightly increased for better coverage
    df_ready = df_ready.sample(n=subset_size, random_state=42)

    # Prepare X and y
    X = df_ready.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])
    y = df_ready['EUI_kWh_per_sqmt']

    # Convert to numpy arrays and ensure float32
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32).reshape(-1)

    return X_np, y_np, X.columns, X

def plot_results(y_true, mu_pred, std_pred, X_df, feature_names):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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

    # Simplified SHAP Analysis
    print("\nGenerating SHAP values...")
    from sklearn.ensemble import RandomForestRegressor
    surrogate = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced trees
    surrogate.fit(X_df, mu_pred)
    
    # Calculate SHAP values with fewer samples
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_df)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary_{timestamp}.png")
    plt.close()

    # Dependence plots for top 2 features only
    top_features = np.abs(shap_values).mean(0).argsort()[-2:][::-1]
    for idx in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values, X_df, show=False)
        plt.title(f"SHAP Dependence Plot - {feature_names[idx]}")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/shap_dependence_{feature_names[idx]}_{timestamp}.png")
        plt.close()

def plot_posterior_analysis(trace, model):
    """Generate posterior analysis plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot posterior distributions
    plt.figure(figsize=(15, 10))
    az.plot_posterior(trace, var_names=['lengthscale_rbf', 'variance_rbf', 
                                      'variance_linear', 'offset', 'sigma'])
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/posterior_distributions_{timestamp}.png")
    plt.close()
    
    # Plot trace
    plt.figure(figsize=(15, 10))
    az.plot_trace(trace, var_names=['lengthscale_rbf', 'variance_rbf', 
                                  'variance_linear', 'offset', 'sigma'])
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/trace_plot_{timestamp}.png")
    plt.close()
    
    # Print summary statistics
    print("\nPosterior Summary Statistics:")
    print(az.summary(trace, var_names=['lengthscale_rbf', 'variance_rbf', 
                                     'variance_linear', 'offset', 'sigma']))

def plot_calibration(y_true, mu_pred, std_pred):
    """Generate calibration plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate prediction intervals
    intervals = [0.5, 0.8, 0.9, 0.95, 0.99]
    coverage = []
    
    plt.figure(figsize=(10, 6))
    for interval in intervals:
        z_score = stats.norm.ppf((1 + interval) / 2)
        lower = mu_pred - z_score * std_pred
        upper = mu_pred + z_score * std_pred
        coverage.append(np.mean((y_true >= lower) & (y_true <= upper)))
        plt.plot([interval], [coverage[-1]], 'o', label=f'{interval*100}% interval')
    
    # Plot ideal calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Expected coverage')
    plt.ylabel('Actual coverage')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOTS_DIR}/calibration_plot_{timestamp}.png")
    plt.close()
    
    # Print coverage statistics
    print("\nCalibration Coverage:")
    for interval, cov in zip(intervals, coverage):
        print(f"{interval*100}% interval: {cov*100:.1f}% coverage")

def cross_validate_model(X, y, n_splits=5):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {'r2': [], 'rmse': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and fit model
        model, gp = create_gp_model(X_train, y_train, X.shape[1])
        
        with model:
            # Optimized sampling parameters
            trace = pm.sample(
                draws=1000,
                tune=1000,
                target_accept=0.95,
                return_inferencedata=True,
                cores=1,
                chains=4  # Increased to 4 chains
            )
            
            # Generate predictions
            f_pred = gp.conditional("f_pred", X_val)
            posterior_pred = pm.sample_posterior_predictive(
                trace, var_names=["f_pred"], random_seed=42
            )
        
        # Extract predictions
        f_samples = posterior_pred.posterior_predictive["f_pred"].values
        f_samples = f_samples.reshape(-1, f_samples.shape[-1])
        mu_pred = f_samples.mean(axis=0)
        
        # Calculate metrics
        r2 = r2_score(y_val, mu_pred)
        rmse = np.sqrt(mean_squared_error(y_val, mu_pred))
        
        cv_scores['r2'].append(r2)
        cv_scores['rmse'].append(rmse)
        
        print(f"\nFold {fold + 1}/{n_splits}:")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:.2f}")
    
    return cv_scores

def plot_cv_results(cv_scores):
    """Plot cross-validation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 5))
    
    # R² scores
    plt.subplot(1, 2, 1)
    plt.boxplot(cv_scores['r2'])
    plt.title('Cross-Validation R² Scores')
    plt.ylabel('R² Score')
    
    # RMSE scores
    plt.subplot(1, 2, 2)
    plt.boxplot(cv_scores['rmse'])
    plt.title('Cross-Validation RMSE Scores')
    plt.ylabel('RMSE')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/cv_results_{timestamp}.png")
    plt.close()
    
    # Print summary statistics
    print("\nCross-Validation Summary:")
    print(f"Mean R² Score: {np.mean(cv_scores['r2']):.3f} ± {np.std(cv_scores['r2']):.3f}")
    print(f"Mean RMSE: {np.mean(cv_scores['rmse']):.2f} ± {np.std(cv_scores['rmse']):.2f}")

def main():
    # Load and prepare data
    X_np, y_np, feature_names, X_df = load_and_prepare_data()

    # Perform cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_validate_model(X_np, y_np)
    plot_cv_results(cv_scores)

    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    gp_model, gp = create_gp_model(X_np, y_np, X_np.shape[1])

    with gp_model:
        # Optimized sampling parameters
        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.95,
            return_inferencedata=True,
            cores=1,
            chains=4
        )

    # Save the trace
    trace.to_netcdf("gpr_cbecs_trace.nc")

    # Generate predictions
    print("\nGenerating predictions...")
    with gp_model:
        f_pred = gp.conditional("f_pred", X_np)
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
    r2 = r2_score(y_np, mu_pred)
    rmse = np.sqrt(mean_squared_error(y_np, mu_pred))

    print(f"\nModel Performance:")
    print(f"GP R² Score: {r2:.3f}")
    print(f"GP RMSE: {rmse:.2f}")

    # Plot results and generate SHAP analysis
    plot_results(y_np, mu_pred, std_pred, X_df, feature_names)
    
    # Add posterior and calibration analysis
    plot_posterior_analysis(trace, gp_model)
    plot_calibration(y_np, mu_pred, std_pred)

if __name__ == "__main__":
    main() 