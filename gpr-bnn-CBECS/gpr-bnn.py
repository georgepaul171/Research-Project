# Hybrid Bayesian Neural Network + Gaussian Process Regression
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

def bayesian_nn(X, input_dim, hidden_dim=10):
    weights_in_1 = pm.Normal('w_in_1', mu=0, sigma=1, shape=(input_dim, hidden_dim))
    bias_in_1 = pm.Normal('b_in_1', mu=0, sigma=1, shape=(hidden_dim,))
    layer_1 = pm.math.tanh(pm.math.dot(X, weights_in_1) + bias_in_1)

    weights_1_out = pm.Normal('w_1_out', mu=0, sigma=1, shape=(hidden_dim,))
    bias_1_out = pm.Normal('b_1_out', mu=0, sigma=1)

    output = pm.math.dot(layer_1, weights_1_out) + bias_1_out
    return output

def create_bnn_gp_model(X, y_data, input_dim):
    with pm.Model() as model:
        X_data = pm.Data("X_data", X)
        y = pm.Data("y_data", y_data)

        # BNN output as latent features
        bnn_output = bayesian_nn(X_data, input_dim)
        bnn_output_reshaped = bnn_output.reshape((-1, 1))

        # GP on BNN output
        lengthscale = pm.Gamma("lengthscale", alpha=2, beta=1)
        variance = pm.HalfNormal("variance", sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        cov_func = variance**2 * pm.gp.cov.ExpQuad(1, ls=lengthscale)
        gp = pm.gp.Marginal(cov_func=cov_func)

        y_ = gp.marginal_likelihood("y", X=bnn_output_reshaped, y=y, noise=sigma)

    return model, gp

def load_and_prepare_data():
    df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data-CBECS/Data_Ready_to_Model.csv')
    features = [
        'EUI_kWh_per_sqmt', 'SQMT', 'NFLOOR', 'FLCEILHT',
        'MONUSE', 'OCCUPYP', 'WKHRS', 'NWKER',
        'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65',
        'YRCONC', 'PUBCLIM'
    ]
    df_model = df[features].dropna()
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df_model.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])),
        columns=[col for col in df_model.columns if col not in ['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']]
    )
    df_ready = pd.concat([X_scaled, df_model[['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']].reset_index(drop=True)], axis=1)
    df_ready = df_ready.sample(n=250, random_state=42)
    X = df_ready.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])
    y = df_ready['EUI_kWh_per_sqmt']
    X_np = X.values.astype(np.float32)  # Use float32 for better performance
    y_np = y.values.astype(np.float32).reshape(-1)
    return X_np, y_np, X.columns, X

def cross_validate_model(X, y, n_splits=2):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {'r2': [], 'rmse': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nProcessing fold {fold + 1}/{n_splits}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, gp = create_bnn_gp_model(X_train, y_train, X.shape[1])

        with model:
            print("Sampling...")
            # More stable sampling configuration
            trace = pm.sample(
                draws=300,  # Further reduced for stability
                tune=300,   # Further reduced for stability
                target_accept=0.9,  # Slightly reduced for better mixing
                return_inferencedata=True,
                cores=1,  # Single core for stability
                chains=2,
                random_seed=42
            )
            print("Computing predictions...")
            f_pred = gp.conditional("f_pred", X_val.reshape((-1, X_val.shape[1])))
            posterior_pred = pm.sample_posterior_predictive(
                trace,
                var_names=["f_pred"],
                random_seed=42
            )

        f_samples = posterior_pred.posterior_predictive["f_pred"].values
        f_samples = f_samples.reshape(-1, f_samples.shape[-1])
        mu_pred = f_samples.mean(axis=0)

        r2 = r2_score(y_val, mu_pred)
        rmse = np.sqrt(mean_squared_error(y_val, mu_pred))

        cv_scores['r2'].append(r2)
        cv_scores['rmse'].append(rmse)

        print(f"Fold {fold + 1}/{n_splits} Results:")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:.2f}")

    return cv_scores

def main():
    X_np, y_np, feature_names, X_df = load_and_prepare_data()
    print("Performing cross-validation...")
    cv_scores = cross_validate_model(X_np, y_np)
    print("\nCross-Validation Results:")
    print(f"Mean R²: {np.mean(cv_scores['r2']):.3f}, Std: {np.std(cv_scores['r2']):.3f}")
    print(f"Mean RMSE: {np.mean(cv_scores['rmse']):.2f}, Std: {np.std(cv_scores['rmse']):.2f}")

    print("\nTraining final model...")
    model, gp = create_bnn_gp_model(X_np, y_np, X_np.shape[1])
    with model:
        # More stable sampling configuration for final model
        trace = pm.sample(
            draws=300,  # Further reduced for stability
            tune=300,   # Further reduced for stability
            target_accept=0.9,  # Slightly reduced for better mixing
            return_inferencedata=True,
            cores=1,  # Single core for stability
            chains=2,
            random_seed=42
        )
        f_pred = gp.conditional("f_pred", X_np.reshape((-1, X_np.shape[1])))
        posterior_pred = pm.sample_posterior_predictive(
            trace,
            var_names=["f_pred"],
            random_seed=42
        )

    f_samples = posterior_pred.posterior_predictive["f_pred"].values.reshape(-1, len(y_np))
    mu_pred = f_samples.mean(axis=0)
    std_pred = f_samples.std(axis=0)

    mu_pred = np.maximum(mu_pred, 0)

    print("\nFinal Model Performance:")
    print(f"R² Score: {r2_score(y_np, mu_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_np, mu_pred)):.2f}")

if __name__ == "__main__":
    main()