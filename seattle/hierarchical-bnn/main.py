# main.py
# Hierarchical Bayesian Neural Network for EUI Estimation (OOP Version)

import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class HBNNEstimator:
    def __init__(self, X_train, y_train, group_idx, n_groups):
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.group_idx = group_idx.astype("int32")
        self.n_groups = n_groups
        self.model = None
        self.trace = None

    def build_model(self):
        with pm.Model() as model:
            X_data = pm.Data("X_data", self.X_train)
            y_data = pm.Data("y_data", self.y_train)
            group_idx = pm.Data("group_idx", self.group_idx)

            group_mu = pm.Normal("group_mu", mu=0, sigma=1, shape=self.n_groups)
            group_bias = group_mu[group_idx]

            w1 = pm.Normal("w1", mu=0, sigma=1, shape=(self.X_train.shape[1], 32))
            b1 = pm.Normal("b1", mu=0, sigma=1, shape=(32,))
            z1 = pt.tanh(pt.dot(X_data, w1) + b1)

            w2 = pm.Normal("w2", mu=0, sigma=1, shape=(32, 16))
            b2 = pm.Normal("b2", mu=0, sigma=1, shape=(16,))
            z2 = pt.tanh(pt.dot(z1, w2) + b2)

            w_out = pm.Normal("w_out", mu=0, sigma=1, shape=(16,))
            b_out = pm.Normal("b_out", mu=0, sigma=1)
            mu = pt.dot(z2, w_out) + b_out + group_bias

            sigma = pm.HalfNormal("sigma", sigma=1)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

            self.model = model

    def fit(self, draws=500, tune=500, chains=1):
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9)
        return self.trace

    def predict(self, X_test, group_idx_test):
        with self.model:
            pm.set_data({"X_data": X_test.astype(np.float32), "group_idx": group_idx_test.astype("int32")})
            ppc = pm.sample_posterior_predictive(self.trace, var_names=["y_obs"])
        return ppc["y_obs"]

    def evaluate(self, y_true, y_pred_samples):
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_std = y_pred_samples.std(axis=0)

        r2 = r2_score(y_true, y_pred_mean)
        rmse = mean_squared_error(y_true, y_pred_mean, squared=False)
        mae = mean_absolute_error(y_true, y_pred_mean)

        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Avg Sigma": np.mean(y_pred_std),
            "Max Sigma": np.max(y_pred_std)
        }

if __name__ == "__main__":
    data_path = "../data"
    output_path = "results"
    os.makedirs(output_path, exist_ok=True)

    X_train = pd.read_csv(os.path.join(data_path, "X_train_imputed.csv"))
    y_train = pd.read_csv(os.path.join(data_path, "y_train_imputed.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test_imputed.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "y_test_imputed.csv"))

    y_train.columns = ["SiteEUI(kBtu/sf)"] if y_train.columns[0] != "SiteEUI(kBtu/sf)" else y_train.columns
    y_test.columns = ["SiteEUI(kBtu/sf)"] if y_test.columns[0] != "SiteEUI(kBtu/sf)" else y_test.columns
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Group by YearBuilt bucket
    bins = pd.qcut(X_train["YearBuilt"], q=4, retbins=True, labels=False)[1]
    X_train["YearBuilt_Group"] = pd.cut(X_train["YearBuilt"], bins=bins, labels=False, include_lowest=True)
    X_test["YearBuilt_Group"] = pd.cut(X_test["YearBuilt"], bins=bins, labels=False, include_lowest=True)

    group_idx_train = X_train.pop("YearBuilt_Group")
    group_idx_test = X_test.pop("YearBuilt_Group")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    n_groups = len(np.unique(group_idx_train))
    
    estimator = HBNNEstimator(X_train_scaled, y_train, group_idx_train, n_groups)
    estimator.build_model()

    trace_file = os.path.join(output_path, "hbnn_trace.nc")
    if os.path.exists(trace_file):
        print(f"Loading trace from {trace_file}...")
        estimator.trace = az.from_netcdf(trace_file)
    else:
        trace = estimator.fit()
        az.to_netcdf(trace, trace_file)
        print(f"Saved trace to {trace_file}")

    preds = estimator.predict(X_test_scaled, group_idx_test)
    metrics = estimator.evaluate(y_test, preds)

    with open(os.path.join(output_path, "hbnn_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.3f}\n")

    print("Model evaluation complete. Metrics written to file.")