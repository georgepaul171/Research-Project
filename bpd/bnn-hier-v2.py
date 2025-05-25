import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap
from scipy import stats
import seaborn as sns

class HierarchicalBayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Hyperpriors: learnable prior mean and log-variance
        self.prior_mu = nn.Parameter(torch.zeros(1))
        self.prior_logvar = nn.Parameter(torch.zeros(1))
        # Posterior parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias), self.kl_loss()

    def kl_loss(self) -> torch.Tensor:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl_weights = self.kl_divergence(self.weight_mu, weight_sigma)
        kl_bias = self.kl_divergence(self.bias_mu, bias_sigma)
        kl_prior_mu = 0.5 * (self.prior_mu ** 2)
        kl_prior_logvar = 0.5 * (self.prior_logvar ** 2)
        return kl_weights + kl_bias + kl_prior_mu + kl_prior_logvar

    def kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        prior_var = torch.exp(self.prior_logvar)
        prior_mu = self.prior_mu
        kl = torch.log(prior_var.sqrt() / sigma) + (sigma ** 2 + (mu - prior_mu) ** 2) / (2 * prior_var) - 0.5
        return kl.sum()

class HierarchicalBayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(HierarchicalBayesianLinear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        layers.append(HierarchicalBayesianLinear(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_sum = 0
        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
            kl_sum += kl
        return x, kl_sum

    def predict(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0), predictions.std(dim=0)

class ModelEvaluator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate Expected Calibration Error (ECE)
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        ece = 0
        for i in range(num_bins):
            mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_std = y_std[mask]
                bin_residuals = np.abs(y_true[mask] - y_pred[mask])
                empirical_coverage = np.mean(bin_residuals <= 2 * bin_std)
                expected_coverage = 0.95  # For 2 standard deviations
                ece += np.abs(empirical_coverage - expected_coverage) * np.sum(mask) / len(y_true)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'ECE': ece
        }

    def plot_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray):
        num_bins = 10
        bin_edges = np.linspace(0, y_std.max(), num_bins + 1)
        calibration_errors = []
        empirical_coverage = []
        
        for i in range(num_bins):
            mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_std = y_std[mask]
                bin_residuals = np.abs(y_true[mask] - y_pred[mask])
                empirical_coverage.append(np.mean(bin_residuals <= 2 * bin_std))
                calibration_errors.append(np.mean(bin_std))

        plt.figure(figsize=(10, 6))
        plt.plot(calibration_errors, empirical_coverage, 'bo-', label='Model')
        plt.plot([0, max(calibration_errors)], [0, max(calibration_errors)], 'r--', label='Perfect Calibration')
        plt.xlabel('Predicted Uncertainty')
        plt.ylabel('Empirical Coverage')
        plt.title('Uncertainty Calibration Plot')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'calibration_plot.png'))
        plt.close()

    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray, 
                              confidence_intervals: Optional[np.ndarray] = None):
        plt.figure(figsize=(12, 6))
        if confidence_intervals is not None:
            plt.bar(feature_names, importance_scores, yerr=confidence_intervals, capsize=5)
        else:
            plt.bar(feature_names, importance_scores)
        plt.xticks(rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance with Confidence Intervals')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()

    def plot_partial_dependence(self, model, X: np.ndarray, feature_names: List[str], 
                              feature_idx: int, num_points: int = 50):
        feature_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), num_points)
        predictions = []
        
        for value in feature_range:
            X_modified = X.copy()
            X_modified[:, feature_idx] = value
            X_tensor = torch.tensor(X_modified, dtype=torch.float32)
            with torch.no_grad():
                pred, _ = model(X_tensor)
            predictions.append(pred.mean().item())
        
        plt.figure(figsize=(10, 6))
        plt.plot(feature_range, predictions)
        plt.xlabel(feature_names[feature_idx])
        plt.ylabel('Predicted Site EUI')
        plt.title(f'Partial Dependence Plot for {feature_names[feature_idx]}')
        plt.savefig(os.path.join(self.output_dir, f'partial_dependence_{feature_names[feature_idx]}.png'))
        plt.close()

def posterior_predictive_checks(y_true, y_pred, y_std, output_dir):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true.flatten(), label='Actual', fill=True)
    sns.kdeplot(y_pred.flatten(), label='Predicted Mean', fill=True)
    plt.title('Posterior Predictive Check: Target vs. Predicted Mean')
    plt.xlabel('Normalized Site EUI')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ppc_target_vs_pred_mean.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(y_std.flatten(), bins=30, alpha=0.7)
    plt.title('Posterior Predictive Check: Predictive Uncertainty Distribution')
    plt.xlabel('Predictive Std (Uncertainty)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'ppc_pred_std_hist.png'))
    plt.close()

def plot_parameter_histograms(model, output_dir):
    for i, layer in enumerate(model.layers):
        weight_mu = layer.weight_mu.detach().cpu().numpy().flatten()
        weight_sigma = torch.log1p(torch.exp(layer.weight_rho)).detach().cpu().numpy().flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(weight_mu, bins=30, alpha=0.7, label='weight_mu')
        plt.hist(weight_sigma, bins=30, alpha=0.7, label='weight_sigma')
        plt.legend()
        plt.title(f'Posterior Parameter Distributions (Layer {i})')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, f'posterior_param_hist_layer{i}.png'))
        plt.close()

def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                      n_splits: int = 5, output_dir: str = 'results'):
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir)
    
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    fold_metrics = []
    feature_importance_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # Initialize model
        model = HierarchicalBayesianNeuralNetwork(input_dim=X.shape[1], 
                                                hidden_dims=[64, 32], 
                                                output_dim=1)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        kl_weight = 1e-3
        
        for epoch in range(100):
            model.train()
            train_loss = 0
            for batch_x, batch_y in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                batch_size=32, shuffle=True):
                
                optimizer.zero_grad()
                output, kl = model(batch_x)
                mse_loss = F.mse_loss(output, batch_y)
                loss = mse_loss + kl_weight * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_output, val_kl = model(X_val_tensor)
                val_loss = F.mse_loss(val_output, y_val_tensor) + kl_weight * val_kl
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 
                         os.path.join(output_dir, f'best_model_fold_{fold}.pth'))
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred, y_std = model.predict(X_val_tensor)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_val, y_pred.numpy(), y_std.numpy())
        fold_metrics.append(metrics)
        
        # Feature importance analysis
        importance_scores = []
        for i in range(len(feature_names)):
            X_perturbed = X_val.copy()
            X_perturbed[:, i] = X_perturbed[:, i] + 0.1
            X_perturbed_tensor = torch.tensor(X_perturbed, dtype=torch.float32)
            
            with torch.no_grad():
                orig_pred, _ = model(X_val_tensor)
                pert_pred, _ = model(X_perturbed_tensor)
            
            importance = torch.mean(torch.abs(pert_pred - orig_pred)).item()
            importance_scores.append(importance)
        
        feature_importance_scores.append(importance_scores)
        
        # Generate plots for this fold
        evaluator.plot_calibration(y_val, y_pred.numpy(), y_std.numpy())
        evaluator.plot_feature_importance(feature_names, np.array(importance_scores))
        
        # Partial dependence plots
        for i in range(len(feature_names)):
            evaluator.plot_partial_dependence(model, X_val, feature_names, i)
    
    # Aggregate results
    mean_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) 
                   for metric in fold_metrics[0].keys()}
    std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics]) 
                  for metric in fold_metrics[0].keys()}
    
    mean_importance = np.mean(feature_importance_scores, axis=0)
    std_importance = np.std(feature_importance_scores, axis=0)
    
    # Final plots with confidence intervals
    evaluator.plot_feature_importance(feature_names, mean_importance, std_importance)
    
    return mean_metrics, std_metrics, mean_importance, std_importance

def evaluate_linear_regression(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, r2_list = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
    return {
        'RMSE': (np.mean(rmse_list), np.std(rmse_list)),
        'MAE': (np.mean(mae_list), np.std(mae_list)),
        'R2': (np.mean(r2_list), np.std(r2_list)),
        'ECE': (np.nan, np.nan)  # Not applicable for linear regression
    }

if __name__ == "__main__":
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv", 
                     na_values=na_vals, low_memory=False)
    
    features = ["floor_area", "ghg_emissions_int", "fuel_eui", "electric_eui"]
    target = "site_eui"
    
    print(f"Original rows: {len(df)}")
    for col in features + [target]:
        print(f"Missing in {col}: {df[col].isna().sum()}")
    
    df = df.dropna(subset=features + [target])
    print(f"Rows remaining after dropna: {len(df)}")
    
    # Prepare data
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    
    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    
    # Train and evaluate model
    output_dir = os.path.join("bpd", "bnn-hier-v2")
    mean_metrics, std_metrics, mean_importance, std_importance = train_and_evaluate(
        X, y, features, n_splits=5, output_dir=output_dir)

    # Baseline: Linear Regression
    baseline_metrics = evaluate_linear_regression(X, y, n_splits=5)

    # Posterior predictive checks and parameter histograms (on last fold model)
    # Reload last fold model for demonstration
    model = HierarchicalBayesianNeuralNetwork(input_dim=X.shape[1], hidden_dims=[64, 32], output_dim=1)
    model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_fold_4.pth')))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_pred, y_std = model.predict(X_tensor)
    posterior_predictive_checks(y, y_pred.numpy(), y_std.numpy(), output_dir)
    plot_parameter_histograms(model, output_dir)

    # Print results
    print("\nCross-validated Results:")
    for metric in mean_metrics:
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    print("\nFeature Importance (with 95% confidence intervals):")
    for feature, importance, std in zip(features, mean_importance, std_importance):
        ci = 1.96 * std  # 95% confidence interval
        print(f"{feature}: {importance:.4f} ± {ci:.4f}")

    # Print summary table
    print("\nSummary Table (Cross-validated Metrics):")
    print(f"{'Metric':<8} | {'BNN Mean±Std':<20} | {'LinearReg Mean±Std':<20}")
    print("-"*55)
    for metric in ['RMSE', 'MAE', 'R2', 'ECE']:
        bnn_val = f"{mean_metrics.get(metric, float('nan')):.4f} ± {std_metrics.get(metric, float('nan')):.4f}"
        base_val = baseline_metrics[metric]
        if not np.isnan(base_val[0]):
            base_val_str = f"{base_val[0]:.4f} ± {base_val[1]:.4f}"
        else:
            base_val_str = "N/A"
        print(f"{metric:<8} | {bnn_val:<20} | {base_val_str:<20}") 