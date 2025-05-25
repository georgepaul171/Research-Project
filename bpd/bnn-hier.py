import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        # KL for weights and biases with hierarchical prior
        kl_weights = self.kl_divergence(self.weight_mu, weight_sigma)
        kl_bias = self.kl_divergence(self.bias_mu, bias_sigma)
        # KL for hyperpriors (prior_mu ~ N(0, 1), prior_logvar ~ N(0, 1))
        kl_prior_mu = 0.5 * (self.prior_mu ** 2)
        kl_prior_logvar = 0.5 * (self.prior_logvar ** 2)
        return kl_weights + kl_bias + kl_prior_mu + kl_prior_logvar
    def kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Hierarchical prior: N(prior_mu, exp(prior_logvar))
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
    def predict(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        return torch.stack(predictions).mean(dim=0)

if __name__ == "__main__":
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv", na_values=na_vals, low_memory=False)
    features = ["floor_area", "ghg_emissions_int", "fuel_eui", "electric_eui"]
    target = "site_eui"
    print(f"Original rows: {len(df)}")
    for col in features + [target]:
        print(f"Missing in {col}: {df[col].isna().sum()}")
    df = df.dropna(subset=features + [target])
    print(f"Rows remaining after dropna: {len(df)}")
    print("First 5 rows after dropna:")
    print(df[features + [target]].head())
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / (y_std + 1e-8)
    X = torch.tensor(X)
    y = torch.tensor(y)
    n = X.shape[0]
    train_size = int(0.8 * n)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = HierarchicalBayesianNeuralNetwork(input_dim=4, hidden_dims=[64, 32], output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    kl_weight = 1e-3
    train_losses = []
    val_losses = []
    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output, kl = model(batch_x)
            mse_loss = F.mse_loss(output, batch_y)
            loss = mse_loss + kl_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output, kl = model(batch_x)
                mse_loss = F.mse_loss(output, batch_y)
                loss = mse_loss + kl_weight * kl
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_hier_bnn_model.pth')
        print(f'Epoch {epoch+1}/100:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    # Ensure output directory exists
    output_dir = os.path.join("bpd", "bnn-hier")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_losses_hier.png'))
    plt.close()
    model.eval()
    with torch.no_grad():
        val_predictions = []
        val_actuals = []
        for batch_x, batch_y in val_loader:
            pred, _ = model(batch_x)
            val_predictions.extend(pred.numpy())
            val_actuals.extend(batch_y.numpy())
    val_predictions = np.array(val_predictions)
    val_actuals = np.array(val_actuals)
    plt.figure(figsize=(10, 6))
    plt.scatter(val_actuals, val_predictions, alpha=0.5)
    plt.plot([val_actuals.min(), val_actuals.max()], 
             [val_actuals.min(), val_actuals.max()], 
             'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values (Hierarchical BNN)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual_hier.png'))
    plt.close()
    model.eval()
    test_x = X_val[:100]
    predictions = []
    for _ in range(100):
        with torch.no_grad():
            pred, _ = model(test_x)
            predictions.append(pred)
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    plt.figure(figsize=(10, 6))
    plt.hist(std_pred.numpy().flatten(), bins=30)
    plt.xlabel('Prediction Uncertainty (Standard Deviation)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Uncertainties (Hierarchical BNN)')
    plt.savefig(os.path.join(output_dir, 'uncertainty_distribution_hier.png'))
    plt.close()
    print("\nExample predictions with uncertainty:")
    for i in range(min(5, len(mean_pred))):
        print(f"Sample {i+1}: {mean_pred[i].item():.4f} ± {std_pred[i].item():.4f}")

    # Convert predictions and actuals to numpy arrays
    val_predictions = val_predictions.flatten()
    val_actuals = val_actuals.flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
    mae = mean_absolute_error(val_actuals, val_predictions)
    r2 = r2_score(val_actuals, val_predictions)

    print("\nEvaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Residual plot
    residuals = val_actuals - val_predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(val_predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
    plt.close()

    # Feature importance analysis using sensitivity analysis
    feature_importance = []
    for i in range(len(features)):
        # Create perturbed input
        perturbed_x = X_val.clone()
        perturbed_x[:, i] = perturbed_x[:, i] + 0.1  # Small perturbation
        
        # Get predictions for original and perturbed inputs
        with torch.no_grad():
            orig_pred, _ = model(X_val)
            pert_pred, _ = model(perturbed_x)
        
        # Calculate importance as average change in prediction
        importance = torch.mean(torch.abs(pert_pred - orig_pred)).item()
        feature_importance.append(importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(features, feature_importance)
    plt.xticks(rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

    print("\nFeature Importance Scores:")
    for feature, importance in zip(features, feature_importance):
        print(f"{feature}: {importance:.4f}")

    # Get validation set uncertainties
    val_uncertainties = []
    for _ in range(100):  # 100 Monte Carlo samples
        with torch.no_grad():
            pred, _ = model(X_val)
            val_uncertainties.append(pred)
    val_uncertainties = torch.stack(val_uncertainties)
    val_std = val_uncertainties.std(dim=0).numpy().flatten()

    # Calibration plot
    num_bins = 10
    bin_edges = np.linspace(0, val_std.max(), num_bins + 1)
    calibration_errors = []
    empirical_coverage = []

    for i in range(num_bins):
        mask = (val_std >= bin_edges[i]) & (val_std < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_std = val_std[mask]
            bin_residuals = np.abs(residuals[mask])
            empirical_coverage.append(np.mean(bin_residuals <= 2 * bin_std))
            calibration_errors.append(np.mean(bin_std))

    plt.figure(figsize=(10, 6))
    plt.plot(calibration_errors, empirical_coverage, 'bo-')
    plt.plot([0, max(calibration_errors)], [0, max(calibration_errors)], 'r--')
    plt.xlabel('Predicted Uncertainty')
    plt.ylabel('Empirical Coverage')
    plt.title('Calibration Plot')
    plt.savefig(os.path.join(output_dir, 'calibration_plot.png'))
    plt.close() 