"""
ARDOnly.py: Bayesian Neural Network with Automatic Relevance Determination (ARD)
-------------------------------------------------------------------------------
- Implements ARD for feature selection
- Uses non-centered parameterization
- Provides uncertainty estimates
- Focuses on feature importance analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from typing import List, Tuple, Optional

class ARDBayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout_rate)
        
        # Weight and bias parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # ARD parameters
        self.ard_alpha = nn.Parameter(torch.ones(in_features))
        self.ard_beta = nn.Parameter(torch.ones(in_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.constant_(self.bias_rho, -5.0)
        nn.init.constant_(self.ard_alpha, 1.0)
        nn.init.constant_(self.ard_beta, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample weights and biases
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        x = self.dropout(x)
        
        # Apply ARD scaling
        ard_scale = F.softplus(self.ard_alpha) * F.softplus(self.ard_beta)
        x = x * ard_scale
        
        # Linear transformation
        output = F.linear(x, weight, bias)
        
        # KL divergence
        kl = self.kl_loss(weight_sigma, bias_sigma)
        
        return output, kl
    
    def kl_loss(self, weight_sigma: torch.Tensor, bias_sigma: torch.Tensor) -> torch.Tensor:
        # Prior KL divergence for weights
        weight_kl = 0.5 * torch.sum(
            torch.log(weight_sigma**2) + (1/weight_sigma**2) + self.weight_mu**2 - 1
        )
        
        # Prior KL divergence for biases
        bias_kl = 0.5 * torch.sum(
            torch.log(bias_sigma**2) + (1/bias_sigma**2) + self.bias_mu**2 - 1
        )
        
        # ARD KL divergence
        ard_kl = torch.sum(
            F.softplus(self.ard_alpha) + F.softplus(self.ard_beta) - 
            torch.log(F.softplus(self.ard_alpha)) - torch.log(F.softplus(self.ard_beta))
        )
        
        return weight_kl + bias_kl + ard_kl

class ARDNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = ARDBayesianLinear(input_dim, hidden_dims[0], dropout_rate)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(
                ARDBayesianLinear(hidden_dims[i], hidden_dims[i+1], dropout_rate)
            )
        
        # Output layer
        self.output_layer = ARDBayesianLinear(hidden_dims[-1], output_dim, dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_sum = 0
        
        # Input layer
        x, kl = self.input_layer(x)
        kl_sum += kl
        x = F.relu(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x, kl = layer(x)
            kl_sum += kl
            x = F.relu(x)
        
        # Output layer
        x, kl = self.output_layer(x)
        kl_sum += kl
        
        return x, kl_sum
    
    def predict(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        return mean_pred, std_pred
    
    def get_ard_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ARD parameters from the input layer"""
        return (F.softplus(self.input_layer.ard_alpha).detach(),
                F.softplus(self.input_layer.ard_beta).detach())

def train_model(model: ARDNeuralNetwork, X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor, feature_names: List[str],
                num_epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                kl_weight: float = 1e-3, patience: int = 10,
                output_dir: Optional[str] = None) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output, kl = model(batch_X)
            mse_loss = F.mse_loss(output, batch_y)
            loss = mse_loss + kl_weight * kl
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output, val_kl = model(X_val)
            val_mse = F.mse_loss(val_output, y_val)
            val_loss = val_mse + kl_weight * val_kl
            val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if output_dir is not None:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return train_losses, val_losses

if __name__ == "__main__":
    print("[INFO] Starting ARD Neural Network experiments...")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    
    # Feature Engineering
    print("[INFO] Performing feature engineering...")
    
    # 1. Enhanced floor_area transformation
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    
    # 2. Enhanced energy intensity features
    df['total_eui'] = df['electric_eui'] + df['fuel_eui']
    df['electric_ratio'] = df['electric_eui'] / df['total_eui']
    df['fuel_ratio'] = df['fuel_eui'] / df['total_eui']
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    
    # 3. Enhanced building age features
    df['building_age'] = 2024 - df['year_built']
    df['building_age_log'] = np.log1p(df['building_age'])
    
    # 4. Enhanced energy star rating features
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    
    # 5. Handle missing values in ghg_emissions_int
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    
    # Select features
    features = [
        "ghg_emissions_int_log",
        "total_eui",
        "floor_area_log",
        "electric_eui",
        "fuel_eui",
        "energy_star_rating_normalized",
        "energy_mix",
        "building_age_log"
    ]
    feature_names = features.copy()
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Model setup
    input_dim = len(features)
    hidden_dims = [256, 128, 64]
    output_dim = 1
    
    model = ARDNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=0.2
    )
    
    # Train model
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultsARDBNN')
    os.makedirs(results_dir, exist_ok=True)
    
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val, feature_names,
        num_epochs=150,
        batch_size=64,
        learning_rate=0.001,
        kl_weight=5e-4,
        output_dir=results_dir
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred, y_std = model.predict(X_val)
    
    # Calculate and save metrics
    metrics = {
        'rmse': float(np.sqrt(np.mean((y_val.numpy() - y_pred.numpy()) ** 2))),
        'mae': float(np.mean(np.abs(y_val.numpy() - y_pred.numpy()))),
        'r2': float(1 - np.sum((y_val.numpy() - y_pred.numpy()) ** 2) / 
                    np.sum((y_val.numpy() - np.mean(y_val.numpy())) ** 2)),
        'mean_std': float(np.mean(y_std.numpy()))
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("[INFO] Training complete. Metrics saved to metrics.json")
    
    # Get ARD parameters
    ard_alpha, ard_beta = model.get_ard_parameters()
    feature_importance = ard_alpha.numpy() * ard_beta.numpy()
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # 1. Predictions vs Actual Values with Uncertainty
    plt.subplot(2, 2, 1)
    y_val_orig = scaler_y.inverse_transform(y_val.numpy())
    y_pred_orig = scaler_y.inverse_transform(y_pred.numpy())
    y_std_orig = y_std.numpy().flatten() * scaler_y.scale_
    
    plt.scatter(y_val_orig, y_pred_orig, alpha=0.5, label='Predictions')
    plt.errorbar(y_val_orig.flatten(), y_pred_orig.flatten(), yerr=y_std_orig, fmt='none', 
                ecolor='gray', alpha=0.2, label='Uncertainty')
    
    min_val = min(y_val_orig.min(), y_pred_orig.min())
    max_val = max(y_val_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Site EUI')
    plt.ylabel('Predicted Site EUI')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    
    # 2. Feature Importance
    plt.subplot(2, 2, 2)
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(features)), feature_importance[sorted_idx])
    plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
    plt.xlabel('Feature Importance (ARD Scale)')
    plt.title('Feature Importance from ARD Parameters')
    
    # 3. ARD Parameters
    plt.subplot(2, 2, 3)
    x = np.arange(len(features))
    width = 0.35
    
    plt.barh(x - width/2, ard_alpha.numpy()[sorted_idx], width, label='Alpha')
    plt.barh(x + width/2, ard_beta.numpy()[sorted_idx], width, label='Beta')
    plt.yticks(x, [features[i] for i in sorted_idx])
    plt.xlabel('Parameter Value')
    plt.title('ARD Alpha and Beta Parameters')
    plt.legend()
    
    # 4. Uncertainty Analysis
    plt.subplot(2, 2, 4)
    abs_error = np.abs(y_val_orig - y_pred_orig)
    plt.scatter(abs_error, y_std_orig, alpha=0.5)
    plt.xlabel('Absolute Prediction Error')
    plt.ylabel('Prediction Uncertainty')
    plt.title('Uncertainty vs Prediction Error')
    
    z = np.polyfit(abs_error.flatten(), y_std_orig, 1)
    p = np.poly1d(z)
    plt.plot(sorted(abs_error.flatten()), p(sorted(abs_error.flatten())), "r--")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ard_visualizations.png'))
    plt.close()
    
    # Print analysis
    print("\nDetailed Analysis:")
    print("\n1. Feature Importance:")
    for feat, imp, alpha, beta in zip(features, feature_importance, ard_alpha, ard_beta):
        print(f"{feat}:")
        print(f"  - Total Importance: {imp:.4f}")
        print(f"  - Alpha: {alpha:.4f}")
        print(f"  - Beta: {beta:.4f}")
    
    print("\n2. Uncertainty Analysis:")
    print(f"Mean Uncertainty: {np.mean(y_std_orig):.4f}")
    print(f"Std of Uncertainty: {np.std(y_std_orig):.4f}")
    print(f"Correlation with Error: {np.corrcoef(abs_error.flatten(), y_std_orig)[0,1]:.4f}") 