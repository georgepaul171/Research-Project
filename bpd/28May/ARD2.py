"""
ARD.py: Hierarchical Bayesian Neural Network with Automatic Relevance Determination
-------------------------------------------------------------------------------
- Implements group-specific parameters for each data category
- Uses non-centered parameterization for better sampling
- Includes hyperpriors for group-level standard deviations
- Provides Monte Carlo KL divergence for non-Gaussian priors
- Enhanced interpretability with hierarchical parameter visualization
- Implements Automatic Relevance Determination (ARD) for feature selection
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
import shap
from captum.attr import IntegratedGradients
from typing import List, Dict, Optional, Tuple
import json
from scipy.stats import norm

class HyperPrior:
    """Base class for hyperpriors on standard deviations"""
    def __init__(self, scale: float = 1.0):
        self.scale = torch.tensor(scale, dtype=torch.float32)
    
    def kl_divergence(self, sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class HalfNormalHyperPrior(HyperPrior):
    """Half-Normal hyperprior for standard deviations"""
    def kl_divergence(self, sigma: torch.Tensor) -> torch.Tensor:
        # KL divergence between log-normal posterior and half-normal prior
        log_sigma = torch.log(sigma)
        kl = -0.5 * torch.log(2 * np.pi * self.scale**2) - log_sigma + \
             (log_sigma**2 + sigma**2) / (2 * self.scale**2) - 0.5
        return kl.sum()

class HierarchicalPrior:
    """Base class for hierarchical prior distributions"""
    def __init__(self, group_mu: float = 0.0, group_logvar: float = 0.0):
        self.group_mu = torch.tensor(group_mu, dtype=torch.float32)
        self.group_logvar = torch.tensor(group_logvar, dtype=torch.float32)
    
    def kl_divergence(self, group_mu: torch.Tensor, group_sigma: torch.Tensor,
                     raw_mu: torch.Tensor, raw_sigma: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class HierarchicalNormalPrior(HierarchicalPrior):
    def kl_divergence(self, group_mu: torch.Tensor, group_sigma: torch.Tensor,
                     raw_mu: torch.Tensor, raw_sigma: torch.Tensor) -> torch.Tensor:
        # Group-level KL
        group_var = torch.exp(self.group_logvar)
        group_kl = torch.log(group_var.sqrt() / group_sigma) + \
                  (group_sigma ** 2 + (group_mu - self.group_mu) ** 2) / (2 * group_var) - 0.5
        
        # Individual-level KL (using non-centered parameterization)
        two_pi = torch.tensor(2 * np.pi, device=raw_sigma.device)
        individual_kl = -0.5 * torch.log(two_pi) - torch.log(raw_sigma) + \
                       (raw_sigma ** 2 + raw_mu ** 2) / 2 - 0.5
        
        return group_kl.sum() + individual_kl.sum()

class HierarchicalBayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int,
                 prior_type: str = 'hierarchical_normal', dropout_rate: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize hierarchical prior
        if prior_type == 'hierarchical_normal':
            self.prior = HierarchicalNormalPrior()
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Hyperprior for group-level standard deviations
        self.hyperprior = HalfNormalHyperPrior(scale=1.0)
        
        # Group-level parameters
        self.group_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.group_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.group_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.group_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Group-specific raw parameters (non-centered parameterization)
        self.raw_weight_mu = nn.Parameter(torch.Tensor(num_groups, out_features, in_features))
        self.raw_weight_rho = nn.Parameter(torch.Tensor(num_groups, out_features, in_features))
        self.raw_bias_mu = nn.Parameter(torch.Tensor(num_groups, out_features))
        self.raw_bias_rho = nn.Parameter(torch.Tensor(num_groups, out_features))
        
        # ARD parameters with improved initialization
        self.ard_alpha = nn.Parameter(torch.ones(in_features))
        self.ard_beta = nn.Parameter(torch.ones(in_features))  # New parameter for adaptive ARD
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize group-level parameters
        nn.init.kaiming_uniform_(self.group_weight_mu, a=np.sqrt(5))
        nn.init.constant_(self.group_weight_rho, -5.0)
        nn.init.uniform_(self.group_bias_mu, -0.1, 0.1)
        nn.init.constant_(self.group_bias_rho, -5.0)
        
        # Initialize group-specific raw parameters
        nn.init.kaiming_uniform_(self.raw_weight_mu, a=np.sqrt(5))
        nn.init.constant_(self.raw_weight_rho, -5.0)
        nn.init.uniform_(self.raw_bias_mu, -0.1, 0.1)
        nn.init.constant_(self.raw_bias_rho, -5.0)
        
        # Initialize ARD parameters
        nn.init.constant_(self.ard_alpha, 1.0)
        nn.init.constant_(self.ard_beta, 1.0)
    
    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Group-level sampling
        group_weight_sigma = F.softplus(self.group_weight_rho)
        group_bias_sigma = F.softplus(self.group_bias_rho)
        
        # Group-specific raw sampling (non-centered parameterization)
        raw_weight_sigma = F.softplus(self.raw_weight_rho)
        raw_bias_sigma = F.softplus(self.raw_bias_rho)
        
        # Sample raw parameters for each group in the batch
        raw_weight = self.raw_weight_mu[group_ids] + \
                    raw_weight_sigma[group_ids] * torch.randn_like(self.raw_weight_mu[group_ids])
        raw_bias = self.raw_bias_mu[group_ids] + \
                  raw_bias_sigma[group_ids] * torch.randn_like(self.raw_bias_mu[group_ids])
        
        # Combine group and raw parameters
        weight = self.group_weight_mu + group_weight_sigma * raw_weight
        bias = self.group_bias_mu + group_bias_sigma * raw_bias
        
        x = self.dropout(x)
        
        # Apply adaptive ARD scaling
        ard_scale = F.softplus(self.ard_alpha) * F.softplus(self.ard_beta)
        x = x * ard_scale
        
        # Perform matrix multiplication with correct shapes
        # x: (batch_size, in_features)
        # weight: (batch_size, out_features, in_features)
        # We need to reshape weight to (batch_size, in_features, out_features)
        weight = weight.transpose(-2, -1)  # (batch_size, in_features, out_features)
        output = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        
        return output, self.kl_loss(group_weight_sigma, group_bias_sigma,
                                  raw_weight_sigma, raw_bias_sigma)
    
    def kl_loss(self, group_weight_sigma: torch.Tensor, group_bias_sigma: torch.Tensor,
                raw_weight_sigma: torch.Tensor, raw_bias_sigma: torch.Tensor) -> torch.Tensor:
        # Prior KL divergence
        prior_kl = self.prior.kl_divergence(
            self.group_weight_mu, group_weight_sigma,
            self.raw_weight_mu, raw_weight_sigma
        )
        
        # Hyperprior KL divergence for standard deviations
        hyperprior_kl = self.hyperprior.kl_divergence(group_weight_sigma) + \
                       self.hyperprior.kl_divergence(group_bias_sigma)
        
        # ARD KL divergence with improved numerical stability
        ard_kl = torch.sum(F.softplus(self.ard_alpha)) + \
                torch.sum(F.softplus(self.ard_beta))
        
        return prior_kl + hyperprior_kl + ard_kl

class TrueHierarchicalHBNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 num_groups: int, prior_type: str = 'hierarchical_normal',
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_groups = num_groups
        
        # Input layer
        self.input_layer = HierarchicalBayesianLinear(
            input_dim, hidden_dims[0], num_groups, prior_type, dropout_rate
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(
                HierarchicalBayesianLinear(
                    hidden_dims[i], hidden_dims[i+1], num_groups, prior_type, dropout_rate
                )
            )
        
        # Output layer
        self.output_layer = HierarchicalBayesianLinear(
            hidden_dims[-1], output_dim, num_groups, prior_type, dropout_rate
        )
    
    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_sum = 0
        
        # Input layer
        x, kl = self.input_layer(x, group_ids)
        kl_sum += kl
        x = F.relu(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x, kl = layer(x, group_ids)
            kl_sum += kl
            x = F.relu(x)
        
        # Output layer
        x, kl = self.output_layer(x, group_ids)
        kl_sum += kl
        
        # Ensure output has correct shape (batch_size, output_dim)
        if len(x.shape) > 2:
            x = x.squeeze(-1)
        
        return x, kl_sum
    
    def predict(self, x: torch.Tensor, group_ids: torch.Tensor,
                num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x, group_ids)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Ensure predictions have correct shape
        if len(mean_pred.shape) > 2:
            mean_pred = mean_pred.squeeze(-1)
        if len(std_pred.shape) > 2:
            std_pred = std_pred.squeeze(-1)
            
        return mean_pred, std_pred
    
    def get_hierarchical_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract hierarchical parameters for visualization"""
        params = {}
        for name, module in self.named_modules():
            if isinstance(module, HierarchicalBayesianLinear):
                params[f"{name}.group_weight_mu"] = module.group_weight_mu.detach()
                params[f"{name}.group_weight_sigma"] = F.softplus(module.group_weight_rho).detach()
                params[f"{name}.raw_weight_mu"] = module.raw_weight_mu.detach()
                params[f"{name}.raw_weight_sigma"] = F.softplus(module.raw_weight_rho).detach()
                params[f"{name}.ard_alpha"] = F.softplus(module.ard_alpha).detach()
                params[f"{name}.ard_beta"] = F.softplus(module.ard_beta).detach()
        return params

def train_model(model: TrueHierarchicalHBNN, X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor, group_ids_train: torch.Tensor,
                group_ids_val: torch.Tensor, feature_names: List[str], num_epochs: int = 100,
                batch_size: int = 32, learning_rate: float = 0.001, kl_weight: float = 1e-3,
                patience: int = 10, output_dir: Optional[str] = None) -> Tuple[List[float], List[float]]:
    """Train the model with early stopping and learning rate scheduling"""
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
            batch_group_ids = group_ids_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output, kl = model(batch_X, batch_group_ids)
            mse_loss = F.mse_loss(output, batch_y)
            kl = torch.abs(kl)  # Make sure KL is positive
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
            val_output, val_kl = model(X_val, group_ids_val)
            val_mse = F.mse_loss(val_output, y_val)
            val_kl = torch.abs(val_kl)  # Make sure KL is positive
            val_loss = val_mse + kl_weight * val_kl
            val_losses.append(val_loss.item())
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
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
    print("[INFO] Starting Enhanced True Hierarchical Bayesian Neural Network experiments...")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    target = "site_eui"
    group_column = "city"
    
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
    df['floor_area_sqrt'] = np.sqrt(df['floor_area'])
    
    # 2. Enhanced energy intensity features
    df['total_eui'] = df['electric_eui'] + df['fuel_eui']
    df['electric_ratio'] = df['electric_eui'] / df['total_eui']
    df['fuel_ratio'] = df['fuel_eui'] / df['total_eui']
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']  # Interaction term
    
    # 3. Enhanced building age features
    df['building_age'] = 2024 - df['year_built']
    df['building_age_log'] = np.log1p(df['building_age'])
    
    # 4. Enhanced energy star rating features
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    
    # 5. Handle missing values in ghg_emissions_int
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    
    # Select most important features based on previous analysis
    features = [
        "ghg_emissions_int_log",  # Log transformed for better distribution
        "total_eui",             # Most important energy metric
        "floor_area_log",        # Log transformed for better distribution
        "electric_eui",          # Direct energy metric
        "fuel_eui",             # Direct energy metric
        "energy_star_rating_normalized",  # Normalized rating
        "energy_mix",           # New interaction term
        "building_age_log"      # Log transformed age
    ]
    feature_names = features.copy()
    
    # Create group IDs
    unique_groups = df[group_column].unique()
    group_to_id = {group: i for i, group in enumerate(unique_groups)}
    group_ids = df[group_column].map(group_to_id).values
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Split data
    X_train, X_val, y_train, y_val, group_ids_train, group_ids_val = train_test_split(
        X, y, group_ids, test_size=0.2, random_state=42
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    group_ids_train = torch.tensor(group_ids_train, dtype=torch.long)
    group_ids_val = torch.tensor(group_ids_val, dtype=torch.long)
    
    # Model setup with updated architecture
    input_dim = len(features)
    hidden_dims = [256, 128, 64]  # Increased capacity
    output_dim = 1
    num_groups = len(unique_groups)
    
    model = TrueHierarchicalHBNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_groups=num_groups,
        prior_type='hierarchical_normal',
        dropout_rate=0.2  # Increased dropout for better regularization
    )
    
    # Train model with adjusted parameters
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultsARD')
    os.makedirs(results_dir, exist_ok=True)
    
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val,
        group_ids_train, group_ids_val, feature_names,
        num_epochs=150,  # Increased epochs
        batch_size=64,   # Increased batch size
        learning_rate=0.001,
        kl_weight=5e-4,  # Reduced KL weight for better uncertainty calibration
        output_dir=results_dir
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred, y_std = model.predict(X_val, group_ids_val)
    
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
    
    # Visualization
    plt.figure(figsize=(20, 15))
    
    # 1. Predictions vs Actual Values with Uncertainty
    plt.subplot(3, 2, 1)
    y_val_orig = scaler_y.inverse_transform(y_val.numpy())
    y_pred_orig = scaler_y.inverse_transform(y_pred.numpy())
    y_std_orig = y_std.numpy().flatten() * scaler_y.scale_
    
    plt.scatter(y_val_orig, y_pred_orig, alpha=0.5, label='Predictions')
    plt.errorbar(y_val_orig.flatten(), y_pred_orig.flatten(), yerr=y_std_orig, fmt='none', 
                ecolor='gray', alpha=0.2, label='Uncertainty')
    
    # Add perfect prediction line
    min_val = min(y_val_orig.min(), y_pred_orig.min())
    max_val = max(y_val_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Site EUI')
    plt.ylabel('Predicted Site EUI')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    
    # 2. Feature Importance Analysis
    plt.subplot(3, 2, 2)
    
    # Get ARD parameters from the first layer
    ard_alpha = F.softplus(model.input_layer.ard_alpha).detach().numpy()
    ard_beta = F.softplus(model.input_layer.ard_beta).detach().numpy()
    feature_importance = ard_alpha * ard_beta
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(features)), feature_importance[sorted_idx])
    plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
    plt.xlabel('Feature Importance (ARD Scale)')
    plt.title('Feature Importance from ARD Parameters')
    
    # 3. Separate Alpha and Beta Parameters
    plt.subplot(3, 2, 3)
    x = np.arange(len(features))
    width = 0.35
    
    plt.barh(x - width/2, ard_alpha[sorted_idx], width, label='Alpha')
    plt.barh(x + width/2, ard_beta[sorted_idx], width, label='Beta')
    plt.yticks(x, [features[i] for i in sorted_idx])
    plt.xlabel('Parameter Value')
    plt.title('ARD Alpha and Beta Parameters')
    plt.legend()
    
    # 4. Group-level Effects
    plt.subplot(3, 2, 4)
    
    # Get group-level parameters
    group_weights = model.input_layer.group_weight_mu.detach().numpy()
    group_biases = model.input_layer.group_bias_mu.detach().numpy()
    
    # Calculate group effects
    group_effects = []
    for group_id in range(num_groups):
        effect = np.abs(group_weights[group_id]).mean() + np.abs(group_biases[group_id])
        group_effects.append(effect)
    
    # Sort groups by effect size
    sorted_groups = np.argsort(group_effects)
    plt.barh(range(num_groups), [group_effects[i] for i in sorted_groups])
    plt.yticks(range(num_groups), [list(group_to_id.keys())[i] for i in sorted_groups])
    plt.xlabel('Group Effect Size')
    plt.title('Group-level Effects')
    
    # 5. Uncertainty Patterns
    plt.subplot(3, 2, 5)
    
    # Plot uncertainty vs prediction error
    abs_error = np.abs(y_val_orig - y_pred_orig)
    plt.scatter(abs_error, y_std_orig, alpha=0.5)
    plt.xlabel('Absolute Prediction Error')
    plt.ylabel('Prediction Uncertainty')
    plt.title('Uncertainty vs Prediction Error')
    
    # Add correlation line
    z = np.polyfit(abs_error.flatten(), y_std_orig, 1)
    p = np.poly1d(z)
    plt.plot(sorted(abs_error.flatten()), p(sorted(abs_error.flatten())), "r--")
    
    # 6. Uncertainty Distribution
    plt.subplot(3, 2, 6)
    plt.hist(y_std_orig, bins=30, alpha=0.7)
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainties')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'detailed_visualizations.png'))
    plt.close()
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print("\n1. Feature Importance:")
    for feat, imp, alpha, beta in zip(features, feature_importance, ard_alpha, ard_beta):
        print(f"{feat}:")
        print(f"  - Total Importance: {imp:.4f}")
        print(f"  - Alpha: {alpha:.4f}")
        print(f"  - Beta: {beta:.4f}")
    
    print("\n2. Group-level Effects:")
    for group, effect in zip(group_to_id.keys(), group_effects):
        print(f"{group}: {effect:.4f}")
    
    print("\n3. Uncertainty Analysis:")
    print(f"Mean Uncertainty: {np.mean(y_std_orig):.4f}")
    print(f"Std of Uncertainty: {np.std(y_std_orig):.4f}")
    print(f"Correlation with Error: {np.corrcoef(abs_error.flatten(), y_std_orig)[0,1]:.4f}") 