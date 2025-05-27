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

class HierarchicalStudentTPrior(HierarchicalPrior):
    def __init__(self, group_mu: float = 0.0, group_logvar: float = 0.0, df: float = 3.0):
        super().__init__(group_mu, group_logvar)
        self.df = torch.tensor(df, dtype=torch.float32)
    
    def kl_divergence(self, group_mu: torch.Tensor, group_sigma: torch.Tensor,
                     raw_mu: torch.Tensor, raw_sigma: torch.Tensor) -> torch.Tensor:
        # Monte Carlo approximation for Student's t KL divergence
        num_samples = 100
        
        # Ensure raw_mu and raw_sigma have the right shape for broadcasting
        raw_mu = raw_mu.view(-1, 1, 1)  # [num_groups, 1, 1]
        raw_sigma = raw_sigma.view(-1, 1, 1)  # [num_groups, 1, 1]
        
        # Generate samples with correct shape
        noise = torch.randn(num_samples, raw_mu.size(0), 1, 1, device=raw_mu.device)
        samples = raw_mu + raw_sigma * noise  # [num_samples, num_groups, 1, 1]
        
        # Convert constants to tensors
        two_pi = torch.tensor(2 * np.pi, device=raw_sigma.device)
        df_pi = torch.tensor(self.df * np.pi, device=raw_sigma.device)
        
        # Log density of normal posterior
        log_q = -0.5 * torch.log(two_pi) - torch.log(raw_sigma) - \
               0.5 * ((samples - raw_mu) / raw_sigma) ** 2
        
        # Log density of Student's t prior
        log_p = torch.lgamma((self.df + 1) / 2) - torch.lgamma(self.df / 2) - \
                0.5 * torch.log(df_pi) - \
                ((self.df + 1) / 2) * torch.log(1 + samples ** 2 / self.df)
        
        # Monte Carlo KL estimate
        kl = torch.mean(log_q - log_p, dim=0)
        
        return kl.sum()

class HierarchicalMixtureGaussianPrior(HierarchicalPrior):
    def __init__(self, group_mu: float = 0.0, group_logvar: float = 0.0,
                 mus: List[float] = [-1.0, 1.0], logvars: List[float] = [0.0, 0.0],
                 weights: List[float] = [0.5, 0.5]):
        super().__init__(group_mu, group_logvar)
        self.mus = [torch.tensor(m, dtype=torch.float32) for m in mus]
        self.logvars = [torch.tensor(v, dtype=torch.float32) for v in logvars]
        self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
    
    def kl_divergence(self, group_mu: torch.Tensor, group_sigma: torch.Tensor,
                     raw_mu: torch.Tensor, raw_sigma: torch.Tensor) -> torch.Tensor:
        # Monte Carlo approximation for mixture Gaussian KL divergence
        num_samples = 100
        
        # Ensure raw_mu and raw_sigma have the right shape for broadcasting
        raw_mu = raw_mu.view(-1, 1, 1)  # [num_groups, 1, 1]
        raw_sigma = raw_sigma.view(-1, 1, 1)  # [num_groups, 1, 1]
        
        # Generate samples with correct shape
        noise = torch.randn(num_samples, raw_mu.size(0), 1, 1, device=raw_mu.device)
        samples = raw_mu + raw_sigma * noise  # [num_samples, num_groups, 1, 1]
        
        # Convert constants to tensors
        two_pi = torch.tensor(2 * np.pi, device=raw_sigma.device)
        
        # Log density of normal posterior
        log_q = -0.5 * torch.log(two_pi) - torch.log(raw_sigma) - \
               0.5 * ((samples - raw_mu) / raw_sigma) ** 2
        
        # Log density of mixture Gaussian prior
        log_p = torch.zeros_like(samples)
        for m, v, w in zip(self.mus, self.logvars, self.weights):
            prior_var = torch.exp(v)
            log_p += w * (-0.5 * torch.log(two_pi * prior_var) - \
                         0.5 * ((samples - m) ** 2) / prior_var)
        
        # Monte Carlo KL estimate
        kl = torch.mean(log_q - log_p, dim=0)
        
        return kl.sum()

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
        elif prior_type == 'hierarchical_student_t':
            self.prior = HierarchicalStudentTPrior()
        elif prior_type == 'hierarchical_mixture':
            self.prior = HierarchicalMixtureGaussianPrior()
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
        
        # ARD parameters
        self.ard_alpha = nn.Parameter(torch.ones(in_features))
        
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
    
    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Group-level sampling
        group_weight_sigma = torch.log1p(torch.exp(self.group_weight_rho))
        group_bias_sigma = torch.log1p(torch.exp(self.group_bias_rho))
        
        # Group-specific raw sampling (non-centered parameterization)
        raw_weight_sigma = torch.log1p(torch.exp(self.raw_weight_rho))
        raw_bias_sigma = torch.log1p(torch.exp(self.raw_bias_rho))
        
        # Sample raw parameters for each group in the batch
        raw_weight = self.raw_weight_mu[group_ids] + \
                    raw_weight_sigma[group_ids] * torch.randn_like(self.raw_weight_mu[group_ids])
        raw_bias = self.raw_bias_mu[group_ids] + \
                  raw_bias_sigma[group_ids] * torch.randn_like(self.raw_bias_mu[group_ids])
        
        # Combine group and raw parameters
        # raw_weight shape: [batch_size, out_features, in_features]
        weight = self.group_weight_mu + group_weight_sigma * raw_weight
        
        # Reshape weight to [batch_size, out_features, in_features]
        weight = weight.view(batch_size, self.out_features, self.in_features)
        bias = self.group_bias_mu + group_bias_sigma * raw_bias
        
        x = self.dropout(x)
        
        # Apply ARD scaling
        x = x * self.ard_alpha
        
        # Perform batched matrix multiplication
        # x shape: [batch_size, in_features]
        # weight shape: [batch_size, out_features, in_features]
        # output shape: [batch_size, out_features]
        output = torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1) + bias
        
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
        
        # ARD KL divergence
        ard_kl = torch.sum(self.ard_alpha)
        
        return prior_kl + hyperprior_kl + ard_kl

class HierarchicalResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + identity

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
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(HierarchicalResidualBlock(hidden_dims[i], dropout_rate))
            self.hidden_layers.append(
                HierarchicalBayesianLinear(
                    hidden_dims[i], hidden_dims[i+1], num_groups, prior_type, dropout_rate
                )
            )
        
        # Output layer
        self.output_layer = HierarchicalBayesianLinear(
            hidden_dims[-1], output_dim, num_groups, prior_type, dropout_rate
        )
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])
    
    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_sum = 0
        
        # Input layer
        x, kl = self.input_layer(x, group_ids)
        kl_sum += kl
        x = self.bn_layers[0](x)
        x = F.relu(x)
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, HierarchicalResidualBlock):
                x = layer(x)
            else:
                x, kl = layer(x, group_ids)
                kl_sum += kl
                x = self.bn_layers[i//2 + 1](x)
                x = F.relu(x)
        
        # Output layer
        x, kl = self.output_layer(x, group_ids)
        kl_sum += kl
        
        return x, kl_sum
    
    def predict(self, x: torch.Tensor, group_ids: torch.Tensor,
                num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x, group_ids)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0), predictions.std(dim=0)
    
    def get_hierarchical_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract hierarchical parameters for visualization"""
        params = {}
        for name, module in self.named_modules():
            if isinstance(module, HierarchicalBayesianLinear):
                params[f"{name}.group_weight_mu"] = module.group_weight_mu.detach()
                params[f"{name}.group_weight_sigma"] = torch.log1p(torch.exp(module.group_weight_rho)).detach()
                params[f"{name}.raw_weight_mu"] = module.raw_weight_mu.detach()
                params[f"{name}.raw_weight_sigma"] = torch.log1p(torch.exp(module.raw_weight_rho)).detach()
                params[f"{name}.ard_alpha"] = module.ard_alpha.detach()
        return params

class CalibrationMetrics:
    @staticmethod
    def negative_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray,
                              y_std: np.ndarray) -> float:
        """Calculate Negative Log Likelihood (NLL)"""
        nll = 0.5 * np.log(2 * np.pi * y_std**2) + \
              0.5 * ((y_true - y_pred) / y_std)**2
        return np.mean(nll)
    
    @staticmethod
    def prediction_interval_coverage(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_std: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate prediction interval coverage"""
        z_score = 1.96  # for 95% confidence
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage

class PredictionWrapper(nn.Module):
    """Wrapper class for model predictions to be used with IntegratedGradients"""
    def __init__(self, model: TrueHierarchicalHBNN):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        output, _ = self.model(x, group_ids)
        return output

class HierarchicalInterpretabilityTools:
    def __init__(self, model: TrueHierarchicalHBNN, input_dim: int):
        self.model = model
        self.input_dim = input_dim
        self.wrapped_model = PredictionWrapper(model)
        self.ig = IntegratedGradients(self.wrapped_model)
    
    def compute_group_importance(self, x: torch.Tensor, group_ids: torch.Tensor,
                               num_samples: int = 100) -> torch.Tensor:
        """Compute importance of group-level parameters"""
        group_importance = []
        for _ in range(num_samples):
            with torch.no_grad():
                output, _ = self.model(x, group_ids)
                group_params = self.model.get_hierarchical_parameters()
                importance = torch.zeros(self.model.num_groups)
                for param_name, param_value in group_params.items():
                    if "group_weight_mu" in param_name:
                        importance += torch.norm(param_value, dim=(0, 1))
                group_importance.append(importance)
        return torch.stack(group_importance).mean(dim=0)
    
    def visualize_hierarchical_parameters(self, output_dir: str):
        """Visualize hierarchical parameters"""
        params = self.model.get_hierarchical_parameters()
        for name, value in params.items():
            plt.figure(figsize=(10, 6))
            if len(value.shape) == 3:  # Group-specific parameters
                for g in range(value.shape[0]):
                    plt.hist(value[g].flatten().cpu().numpy(), alpha=0.3,
                            label=f'Group {g}')
            else:  # Global parameters
                plt.hist(value.flatten().cpu().numpy())
            plt.title(f'Distribution of {name}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            if len(value.shape) == 3:
                plt.legend()
            plt.savefig(os.path.join(output_dir, f'{name}_distribution.png'))
            plt.close()

class PerformanceMetrics:
    """Comprehensive performance metrics for HBNN with ARD"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Explained variance score"""
        return 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    @staticmethod
    def mean_prediction_interval_width(y_std: np.ndarray, confidence: float = 0.95) -> float:
        """Mean width of prediction intervals"""
        z_score = 1.96  # for 95% confidence
        return np.mean(2 * z_score * y_std)
    
    @staticmethod
    def calibration_error(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_std: np.ndarray, num_bins: int = 10) -> float:
        """Calculate calibration error using quantile-based approach"""
        # Create bins based on predicted probabilities
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.percentile(y_std, quantiles * 100)
        
        # Calculate empirical coverage for each bin
        empirical_coverage = []
        predicted_coverage = []
        
        for i in range(num_bins):
            mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])
            if np.sum(mask) > 0:
                z_scores = np.abs(y_true[mask] - y_pred[mask]) / y_std[mask]
                empirical_coverage.append(np.mean(z_scores <= 1.96))
                predicted_coverage.append(0.95)  # For 95% confidence interval
        
        return np.mean(np.abs(np.array(empirical_coverage) - np.array(predicted_coverage)))
    
    @staticmethod
    def feature_importance(model: TrueHierarchicalHBNN, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on ARD parameters"""
        importance = {}
        for name, module in model.named_modules():
            if isinstance(module, HierarchicalBayesianLinear):
                ard_values = module.ard_alpha.detach().cpu().numpy()
                for i, feature in enumerate(feature_names):
                    importance[feature] = float(ard_values[i])
        return importance
    
    @staticmethod
    def group_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_std: np.ndarray, group_ids: np.ndarray) -> Dict:
        """Calculate metrics for each group"""
        unique_groups = np.unique(group_ids)
        metrics = {}
        
        for group in unique_groups:
            mask = group_ids == group
            metrics[f'group_{group}'] = {
                'rmse': PerformanceMetrics.rmse(y_true[mask], y_pred[mask]),
                'mae': PerformanceMetrics.mae(y_true[mask], y_pred[mask]),
                'r2': PerformanceMetrics.r2_score(y_true[mask], y_pred[mask]),
                'coverage': CalibrationMetrics.prediction_interval_coverage(
                    y_true[mask], y_pred[mask], y_std[mask]
                ),
                'mean_interval_width': PerformanceMetrics.mean_prediction_interval_width(y_std[mask])
            }
        
        return metrics

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
    
    # After training, calculate and save comprehensive metrics
    model.eval()
    with torch.no_grad():
        y_pred, y_std = model.predict(X_val, group_ids_val)
        
        # Convert tensors to numpy arrays
        y_val_np = y_val.numpy()
        y_pred_np = y_pred.numpy()
        y_std_np = y_std.numpy()
        group_ids_val_np = group_ids_val.numpy()
        
        # Calculate all metrics
        metrics = {
            'overall': {
                'rmse': float(PerformanceMetrics.rmse(y_val_np, y_pred_np)),
                'mae': float(PerformanceMetrics.mae(y_val_np, y_pred_np)),
                'r2': float(PerformanceMetrics.r2_score(y_val_np, y_pred_np)),
                'explained_variance': float(PerformanceMetrics.explained_variance(y_val_np, y_pred_np)),
                'nll': float(CalibrationMetrics.negative_log_likelihood(y_val_np, y_pred_np, y_std_np)),
                'coverage': float(CalibrationMetrics.prediction_interval_coverage(y_val_np, y_pred_np, y_std_np)),
                'mean_interval_width': float(PerformanceMetrics.mean_prediction_interval_width(y_std_np)),
                'calibration_error': float(PerformanceMetrics.calibration_error(y_val_np, y_pred_np, y_std_np))
            },
            'group_wise': {
                group_id: {
                    metric: float(value) 
                    for metric, value in group_metrics.items()
                }
                for group_id, group_metrics in PerformanceMetrics.group_wise_metrics(
                    y_val_np, y_pred_np, y_std_np, group_ids_val_np
                ).items()
            },
            'feature_importance': {
                feature: float(importance)
                for feature, importance in PerformanceMetrics.feature_importance(model, feature_names).items()
            }
        }
        
        # Save metrics to file
        if output_dir is not None:
            with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            importance = metrics['feature_importance']
            plt.bar(range(len(importance)), list(importance.values()))
            plt.xticks(range(len(importance)), list(importance.keys()), rotation=45)
            plt.title('Feature Importance (ARD)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()
            
            # Plot calibration curve
            plt.figure(figsize=(10, 6))
            z_scores = np.abs(y_val_np - y_pred_np) / y_std_np
            empirical_coverage = []
            predicted_coverage = np.linspace(0, 1, 100)
            for p in predicted_coverage:
                empirical_coverage.append(float(np.mean(z_scores <= norm.ppf((1 + p) / 2))))
            plt.plot(predicted_coverage, empirical_coverage, 'b-', label='Empirical')
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            plt.xlabel('Predicted Coverage')
            plt.ylabel('Empirical Coverage')
            plt.title('Calibration Curve')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
            plt.close()
    
    return train_losses, val_losses

if __name__ == "__main__":
    print("[INFO] Starting Enhanced True Hierarchical Bayesian Neural Network experiments...")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    features = ["floor_area", "ghg_emissions_int", "fuel_eui", "electric_eui", 
                "energy_star_rating", "heating_fuel"]
    target = "site_eui"
    group_column = "city"  # Assuming 'city' is the grouping variable
    feature_names = features.copy()
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    df = df.dropna(subset=features + [target, group_column])
    
    # Handle categorical features
    df['heating_fuel'] = df['heating_fuel'].astype('category').cat.codes
    
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
    
    # Experiment setup
    priors = ['hierarchical_normal', 'hierarchical_student_t', 'hierarchical_mixture']
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultsARD')
    os.makedirs(results_dir, exist_ok=True)
    summary_rows = []
    
    for prior in priors:
        print(f"[INFO] Running experiment with prior: {prior}")
        prior_dir = os.path.join(results_dir, prior)
        os.makedirs(prior_dir, exist_ok=True)
        
        # Model setup
        input_dim = len(features)
        hidden_dims = [128, 64, 32]
        output_dim = 1
        num_groups = len(unique_groups)
        
        model = TrueHierarchicalHBNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_groups=num_groups,
            prior_type=prior,
            dropout_rate=0.1
        )
        
        # Train model
        train_losses, val_losses = train_model(
            model, X_train, y_train, X_val, y_val,
            group_ids_train, group_ids_val, feature_names,
            num_epochs=100,
            batch_size=32,
            learning_rate=0.001,
            kl_weight=1e-3,
            output_dir=prior_dir
        )
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            y_pred, y_std = model.predict(X_val, group_ids_val)
        
        # Calculate calibration metrics
        nll = CalibrationMetrics.negative_log_likelihood(
            y_val.numpy(), y_pred.numpy(), y_std.numpy()
        )
        coverage = CalibrationMetrics.prediction_interval_coverage(
            y_val.numpy(), y_pred.numpy(), y_std.numpy()
        )
        
        # Save calibration metrics
        with open(os.path.join(prior_dir, 'calibration_metrics.txt'), 'w') as f:
            f.write(f"Negative Log Likelihood: {nll:.4f}\n")
            f.write(f"95% Prediction Interval Coverage: {coverage:.4f}\n")
        
        # Visualize hierarchical parameters
        interpretability = HierarchicalInterpretabilityTools(model, input_dim)
        interpretability.visualize_hierarchical_parameters(prior_dir)
        
        # Compute group importance
        group_importance = interpretability.compute_group_importance(X_val, group_ids_val)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(unique_groups)), group_importance.cpu().numpy())
        plt.xticks(range(len(unique_groups)), unique_groups, rotation=45)
        plt.xlabel('Group')
        plt.ylabel('Importance')
        plt.title(f'Group Importance ({prior})')
        plt.tight_layout()
        plt.savefig(os.path.join(prior_dir, 'group_importance.png'))
        plt.close()
        
        # Add to summary
        summary_rows.append({
            'prior': prior,
            'nll': nll,
            'coverage': coverage,
            **{f'group_{group}': float(imp) for group, imp in zip(unique_groups, group_importance)}
        })
    
    # Output comparison summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, 'compare_priors.csv'), index=False)
    print(f"[INFO] Comparison summary saved to {os.path.join(results_dir, 'compare_priors.csv')}")
    print("[INFO] All experiments complete.") 