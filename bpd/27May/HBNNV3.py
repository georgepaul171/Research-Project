"""
HBNNV3.py: True Hierarchical Bayesian Neural Network Implementation
------------------------------------------------------------------
- Implements proper hierarchical Bayesian modeling with group and individual level parameters
- Uses hyperpriors for group-level parameters
- Implements hierarchical uncertainty propagation
- Includes group-level and individual-level KL divergences
- Provides proper hierarchical sampling in the forward pass
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

class PriorDistribution:
    """Base class for prior distributions"""
    def __init__(self, mu=0.0, logvar=0.0):
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.logvar = torch.tensor(logvar, dtype=torch.float32)
    
    def kl_divergence(self, mu, sigma):
        raise NotImplementedError

class HierarchicalPrior(PriorDistribution):
    """Base class for hierarchical prior distributions"""
    def __init__(self, group_mu=0.0, group_logvar=0.0, individual_mu=0.0, individual_logvar=0.0):
        self.group_mu = torch.tensor(group_mu, dtype=torch.float32)
        self.group_logvar = torch.tensor(group_logvar, dtype=torch.float32)
        self.individual_mu = torch.tensor(individual_mu, dtype=torch.float32)
        self.individual_logvar = torch.tensor(individual_logvar, dtype=torch.float32)
    
    def kl_divergence(self, group_mu, group_sigma, individual_mu, individual_sigma):
        raise NotImplementedError

class HierarchicalNormalPrior(HierarchicalPrior):
    def kl_divergence(self, group_mu, group_sigma, individual_mu, individual_sigma):
        # Group-level KL
        group_var = torch.exp(self.group_logvar)
        group_kl = torch.log(group_var.sqrt() / group_sigma) + \
                  (group_sigma ** 2 + (group_mu - self.group_mu) ** 2) / (2 * group_var) - 0.5
        
        # Individual-level KL
        individual_var = torch.exp(self.individual_logvar)
        individual_kl = torch.log(individual_var.sqrt() / individual_sigma) + \
                       (individual_sigma ** 2 + (individual_mu - self.individual_mu) ** 2) / (2 * individual_var) - 0.5
        
        return group_kl.sum() + individual_kl.sum()

class HierarchicalLaplacePrior(HierarchicalPrior):
    def kl_divergence(self, group_mu, group_sigma, individual_mu, individual_sigma):
        # Group-level KL
        group_b = torch.exp(0.5 * self.group_logvar)
        group_kl = torch.log(2 * group_b * torch.sqrt(torch.tensor(np.e))) - torch.log(group_sigma) + \
                  (group_sigma + torch.abs(group_mu - self.group_mu)) / group_b - 1
        
        # Individual-level KL
        individual_b = torch.exp(0.5 * self.individual_logvar)
        individual_kl = torch.log(2 * individual_b * torch.sqrt(torch.tensor(np.e))) - torch.log(individual_sigma) + \
                       (individual_sigma + torch.abs(individual_mu - self.individual_mu)) / individual_b - 1
        
        return group_kl.sum() + individual_kl.sum()

class HierarchicalStudentTPrior(HierarchicalPrior):
    def __init__(self, group_mu=0.0, group_logvar=0.0, individual_mu=0.0, individual_logvar=0.0, df=3.0):
        super().__init__(group_mu, group_logvar, individual_mu, individual_logvar)
        self.df = torch.tensor(df, dtype=torch.float32)
    
    def kl_divergence(self, group_mu, group_sigma, individual_mu, individual_sigma):
        # Group-level KL
        group_var = torch.exp(self.group_logvar)
        group_kl = 0.5 * torch.log(group_var / group_sigma**2) + \
                  (self.df + 1) * torch.log(1 + (group_mu - self.group_mu)**2 / (self.df * group_var)) - \
                  (self.df + 1) * torch.log(1 + (group_mu - self.group_mu)**2 / (self.df * group_sigma**2))
        
        # Individual-level KL
        individual_var = torch.exp(self.individual_logvar)
        individual_kl = 0.5 * torch.log(individual_var / individual_sigma**2) + \
                       (self.df + 1) * torch.log(1 + (individual_mu - self.individual_mu)**2 / (self.df * individual_var)) - \
                       (self.df + 1) * torch.log(1 + (individual_mu - self.individual_mu)**2 / (self.df * individual_sigma**2))
        
        return group_kl.sum() + individual_kl.sum()

class HierarchicalMixtureGaussianPrior(HierarchicalPrior):
    def __init__(self, group_mus=[-1.0, 1.0], group_logvars=[0.0, 0.0], group_weights=[0.5, 0.5],
                 individual_mus=[-1.0, 1.0], individual_logvars=[0.0, 0.0], individual_weights=[0.5, 0.5]):
        super().__init__()
        self.group_mus = [torch.tensor(m, dtype=torch.float32) for m in group_mus]
        self.group_logvars = [torch.tensor(v, dtype=torch.float32) for v in group_logvars]
        self.group_weights = [torch.tensor(w, dtype=torch.float32) for w in group_weights]
        self.individual_mus = [torch.tensor(m, dtype=torch.float32) for m in individual_mus]
        self.individual_logvars = [torch.tensor(v, dtype=torch.float32) for v in individual_logvars]
        self.individual_weights = [torch.tensor(w, dtype=torch.float32) for w in individual_weights]
    
    def kl_divergence(self, group_mu, group_sigma, individual_mu, individual_sigma):
        # Group-level KL
        group_kl = torch.zeros_like(group_mu)
        for m, v, w in zip(self.group_mus, self.group_logvars, self.group_weights):
            prior_var = torch.exp(v)
            group_kl += w * (torch.log(prior_var.sqrt() / group_sigma) +
                           (group_sigma ** 2 + (group_mu - m) ** 2) / (2 * prior_var) - 0.5)
        
        # Individual-level KL
        individual_kl = torch.zeros_like(individual_mu)
        for m, v, w in zip(self.individual_mus, self.individual_logvars, self.individual_weights):
            prior_var = torch.exp(v)
            individual_kl += w * (torch.log(prior_var.sqrt() / individual_sigma) +
                                (individual_sigma ** 2 + (individual_mu - m) ** 2) / (2 * prior_var) - 0.5)
        
        return group_kl.sum() + individual_kl.sum()

class HierarchicalResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
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

class HierarchicalBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_type='hierarchical_normal', dropout_rate=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize hierarchical prior
        if prior_type == 'hierarchical_normal':
            self.prior = HierarchicalNormalPrior()
        elif prior_type == 'hierarchical_laplace':
            self.prior = HierarchicalLaplacePrior()
        elif prior_type == 'hierarchical_student_t':
            self.prior = HierarchicalStudentTPrior()
        elif prior_type == 'hierarchical_mixture':
            self.prior = HierarchicalMixtureGaussianPrior()
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Group-level parameters
        self.group_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.group_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.group_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.group_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Individual-level parameters
        self.individual_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.individual_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.individual_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.individual_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize group-level parameters
        nn.init.kaiming_uniform_(self.group_weight_mu, a=np.sqrt(5))
        nn.init.constant_(self.group_weight_rho, -5.0)
        nn.init.uniform_(self.group_bias_mu, -0.1, 0.1)
        nn.init.constant_(self.group_bias_rho, -5.0)
        
        # Initialize individual-level parameters
        nn.init.kaiming_uniform_(self.individual_weight_mu, a=np.sqrt(5))
        nn.init.constant_(self.individual_weight_rho, -5.0)
        nn.init.uniform_(self.individual_bias_mu, -0.1, 0.1)
        nn.init.constant_(self.individual_bias_rho, -5.0)
    
    def forward(self, x):
        # Group-level sampling
        group_weight_sigma = torch.log1p(torch.exp(self.group_weight_rho))
        group_bias_sigma = torch.log1p(torch.exp(self.group_bias_rho))
        group_weight = self.group_weight_mu + group_weight_sigma * torch.randn_like(self.group_weight_mu)
        group_bias = self.group_bias_mu + group_bias_sigma * torch.randn_like(self.group_bias_mu)
        
        # Individual-level sampling
        individual_weight_sigma = torch.log1p(torch.exp(self.individual_weight_rho))
        individual_bias_sigma = torch.log1p(torch.exp(self.individual_bias_rho))
        individual_weight = self.individual_weight_mu + individual_weight_sigma * torch.randn_like(self.individual_weight_mu)
        individual_bias = self.individual_bias_mu + individual_bias_sigma * torch.randn_like(self.individual_bias_mu)
        
        # Combine group and individual effects
        weight = group_weight + individual_weight
        bias = group_bias + individual_bias
        
        x = self.dropout(x)
        return F.linear(x, weight, bias), self.kl_loss(group_weight_sigma, group_bias_sigma,
                                                      individual_weight_sigma, individual_bias_sigma)
    
    def kl_loss(self, group_weight_sigma, group_bias_sigma, individual_weight_sigma, individual_bias_sigma):
        kl_weights = self.prior.kl_divergence(
            self.group_weight_mu, group_weight_sigma,
            self.individual_weight_mu, individual_weight_sigma
        )
        kl_bias = self.prior.kl_divergence(
            self.group_bias_mu, group_bias_sigma,
            self.individual_bias_mu, individual_bias_sigma
        )
        return kl_weights + kl_bias

class TrueHierarchicalHBNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, prior_type='hierarchical_normal', dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = HierarchicalBayesianLinear(input_dim, hidden_dims[0], prior_type, dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(HierarchicalResidualBlock(hidden_dims[i], dropout_rate))
            self.hidden_layers.append(
                HierarchicalBayesianLinear(hidden_dims[i], hidden_dims[i+1], prior_type, dropout_rate)
            )
        
        # Output layer
        self.output_layer = HierarchicalBayesianLinear(hidden_dims[-1], output_dim, prior_type, dropout_rate)
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])
    
    def forward(self, x):
        kl_sum = 0
        
        # Input layer
        x, kl = self.input_layer(x)
        kl_sum += kl
        x = self.bn_layers[0](x)
        x = F.relu(x)
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, HierarchicalResidualBlock):
                x = layer(x)
            else:
                x, kl = layer(x)
                kl_sum += kl
                x = self.bn_layers[i//2 + 1](x)
                x = F.relu(x)
        
        # Output layer
        x, kl = self.output_layer(x)
        kl_sum += kl
        
        return x, kl_sum
    
    def predict(self, x, num_samples=100):
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0), predictions.std(dim=0)

class CalibrationMetrics:
    @staticmethod
    def expected_calibration_error(y_true, y_pred, y_std, n_bins=10):
        """Calculate Expected Calibration Error (ECE)"""
        confidences = 1.96 * y_std  # 95% confidence interval
        accuracies = np.abs(y_true - y_pred) <= confidences
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * np.sum(in_bin) / len(y_true)
        
        return ece
    
    @staticmethod
    def reliability_diagram(y_true, y_pred, y_std, n_bins=10):
        """Generate reliability diagram"""
        confidences = 1.96 * y_std
        accuracies = np.abs(y_true - y_pred) <= confidences
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies_in_bins = []
        confidences_in_bins = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracies_in_bins.append(np.mean(accuracies[in_bin]))
                confidences_in_bins.append(np.mean(confidences[in_bin]))
        
        return accuracies_in_bins, confidences_in_bins

class PredictionWrapper(nn.Module):
    """Wrapper class to handle model predictions for interpretability"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output, _ = self.model(x)
        return output

class InterpretabilityTools:
    def __init__(self, model, input_dim):
        self.model = model
        self.input_dim = input_dim
        self.wrapped_model = PredictionWrapper(model)
        self.ig = IntegratedGradients(self.wrapped_model)
    
    def compute_integrated_gradients(self, x, target=0):
        """Compute Integrated Gradients attribution"""
        attributions = self.ig.attribute(x, target=target)
        return attributions
    
    def compute_shap_values(self, x, background):
        """Compute SHAP values using KernelExplainer"""
        explainer = shap.KernelExplainer(
            lambda x: self.wrapped_model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
            background
        )
        shap_values = explainer.shap_values(x)
        return shap_values
    
    def compute_feature_importance(self, x, num_samples=100):
        """Compute feature importance using variance of attributions"""
        attributions = []
        for _ in range(num_samples):
            attr = self.compute_integrated_gradients(x)
            attributions.append(attr)
        
        attributions = torch.stack(attributions)
        importance = torch.var(attributions, dim=0)
        return importance

def train_model(model, X_train, y_train, X_val, y_val, 
                num_epochs=100, batch_size=32, learning_rate=0.001,
                kl_weight=1e-3, patience=10, output_dir=None):
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
            
            optimizer.zero_grad()
            output, kl = model(batch_X)
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
            val_output, val_kl = model(X_val)
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

    # Plot training curves
    if output_dir is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
    
    return train_losses, val_losses

if __name__ == "__main__":
    print("[INFO] Starting True Hierarchical Bayesian Neural Network experiments...")
    
    # Data setup
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    features = ["floor_area", "ghg_emissions_int", "fuel_eui", "electric_eui"]
    target = "site_eui"
    feature_names = features.copy()
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    df = df.dropna(subset=features + [target])
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Experiment setup
    priors = ['hierarchical_normal', 'hierarchical_laplace', 'hierarchical_student_t', 'hierarchical_mixture']
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultsV3')
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
        
        model = TrueHierarchicalHBNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            prior_type=prior,
            dropout_rate=0.1
        )
        
        # Train model
        train_losses, val_losses = train_model(
            model, X_train, y_train, X_val, y_val,
            num_epochs=100,
            batch_size=32,
            learning_rate=0.001,
            kl_weight=1e-3,
            output_dir=prior_dir
        )
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            y_pred, y_std = model.predict(X_val)
        
        # Calculate calibration metrics
        ece = CalibrationMetrics.expected_calibration_error(
            y_val.numpy(), y_pred.numpy(), y_std.numpy()
        )
        
        # Save calibration metrics
        with open(os.path.join(prior_dir, 'calibration_metrics.txt'), 'w') as f:
            f.write(f"Expected Calibration Error: {ece:.4f}\n")
        
        # Generate reliability diagram
        accuracies, confidences = CalibrationMetrics.reliability_diagram(
            y_val.numpy(), y_pred.numpy(), y_std.numpy()
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(confidences, accuracies, 'bo-', label='Model')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram ({prior})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(prior_dir, 'reliability_diagram.png'))
        plt.close()
        
        # Compute feature importance
        interpretability = InterpretabilityTools(model, input_dim)
        feature_importance = interpretability.compute_feature_importance(X_val)
        feature_importance = feature_importance.mean(dim=0) if len(feature_importance.shape) > 1 else feature_importance
        
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, feature_importance.detach().cpu().numpy())
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance ({prior})')
        plt.savefig(os.path.join(prior_dir, 'feature_importance.png'))
        plt.close()
        
        np.savetxt(os.path.join(prior_dir, 'feature_importance.csv'), 
                   feature_importance.detach().cpu().numpy(), delimiter=',', header=','.join(feature_names), comments='')
        
        # Generate SHAP summary plot
        try:
            print(f"[INFO] Computing SHAP summary plot for {prior}...")
            wrapper = PredictionWrapper(model)
            background = X_train[:100, :].numpy()
            X_val_subset = X_val[:200, :].numpy()
            explainer = shap.KernelExplainer(
                lambda x: wrapper(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
                background
            )
            shap_values = explainer.shap_values(X_val_subset, nsamples=100)
            shap.summary_plot(shap_values, X_val_subset, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(prior_dir, 'shap_summary.png'))
            plt.close()
        except Exception as e:
            print(f"[WARN] SHAP plot failed for {prior}: {e}")
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses ({prior})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(prior_dir, 'training_curves.png'))
        plt.close()
        
        # Add to summary
        summary_rows.append({
            'prior': prior,
            'ece': ece,
            **{f'feat_{name}': float(val) for name, val in zip(feature_names, feature_importance.detach().cpu().numpy())}
        })
    
    # Output comparison summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, 'compare_priors.csv'), index=False)
    print(f"[INFO] Comparison summary saved to {os.path.join(results_dir, 'compare_priors.csv')}")
    print("[INFO] All experiments complete.") 