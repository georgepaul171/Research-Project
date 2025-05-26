"""
Enhanced Hierarchical Bayesian Neural Network (HBNN)
--------------------------------------------------
This implementation includes:
- Advanced architectural features (residual connections, dropout, batch norm)
- Multiple prior distributions (Normal, Laplace, Student's t, Mixture of Gaussians)
- Calibration metrics and evaluation
- Enhanced interpretability methods
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import shap
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.metrics import infidelity, sensitivity_max

# Set up output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PriorDistribution:
    """Base class for prior distributions"""
    def __init__(self, mu=0.0, logvar=0.0):
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.logvar = torch.tensor(logvar, dtype=torch.float32)
    
    def kl_divergence(self, mu, sigma):
        raise NotImplementedError

class NormalPrior(PriorDistribution):
    def kl_divergence(self, mu, sigma):
        prior_var = torch.exp(self.logvar)
        prior_mu = self.mu
        kl = torch.log(prior_var.sqrt() / sigma) + (sigma ** 2 + (mu - prior_mu) ** 2) / (2 * prior_var) - 0.5
        return kl.sum()

class LaplacePrior(PriorDistribution):
    def kl_divergence(self, mu, sigma):
        b = torch.exp(0.5 * self.logvar)
        kl = torch.log(2 * b * torch.sqrt(torch.tensor(np.e))) - torch.log(sigma) + (sigma + torch.abs(mu - self.mu)) / b - 1
        return kl.sum()

class StudentTPrior(PriorDistribution):
    def __init__(self, mu=0.0, logvar=0.0, df=3.0):
        super().__init__(mu, logvar)
        self.df = torch.tensor(df, dtype=torch.float32)
    
    def kl_divergence(self, mu, sigma):
        # Approximate KL divergence for Student's t
        prior_var = torch.exp(self.logvar)
        kl = 0.5 * torch.log(prior_var / sigma**2) + \
             (self.df + 1) * torch.log(1 + (mu - self.mu)**2 / (self.df * prior_var)) - \
             (self.df + 1) * torch.log(1 + (mu - self.mu)**2 / (self.df * sigma**2))
        return kl.sum()

class MixtureGaussianPrior(PriorDistribution):
    def __init__(self, mus=[-1.0, 1.0], logvars=[0.0, 0.0], weights=[0.5, 0.5]):
        super().__init__()
        self.mus = [torch.tensor(m, dtype=torch.float32) for m in mus]
        self.logvars = [torch.tensor(v, dtype=torch.float32) for v in logvars]
        self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
    
    def kl_divergence(self, mu, sigma):
        kl = torch.tensor(0.0, dtype=torch.float32)
        for m, v, w in zip(self.mus, self.logvars, self.weights):
            prior_var = torch.exp(v)
            kl += w * (torch.log(prior_var.sqrt() / sigma) + 
                      (sigma ** 2 + (mu - m) ** 2) / (2 * prior_var) - 0.5)
        return kl.sum()

class ResidualBlock(nn.Module):
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
    def __init__(self, in_features, out_features, prior_type='normal', dropout_rate=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize prior
        if prior_type == 'normal':
            self.prior = NormalPrior()
        elif prior_type == 'laplace':
            self.prior = LaplacePrior()
        elif prior_type == 'student_t':
            self.prior = StudentTPrior()
        elif prior_type == 'mixture':
            self.prior = MixtureGaussianPrior()
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Initialize parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.constant_(self.bias_rho, -5.0)
    
    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        x = self.dropout(x)
        return F.linear(x, weight, bias), self.kl_loss(weight_sigma, bias_sigma)
    
    def kl_loss(self, weight_sigma, bias_sigma):
        kl_weights = self.prior.kl_divergence(self.weight_mu, weight_sigma)
        kl_bias = self.prior.kl_divergence(self.bias_mu, bias_sigma)
        return kl_weights + kl_bias

class EnhancedHBNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, prior_type='normal', dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = HierarchicalBayesianLinear(input_dim, hidden_dims[0], prior_type, dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(ResidualBlock(hidden_dims[i], dropout_rate))
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
            if isinstance(layer, ResidualBlock):
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
                kl_weight=1e-3, patience=10):
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
            # Ensure KL loss is positive and scale it appropriately
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
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()
    
    return train_losses, val_losses

if __name__ == "__main__":
    print("[INFO] Starting HBNN pipeline...")
    # Load your real data
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("[INFO] Loading data from:", "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv")
    data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"

    features = ["floor_area", "ghg_emissions_int", "fuel_eui", "electric_eui"]
    target = "site_eui"
    print(f"[INFO] Features: {features}")
    print(f"[INFO] Target: {target}")

    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    print(f"[INFO] Loaded dataframe shape: {df.shape}")
    df = df.dropna(subset=features + [target])
    print(f"[INFO] Dataframe shape after dropping NA: {df.shape}")

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    print("[INFO] Data standardized.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"[INFO] y_train: {y_train.shape}, y_val: {y_val.shape}")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    print("[INFO] Data converted to torch tensors.")

    input_dim = 4
    hidden_dims = [128, 64, 32]
    output_dim = 1
    print(f"[INFO] Model input_dim: {input_dim}, hidden_dims: {hidden_dims}, output_dim: {output_dim}")

    model = EnhancedHBNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        prior_type='student_t',  # Try different priors: 'normal', 'laplace', 'student_t', 'mixture'
        dropout_rate=0.1
    )
    print("[INFO] Model instantiated.")

    print("[INFO] Beginning training...")
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        kl_weight=1e-3
    )
    print("[INFO] Training complete.")

    print("[INFO] Making predictions on validation set...")
    model.eval()
    with torch.no_grad():
        y_pred, y_std = model.predict(X_val)
    print(f"[INFO] y_pred shape: {y_pred.shape}, y_std shape: {y_std.shape}")

    print("[INFO] Calculating calibration metrics...")
    ece = CalibrationMetrics.expected_calibration_error(
        y_val.numpy(), y_pred.numpy(), y_std.numpy()
    )
    print(f"[RESULT] Expected Calibration Error: {ece:.4f}")

    with open(os.path.join(OUTPUT_DIR, 'calibration_metrics.txt'), 'w') as f:
        f.write(f"Expected Calibration Error: {ece:.4f}\n")
    print(f"[INFO] Calibration metrics saved to {os.path.join(OUTPUT_DIR, 'calibration_metrics.txt')}")

    print("[INFO] Generating reliability diagram...")
    accuracies, confidences = CalibrationMetrics.reliability_diagram(
        y_val.numpy(), y_pred.numpy(), y_std.numpy()
    )
    plt.figure(figsize=(8, 6))
    plt.plot(confidences, accuracies, 'bo-', label='Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'reliability_diagram.png'))
    plt.close()
    print(f"[INFO] Reliability diagram saved to {os.path.join(OUTPUT_DIR, 'reliability_diagram.png')}")

    print("[INFO] Computing feature importance...")
    interpretability = InterpretabilityTools(model, input_dim)
    feature_importance = interpretability.compute_feature_importance(X_val)
    feature_importance = feature_importance.mean(dim=0) if len(feature_importance.shape) > 1 else feature_importance
    print(f"[INFO] Feature importance shape: {feature_importance.shape}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(input_dim), feature_importance.detach().cpu().numpy())
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()
    print(f"[INFO] Feature importance plot saved to {os.path.join(OUTPUT_DIR, 'feature_importance.png')}")

    np.savetxt(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), 
               feature_importance.detach().cpu().numpy(), delimiter=',')
    print(f"[INFO] Feature importance values saved to {os.path.join(OUTPUT_DIR, 'feature_importance.csv')}")

    print(f"[INFO] All results have been saved to: {OUTPUT_DIR}") 