"""
HBNNFD.py: Further Directions for Hierarchical Bayesian Neural Networks (HBNN)
-----------------------------------------------------------------------------
This script demonstrates advanced research directions for HBNNs:
- Deeper/wider architectures
- Alternative priors
- OOD detection
- SHAP interpretability
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

# 1. Utility Classes
# ------------------------------------------------
class HierarchicalBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_type='normal'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type
        self.prior_mu = nn.Parameter(torch.zeros(1))
        self.prior_logvar = nn.Parameter(torch.zeros(1))
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
        return F.linear(x, weight, bias), self.kl_loss()
    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl_weights = self.kl_divergence(self.weight_mu, weight_sigma)
        kl_bias = self.kl_divergence(self.bias_mu, bias_sigma)
        kl_prior_mu = 0.5 * (self.prior_mu ** 2)
        kl_prior_logvar = 0.5 * (self.prior_logvar ** 2)
        return kl_weights + kl_bias + kl_prior_mu + kl_prior_logvar
    def kl_divergence(self, mu, sigma):
        if self.prior_type == 'normal':
            prior_var = torch.exp(self.prior_logvar)
            prior_mu = self.prior_mu
            kl = torch.log(prior_var.sqrt() / sigma) + (sigma ** 2 + (mu - prior_mu) ** 2) / (2 * prior_var) - 0.5
        elif self.prior_type == 'laplace':
            b = torch.exp(0.5 * self.prior_logvar)
            kl = torch.log(2 * b * np.sqrt(np.e)) - torch.log(sigma) + (sigma + torch.abs(mu - self.prior_mu)) / b - 1
        else:
            raise ValueError("Unknown prior type")
        return kl.sum()

class DeepHierarchicalBayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, prior_type='normal'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(HierarchicalBayesianLinear(prev_dim, hidden_dim, prior_type=prior_type))
            prev_dim = hidden_dim
        layers.append(HierarchicalBayesianLinear(prev_dim, output_dim, prior_type=prior_type))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        kl_sum = 0
        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
            kl_sum += kl
        return x, kl_sum
    def predict(self, x, num_samples=100):
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0), predictions.std(dim=0)

class PredictOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out, _ = self.model(x)
        return out

# 2. OOD Detection
# ------------------------------------------------
def ood_detection(model, X_train, X_ood, output_dir):
    model.eval()
    with torch.no_grad():
        _, std_train = model.predict(torch.tensor(X_train, dtype=torch.float32))
        _, std_ood = model.predict(torch.tensor(X_ood, dtype=torch.float32))
    plt.hist(std_train.numpy().flatten(), bins=30, alpha=0.7, label='Train')
    plt.hist(std_ood.numpy().flatten(), bins=30, alpha=0.7, label='OOD')
    plt.legend()
    plt.title('Predictive Uncertainty: Train vs. OOD')
    plt.xlabel('Predictive Std')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'ood_uncertainty.png'))
    plt.close()

# 3. SHAP Interpretability
# ------------------------------------------------
def compute_shap(model, X_train, X_val, features, output_dir):
    try:
        import shap
        print('Using SHAP KernelExplainer (model-agnostic, may be slow)...')
        wrapper = PredictOnlyWrapper(model)
        # Use a small background set for speed, but keep all features
        background = X_train[:100, :]  # Ensure 2D
        X_val_subset = X_val[:200, :]  # Ensure 2D
        print('X_train shape:', background.shape)
        print('X_val_subset shape:', X_val_subset.shape)
        print('Features:', features)
        explainer = shap.KernelExplainer(lambda x: wrapper(torch.tensor(x, dtype=torch.float32)).detach().numpy(), background)
        shap_values = explainer.shap_values(X_val_subset, nsamples=100)
        shap.summary_plot(shap_values, X_val_subset, feature_names=features, show=False)
        plt.savefig(os.path.join(output_dir, 'shap_summary_kernel.png'))
        plt.close()
    except ImportError:
        print('shap not installed, skipping SHAP analysis.')

# 4. Main Experiment Block
# ------------------------------------------------
if __name__ == "__main__":
    print('Loading and preprocessing data...')
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv("../cleaned_office_buildings.csv", na_values=na_vals, low_memory=False)
    features = ["floor_area", "ghg_emissions_int", "fuel_eui", "electric_eui"]
    target = "site_eui"
    df = df.dropna(subset=features + [target])
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    print('Training a deeper HBNN with alternative prior...')
    model = DeepHierarchicalBayesianNeuralNetwork(input_dim=4, hidden_dims=[128, 64, 32], output_dim=1, prior_type='normal')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    kl_weight = 1e-3
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output, kl = model(X_train_tensor)
        mse_loss = F.mse_loss(output, y_train_tensor)
        loss = mse_loss + kl_weight * kl
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={loss.item():.4f}')
    print('Training complete.')

    print('Evaluating model on validation set...')
    model.eval()
    with torch.no_grad():
        y_pred, y_std = model.predict(X_val_tensor)
    print('Validation RMSE:', np.sqrt(mean_squared_error(y_val, y_pred.numpy())))
    print('Validation R2:', r2_score(y_val, y_pred.numpy()))

    print('Running OOD detection (adding noise to validation set)...')
    X_ood = X_val + np.random.normal(0, 2, X_val.shape)
    ood_detection(model, X_val, X_ood, output_dir)
    print('OOD detection plot saved.')

    print('Running SHAP interpretability analysis (this may take a few minutes)...')
    compute_shap(model, X_train, X_val, features, output_dir)
    print('SHAP analysis complete.')

    print('All plots and results saved in:', output_dir) 