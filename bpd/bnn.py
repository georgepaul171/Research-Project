import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma_1: float = 1.0,
                 prior_sigma_2: float = 0.002, prior_pi: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Prior parameters
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.constant_(self.bias_rho, -5.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample weights from the posterior
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias), self.kl_loss()
    
    def kl_loss(self) -> torch.Tensor:
        # KL divergence between posterior and prior
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_weights = self.kl_divergence(self.weight_mu, weight_sigma)
        kl_bias = self.kl_divergence(self.bias_mu, bias_sigma)
        
        return kl_weights + kl_bias
    
    def kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        kl = self.prior_pi * self.kl_gaussian(mu, sigma, self.prior_sigma_1) + \
             (1 - self.prior_pi) * self.kl_gaussian(mu, sigma, self.prior_sigma_2)
        return kl.sum()
    
    def kl_gaussian(self, mu: torch.Tensor, sigma: torch.Tensor, prior_sigma: float) -> torch.Tensor:
        return 0.5 * (torch.log(prior_sigma**2 / sigma**2) + 
                     (sigma**2 + mu**2) / prior_sigma**2 - 1)

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(BayesianLinear(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_sum = 0
        for layer in self.layers:
            x, kl = layer(x)
            x = F.relu(x)
            kl_sum += kl
        
        return x, kl_sum
    
    def predict(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        predictions = []
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        return torch.stack(predictions).mean(dim=0)

def train_bnn(model: BayesianNeuralNetwork, 
             train_loader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             num_epochs: int,
             kl_weight: float = 1.0) -> None:
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output, kl = model(batch_x)
            
            # Compute loss
            mse_loss = F.mse_loss(output, batch_y)
            loss = mse_loss + kl_weight * kl
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Load real data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv", na_values=na_vals, low_memory=False)
    features = ["floor_area"]
    target = "site_eui"
    print(f"Original rows: {len(df)}")
    for col in features + [target]:
        print(f"Missing in {col}: {df[col].isna().sum()}")
    # Drop rows with missing values in features or target
    df = df.dropna(subset=features + [target])
    print(f"Rows remaining after dropna: {len(df)}")
    print("First 5 rows after dropna:")
    print(df[features + [target]].head())
    # Extract features and target
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)
    # Normalize features and target
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / (y_std + 1e-8)
    # Convert to torch tensors
    X = torch.tensor(X)
    y = torch.tensor(y)
    # Split into train and validation sets
    n = X.shape[0]
    train_size = int(0.8 * n)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Initialize model
    model = BayesianNeuralNetwork(input_dim=1, hidden_dims=[64, 32], output_dim=1)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop with validation
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
            torch.save(model.state_dict(), 'best_bnn_model.pth')
        print(f'Epoch {epoch+1}/100:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('training_losses.png')
    plt.close()
    # Plot predictions vs actual values
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
    plt.title('Predictions vs Actual Values')
    plt.legend()
    plt.savefig('predictions_vs_actual.png')
    plt.close()
    # Plot uncertainty distribution
    model.eval()
    test_x = X_val[:100]  # Use 100 validation samples for uncertainty
    predictions = []
    for _ in range(100):  # 100 forward passes
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
    plt.title('Distribution of Prediction Uncertainties')
    plt.savefig('uncertainty_distribution.png')
    plt.close()
    # Print example predictions
    print("\nExample predictions with uncertainty:")
    for i in range(min(5, len(mean_pred))):
        print(f"Sample {i+1}: {mean_pred[i].item():.4f} ± {std_pred[i].item():.4f}")
