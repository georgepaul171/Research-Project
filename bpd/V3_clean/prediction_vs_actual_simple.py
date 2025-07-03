import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler

# --- Load data and features (same as V3.py) ---
data_csv_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
target = "site_eui"
na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)

def feature_engineering_no_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    df['building_age'] = 2025 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    return df

df = feature_engineering_no_interactions(df)
features = [
    "ghg_emissions_int_log",
    "floor_area_log",
    "electric_eui",
    "fuel_eui",
    "energy_star_rating_normalized",
    "energy_mix",
    "building_age_log",
    "floor_area_squared",
    "energy_intensity_ratio",
    "building_age_squared",
    "energy_star_rating_squared",
    "ghg_per_area"
]
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1)

# --- Scaling (same as V3.py) ---
scaler_X = RobustScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# --- Fit Linear Regression ---
lr = LinearRegression()
lr.fit(X_scaled, y_scaled)
y_pred_lr = lr.predict(X_scaled)

# --- Fit Bayesian Ridge Regression ---
br = BayesianRidge()
br.fit(X_scaled, y_scaled)
y_pred_br, y_std_br = br.predict(X_scaled, return_std=True)

# --- Inverse transform predictions ---
y_pred_lr_orig = scaler_y.inverse_transform(y_pred_lr.reshape(-1, 1)).ravel()
y_pred_br_orig = scaler_y.inverse_transform(y_pred_br.reshape(-1, 1)).ravel()
y_std_br_orig = y_std_br * scaler_y.scale_[0]  # std in original units

# --- Results directory ---
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_simple_model')
os.makedirs(results_dir, exist_ok=True)

# --- Plot: Linear Regression ---
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_lr_orig, alpha=0.5, label='Linear Regression')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal')
plt.xlabel('Actual site_eui')
plt.ylabel('Predicted site_eui')
plt.title('Linear Regression: Prediction vs Actual')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_vs_actual_linear.png'), dpi=300)
plt.close()

# --- Plot: Bayesian Ridge Regression with error bars ---
plt.figure(figsize=(8, 6))
plt.errorbar(y, y_pred_br_orig, yerr=2*y_std_br_orig, fmt='o', alpha=0.5, label='BayesianRidge ±2σ')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal')
plt.xlabel('Actual site_eui')
plt.ylabel('Predicted site_eui')
plt.title('Bayesian Ridge: Prediction vs Actual with Uncertainty (±2σ)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_vs_actual_bayesridge.png'), dpi=300)
plt.close()

print(f"Plots saved to {results_dir}") 