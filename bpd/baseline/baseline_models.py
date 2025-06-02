import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from typing import Tuple, List, Optional, Dict, Union
import logging
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import xgboost as xgb

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Remove any existing handlers to avoid duplicate messages
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Remove root logger handlers
for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

# Add a new handler with the correct format
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Prevent propagation to root logger
logger.propagate = False

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset"""
    # Floor area features with scaling
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)
    
    # Energy intensity features
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    
    # Building age features
    df['building_age'] = 2025 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)
    
    # Energy star rating features
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    
    # GHG emissions features
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    
    # Interaction features
    df['age_energy_star_interaction'] = df['building_age_log'] * df['energy_star_rating_normalized']
    df['area_energy_star_interaction'] = df['floor_area_log'] * df['energy_star_rating_normalized']
    df['age_ghg_interaction'] = df['building_age_log'] * df['ghg_emissions_int_log']
    
    return df

def train_and_evaluate_models(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                            output_dir: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Train and evaluate XGBoost and Random Forest models
    
    Args:
        X: Input features
        y: Target values
        feature_names: List of feature names
        output_dir: Optional directory to save results
        
    Returns:
        xgb_metrics: XGBoost model metrics
        rf_metrics: Random Forest model metrics
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb_cv_metrics = []
    rf_cv_metrics = []
    
    # Train and evaluate models
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train XGBoost
        xgb_model.fit(
            X_train_scaled, y_train,
            verbose=False
        )
        
        # Train Random Forest
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate XGBoost
        xgb_pred = xgb_model.predict(X_val_scaled)
        xgb_metrics = {
            'fold': fold,
            'rmse': np.sqrt(mean_squared_error(y_val, xgb_pred)),
            'mae': mean_absolute_error(y_val, xgb_pred),
            'r2': r2_score(y_val, xgb_pred)
        }
        xgb_cv_metrics.append(xgb_metrics)
        
        # Evaluate Random Forest
        rf_pred = rf_model.predict(X_val_scaled)
        rf_metrics = {
            'fold': fold,
            'rmse': np.sqrt(mean_squared_error(y_val, rf_pred)),
            'mae': mean_absolute_error(y_val, rf_pred),
            'r2': r2_score(y_val, rf_pred)
        }
        rf_cv_metrics.append(rf_metrics)
    
    # Calculate average metrics
    xgb_avg_metrics = pd.DataFrame(xgb_cv_metrics).mean().to_dict()
    rf_avg_metrics = pd.DataFrame(rf_cv_metrics).mean().to_dict()
    
    if output_dir is not None:
        # Save metrics
        metrics = {
            'xgb': xgb_avg_metrics,
            'rf': rf_avg_metrics
        }
        with open(os.path.join(output_dir, 'baseline_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Feature Importance Comparison
        plt.subplot(2, 2, 1)
        xgb_importance = xgb_model.feature_importances_
        rf_importance = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'XGBoost': xgb_importance,
            'Random Forest': rf_importance
        })
        importance_df = importance_df.sort_values('XGBoost', ascending=False)
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        plt.bar(x - width/2, importance_df['XGBoost'], width, label='XGBoost')
        plt.bar(x + width/2, importance_df['Random Forest'], width, label='Random Forest')
        plt.xticks(x, importance_df['Feature'], rotation=45, ha='right')
        plt.title('Feature Importance Comparison')
        plt.legend()
        
        # SHAP Values for XGBoost
        plt.subplot(2, 2, 2)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_val_scaled)
        shap.summary_plot(shap_values, X_val_scaled, feature_names=feature_names, show=False)
        plt.title('XGBoost SHAP Values')
        
        # Performance Metrics Comparison
        plt.subplot(2, 2, 3)
        metrics_df = pd.DataFrame({
            'Model': ['XGBoost', 'Random Forest'] * 3,
            'Metric': ['RMSE', 'RMSE', 'MAE', 'MAE', 'R²', 'R²'],
            'Value': [
                xgb_avg_metrics['rmse'], rf_avg_metrics['rmse'],
                xgb_avg_metrics['mae'], rf_avg_metrics['mae'],
                xgb_avg_metrics['r2'], rf_avg_metrics['r2']
            ]
        })
        sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model')
        plt.title('Model Performance Comparison')
        
        # Prediction vs Actual
        plt.subplot(2, 2, 4)
        plt.scatter(y_val, xgb_pred, alpha=0.5, label='XGBoost')
        plt.scatter(y_val, rf_pred, alpha=0.5, label='Random Forest')
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs Actual')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baseline_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save models
        joblib.dump(xgb_model, os.path.join(output_dir, 'xgb_model.joblib'))
        joblib.dump(rf_model, os.path.join(output_dir, 'rf_model.joblib'))
    
    return xgb_avg_metrics, rf_avg_metrics

if __name__ == "__main__":
    logger.info("Starting baseline models analysis")
    
    # Data setup
    data_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned_office_buildings.csv")
    target = "site_eui"
    
    # Load and preprocess data
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_csv_path, na_values=na_vals, low_memory=False)
    
    # Feature engineering
    logger.info("Performing feature engineering")
    df = feature_engineering(df)
    
    # Select features
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
        "ghg_per_area",
        "age_energy_star_interaction",
        "area_energy_star_interaction",
        "age_ghg_interaction"
    ]
    feature_names = features.copy()
    
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1)
    
    # Train and evaluate models
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'baseline')
    xgb_metrics, rf_metrics = train_and_evaluate_models(X, y, feature_names, output_dir=results_dir)
    
    logger.info("XGBoost Metrics:")
    for metric, value in xgb_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nRandom Forest Metrics:")
    for metric, value in rf_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("Complete. Results saved to %s", results_dir) 