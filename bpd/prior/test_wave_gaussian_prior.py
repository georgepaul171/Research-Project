"""
Test script for the Wave-Gaussian Mixture Prior.

This script demonstrates the usage of the novel Wave-Gaussian Mixture Prior
with the cleaned office buildings dataset, using the same features and preprocessing
as the main ARD model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wave_gaussian_prior import WaveGaussianPrior, WaveGaussianPriorConfig
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import norm

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering for building energy performance analysis.
    Using the same feature engineering as the main ARD model.
    """
    # Floor area features with robust scaling
    df['floor_area'] = df['floor_area'].clip(
        lower=df['floor_area'].quantile(0.01),
        upper=df['floor_area'].quantile(0.99)
    )
    df['floor_area_log'] = np.log1p(df['floor_area'])
    df['floor_area_squared'] = np.log1p(df['floor_area'] ** 2)
    
    # Energy intensity features with ratio analysis
    df['electric_ratio'] = df['electric_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['fuel_ratio'] = df['fuel_eui'] / (df['electric_eui'] + df['fuel_eui'])
    df['energy_mix'] = df['electric_ratio'] * df['fuel_ratio']
    df['energy_intensity_ratio'] = np.log1p((df['electric_eui'] + df['fuel_eui']) / df['floor_area'])
    
    # Building age features with non-linear transformations
    df['building_age'] = 2025 - df['year_built']
    df['building_age'] = df['building_age'].clip(
        lower=df['building_age'].quantile(0.01),
        upper=df['building_age'].quantile(0.99)
    )
    df['building_age_log'] = np.log1p(df['building_age'])
    df['building_age_squared'] = np.log1p(df['building_age'] ** 2)
    
    # Energy star rating features with normalisation
    df['energy_star_rating'] = pd.to_numeric(df['energy_star_rating'], errors='coerce')
    df['energy_star_rating'] = df['energy_star_rating'].fillna(df['energy_star_rating'].median())
    df['energy_star_rating_normalized'] = df['energy_star_rating'] / 100
    df['energy_star_rating_squared'] = df['energy_star_rating_normalized'] ** 2
    
    # GHG emissions features with intensity metrics
    df['ghg_emissions_int'] = pd.to_numeric(df['ghg_emissions_int'], errors='coerce')
    df['ghg_emissions_int'] = df['ghg_emissions_int'].fillna(df['ghg_emissions_int'].median())
    df['ghg_emissions_int_log'] = np.log1p(df['ghg_emissions_int'])
    df['ghg_per_area'] = np.log1p(df['ghg_emissions_int'] / df['floor_area'])
    
    # Interaction features capturing complex relationships
    df['age_energy_star_interaction'] = df['building_age_log'] * df['energy_star_rating_normalized']
    df['area_energy_star_interaction'] = df['floor_area_log'] * df['energy_star_rating_normalized']
    df['age_ghg_interaction'] = df['building_age_log'] * df['ghg_emissions_int_log']
    
    return df

def test_building_data():
    """
    Test the prior with the cleaned office buildings dataset.
    """
    # Load and preprocess data
    data_path = "/Users/georgepaul/Desktop/Research-Project/bpd/cleaned_office_buildings.csv"
    na_vals = ['No Value', '', 'NA', 'N/A', 'null', 'Null', 'nan', 'NaN']
    df = pd.read_csv(data_path, na_values=na_vals, low_memory=False)
    
    # Perform feature engineering
    df = feature_engineering(df)
    
    # Select features (same as main ARD model)
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
    
    # Initialize scalers
    scaler_X = RobustScaler()
    scaler_y = StandardScaler()
    
    # Prepare data
    X = df[features].values
    y = df['site_eui'].values.reshape(-1, 1)
    
    # Scale data
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()
    
    # Initialize prior
    config = WaveGaussianPriorConfig(
        n_components=3,
        n_waves=2,
        adaptation_rate=0.1
    )
    prior = WaveGaussianPrior(config)
    
    # Create output directory for visualizations
    os.makedirs('prior_visualizations', exist_ok=True)
    
    # Test prior for each feature
    for i, feature in enumerate(features):
        # Get feature data
        feature_data = X_scaled[:, i]
        
        # Visualize initial prior
        prior.visualize_prior(i, f'prior_visualizations/initial_prior_{feature}.png')
        
        # Update prior parameters
        for j in range(10):
            prior.update_parameters(feature_data, i, j)
        
        # Visualize adapted prior
        prior.visualize_prior(i, f'prior_visualizations/adapted_prior_{feature}.png')
        
        # Save prior parameters
        params = prior.get_prior_parameters()
        np.save(f'prior_visualizations/prior_params_{feature}.npy', params)
        
        print(f"Processed feature: {feature}")
        print(f"Wave frequencies: {params['feature_params'][i]['frequencies']}")
        print(f"Wave amplitudes: {params['feature_params'][i]['amplitudes']}")
        print(f"Mixture weights: {params['mixture_params']['weights']}")
        print("-" * 50)
    
    print("\nBuilding data test completed. Check the 'prior_visualizations' directory for results.")

def analyze_prior_behavior():
    """
    Analyze how the prior adapts to different features.
    """
    # Load saved parameters
    feature_params = {}
    for feature in features:
        params = np.load(f'prior_visualizations/prior_params_{feature}.npy', allow_pickle=True).item()
        feature_params[feature] = params
    
    # Create a comprehensive analysis figure
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Wave Frequencies Analysis
    plt.subplot(2, 2, 1)
    frequencies = [params['feature_params'][i]['frequencies'] for i in range(len(features))]
    plt.boxplot(frequencies, labels=features, rot=45)
    plt.title('Wave Frequencies Across Features', fontsize=12, pad=20)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Wave Amplitudes Analysis
    plt.subplot(2, 2, 2)
    amplitudes = [params['feature_params'][i]['amplitudes'] for i in range(len(features))]
    plt.boxplot(amplitudes, labels=features, rot=45)
    plt.title('Wave Amplitudes Across Features', fontsize=12, pad=20)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 3. Mixture Weights Analysis
    plt.subplot(2, 2, 3)
    weights = [params['mixture_params']['weights'] for params in feature_params.values()]
    plt.boxplot(weights, labels=features, rot=45)
    plt.title('Mixture Weights Across Features', fontsize=12, pad=20)
    plt.ylabel('Weight', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 4. Feature Importance Analysis
    plt.subplot(2, 2, 4)
    importance = [np.mean(params['feature_params'][i]['amplitudes']) * 
                 np.mean(params['mixture_params']['weights'])
                 for i, params in enumerate(feature_params.values())]
    plt.barh(features, importance)
    plt.title('Feature Importance Based on Prior Adaptation', fontsize=12, pad=20)
    plt.xlabel('Importance Score', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prior_visualizations/prior_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature-specific analysis plots
    for i, feature in enumerate(features):
        params = feature_params[feature]
        
        # Create feature-specific figure
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Prior Components
        plt.subplot(2, 2, 1)
        x = np.linspace(-3, 3, 1000)
        wave = np.zeros_like(x)
        for j in range(len(params['feature_params'][i]['frequencies'])):
            wave += params['feature_params'][i]['amplitudes'][j] * np.sin(
                2 * np.pi * params['feature_params'][i]['frequencies'][j] * x + 
                params['feature_params'][i]['phases'][j]
            )
        plt.plot(x, wave, label='Wave Component')
        plt.title(f'Wave Component for {feature}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Gaussian Mixture
        plt.subplot(2, 2, 2)
        mixture = np.zeros_like(x)
        for j in range(len(params['mixture_params']['weights'])):
            mixture += params['mixture_params']['weights'][j] * norm.pdf(
                x,
                loc=params['mixture_params']['means'][j],
                scale=np.sqrt(params['mixture_params']['variances'][j])
            )
        plt.plot(x, mixture, label='Gaussian Mixture')
        plt.title(f'Gaussian Mixture for {feature}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Combined Prior
        plt.subplot(2, 2, 3)
        prior = mixture * (1 + wave)
        plt.plot(x, prior, label='Combined Prior')
        plt.title(f'Combined Prior for {feature}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. Parameter Evolution
        plt.subplot(2, 2, 4)
        plt.bar(range(len(params['mixture_params']['weights'])), 
                params['mixture_params']['weights'],
                label='Mixture Weights')
        plt.title(f'Mixture Weights for {feature}', fontsize=12)
        plt.xlabel('Component Index')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'prior_visualizations/feature_analysis_{feature}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create summary statistics
    summary_stats = {
        'feature': features,
        'mean_frequency': [np.mean(params['feature_params'][i]['frequencies']) 
                          for i in range(len(features))],
        'mean_amplitude': [np.mean(params['feature_params'][i]['amplitudes']) 
                          for i in range(len(features))],
        'dominant_mixture_weight': [np.max(params['mixture_params']['weights']) 
                                  for params in feature_params.values()],
        'importance_score': importance
    }
    
    # Save summary statistics
    pd.DataFrame(summary_stats).to_csv('prior_visualizations/prior_summary_stats.csv', 
                                     index=False)
    
    print("\nAnalysis completed. Check the 'prior_visualizations' directory for:")
    print("1. Overall prior behavior analysis (prior_behavior_analysis.png)")
    print("2. Feature-specific analyses (feature_analysis_*.png)")
    print("3. Summary statistics (prior_summary_stats.csv)")

if __name__ == "__main__":
    print("Testing Wave-Gaussian Mixture Prior with cleaned office buildings data...")
    test_building_data()
    analyze_prior_behavior() 