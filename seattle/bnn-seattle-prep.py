# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Bayesian Neural Network for EUI Estimation (with Uncertainty)
# 
# This notebook loads pre-cleaned Seattle building data and sets up the input matrix
# for Bayesian neural network modeling with PyMC. We include feature preprocessing and a data split.

# ## Environment setup (run once in terminal)
# python3 -m venv bpd_env
# source bpd_env/bin/activate
# pip install pandas scikit-learn pymc arviz matplotlib

# %%
# ## 1. Load libraries and dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Path to cleaned data
file_path = "/Users/georgepaul/Desktop/Research-Project/seattle/seattle-data-cleaned.csv"

# Load dataset
df = pd.read_csv(file_path)
print("Data shape:", df.shape)
df.head()
# %%
# ## 2. Define relevant features

target = "SiteEUI(kBtu/sf)"

numerical_features = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings",
    "PropertyGFATotal", "ENERGYSTARScore",
    "Electricity(kWh)", "NaturalGas(kBtu)",
    "SteamUse(kBtu)", "GHGEmissionsIntensity"
]

# Automatically detect one-hot encoded categorical features
categorical_features = [col for col in df.columns if col.startswith("EPAPropertyType_") or col.startswith("LargestPropertyUseType_")]

all_features = numerical_features + categorical_features

X = df[all_features]
y = df[target]

print("Feature matrix shape:", X.shape)
# %%
# ## 3. Scale numeric features

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

# %%
# ## 4. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)
# %%
# ## Ready for BNN Modeling with PyMC

# This is the end of the data preparation stage.
# In the next notebook/file, we will define and train a BNN using PyMC.
# %%
# ## 7. Save Train/Test Splits to CSV for BNN model input

X_train.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_train.csv", index=False)
X_test.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/X_test.csv", index=False)
y_train.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_train.csv", index=False)
y_test.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/y_test.csv", index=False)

print("Saved X_train, X_test, y_train, y_test to /seattle/")
# %%