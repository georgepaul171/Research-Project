# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Filtered Data Analysis
# This notebook-style script performs exploratory data analysis and preprocessing for modeling Energy Use Intensity (EUI) using the CBECS dataset.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the dataset
df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data/Usable_data_Offices_Clean.csv')
print(df.columns.tolist())

# %%
# Basic info
print("Basic Info:")
print(df.info())

# %%
# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# %%
# Missing values
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# %%
# Correlation with EUI (numerical only)
eui_corr = df.corr(numeric_only=True)['EUI_kWh_per_sqmt'].sort_values(ascending=False)
print("\nTop correlations with EUI (kWh/m²):")
print(eui_corr.head(10))
print("\nBottom correlations with EUI (kWh/m²):")
print(eui_corr.tail(10))

# %% [markdown]
# ## Key Visualizations

# %%
# Histogram of EUI
plt.figure(figsize=(8, 4))
sns.histplot(df['EUI_kWh_per_sqmt'], bins=50, kde=True)
plt.title('Distribution of Energy Use Intensity (kWh/m²)')
plt.xlabel('EUI (kWh/m²)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
# Scatter: EUI vs. SQFT
plt.figure(figsize=(6, 4))
sns.scatterplot(x='SQFT', y='EUI_kWh_per_sqmt', data=df)
plt.title('EUI vs. Gross Floor Area (ft²)')
plt.xlabel('Square Footage')
plt.ylabel('EUI (kWh/m²)')
plt.tight_layout()
plt.show()

# %%
# Boxplot: EUI by year of construction
plt.figure(figsize=(10, 5))
sns.boxplot(x='YRCONC', y='EUI_kWh_per_sqmt', data=df)
plt.title('EUI by Year of Construction Category')
plt.xlabel('Construction Year Category')
plt.ylabel('EUI (kWh/m²)')
plt.tight_layout()
plt.show()

# %%
# Boxplot: EUI by main heating type
plt.figure(figsize=(12, 5))
sns.boxplot(x='MAINHT', y='EUI_kWh_per_sqmt', data=df)
plt.title('EUI by Main Heating Type')
plt.xlabel('Main Heating Type')
plt.ylabel('EUI (kWh/m²)')
plt.tight_layout()
plt.show()

# %%
# Pairplot of key continuous variables
key_vars = ['EUI_kWh_per_sqmt', 'SQFT', 'NFLOOR', 'FLCEILHT', 'WKHRS', 'NWKER', 'HDD65', 'CDD65']
sns.pairplot(df[key_vars].dropna())
plt.suptitle("Pairwise Relationships (Numerical Variables)", y=1.02)
plt.show()

# %% [markdown]
# ## Data Cleaning and Feature Selection

# %%
# Step 1: Drop columns that are entirely missing
df_cleaned = df.dropna(axis=1, how='all')

# %%
# Step 2: Select relevant features
selected_columns = [
    'EUI_kWh_per_sqmt',     # Target variable
    'SQMT',                 # Gross floor area (m²)
    'NFLOOR',               # Number of floors
    'FLCEILHT',             # Floor to ceiling height
    'YRCONC',               # Year of construction category (categorical)
    'MONUSE',               # Months in use
    'OCCUPYP',              # Percent occupancy
    'WKHRS',                # Hours open per week
    'NWKER',                # Number of employees
    'HEATP',                # Percent heated
    'COOLP',                # Percent cooled
    'DAYLTP',               # Percent daylight
    'HDD65',                # Heating degree days
    'CDD65',                # Cooling degree days
    'PUBCLIM'               # Public climate zone (categorical)
]

# %%
# Step 3: Keep only selected columns
df_model = df_cleaned[selected_columns].copy()

# %%
# Step 4: Check and report missing values
missing_summary = df_model.isnull().sum()
print("Missing values per column:\n", missing_summary)

# %%
# Step 5: Drop rows with missing target or any missing features
df_model = df_model.dropna()

# %%
# Step 6: Confirm shape and preview cleaned data
print("\nCleaned dataset shape:", df_model.shape)
print(df_model.head())

# %% [markdown]
# ## Final Export
# We now export the final cleaned and filtered dataset to a CSV file: Data_Ready_to_Model.csv

# %%
# Export filtered dataset
df_model.to_csv('/Users/georgepaul/Desktop/Research-Project/Data/Data_Ready_to_Model.csv', index=False)
# %%
