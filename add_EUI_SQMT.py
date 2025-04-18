# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # CBECS Office Buildings Energy Use Cleaning
# This script loads, filters, cleans, and calculates EUI values for office buildings in the CBECS dataset.

# ## Step 1: Load Data
# %% 
import pandas as pd

# Load the original dataset
df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data/dataCBECS.csv')

# Show all column names
print(df.columns.tolist())
# %% 
# ## Step 2: Filter Office Buildings

# Filter to only office buildings (PBA code 2)
df = df[df['PBA'] == 2]
# %% 
# ## Step 3: Handle Energy Fields

# Drop rows where all three consumption fields are missing
df = df.dropna(subset=['ELCNS', 'NGCNS', 'FKCNS'], how='all')

# Fill NaN with 0 for calculation
df[['ELCNS', 'NGCNS', 'FKCNS']] = df[['ELCNS', 'NGCNS', 'FKCNS']].fillna(0)
# %% 
# ## Step 4: Convert Energy to kWh

df['Electricity_kWh'] = df['ELCNS']
df['Gas_kWh'] = df['NGCNS'] * 29.3
df['FuelOil_kWh'] = df['FKCNS'] * 40.7

# Total energy consumption
# HIIIIIIII
df['Total_Energy_kWh'] = df['Electricity_kWh'] + df['Gas_kWh'] + df['FuelOil_kWh']
# %% 
# ## Step 5: Area Cleaning and EUI Calculation (Version 1)

# Replace zero SQFT with NA to avoid divide-by-zero
df['SQFT'] = df['SQFT'].replace(0, pd.NA)

# Add SQMT column
df['SQMT'] = df['SQFT'] * 0.092903

# EUI calculations
df['EUI_kWh_per_sqft'] = df['Total_Energy_kWh'] / df['SQFT']
df['EUI_kWh_per_sqmt'] = df['Total_Energy_kWh'] / df['SQMT']

# Save version 1
df.to_csv('/Users/georgepaul/Desktop/Research-Project/Data/data_Offices_Clean.csv', index=False)

# Output shape
print("Dataset shape (rows, columns):", df.shape)
# %% 
# ## Step 6: Alternative Calculation Using MFBTU (Version 2)

df = pd.read_csv('/Users/georgepaul/Desktop/Research-Project/Data/dataCBECS.csv')

# Filter to only office buildings
df = df[df['PBA'] == 2]

# Drop if MFBTU is missing or zero
df = df[df['MFBTU'].notna() & (df['MFBTU'] > 0)]

# Clean floor area
df['SQFT'] = df['SQFT'].replace(0, pd.NA)
df['SQMT'] = df['SQFT'] * 0.092903

# Convert MFBTU to kWh (1 kBtu = 0.293071 kWh)
df['Total_Energy_kWh'] = df['MFBTU'] * 0.293071

# EUI calculations
df['EUI_kWh_per_sqft'] = df['Total_Energy_kWh'] / df['SQFT']
df['EUI_kWh_per_sqmt'] = df['Total_Energy_kWh'] / df['SQMT']

# Save version 2
df.to_csv('/Users/georgepaul/Desktop/Research-Project/Data/Usable_data_Offices_Clean.csv', index=False)

# Output shape
print("Usable data filtered for offices and cleaning has shape (rows, columns):", df.shape)
# %%
