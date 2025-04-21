# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Seattle Building Energy Data: Cleaning and EDA with Plotly

# ## Virtual Environment Setup
# Run the following in your terminal:
# python3 -m venv bpd_env
# source bpd_env/bin/activate
# pip install pandas plotly scikit-learn matplotlib

# %%
# ## 1. Load Libraries and Dataset

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Replace with your actual path
file_path = "/Users/georgepaul/Desktop/Research-Project/data/seattle/seattle-data.csv"

# Load data
df = pd.read_csv(file_path)

print("Initial shape:", df.shape)
df.head()
# %%
# ## 2. Clean the Dataset

drop_cols = [
    "OSEBuildingID", "BuildingName", "TaxParcelIdentificationNumber",
    "Address", "City", "State", "ZipCode", "Neighborhood",
    "ComplianceIssue", "Demolished"
]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "SteamUse(kBtu)", "GHGEmissionsIntensity", "SiteEUI(kBtu/sf)"
]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df = df[df["SiteEUI(kBtu/sf)"].notna()]
# df = df.dropna(subset=numeric_columns)
upper_limit = df["SiteEUI(kBtu/sf)"].quantile(0.99)
df = df[df["SiteEUI(kBtu/sf)"] <= upper_limit]
df = df[(df["PropertyGFATotal"] > 0) & (df["SiteEUI(kBtu/sf)"] > 0)]
df.reset_index(drop=True, inplace=True)

print("Cleaned data shape:", df.shape)
# %%
# ## 3. Interactive EDA with Plotly

# Histogram of Site EUI

fig_hist = px.histogram(
    df,
    x="SiteEUI(kBtu/sf)",
    nbins=50,
    title="Distribution of Site EUI (kBtu/sf)",
    labels={"SiteEUI(kBtu/sf)": "Site EUI (kBtu/sf)"}
)
fig_hist.update_layout(bargap=0.1)
fig_hist.show()
# %%
# ## 4. Pairplot (Scatter Matrix) for Selected Features

selected_features = [
    "SiteEUI(kBtu/sf)",
    "YearBuilt",
    "NumberofFloors",
    "NumberofBuildings",
    "PropertyGFATotal",
    "ENERGYSTARScore",
    "Electricity(kWh)",
    "NaturalGas(kBtu)",
    "SteamUse(kBtu)",
    "GHGEmissionsIntensity"
]

plt.figure(figsize=(12, 10))
sns.pairplot(df[selected_features], corner=True)
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()
# %%
# ## 5. Correlation Heatmap


corr_matrix = df[selected_features].corr().round(2)
z = corr_matrix.values
x = y = corr_matrix.columns.tolist()

fig_corr = ff.create_annotated_heatmap(
    z=z,
    x=x,
    y=y,
    annotation_text=corr_matrix.values.astype(str),
    colorscale="Viridis",
    showscale=True
)
fig_corr.update_layout(title="Correlation Matrix", height=700)
fig_corr.show()
# %%
# ## 6. Feature Engineering (for modeling)

features = [
    "DataYear", "YearBuilt", "NumberofFloors", "NumberofBuildings",
    "PropertyGFATotal", "ENERGYSTARScore",
    "Electricity(kWh)", "NaturalGas(kBtu)", "SteamUse(kBtu)",
    "GHGEmissionsIntensity"
]
categoricals = [
    "EPAPropertyType", "LargestPropertyUseType"
]

df = pd.get_dummies(df, columns=categoricals, drop_first=True)
model_features = features + [col for col in df.columns if col.startswith("EPAPropertyType_") or col.startswith("LargestPropertyUseType_")]

X = df[model_features]
y = df["SiteEUI(kBtu/sf)"]

print("Final input shape:", X.shape)
df.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/seattle-data-cleaned.csv", index=False)
# %%