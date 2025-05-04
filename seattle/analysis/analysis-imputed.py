import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data path (still reading from /data/)
data_path = "/Users/georgepaul/Desktop/Research-Project/seattle/data"
X_train = pd.read_csv(os.path.join(data_path, "X_train_imputed.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train_imputed.csv"))
X_test = pd.read_csv(os.path.join(data_path, "X_test_imputed.csv"))
y_test = pd.read_csv(os.path.join(data_path, "y_test_imputed.csv"))

# Output path for graphs and analysis
output_path = "/Users/georgepaul/Desktop/Research-Project/seattle/analysis/imputed-graphs"
os.makedirs(output_path, exist_ok=True)  # Create folder if it doesn't exist

# Check and fix y_train column name
if y_train.columns[0] != "SiteEUI(kBtu/sf)":
    y_train.columns = ["SiteEUI(kBtu/sf)"]
    print("Renamed y_train column to 'SiteEUI(kBtu/sf)'.")

# Merge features and target
train_full = pd.concat([X_train, y_train], axis=1)

# 1. Descriptive stats
print("=== Descriptive Statistics ===")
print(train_full.describe())

# 2. Correlation heatmap
print("\n=== Generating Correlation Heatmap ===")
corr = train_full.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Train Set)")
plt.tight_layout()
heatmap_path = os.path.join(output_path, "correlation_heatmap.png")
plt.savefig(heatmap_path)
print(f"Saved heatmap to: {heatmap_path}")
plt.show()

# 3. Feature distributions
print("\n=== Generating Feature Distributions ===")
for col in X_train.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_full[col], kde=True, bins=30)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    dist_path = os.path.join(output_path, f"distribution_{col}.png")
    plt.savefig(dist_path)
    print(f"Saved: {dist_path}")
    plt.close()

# 4. Feature vs SiteEUI plots
print("\n=== Generating Feature vs SiteEUI Scatterplots ===")
for col in X_train.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=train_full, x=col, y="SiteEUI(kBtu/sf)")
    plt.title(f"{col} vs SiteEUI")
    plt.tight_layout()
    scatter_path = os.path.join(output_path, f"{col}_vs_SiteEUI.png")
    plt.savefig(scatter_path)
    print(f"Saved: {scatter_path}")
    plt.close()

# 5. Save correlation matrix as CSV
corr_csv_path = os.path.join(output_path, "correlation_matrix.csv")
corr.to_csv(corr_csv_path)
print(f"\nSaved correlation matrix CSV to: {corr_csv_path}")

print("\nAll analysis outputs saved in 'imputed-graphs' folder.")
# %%
# 6. SHAP Value Analysis
print("\n=== Running SHAP Analysis ===")

import shap
import xgboost as xgb
import numpy as np

# Fit a simple XGBoost regressor
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Create SHAP explainer and compute values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Summary plot (feature importance)
summary_path = os.path.join(output_path, "shap_summary_plot.png")
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(summary_path)
print(f"Saved SHAP summary plot to: {summary_path}")
plt.close()

# SHAP bar plot (mean absolute SHAP values per feature)
bar_path = os.path.join(output_path, "shap_bar_plot.png")
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(bar_path)
print(f"Saved SHAP bar plot to: {bar_path}")
plt.close()

# SHAP dependence plots for top features
print("\n=== Generating SHAP Dependence Plots ===")
top_features = np.argsort(np.abs(shap_values.values).mean(0))[-5:][::-1]
for i in top_features:
    feature_name = X_train.columns[i]
    plt.figure()
    shap.plots.scatter(shap_values[:, i], color=shap_values, show=False)
    plt.title(f"SHAP Dependence: {feature_name}")
    dep_path = os.path.join(output_path, f"shap_dependence_{feature_name}.png")
    plt.tight_layout()
    plt.savefig(dep_path)
    print(f"Saved: {dep_path}")
    plt.close()