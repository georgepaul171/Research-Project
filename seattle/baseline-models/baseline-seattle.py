import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import os
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns



# Load and clean
numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "SteamUse(kBtu)", "GHGEmissionsIntensity", "SiteEUI(kBtu/sf)"
]

data_path = "/Users/georgepaul/Desktop/Research-Project/seattle/data/seattle-data-cleaned.csv"
results_path = "/Users/georgepaul/Desktop/Research-Project/seattle/baseline-models/results"
os.makedirs(results_path, exist_ok=True)

df = pd.read_csv(data_path)[numeric_columns]
df_clean = df.apply(pd.to_numeric, errors="coerce").dropna()

X = df_clean.drop(columns=["SiteEUI(kBtu/sf)"])
y = df_clean["SiteEUI(kBtu/sf)"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

# Evaluation
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    rmse = ((y_test - preds) ** 2).mean() ** 0.5  # Manual RMSE
    results.append({
        "Model": name,
        "R2 Score": round(r2, 3),
        "RMSE": round(rmse, 2)
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_path, "baseline_summary.csv"), index=False)
print(results_df)


# R² Score Bar Plot
fig_r2 = go.Figure(data=[
    go.Bar(name='R² Score', x=results_df['Model'], y=results_df['R2 Score'], marker_color='mediumseagreen')
])
fig_r2.update_layout(
    title="R² Score Comparison",
    xaxis_title="Model",
    yaxis_title="R² Score",
    template="plotly_white"
)
fig_r2.write_image(f"{results_path}/r2_comparison.png")

# RMSE Bar Plot
fig_rmse = go.Figure(data=[
    go.Bar(name='RMSE', x=results_df['Model'], y=results_df['RMSE'], marker_color='coral')
])
fig_rmse.update_layout(
    title="RMSE Comparison",
    xaxis_title="Model",
    yaxis_title="RMSE",
    template="plotly_white"
)
fig_rmse.write_image(f"{results_path}/rmse_comparison.png")

# Random Forest Feature Importances
rf_model = models["Random Forest"]
rf_importances = rf_model.feature_importances_
rf_series = pd.Series(rf_importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=rf_series.values[:10], y=rf_series.index[:10], palette="crest")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{results_path}/rf_feature_importance.png")
plt.close()

# XGBoost Feature Importances
xgb_model = models["XGBoost"]
xgb_importances = xgb_model.feature_importances_
xgb_series = pd.Series(xgb_importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_series.values[:10], y=xgb_series.index[:10], palette="flare")
plt.title("Top 10 Feature Importances - XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{results_path}/xgb_feature_importance.png")
plt.close()