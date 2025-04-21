import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# File and save paths
data_path = '/Users/georgepaul/Desktop/Research-Project/Data-CBECS/Data_Ready_to_Model.csv'
results_path = '/Users/georgepaul/Desktop/Research-Project/Data-CBECS/baseline-models/results'
os.makedirs(results_path, exist_ok=True)

# Feature selection
features = [
    'EUI_kWh_per_sqmt', 'SQMT', 'NFLOOR', 'FLCEILHT',
    'MONUSE', 'OCCUPYP', 'WKHRS', 'NWKER',
    'HEATP', 'COOLP', 'DAYLTP', 'HDD65', 'CDD65',
    'YRCONC', 'PUBCLIM'
]

# Load and prepare data
df = pd.read_csv(data_path)
df_model = df[features].dropna()

# Scale numeric features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df_model.drop(columns=['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM'])),
    columns=[col for col in df_model.columns if col not in ['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']]
)

# Merge scaled with categorical
df_ready = pd.concat([X_scaled, df_model[['EUI_kWh_per_sqmt', 'YRCONC', 'PUBCLIM']].reset_index(drop=True)], axis=1)
df_ready['YRCONC'] = df_ready['YRCONC'].astype("category")
df_ready['PUBCLIM'] = df_ready['PUBCLIM'].astype("category")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df_ready, columns=["YRCONC", "PUBCLIM"], drop_first=True)

# Split features and target
X = df_encoded.drop(columns=["EUI_kWh_per_sqmt"])
y = df_encoded["EUI_kWh_per_sqmt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = ((y_test - preds) ** 2).mean() ** 0.5
    results.append({"Model": name, "R2 Score": round(r2, 3), "RMSE": round(rmse, 2)})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_path, "baseline_summary.csv"), index=False)

# Plot R²
fig_r2 = go.Figure([
    go.Bar(x=results_df['Model'], y=results_df['R2 Score'], marker_color='mediumseagreen')
])
fig_r2.update_layout(title="R² Score Comparison", xaxis_title="Model", yaxis_title="R² Score", template="plotly_white")
fig_r2.write_image(os.path.join(results_path, "r2_comparison.png"))

# Plot RMSE
fig_rmse = go.Figure([
    go.Bar(x=results_df['Model'], y=results_df['RMSE'], marker_color='coral')
])
fig_rmse.update_layout(title="RMSE Comparison", xaxis_title="Model", yaxis_title="RMSE", template="plotly_white")
fig_rmse.write_image(os.path.join(results_path, "rmse_comparison.png"))

print("Baseline models trained, evaluated, and results saved.")