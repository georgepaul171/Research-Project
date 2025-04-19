# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %% [markdown]
# ## Load, Clean, Filter Identifiers, and Plot Using Selected Base Features (with Plotly)

# %%
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import time

# Path to your CSV
target_path = '/Users/georgepaul/Desktop/Research-Project/LL87/LL87_data_through_2018.csv'

# 1) Load dataset
df = pd.read_csv(target_path, encoding='latin-1')
print(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} cols")

# %%
# 2) Basic cleaning
thresh = len(df) * 0.5
df = df.dropna(axis=1, thresh=thresh)
df = df.drop(columns=[c for c in df.columns if df[c].nunique() <= 1])

def smart_coerce(col):
    s = df[col].astype(str).str.replace(',', '').str.strip()
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().sum() / len(df) > 0.8:
        df[col] = num
for c in df.select_dtypes(include=['object']).columns:
    smart_coerce(c)

num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.fillna('Unknown')
print(f"Cleaned data: {df.shape[0]} rows × {df.shape[1]} cols")

# %%
# 3) Define target and predictor set (only base features)
target_col = 'Existing Annual Energy Use_Total Site Energy Use_Total per sq'

base_cols = [
    'Building Characteristics_Gross Floor Area',
    'Exterior Walls_total exposed above grade wall area',
    'Exterior Walls_vertical glazing % of wall',
    'Building Characteristics_# of above grade floors',
    'Building Characteristics_Year of construction/substantial Rehabilitation'
]

# Ensure features exist
predictors = [c for c in base_cols if c in df.columns]

# Extract features and target
y = df[target_col]
X = df[predictors].copy()
print(f"Feature matrix shape: {X.shape[0]} rows × {X.shape[1]} cols")

# %%
# 4) Preprocess and train Random Forest
X_proc = pd.get_dummies(X, drop_first=True)
rf = RandomForestRegressor(
    n_estimators=50,
    max_features='sqrt',
    max_depth=10,
    random_state=0,
    n_jobs=-1
)
start = time.time()
rf.fit(X_proc, y)
print(f"Training took {time.time() - start:.2f} seconds.")

importances = pd.Series(rf.feature_importances_, index=X_proc.columns)
top20 = importances.nlargest(20)

# %%
# 5) Remove outliers from predictors and target (IQR method)
# 5a) Predictor outliers
mask = np.ones(len(X_proc), dtype=bool)
for col in X_proc.select_dtypes(include=[np.number]).columns:
    Q1, Q3 = X_proc[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    mask &= X_proc[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
X_clean = X_proc.loc[mask]
y_clean = y.loc[mask]

# 5b) Target outliers
Q1_y, Q3_y = y_clean.quantile([0.25, 0.75])
IQR_y = Q3_y - Q1_y
lower_y, upper_y = Q1_y - 1.5 * IQR_y, Q3_y + 1.5 * IQR_y
mask_y = y_clean.between(lower_y, upper_y)
X_clean = X_clean.loc[mask_y]
y_clean = y_clean.loc[mask_y]
print(f"After outlier removal: {X_clean.shape[0]} samples remain.")

# %%
# 6) Plot histogram of cleaned EUI with Plotly
fig_hist = px.histogram(
    y_clean,
    nbins=200,
    title='Filtered Total Site Energy Use EUI Distribution',
    labels={'value': 'EUI (per sq ft)'}
)
fig_hist.show()

# %%
# 7) Bar chart of top‑20 feature importances with Plotly
fig_bar = px.bar(
    x=top20.values,
    y=top20.index,
    orientation='h',
    title='Top 20 Feature Importances',
    labels={'x': 'Importance', 'y': 'Feature'}
)
fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_bar.show()

# %%
# 8) Scatter plots (top-5 features)
top5 = top20.index[:5]
for feat in top5:
    fig_scatter = px.scatter(
        x=X_clean[feat],
        y=y_clean,
        trendline='ols',
        title=f'{feat} vs EUI (filtered)',
        labels={'x': feat, 'y': 'EUI'}
    )
    fig_scatter.show()

# %%
