# ---
# jupytext:
#   formats: ipynb,py:percent
#   main_language: python
# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %% [markdown]
# Monte Carlo Dropout BNN for High‑EUI Classification (All‑in‑One Notebook)
# This single notebook loads the LL87 audit data, cleans and preprocesses it, and trains a Bayesian Neural Network via Monte Carlo Dropout to classify buildings as high‑EUI vs low‑EUI.

# %%
# 1) Imports and Setup
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# %%
# 2) Load and Clean Data
file_path = '/Users/georgepaul/Desktop/Research-Project/LL87/LL87_data_through_2018.csv'
# Read CSV with latin-1 encoding to avoid decode errors
df = pd.read_csv(file_path, encoding='latin-1')

# a) Drop columns with >50% missing
df = df.dropna(axis=1, thresh=len(df) * 0.5)
# b) Drop constant columns
df = df.loc[:, df.nunique() > 1]
# c) Convert mostly-numeric strings to numeric
def smart_coerce(col):
    s = df[col].astype(str).str.replace(',', '').str.strip()
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().sum() / len(df) > 0.8:
        df[col] = num
for col in df.select_dtypes(include=['object']):
    smart_coerce(col)
# d) Impute numeric NaNs with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
# e) Fill remaining NaNs with 'Unknown'
df = df.fillna('Unknown')
print(f"Cleaned data: {df.shape[0]} rows × {df.shape[1]} cols")

# %%
# 3) Define Predictors and Target
# Drop identifier-like columns if present
ident_cols = [c for c in df.columns if c=='BBL' or 'BIN' in c or 'Address' in c]
df = df.drop(columns=ident_cols, errors='ignore')
# Base numeric features (physical)
base_cols = [
    'Building Characteristics_Total conditioned Area',
    'Building Characteristics_Conditioned Area, heated & cooled',
    'Building Characteristics_Conditioned Area Heated only',
    'Exterior Walls_total exposed above grade wall area',
    'Exterior Walls_vertical glazing % of wall',
    'Building Characteristics_# of above grade floors',
    'Building Characteristics_Year of construction/substantial Rehabilitation'
]
# Space type categorical flags
space_cols = [c for c in df.columns if c.startswith('Space Types_')]
# Combine predictors
predictor_cols = base_cols + space_cols
X_raw = df[predictor_cols]
# Binary classification target: high EUI if above median
target_col = 'Existing Annual Energy Use_Total Site Energy Use_Total per sq'
median_val = df[target_col].median()
y = (df[target_col] > median_val).astype(int)
print(f"High‑EUI threshold (median): {median_val:.2f}")

# %%
# 4) Train/Test Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=0, stratify=y
)
print(f"Raw Train: {X_train_raw.shape}, Raw Test: {X_test_raw.shape}")

# %%
# 5) Scale Numeric Features
num_features = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train_raw[num_features]),
    columns=num_features,
    index=X_train_raw.index
)
X_test_num = pd.DataFrame(
    scaler.transform(X_test_raw[num_features]),
    columns=num_features,
    index=X_test_raw.index
)

# %%
# 6) Encode Categorical Features (Space Types)
cat_features = space_cols
# Cast to str to ensure uniform type
X_train_raw[cat_features] = X_train_raw[cat_features].astype(str)
X_test_raw[cat_features] = X_test_raw[cat_features].astype(str)
# One-hot encode, ignore unknown categories in test
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_train_cat = pd.DataFrame(
    ohe.fit_transform(X_train_raw[cat_features]),
    columns=ohe.get_feature_names_out(cat_features),
    index=X_train_raw.index
)
X_test_cat = pd.DataFrame(
    ohe.transform(X_test_raw[cat_features]),
    columns=ohe.get_feature_names_out(cat_features),
    index=X_test_raw.index
)

# %%
# 7) Combine Processed Features
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
print(f"Processed Train: {X_train.shape}, Processed Test: {X_test.shape}")

# %%
# 8) Build MC Dropout Model
def build_model(input_dim, dropout_rate=0.3):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(X_train.shape[1])
model.summary()

# %%
# 9) Train with Early Stopping
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[es],
    verbose=2
)
print(f"Training completed in {time.time() - start_time:.2f}s")

# %%
# 10) Evaluate on Test Set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}")

# %%
# 11) Monte Carlo Predictions for Uncertainty
def mc_predict(model, X, n_samples=50):
    preds = np.stack([
        model(X, training=True).numpy().flatten() for _ in range(n_samples)
    ], axis=0)
    return preds, preds.mean(axis=0), preds.std(axis=0)

mc_preds, mc_mean, mc_std = mc_predict(model, X_test.values, n_samples=50)
for i in range(5):
    print(f"Sample {i}: mean={mc_mean[i]:.3f}, std={mc_std[i]:.3f}, true={y_test.iloc[i]}")

# %%
