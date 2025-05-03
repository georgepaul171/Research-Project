import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import joblib

# Load cleaned full dataset
df = pd.read_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/seattle-data-cleaned.csv")

# Define numeric columns
all_numeric_columns = [
    "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal",
    "ENERGYSTARScore", "Electricity(kWh)", "NaturalGas(kBtu)",
    "SteamUse(kBtu)", "GHGEmissionsIntensity", "SiteEUI(kBtu/sf)"
]

target_col = "SiteEUI(kBtu/sf)"

# Subset to numeric columns
df_numeric = df[all_numeric_columns]

# Drop columns with >50% missing
missing_ratios = df_numeric.isna().mean()
cols_to_keep = missing_ratios[missing_ratios <= 0.5].index.tolist()
df_numeric = df_numeric[cols_to_keep]

# Separate features and target
if target_col not in df_numeric.columns:
    raise ValueError(f"Target column '{target_col}' was dropped. Cannot proceed.")
    
X = df_numeric.drop(columns=[target_col])
y = df_numeric[target_col]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit KNN imputer on training data
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Drop any rows with NaNs (rare but safe check)
train_mask = ~X_train_imputed.isna().any(axis=1) & ~y_train.isna()
X_train_imputed = X_train_imputed[train_mask]
y_train = y_train[train_mask]

test_mask = ~X_test_imputed.isna().any(axis=1) & ~y_test.isna()
X_test_imputed = X_test_imputed[test_mask]
y_test = y_test[test_mask]

# Save all files
X_train_imputed.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_train_imputed.csv", index=False)
y_train.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_train_imputed.csv", index=False)
X_test_imputed.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/X_test_imputed.csv", index=False)
y_test.to_csv("/Users/georgepaul/Desktop/Research-Project/seattle/data/y_test_imputed.csv", index=False)
joblib.dump(imputer, "/Users/georgepaul/Desktop/Research-Project/seattle/data/knn_imputer.joblib")

# Summary
print("Split + KNN Imputation complete.")
print(f"Columns kept: {list(cols_to_keep)}")
print(f"X_train: {X_train_imputed.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test_imputed.shape}, y_test : {y_test.shape}")