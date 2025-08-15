import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------ PIMA dataset ------------------
pima = pd.read_csv("../data-preprocessing/data/raw/diabetes.csv")

# Handle missing values
missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in missing_cols:
    pima[col] = pima[col].replace(0, np.nan)
    pima[col] = pima[col].fillna(pima[col].median())

# Features & target
X_pima = pima.drop(columns=['Outcome'])
y_pima = pima['Outcome']

# Train-test split
X_pima_train, X_pima_test, y_pima_train, y_pima_test = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=42, stratify=y_pima
)

# Save
train_df = pd.DataFrame(X_pima_train, columns=X_pima.columns)
train_df["Outcome"] = y_pima_train.values
test_df = pd.DataFrame(X_pima_test, columns=X_pima.columns)
test_df["Outcome"] = y_pima_test.values

train_df.to_csv("pima_train.csv", index=False)
test_df.to_csv("pima_test.csv", index=False)
print("✅ PIMA train/test CSV saved successfully")
"""
# ------------------ Health dataset ------------------
health = pd.read_csv("../data/raw/diabetes_health.csv")
health_features = ['BMI', 'HighBP', 'HighChol', 'Age']

# Handle zeros as missing for specific columns
cols_with_zero_as_missing = ['BMI', 'Age']
for col in cols_with_zero_as_missing:
    health[col] = health[col].replace(0, np.nan)
    health[col] = health[col].fillna(health[col].median())

# Features & target
X_health = health[health_features]
y_health = health["Diabetes_binary"]

# Train-test split
X_health_train, X_health_test, y_health_train, y_health_test = train_test_split(
    X_health, y_health, test_size=0.2, random_state=42, stratify=y_health
)

# Save
train_df = pd.DataFrame(X_health_train, columns=health_features)
train_df["Diabetes_binary"] = y_health_train.values
test_df = pd.DataFrame(X_health_test, columns=health_features)
test_df["Diabetes_binary"] = y_health_test.values

train_df.to_csv("health_train.csv", index=False)
test_df.to_csv("health_test.csv", index=False)
print("✅ Health train/test CSV saved successfully")
"""