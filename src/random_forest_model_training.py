# phase_b_model_training.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# File paths
# -------------------------------
TRAIN_PATH = "../data-preprocessing/data/processed/pima_train.csv"
TEST_PATH = "../data-preprocessing/data/processed/pima_test.csv"
MODEL_SAVE_PATH = "../models/best_model_rf.pkl"
RISK_TRAIN_PATH = "../data-preprocessing/data/processed/risk_scoring_train.csv"
RISK_TEST_PATH = "../data-preprocessing/data/processed/risk_scoring_test.csv"

# -------------------------------
# Step 1: Load data
# -------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

risk_features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

X_train = train_df[risk_features]
y_train = train_df['Outcome']

# -------------------------------
# Step 2: Train Random Forest
# -------------------------------
rf_risk = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf_risk.fit(X_train, y_train)

# -------------------------------
# Step 3: Save model
# -------------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(rf_risk, MODEL_SAVE_PATH)
print(f"✅ Risk scoring model saved to {MODEL_SAVE_PATH}")

# -------------------------------
# Step 4: Save train & test datasets
# -------------------------------
train_risk = train_df[risk_features + ['Outcome']]
test_risk = test_df[risk_features + ['Outcome']]

os.makedirs(os.path.dirname(RISK_TRAIN_PATH), exist_ok=True)
train_risk.to_csv(RISK_TRAIN_PATH, index=False)
test_risk.to_csv(RISK_TEST_PATH, index=False)

print(f"✅ Risk scoring datasets saved to:\n  {RISK_TRAIN_PATH}\n  {RISK_TEST_PATH}")
