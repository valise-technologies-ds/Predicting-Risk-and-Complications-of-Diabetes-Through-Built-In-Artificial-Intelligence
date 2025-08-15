import pandas as pd
import joblib
import os

# -------------------------------
# File paths
# -------------------------------
MODEL_PATH = "../models/best_model_rf.pkl"
DATA_PATH = "../data-preprocessing/data/processed/risk_scoring_dataset.csv"
OUTPUT_PATH = "../results/risk_scores.csv"

# -------------------------------
# Step 1: Load trained model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
best_rf = joblib.load(MODEL_PATH)
print("✅ Loaded trained Random Forest model.")

# -------------------------------
# Step 2: Load risk scoring dataset (4 features)
# -------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Risk scoring dataset not found: {DATA_PATH}")
risk_df = pd.read_csv(DATA_PATH)

features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
X_risk = risk_df[features]
y_true = risk_df['Outcome']

# -------------------------------
# Step 3: Predict risk scores
# -------------------------------
risk_scores = best_rf.predict_proba(X_risk)[:, 1]  # Probability of Outcome = 1
risk_df['Risk_Score'] = risk_scores

# -------------------------------
# Step 4: Classify based on threshold
# -------------------------------
threshold = 0.5
risk_df['Risk_Class'] = (risk_df['Risk_Score'] >= threshold).astype(int)

# -------------------------------
# Step 5: Save results
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
risk_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Risk scores saved to {OUTPUT_PATH}")

# Optional: Summary
print("\n📊 Risk Score Summary:")
print(risk_df[['Risk_Score', 'Risk_Class']].describe())
