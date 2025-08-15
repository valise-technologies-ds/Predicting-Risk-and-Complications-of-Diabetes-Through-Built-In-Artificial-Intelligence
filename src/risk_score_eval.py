# risk_score_eval.py
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

# -------------------------------
# File paths
# -------------------------------
MODEL_PATH = "../models/best_model_rf.pkl"
DATA_PATH = "../data-preprocessing/data/processed/risk_scoring_test.csv"
OUTPUT_CSV = "../results/risk_scores.csv"
HIST_PATH = "../results/risk_score_distribution.png"
ROC_PATH = "../results/roc_curve.png"
CM_PATH = "../results/confusion_matrix.png"

# -------------------------------
# Step 1: Load model & test data
# -------------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
X = df[features]
y_true = df['Outcome']

# -------------------------------
# Step 2: Predict risk scores
# -------------------------------
df['Risk_Score'] = model.predict_proba(X)[:, 1]
threshold = 0.5
df['Risk_Class'] = (df['Risk_Score'] >= threshold).astype(int)

# -------------------------------
# Step 3: Evaluate
# -------------------------------
accuracy = accuracy_score(y_true, df['Risk_Class'])
precision = precision_score(y_true, df['Risk_Class'])
recall = recall_score(y_true, df['Risk_Class'])
f1 = f1_score(y_true, df['Risk_Class'])
auc = roc_auc_score(y_true, df['Risk_Score'])

print("\n📊 Model Evaluation Metrics (Test Set, Threshold = 0.5):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

# -------------------------------
# Step 4: Save risk scores CSV
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Risk scores saved to {OUTPUT_CSV}")

# -------------------------------
# Step 5: Plot Risk Score Distribution
# -------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df['Risk_Score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Risk Score Distribution (Test Set)')
plt.xlabel('Risk Score')
plt.ylabel('Number of Patients')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(HIST_PATH)
plt.close()
print(f"📈 Risk score distribution saved to {HIST_PATH}")

# -------------------------------
# Step 6: Plot ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_true, df['Risk_Score'])
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(ROC_PATH)
plt.close()
print(f"📈 ROC curve saved to {ROC_PATH}")

# -------------------------------
# Step 7: Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, df['Risk_Class'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.savefig(CM_PATH)
plt.close()
print(f"📈 Confusion matrix saved to {CM_PATH}")
