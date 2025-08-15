import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------------
# Step 1: Load cleaned PIMA datasets (all features)
# -------------------------------
train_df = pd.read_csv("../data-preprocessing/data/processed/pima_train.csv")
test_df = pd.read_csv("../data-preprocessing/data/processed/pima_test.csv")

# Combine train and test for cross-validation
data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Features = all columns except Outcome
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# -------------------------------
# Step 2: Define models with tuned parameters
# -------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,  # more trees for better generalization
        max_features='sqrt',  # common choice in the paper
        class_weight='balanced',  # handles class imbalance
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_features='sqrt', random_state=42
    ),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(
            n_neighbors=5, metric='manhattan'
        ))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='sigmoid', C=1, probability=True, random_state=42
        ))
    ]),
    "Logistic Regression": LogisticRegression(
        penalty='l2', solver='liblinear', class_weight='balanced', random_state=42
    ),
    "Naive Bayes": GaussianNB()
}

# -------------------------------
# Step 3: Perform 10-Fold Stratified Cross-Validation
# -------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for train_idx, test_idx in kf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)

        acc_scores.append(accuracy_score(y_test_cv, y_pred_cv))
        prec_scores.append(precision_score(y_test_cv, y_pred_cv))
        rec_scores.append(recall_score(y_test_cv, y_pred_cv))
        f1_scores.append(f1_score(y_test_cv, y_pred_cv))

    results.append({
        "Model": name,
        "Accuracy": round(np.mean(acc_scores), 4),
        "Precision": round(np.mean(prec_scores), 4),
        "Recall": round(np.mean(rec_scores), 4),
        "F1 Score": round(np.mean(f1_scores), 4)
    })

# -------------------------------
# Step 4: Display & Save Results
# -------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("\n10-Fold Cross-Validation Performance (Paper-Matched Tuned Settings):")
print(results_df)

# Save results in current directory
results_df.to_csv("./model_comparison_cv_paper.csv", index=False)
print("\n✅ Cross-validation results saved as 'model_comparison_cv_paper.csv' in current directory.")
