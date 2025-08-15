import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# File paths
# -------------------------------
DATA_PATH = "../data-preprocessing/data/raw/diabetes_health.csv"
RESULTS_DIR = "../results"

# -------------------------------
# Step 1: Load data
# -------------------------------
df = pd.read_csv(DATA_PATH)

# Keep only diabetic patients
df = df[df["Diabetes_binary"] == 1]

# -------------------------------
# Step 2: Complication columns from paper
# -------------------------------
complications = {
    "HeartDisease": "HeartDiseaseorAttack",
    "Stroke": "Stroke",
    "KidneyDisease": "KidneyDisease",
    "VisionProblem": "Blind"
}

# Features (all except target & ID cols)
drop_cols = ["Diabetes_binary"]
features = [col for col in df.columns if col not in drop_cols]

# -------------------------------
# Step 3: Model definitions
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

# -------------------------------
# Step 4: Evaluation function
# -------------------------------
def evaluate_model_cv(model, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score)
    }
    scores = {}
    for metric_name, scorer in scoring.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        scores[metric_name] = np.mean(cv_scores)
    return scores

# -------------------------------
# Step 5: Run for each complication
# -------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

for comp_name, comp_col in complications.items():
    print(f"\n🔍 Predicting complication: {comp_name} ({comp_col})")

    # Prepare X, y
    X = df[features].drop(columns=[comp_col])
    y = df[comp_col]

    results = []

    for model_name, model in models.items():
        scores = evaluate_model_cv(model, X, y)
        results.append({
            "Model": model_name,
            "Accuracy": round(scores["accuracy"], 4),
            "Precision": round(scores["precision"], 4),
            "Recall": round(scores["recall"], 4),
            "F1 Score": round(scores["f1"], 4)
        })

        # Print progress after each model
        print(f"✅ Finished {model_name} for {comp_name}")
        results_df = pd.DataFrame(results)
        print(results_df)

        # Save partial results so far
        results_file = os.path.join(RESULTS_DIR, f"{comp_name}_model_performance.csv")
        results_df.to_csv(results_file, index=False)

    print(f"📄 Final results for {comp_name} saved to {results_file}")
