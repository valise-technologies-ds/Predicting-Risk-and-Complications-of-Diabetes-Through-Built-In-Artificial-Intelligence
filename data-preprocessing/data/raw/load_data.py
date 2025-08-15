import kagglehub
import pandas as pd
import shutil
import os

# Get current directory
current_dir = os.getcwd()

# -----------------------------
# Download PIMA dataset
# -----------------------------
pima_path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
src_pima_file = os.path.join(pima_path, "diabetes.csv")
dst_pima_file = os.path.join(current_dir, "diabetes.csv")
shutil.copy(src_pima_file, dst_pima_file)
print(f"PIMA dataset saved to: {dst_pima_file}")

# -----------------------------
# Download Health Indicators dataset
# -----------------------------
health_path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
src_health_file = os.path.join(health_path, "diabetes_binary_health_indicators_BRFSS2015.csv")
dst_health_file = os.path.join(current_dir, "diabetes_health.csv")
shutil.copy(src_health_file, dst_health_file)
print(f"Health dataset saved to: {dst_health_file}")

# -----------------------------
# Test loading them from current directory
# -----------------------------
pima = pd.read_csv("diabetes.csv")
health = pd.read_csv("diabetes_health.csv")

print("PIMA shape:", pima.shape)
print("Health shape:", health.shape)
