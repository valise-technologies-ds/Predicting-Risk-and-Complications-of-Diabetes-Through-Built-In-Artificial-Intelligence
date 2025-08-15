diabetes_ai_project/
│
├── data/
│   ├── raw/
│   │   ├── diabetes.csv                   # Original PIMA dataset
│   │   ├── diabetes_health.csv            # Original Health Indicators dataset
│   │
│   ├── processed/
│   │   ├── pima_train.csv                  # Cleaned & split PIMA train data
│   │   ├── pima_test.csv                   # Cleaned & split PIMA test data
│   │   ├── health_train.csv                # Cleaned & split Health Indicators train data
│   │   ├── health_test.csv                 # Cleaned & split Health Indicators test data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb              # For EDA & cleaning both datasets
│   ├── 02_model_training.ipynb             # For Step 3 & Step 4 (ML training & evaluation)
│   ├── 03_risk_scoring.ipynb                # For Phase B risk scoring & complication prediction
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py               # Cleaning functions for PIMA & Health datasets
│   ├── model_training.py                   # Functions to train and evaluate ML models
│   ├── risk_scoring.py                      # Functions to calculate risk levels & complications
│   ├── utils.py                            # Helper functions (e.g., metrics, plots)
│
├── models/
│   ├── diabetes_rf_model.pkl               # Saved best ML model for diabetes prediction
│   ├── complication_ann_models/            # Folder for saved ANN models for each complication
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison.png            # Accuracy comparison chart
│   │   ├── risk_distribution.png           # Risk score distribution
│   ├── final_report.pdf                    # Project summary report
│
├── requirements.txt                         # All Python dependencies
├── README.md                                # Project overview & instructions
└── main.py                                  # Entry point script to run full pipeline
