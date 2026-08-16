# 🏡 House Price Prediction Project

A professional machine learning project that predicts residential home prices using the Kaggle Ames, Iowa Housing Dataset. This project transitions exploratory data analysis from loose Jupyter Notebooks into a reusable python modules for reproducability.

## 🛠️ Tech Stack & MLOps Tools
*   **Core Logic:** Python, Pandas, NumPy, Scikit-Learn
*   **Machine Learning:** XGBoost Regression


## 📁 Project Structure
```text
HousePrices/
├── .venv/                  # Local virtual environment, ignored by git
│
├── data/                      # Strictly ignored by Git
│   ├── raw/                   # Immutable original data (train.csv, test.csv, data_description.txt)
│   └── processed/             # Generated predictions (baseline_submission.csv)
│
│
├── models/                    # Strictly ignored by Git (unless small)
│   └── xgboost_baseline_v1.joblib  # Serialized, deployment-ready pipeline
│
├── notebooks/                 # Strictly for exploration and reporting purpose
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_deep_eda_and_statistics.ipynb
│   ├── 03_feature_engineering_sandbox.ipynb
│   ├── 04_model_prototyping.ipynb
│   └── 05_model_evaluation.ipynb
│
├── reports/                   # Final reports and visualizations
│   └── figures/               # Saved plots from EDA and model diagnostics
│
├── src/                       # The heart of your production codebase
│   ├── __init__.py
│   ├── config.yaml            # Feature lists and model hyperparameters
│   │
│   ├── features/              # Modular feature engineering code
│   │   ├── __init__.py        # AmesFeatureEngineer, custom imputers, etc.
│   │   └── preprocessing.py   # Handles missing values & encoding categorical features
│   │
│   └── models/                # Execution scripts
│       ├── __init__.py
│       ├── train_model.py     # Pulls data, builds pipeline, trains, saves .joblib
│       └── predict_model.py   # CLI tool to load .joblib and predict on new CSVs
│
│
├── .gitignore                 # Prevents data, models, and virtual environments from leaking to GitHub
├── PROJECT_LOG.md             # Running diary of architectural decisions
├── README.md                  # Instructions for setting up the environment and running the code
└── requirements.txt           # Exact package versions to reproduce environment
```
## 🚀 How to Run the Project

### Prerequisites

Before starting, make sure you have:

* Python 3.11 or later installed.
* Jupyter Notebook support in your preferred code editor (e.g., the Jupyter extension for VS Code) if you want to run the notebooks interactively.
* Git installed for cloning the repository.

### 1. Clone the Repository and Download the Dataset

```bash
git clone https://github.com/shivdt/house-price-prediction
cd house-price-prediction
```
* Download train.csv and test.csv from the Kaggle House Prices – Advanced Regression Techniques competition page and place them in the data/raw/ directory.

Your directory structure should look like:

```text
house-price-prediction/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── ...
└── requirements.txt
```

### 2. Configure the Environment
Open the terminal inside your Code Editor and execute the setup script to create an isolated environment and install the required data science packages:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\Activate.ps1 # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
To ingest the raw data, execute the modular feature engineering transformations, and train the optimized XGBoost regression model, run the execution file:

```bash
python -m src.models.train_model
```

## 📊 Results & Model Performance
*   **Baseline Model & Production Model (XGBoost):** RMSE: \$18,200 | $R^2$: 0.89
*   *Key Features Driving Evaluation:* Ground Living Area (`GrLivArea`), Overall House Quality (`OverallQual`), and Neighborhood location.
