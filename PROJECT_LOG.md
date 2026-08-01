# Project Log

**Project:** House Prices – Advanced Regression Techniques (Kaggle)
**Author:** Shiv Shankar Dubey
**Type:** End-to-End Machine Learning Regression Project
**Status:** Phase 5 – Data Preparation (In Progress)

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Business Understanding | ✅ Complete |
| 2 | Project Setup | ✅ Complete |
| 3 | Data Understanding | ✅ Complete |
| 4 | Exploratory Data Analysis | ✅ Complete |
| 5 | Data Preparation | ⏳ In Progress |
| 6 | Baseline Model | ⬜ Pending |
| 7 | Model Improvement | ⬜ Pending |
| 8 | Model Evaluation | ⬜ Pending |
| 9 | Model Explainability | ⬜ Pending |
| 10 | Final Report | ⬜ Pending |

---

## Phase 1 – Business Understanding

### Objective
Develop a machine learning model to predict the sale price of residential houses in Ames, Iowa using 79 explanatory variables.

### Business Context
The model can assist real estate agents, property investors, homeowners, and mortgage providers by estimating a property's market value to support pricing and investment decisions.

### Key Learnings
- ML models support decision-making; they do not replace human judgment.
- Model errors have real consequences: overestimation leads to overpriced, slow-selling properties; underestimation causes financial losses for sellers.
- Human review remains important, particularly for unusual or unseen cases.

---

## Phase 2 – Project Setup

### Objective
Establish a professional, reproducible, and maintainable project structure before beginning analysis.

### Completed Tasks
- Created a professional project directory structure.
- Initialized a Git repository with a configured `.gitignore` (excluding `.venv`, caches, and generated files).
- Created and activated a Python virtual environment (`.venv`).
- Installed initial dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`.
- Generated `requirements.txt` via `pip freeze`.
- Downloaded the Kaggle dataset and placed original files in `data/raw/`.
- Made initial Git commits documenting the setup.

### Engineering Decisions

| Decision | Rationale |
|---|---|
| Virtual environment | Isolates project dependencies and avoids conflicts with other projects. |
| `.gitignore` | Prevents tracking of generated files, caches, and the virtual environment. |
| `requirements.txt` | Allows anyone cloning the repository to recreate the same environment. |
| Separate `raw/` and `processed/` directories | Preserves the original dataset; all cleaning produces new files in `data/processed/`. |

---

## Phase 3 – Data Understanding

### Objective
Develop a thorough understanding of the dataset before analysis, preprocessing, or model building.

### Completed Tasks
- Reviewed the competition objective and evaluation metric (RMSE on log-transformed `SalePrice`).
- Loaded `train.csv` and `test.csv` into pandas and verified dimensions:
  - Training set: **1,460 rows × 81 columns**
  - Test set: **1,459 rows × 80 columns**
- Confirmed each row represents one residential property; `SalePrice` is the target; `Id` is an identifier.
- Studied `data_description.txt` to understand feature semantics before writing any code.
- Investigated missing values conceptually, identifying that many represent the absence of a property feature (e.g., no pool, no garage) rather than unknown information.
- Began semantic feature classification: Identifier, Target, Numerical, Nominal, Ordinal, Temporal.

### Key Learnings
- Data understanding must precede cleaning and modeling.
- The semantic meaning of a feature is often more important than its storage data type.
- Numeric-looking variables (e.g., `MSSubClass`) may represent categorical information.
- Ordinal features require different preprocessing strategies than nominal ones.
- Missing values should always be interpreted before deciding how to handle them.

---

## Phase 4 – Exploratory Data Analysis

### Objective
Explore the dataset systematically to understand target variable distribution, feature patterns, and relationships — and generate hypotheses for preprocessing and modeling.

### Completed Tasks
- Analyzed the distribution of `SalePrice`: histogram, box plot, summary statistics (median, IQR).
- Generated a dataset-wide feature summary covering cardinality, missing values, and correlation with the target.
- Performed univariate analysis of `OverallQual` (frequency distribution, summary statistics).
- Conducted bivariate analyses between `SalePrice` and `OverallQual`, `TotalBsmtSF`, and `GarageArea` using scatter plots, regression plots, box plots, and violin plots.
- Extended analyses to multivariate exploration by encoding `OverallQual` as a color dimension.
- Flagged outlier candidates for investigation during preprocessing rather than immediate removal.

### Key Findings

**Target variable:** `SalePrice` is positively skewed; the median is a more representative measure of central tendency than the mean. A log transformation is likely to improve model performance.

**Strong predictive features:**
- `OverallQual` — strong positive relationship with sale price.
- `TotalBsmtSF` — larger basements associated with higher prices, especially in high-quality homes.
- `GarageArea` — positive relationship, though garage area alone is a weak quality predictor.

**Missing values:** Require semantic interpretation; many represent absent features rather than data gaps.

**Outliers:** Several observations show unusually large basement or garage areas relative to their price. Treatment deferred to preprocessing.

### Key Learnings
- EDA is question-driven, not feature-driven — ask a clear question before each analysis step.
- Visualization reveals relationships that correlation coefficients alone cannot capture.
- EDA generates hypotheses; it does not make irreversible preprocessing decisions.

---

## Phase 5 – Data Preparation

### Objective
Prepare the dataset for machine learning with a reproducible, leakage-free preprocessing pipeline that preserves semantic meaning.

### Completed Tasks

**Train/Validation Split**
- Split the original training data into training and validation sets before any learned preprocessing.
- Established a leakage-free workflow where all `fit()` operations are applied exclusively to the training data.

**Missing Value Treatment**
- Distinguished between structural missing values (absent property features) and genuinely unknown values.
- Replaced structural missing values using domain knowledge:
  - Categorical structural features → `"None"`
  - Numerical structural features → `0`
- Investigated ambiguous cases (`LotFrontage`, `Electrical`, `GarageYrBlt`, `MasVnrType`) individually before selecting imputation strategies.
- Implemented `SimpleImputer` for simple statistical imputation (`Electrical`).
- Designed custom scikit-learn-compatible transformers:
  - `NeighborhoodLotFrontageImputer` — imputes `LotFrontage` using neighborhood medians.
  - `Exterior1stMasVnrTypeImputer` — resolves `MasVnrType` using domain knowledge and rule-based logic.
- Implemented a custom `MedianImputer` to reinforce understanding of the `fit()` / `transform()` estimator pattern.

**Feature Engineering**
- Designed an engineering plan before implementation, ensuring each new feature encodes a meaningful domain concept.

| Feature | Description |
|---|---|
| `HouseAge` | Age of the house at time of sale |
| `GarageAge` | Age of the garage at time of sale |
| `YearsSinceRemodel` | Years elapsed since last remodel |
| `HasGarage` | Binary indicator for garage presence |
| `HasBsmt` | Binary indicator for basement presence |
| `TotalBathrooms` | Combined bathroom count (full + half-weighted) |
| `TotalUsableSF` | Total usable square footage across floors |
| `BsmtFinishedRatio` | Ratio of finished basement area to total basement area |
| `OverallQual_x_TotalUsableSF` | Interaction term: quality × usable area |

- Classified all features into a semantic inventory: Ordinal, Nominal, Continuous Numerical, Discrete Numerical, Binary, and Candidates for Future Removal.
- Deferred irreversible feature removal until after benchmarking, following an evidence-driven workflow.

**Encoding, Transformation & Scaling Policy**
- Finalised a production-oriented categorical encoding policy:
  - Explicit ordinal mappings based on domain semantics.
  - One-Hot Encoding for nominal features.
  - Explicit binary mappings for binary categorical features.
  - Defined a strategy for handling unseen categories at inference time.
- Evaluated target transformation options (`log`, `log1p`, Box–Cox, Yeo–Johnson) and selected a log-based transformation consistent with the competition's RMSE-on-log objective.
- Designed a model-dependent scaling policy, distinguishing preprocessing requirements for linear, distance-based, and tree-based algorithms.

**Outlier Strategy**
- Established an evidence-driven outlier policy separating three categories: data recording errors, influential observations, and legitimate rare observations.
- Deferred outlier removal decisions until empirical model benchmarking.

**Preprocessing Architecture Design**
- Designed the overall pipeline architecture using custom transformers, `ColumnTransformer`, and `Pipeline` to ensure reproducibility and eliminate data leakage end-to-end.

### Remaining Tasks
- Implement the preprocessing architecture (custom transformers, `ColumnTransformer`, `Pipeline`).
- Train baseline models.
- Evaluate preprocessing choices empirically.

### Key Learnings
- Preprocessing decisions must be driven by feature semantics, not data types.
- Leakage prevention begins at the train/validation split — every learned transformation is fitted on training data only.
- Custom transformers integrating into scikit-learn require `BaseEstimator` and `TransformerMixin`.
- Feature engineering should encode domain concepts, not arbitrary mathematical combinations.
- New engineered features should not replace original features without empirical validation.
- Scaling requirements depend on the learning algorithm, not the feature.
- Encoding strategies must account for inference-time unseen categories, not just training data.
- Target transformation choices should be aligned to the evaluation metric, not applied by default.
- Outlier treatment requires intent classification before any removal decision is made.
- A robust preprocessing workflow prioritises reproducibility, maintainability, and deployment readiness.

---

## Overall Reflection

This project is developed with an emphasis on professional data science practices over optimising a Kaggle score. Each phase reinforces the full end-to-end ML workflow: business framing, reproducibility, version control, systematic documentation, and evidence-driven decision-making.

The guiding principle throughout has been to understand before acting — reading documentation before writing code, interpreting missing values before imputing them, designing feature engineering before implementing it, and deferring irreversible decisions until empirical evidence supports them.
