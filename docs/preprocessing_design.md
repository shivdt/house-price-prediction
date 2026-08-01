# Preprocessing Design Document

**Project:** House Prices – Advanced Regression Techniques (Kaggle)

**Author:** Shiv Shankar Dubey

**Status:** Design Approved (Pre-Implementation)

---

# Purpose

This document records the engineering decisions made during the design of the preprocessing pipeline.

Its purpose is to explain **why** specific preprocessing strategies were chosen before implementation.

This document intentionally separates design rationale from implementation details.

Implementation lives in source code.

Progress lives in `PROJECT_LOG.md`.

---

# Design Principles

The preprocessing system is designed around the following principles:

- Prevent data leakage.
- Preserve feature semantics.
- Prefer reproducible, sklearn-compatible components.
- Keep preprocessing modular and maintainable.
- Delay irreversible decisions until supported by evidence.
- Optimize first for correctness and maintainability, then for Kaggle performance.

---

# Overall Preprocessing Workflow

The planned preprocessing workflow is:

Raw Data

↓

Cleaning

↓

Feature Engineering

↓

ColumnTransformer

↓

Estimator

Target transformation is handled independently using `TransformedTargetRegressor`.

---

# 1. Train / Validation Strategy

## Decision

The original training dataset is split into training and validation sets before any learned preprocessing.

## Rationale

All preprocessing components that learn from data (imputers, encoders, scalers, etc.) are fitted only on the training partition.

This prevents train/validation leakage.

---

# 2. Missing Value Strategy

Missing values are treated according to their semantic meaning rather than by applying a universal imputation strategy.

## Structural Missing Values

Missing values representing the absence of a property feature are preserved as meaningful information.

Examples:

- No garage
- No basement
- No fireplace
- No pool

### Decision

Categorical structural missing values

→ `"None"`

Numerical structural missing values

→ `0`

---

## Genuine Missing Values

True missing information is handled individually.

### Electrical

Strategy:

- Most Frequent Imputation

---

### LotFrontage

Strategy:

Neighborhood median

Reason:

Lot frontage depends strongly on neighborhood characteristics.

Implemented through a custom sklearn transformer.

---

### MasVnrType

Strategy:

Rule-based imputation using `Exterior1st` where appropriate.

Remaining structural cases become `"None"`.

Implemented through a custom transformer.

---

### GarageYrBlt

No direct imputation.

Instead:

Feature engineering derives `GarageAge` together with `HasGarage`.

---

# 3. Feature Engineering

The project adopts the principle:

> One engineered feature should represent one meaningful domain concept.

Implemented features:

- HouseAge
- GarageAge
- YearsSinceRemodel
- HasGarage
- HasBsmt
- TotalBathrooms
- TotalUsableSF
- BsmtFinishedRatio
- OverallQual × TotalUsableSF

Deferred features:

- YearsUntilFirstRemodel
- PrimaryAmenityScore
- LotAccessibility

Reason:

Insufficient evidence that they improve model quality.

---

# 4. Categorical Encoding

Encoding decisions are based on feature semantics rather than storage datatype.

## Ordinal Features

Strategy:

Explicit manually-defined ordering.

Reason:

Domain semantics must not depend on automatically discovered category ordering.

Mappings are organized by semantic scale rather than duplicated per feature.

Examples:

- Quality scale
- Finish scale
- Exposure scale

---

## Nominal Features

Strategy:

One-Hot Encoding

Reason:

No natural ordering exists.

High-cardinality features may be revisited after baseline evaluation if justified.

---

## Binary Features

Numeric binary variables

→ kept unchanged

String binary variables

→ explicit mapping

---

## Unknown Categories

Nominal variables:

`OneHotEncoder(handle_unknown="ignore")`

Ordinal variables:

Dedicated unknown category defined explicitly where appropriate.

The system should never fail solely because an unseen category appears during inference.

---

# 5. Target Transformation

Target:

SalePrice

Decision:

Log-based transformation.

Reasons:

- Reduce right skew.
- Reduce heteroscedasticity.
- Match Kaggle evaluation metric.
- Improve numerical behavior for linear models.

Predictions are converted back using the corresponding inverse transformation.

---

# 6. Scaling Strategy

Scaling depends on the downstream estimator.

## Linear / Distance Models

Scaling enabled.

Preferred scaler:

RobustScaler

Reason:

Continuous housing variables contain genuine extreme values.

---

## Tree Models

Scaling disabled.

Reason:

Decision trees depend on ordering rather than feature magnitude.

Scaling adds unnecessary computation.

---

# 7. Outlier Strategy

Outliers are separated into three categories.

## Data Errors

Correct or remove.

---

## Rare but Valid Observations

Retain.

---

## Influential Observations

Investigate individually.

For the Ames dataset:

Training observations with

GrLivArea > 4000

are removed before fitting.

Other legitimate extremes remain.

---

# 8. ColumnTransformer Architecture

The preprocessing system uses ColumnTransformer to isolate preprocessing by feature type.

Branches:

- Numeric
- Ordinal
- Nominal
- Binary

Each branch owns only the transformations relevant to that feature family.

---

# 9. Pipeline Design

The complete preprocessing system is implemented as a sklearn Pipeline.

Objectives:

- eliminate leakage
- guarantee reproducibility
- ensure identical training and inference behavior
- simplify cross-validation
- support estimator swapping

---

# Planned Final Architecture

Raw Data

↓

Cleaning Transformer

↓

Feature Engineering Transformer

↓

ColumnTransformer

├── Numeric Pipeline

├── Ordinal Pipeline

├── Nominal Pipeline

└── Binary Pipeline

↓

Estimator

↓

Predictions

---

# Deferred Decisions

The following topics will be revisited only after establishing a baseline model.

- Target Encoding
- Feature Selection
- Power Transformations
- Winsorization
- Advanced Interaction Features
- Dimensionality Reduction

The guiding principle is:

> Do not introduce additional preprocessing complexity until empirical evidence demonstrates a measurable benefit.

---

# Revision History

| Date | Version | Notes |
|------|---------|------|
| July 2026 | 1.0 | Initial preprocessing design before implementation |



# Preprocessing Design Document

**Project:** House Prices – Advanced Regression Techniques (Kaggle)  
**Author:** Shiv Shankar Dubey  
**Status:** Design Approved (Pre-Implementation)

---

# Purpose

This document records the engineering decisions made during the design of the preprocessing pipeline.

Its purpose is to explain **why** specific preprocessing strategies were chosen before implementation.

This document intentionally separates design rationale from implementation details.

Implementation lives in source code.

Progress lives in `PROJECT_LOG.md`.

---

# Design Principles

The preprocessing system is designed around the following principles:

- Prevent data leakage.
- Preserve feature semantics.
- Prefer reproducible, sklearn-compatible components.
- Keep preprocessing modular and maintainable.
- Delay irreversible decisions until supported by evidence.
- Optimize first for correctness and maintainability, then for Kaggle performance.

---

# Overall Preprocessing Workflow

The planned preprocessing workflow is:

Raw Data  
↓  
Row-Level Outlier Pruning (Training set only: `GrLivArea > 4000` & `SalePrice < 200000`)  
↓  
`TransformedTargetRegressor` ($\log1p$ target scaling wrapper)  
↓  
`ColumnTransformer` (Parallel preprocessing branches)  
↓  
Estimator  

Target transformation and inverse prediction ($\text{expm1}$) are handled transparently via `TransformedTargetRegressor`.

---

# 1. Train / Validation Strategy

## Decision

The original training dataset is split into training and validation sets before any learned preprocessing.

## Rationale

All preprocessing components that learn from data (imputers, encoders, scalers, etc.) are fitted only on the training partition.

This prevents train/validation leakage.

---

# 2. Missing Value Strategy

Missing values are treated according to their semantic meaning rather than by applying a universal imputation strategy.

## Structural Missing Values

Missing values representing the absence of a property feature are preserved as meaningful information.

Examples:
- No garage
- No basement
- No fireplace
- No pool

### Decision

Categorical structural missing values  
→ `"None"`

Numerical structural missing values  
→ `0`

---

## Genuine Missing Values

True missing information is handled individually.

### Electrical & General Binaries

Strategy:
- Most Frequent Imputation (Mode)

---

### LotFrontage

Strategy:
- Neighborhood median imputation

Reason:
- Lot frontage depends strongly on neighborhood characteristics.

Implemented inside the numerical sub-pipeline.

---

### MasVnrType

Strategy:
- Rule-based imputation where appropriate; structural missing cases become `"None"`.

---

### GarageYrBlt

No direct standalone imputation.

Instead:
- Feature engineering derives `GarageAge` alongside structural presence indicator flags.

---

# 3. Feature Engineering

The project adopts the principle:

> One engineered feature should represent one meaningful domain concept.

Implemented features (via custom `DomainFeatureEngineer` transformer):

- `TotalSF`: `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
- `TotalBaths`: `FullBath` + ($0.5 \times$ `HalfBath`) + `BsmtFullBath` + ($0.5 \times$ `BsmtHalfBath`)
- `HouseAge`: `YrSold` - `YearBuilt`
- `GarageAge`: `YrSold` - `GarageYrBlt`
- `HasGarage` / `HasBsmt`: Binary flags indicating structural presence.

Deferred features:
- `YearsUntilFirstRemodel`
- `PrimaryAmenityScore`
- `LotAccessibility`

Reason:
- Insufficient evidence that they improve model quality.

---

# 4. Categorical Encoding

Encoding decisions are based on feature semantics rather than storage datatype.

## Ordinal Features

Strategy:
- Explicit dictionary ordering via `ExplicitMapTransformer`.

Reason:
- Domain semantics must not depend on automatically discovered category ordering.
- Mappings are standardized across common scales (e.g., `QUALITY_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}`).

Examples:
- `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`.

---

## Nominal Features

Strategy:
- Split based on cardinality to balance spatial dimensionality and signal preservation:
  - **Low-Cardinality ($K \le 10$, 20 columns):** `OneHotEncoder(min_frequency=0.01, drop='first', handle_unknown='ignore', sparse_output=False)`
  - **High-Cardinality ($K > 10$, 3 columns: `Neighborhood`, `Exterior1st`, `Exterior2nd`):** `TargetEncoder(smooth="auto", cv=5, handle_unknown='value')`

Reason:
- Prevents extreme column space explosion for high-cardinality locations while preserving non-linear linear pricing boundaries.

---

## Binary Features

- **Numeric binary variables:** Kept unchanged.
- **Categorical binary strings (`CentralAir`, `Street`, `Utilities`):** Explicit dictionary mapping ($0/1$) via custom `ExplicitBinaryMapper`.

---

## Unknown Categories

- **Nominal variables:** Handled via `handle_unknown="ignore"` (One-Hot) or `handle_unknown="value"` (Target Encoding).
- **Ordinal variables:** Dedicated zero category mapped explicitly (`'None': 0`).

The system will never fail solely because an unseen category appears during inference.

---

# 5. Target Transformation

Target:
- `SalePrice`

Decision:
- Log-based transformation ($\log1p$).

Reasons:
- Reduce right skew.
- Reduce heteroscedasticity.
- Directly match Kaggle evaluation metric (RMSLE).
- Improve numerical behavior for linear models.

Predictions are automatically converted back to USD using $\text{expm1}$ inside `TransformedTargetRegressor`.

---

# 6. Scaling Strategy

Scaling depends on the downstream estimator.

## Linear / Distance Models (Pipeline Alpha)

Scaling enabled.

Preferred scaler:
- `RobustScaler`

Reason:
- Continuous housing variables contain genuine extreme values; median and IQR scaling prevents leverage distortions during regularized optimization.

---

## Tree Models (Pipeline Beta)

Scaling disabled (`'passthrough'`).

Reason:
- Decision trees depend on relative order rather than feature magnitude.
- Scaling adds unnecessary computation and destroys feature readability.

---

# 7. Outlier Strategy

Outliers are separated into three categories:

- **Data Errors:** Correct or remove.
- **Rare but Valid Observations:** Retain (e.g., luxury estates $> \$600,000$).
- **Influential Observations:** Investigate individually.

Ames Specific Rule:
- Training observations with `GrLivArea > 4000 sq ft` AND `SalePrice < $200,000` (Edwards neighborhood partial sales, PIDs `908154190` and `908154235`) are pruned prior to fitting. 
- Legitimate extreme luxury homes are retained and dampened gracefully by the $\log1p$ target transformation.

---

# 8. ColumnTransformer & Pipeline Architecture

The complete system is encapsulated within a `TransformedTargetRegressor` containing a master `Pipeline` and `ColumnTransformer`.

Branches:
- **Numeric Branch:** `SimpleImputer(median)` → `DomainFeatureEngineer` → `RobustScaler` / `'passthrough'`
- **Ordinal Branch:** `SimpleImputer('None')` → `ExplicitMapTransformer` → `RobustScaler` / `'passthrough'`
- **Nominal Low-Card Branch:** `SimpleImputer('Missing')` → `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Nominal High-Card Branch:** `SimpleImputer('Missing')` → `TargetEncoder(smooth='auto', cv=5)`
- **Binary Branch:** `SimpleImputer(mode)` → `ExplicitBinaryMapper`

Each branch owns only the transformations relevant to that feature family.

---

# Planned Final Architecture

Raw Input Data $(X, y)$  
│  
▼  
[ Outlier Pruner ] (Training Only: `GrLivArea > 4000` & `SalePrice < 200k`)  
│  
▼  
[ TransformedTargetRegressor ] ──(Target Trans: $y_{\text{log}} = \log1p(y)$)  
│  
▼  
[ Master ColumnTransformer ]  
 ├── Numeric Branch  ──► SimpleImputer(median) ──► DomainFeatureEngineer ──► RobustScaler / Passthrough  
 ├── Ordinal Branch  ──► SimpleImputer('None')  ──► ExplicitMapTransformer ──► RobustScaler / Passthrough  
 ├── Nom-Low Branch  ──► SimpleImputer('Missing') ──► OneHotEncoder(drop='first', handle_unknown='ignore')  
 ├── Nom-High Branch ──► SimpleImputer('Missing') ──► TargetEncoder(smooth='auto', cv=5)  
 └── Binary Branch   ──► SimpleImputer(mode)    ──► ExplicitBinaryMapper  
│  
▼  
[ Downstream Estimator ] (Pipeline Alpha: `Ridge` / Pipeline Beta: `XGBRegressor`)  
│  
▼  
Inverse Prediction Engine ──($\text{expm1}(y_{\text{pred\_log}}) \longrightarrow \text{USD Real Dollar Output}$)

---

# Deferred Decisions

The following topics will be revisited only after establishing a baseline model.

- Feature Selection Techniques (e.g., Lasso / Recursive Feature Elimination)
- Power Transformations (e.g., Box-Cox / Yeo-Johnson on continuous inputs)
- Winsorization
- Advanced Polynomial Interaction Features
- Dimensionality Reduction (PCA / UMAP)

The guiding principle is:

> Do not introduce additional preprocessing complexity until empirical evidence demonstrates a measurable benefit.

---

# Revision History

| Date | Version | Notes |
|------|---------|------|
| July 2026 | 1.0 | Initial preprocessing design before implementation |
| July 2026 | 1.1 | Updated with exact 4-branch ColumnTransformer specs, high-cardinality TargetEncoder split, custom transformer classes, and precise Dean De Cock outlier filtering criteria. |

Missing Value Philosophy

Structural Missing Values

LotFrontage Strategy

MasVnrType Strategy

GarageYrBlt Strategy

Feature Engineering Decisions

Rejected Features

Encoding Policy

Ordinal Policy

Nominal Policy

Unknown Categories

Scaling Policy

Target Transformation

Outlier Strategy

ColumnTransformer Architecture

Pipeline Architecture