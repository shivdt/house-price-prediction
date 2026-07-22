# Project Log

**Project:** House Prices – Advanced Regression Techniques (Kaggle)
**Author:** Shiv Shankar Dubey
**Project Type:** End-to-End Machine Learning Regression Project
**Status:** In Progress

---

# Phase 1 – Business Understanding

## Objective

Develop a machine learning model to predict the sale price of residential houses in Ames, Iowa using 79 explanatory variables.

## Business Perspective

The model can assist stakeholders such as:

* Real estate agents
* Property investors
* Homeowners
* Banks and mortgage providers

by providing an estimated market value of a property to support pricing and investment decisions.

## Key Learnings

* Machine learning models support decision-making rather than replace human judgment.
* Model errors have real business consequences:

  * Overestimation may result in overpriced properties and slower sales.
  * Underestimation may lead to financial losses for sellers.
* Human review remains important, especially for unusual or unseen cases.

---

# Phase 2 – Project Setup

## Objective

Create a professional, reproducible, and maintainable project structure before beginning data analysis.

## Completed Tasks

* Created a professional project directory structure.
* Initialized a Git repository.
* Created and activated a Python virtual environment (`.venv`).
* Configured `.gitignore` to exclude unnecessary files such as the virtual environment and cache files.
* Installed initial project dependencies:

  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn
  * jupyter
* Generated `requirements.txt` using `pip freeze`.
* Made initial Git commits documenting project setup.
* Downloaded the Kaggle dataset and placed the original files in `data/raw/`.

## Engineering Decisions

### Why use a virtual environment?

To isolate project dependencies and avoid conflicts with other Python projects.

### Why use `.gitignore`?

To prevent tracking generated files, caches, and the virtual environment in Git.

### Why generate `requirements.txt`?

To allow anyone cloning the repository to recreate the same Python environment.

### Why keep `raw` and `processed` data separate?

The original dataset should remain unchanged. Any cleaning or preprocessing will generate new datasets inside `data/processed/`, ensuring reproducibility.

---

# Current Project Status

✅ Business Understanding

✅ Project Setup

⬜ Data Understanding

⬜ Exploratory Data Analysis (EDA)

⬜ Data Cleaning

⬜ Feature Engineering

⬜ Baseline Model

⬜ Model Improvement

⬜ Model Evaluation

⬜ Model Explainability

⬜ Final Report

---

# Reflection

This project is being developed with an emphasis on learning professional data science practices rather than only achieving a good Kaggle score.

The primary focus is to understand the complete end-to-end machine learning workflow, including business understanding, reproducibility, version control, documentation, engineering best practices, and systematic decision-making.

Future log entries will document important technical decisions, observations from the data, modeling experiments, and lessons learned throughout the project.

## Phase 2 – Project Setup

### Completed
- Created project directory structure
- Initialized Git repository
- Created Python virtual environment
- Configured `.gitignore`
- Installed initial project dependencies
- Generated `requirements.txt`
- Added first two Git commits

### Notes
Learned the purpose of virtual environments, reproducibility, Git commit practices, and dependency management.


# Phase 3 – Data Understanding

## Objective

Develop a thorough understanding of the dataset before performing exploratory analysis, preprocessing, or model building.

## Completed Tasks

* Reviewed the Kaggle competition objective and evaluation metric.
* Loaded `train.csv` and `test.csv` into pandas.
* Verified dataset dimensions:

  * Training set: **1460 rows × 81 columns**
  * Test set: **1459 rows × 80 columns**
* Confirmed that:

  * Each row represents one residential house.
  * `SalePrice` is the prediction target.
  * `Id` is an identifier.
  * `SalePrice` is absent from the test dataset.
* Inspected feature names and data types.
* Studied `data_description.txt` to understand feature meanings.
* Investigated missing values conceptually rather than immediately applying preprocessing.
* Identified that many missing values represent "Not Applicable" rather than missing information (e.g., no pool, no garage, no basement).
* Began semantic feature classification:

  * Identifier
  * Target
  * Numerical
  * Nominal
  * Ordinal
  * Temporal

## Key Learnings

* Data understanding must precede data cleaning and modeling.
* Missing values should always be interpreted before deciding how to handle them.
* The semantic meaning of a feature is often more important than its storage data type.
* Numeric-looking variables (such as `MSSubClass`) may actually represent categorical information.
* Ordered categories (ordinal features) should be distinguished from nominal categories because they often require different preprocessing strategies.

## Professional Practices Learned

* Read the dataset documentation before writing preprocessing code.
* Build a mental model of the dataset before running extensive analysis.
* Separate observations, hypotheses, evidence, and conclusions.
* Perform disciplined exploration by asking a clear question before executing each analysis step.

## Current Project Status

* ✅ Phase 1 – Business Understanding
* ✅ Phase 2 – Project Setup
* ✅ Phase 3 – Data Understanding
* ⏳ Phase 4 – Exploratory Data Analysis (Next)

## Reflection

This phase reinforced that successful machine learning projects begin with understanding the data rather than immediately building models. Careful interpretation of features, documentation, and missing values provides the foundation for meaningful exploratory analysis and informed modeling decisions in the next phase.



# Phase 4 – Exploratory Data Analysis (EDA)

## Objective

Explore the dataset systematically to understand the distribution of the target variable, identify important feature patterns, investigate relationships with the target, and generate hypotheses for preprocessing and model development.

## Completed Tasks

* Performed descriptive statistical analysis of the target variable (`SalePrice`).
* Examined the distribution of `SalePrice` using histograms and box plots.
* Identified positive skewness in the target distribution.
* Calculated summary statistics including median and interquartile range (IQR).
* Generated a dataset-wide feature summary:
  * Numerical and categorical feature identification.
  * Missing value analysis.
  * Feature cardinality analysis.
  * Initial correlation analysis with the target variable.
* Performed detailed univariate analysis of `OverallQual`:
  * Frequency distribution.
  * Summary statistics.
  * Missing value assessment.
* Conducted bivariate analysis between `SalePrice` and representative features including:
  * `OverallQual`
  * `TotalBsmtSF`
  * `GarageArea`
* Used scatter plots, regression plots, box plots, and violin plots to investigate relationships.
* Extended the analysis by incorporating `OverallQual` as a third variable (color encoding) to perform multivariate exploratory analysis.
* Identified potential outlier candidates for further investigation rather than immediate removal.

## Key Findings

### Target Variable

* `SalePrice` is positively skewed.
* Median is a more representative measure of central tendency than the mean.
* A logarithmic transformation may improve model performance.

### Important Predictive Features

The following variables exhibit strong positive relationships with `SalePrice`:

* OverallQual
* TotalBsmtSF
* GarageArea

### Missing Values

* Missing values require semantic interpretation before preprocessing.
* Many missing values correspond to the absence of a property feature (e.g., no pool or no garage) rather than incomplete data.

### Outliers

* Several observations exhibit unusually large basement or garage areas relative to their sale prices.
* Outliers will be investigated during preprocessing rather than automatically removed.

### Multivariate Insights

* Coloring scatter plots by `OverallQual` revealed clearer patterns than bivariate plots alone.
* Larger basement areas tend to be associated with higher-quality homes.
* Garage area alone does not strongly determine house quality.

## Professional Practices Learned

* Exploratory Data Analysis is question-driven rather than feature-driven.
* Representative features can provide sufficient understanding without exhaustively visualizing every variable.
* Visualization often reveals relationships that correlation coefficients alone cannot.
* Outliers should be investigated before deciding on any treatment.
* EDA generates hypotheses that guide preprocessing and feature engineering rather than making irreversible decisions.

## Reflection

This phase reinforced that exploratory data analysis is not about producing as many plots as possible. Instead, its purpose is to answer meaningful questions about the dataset, understand important relationships, identify potential data quality issues, and build intuition that informs preprocessing and model development. The analyses performed during this phase provide a solid foundation for designing a principled preprocessing pipeline.