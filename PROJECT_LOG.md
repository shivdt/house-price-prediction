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