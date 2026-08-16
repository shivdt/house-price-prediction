import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# 1. Elapsed Time & Feature Engineering Transformer
# ======================================================
class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns=None, central_air_map=None): # This avoids I/O bottleneck during transform(), by avoiding importing yaml file
        # Store the configurations passed from train_model.py
        self.drop_columns = drop_columns if drop_columns else []
        self.central_air_map = central_air_map if central_air_map else {'N': 0, 'Y': 1}

    def fit(self, X, y=None):
        self.is_fitted_ = True # prevents scikit-learn from throwing a NotFittedError: This Pipeline instance is not fitted yet.
        return self
        
    def transform(self, X):
        X_new = X.copy()
        
        # Elapsed Time Features
        X_new['HouseAge'] = X_new['YrSold'] - X_new['YearBuilt']
        X_new['YearsSinceRemodel'] = X_new['YrSold'] - X_new['YearRemodAdd']
        X_new['GarageAge'] = X_new['YrSold'] - X_new['GarageYrBlt']
        X_new['GarageAge'] = X_new['GarageAge'].fillna(0) 
        
        # Binary Features & Mappings
        X_new['HasGarage'] = (X_new['GarageType'] != "None").astype(int)
        X_new['HasBsmt'] = (X_new['TotalBsmtSF'] > 0).astype(int)
        X_new['CentralAir'] = X_new['CentralAir'].map(self.central_air_map)
        
        # Area and Room Features
        X_new['TotalBathrooms'] = (
            X_new['BsmtFullBath'] + (0.5 * X_new['BsmtHalfBath']) + 
            X_new['FullBath'] + (0.5 * X_new['HalfBath'])
        )
        X_new['TotalUsableSF'] = X_new['GrLivArea'] + X_new['TotalBsmtSF']
        
        # Ratios & Interactions
        X_new['BsmtFinishedRatio'] = np.where(
            X_new['TotalBsmtSF'] == 0, 0, 
            (X_new['BsmtFinSF1'] + X_new['BsmtFinSF2']) / X_new['TotalBsmtSF']
        )
        X_new['OverallQual_x_TotalUsableSF'] = X_new['OverallQual'] * X_new['TotalUsableSF']

        X_new = X_new.drop(columns = self.drop_columns, errors='ignore') # Got KeyError, 'Id' not found, but somehow adding errors = 'ignore' fixed it!
        return X_new

# ======================================================
# 2. Grouped Imputation: Lot Frontage
# ======================================================
class NeighborhoodLotFrontageImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.global_median_ = np.nan
        self.group_median_ = np.nan 

    def fit(self, X, y=None):
        self.global_median_ = X['LotFrontage'].median()
        self.group_median_ = X.groupby('Neighborhood')['LotFrontage'].median()
        self.is_fitted_ = True
        return self 

    def transform(self, X):
        X = X.copy()
        mapped_medians = X['Neighborhood'].map(self.group_median_).fillna(self.global_median_)
        X['LotFrontage'] = X['LotFrontage'].fillna(mapped_medians)
        return X

# ======================================================
# 3. Grouped Imputation: Masonry Veneer Type
# ======================================================
def get_first_mode(series):
    modes = series.mode()
    if not modes.empty:
        return modes[0]
    return None

class MasVnrImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.global_mode_ = None
        self.group_mode_ = None

    def fit(self, X, y=None):

        veneer_homes = X[X["MasVnrArea"] > 0]

        # Global mode among homes that actually have veneer
        if not veneer_homes["MasVnrType"].dropna().empty:
            self.global_mode_ = veneer_homes["MasVnrType"].mode()[0]

        # Exterior1st-specific mode
        self.group_mode_ = (
            veneer_homes
            .groupby("Exterior1st")["MasVnrType"]
            .apply(get_first_mode)
        )

        return self

    def transform(self, X):

        X = X.copy()

        veneer_mask = X["MasVnrArea"] > 0
        no_veneer_mask = X["MasVnrArea"] == 0

        # Exterior-specific mode → global mode fallback
        mapped_mode = (
            X["Exterior1st"]
            .map(self.group_mode_)
            .fillna(self.global_mode_)
        )

        # Case 1:
        # House has veneer but MasVnrType is missing
        X.loc[veneer_mask, "MasVnrType"] = (
            X.loc[veneer_mask, "MasVnrType"]
            .fillna(mapped_mode.loc[veneer_mask])
        )

        # Case 2:
        # House has no veneer → MasVnrType must be None
        X.loc[no_veneer_mask, "MasVnrType"] = "None"

        # Case 3:
        # Any remaining missing values
        X["MasVnrType"] = X["MasVnrType"].fillna("None")

        return X

# ======================================================
# 4. Structural Default Imputer
# ======================================================
class StructuralImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_cols = [
    'Alley', 'BsmtQual', 'BsmtCond', 
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
    'FireplaceQu', 'GarageType', 'GarageFinish', 
    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 
    'MiscFeature'
    ]
        self.num_cols = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 
    'GarageCars', 'GarageArea', 'MasVnrArea', 
    'PoolArea', 'Fireplaces', 'MiscVal'
    ]
        
    def fit(self, X, y=None):
        self.is_fitted_ = True  # As the transformer is deterministic, it enables calling trasform function directly.
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in self.num_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        return X


# ======================================================
# 5. Training Dataset Outlier Remover
# ======================================================

def remove_training_outliers(X_train, y_train, contamination=0.01, random_state=42):

    from sklearn.ensemble import IsolationForest
    
    # 1. Isolate numerical columns
    numerical_df = X_train.select_dtypes(include=['int64', 'float64'])
    
    # 2. Fit and predict strictly on the training set
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(numerical_df)
    
    # 3. Create a boolean mask (True for normal data, False for outliers)
    mask = preds != -1
    
    # 4. Filter BOTH X and y simultaneously
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]
    
    return X_train_clean, y_train_clean