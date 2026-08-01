import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# 1. Elapsed Time & Feature Engineering Transformer
# ======================================================
class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.is_fitted_ = True 
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
        X_new['CentralAir'] = X_new['CentralAir'].map({'N': 0, 'Y': 1})
        
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

class Exterior1stMasVnrTypeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.global_mode_ = None
        self.group_mode_ = None

    def fit(self, X, y=None):
        veneer_homes = X[X['MasVnrArea'] > 0]
        if not veneer_homes['MasVnrType'].dropna().empty:
            self.global_mode_ = veneer_homes["MasVnrType"].mode()[0]
        else:
            self.global_mode_ = X["MasVnrType"].mode()[0]
        self.group_mode_ = veneer_homes.groupby('Exterior1st')['MasVnrType'].apply(get_first_mode)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        mapped_mode = X['Exterior1st'].map(self.group_mode_).fillna(self.global_mode_)
        mask = X['MasVnrArea'] > 0
        X.loc[mask, 'MasVnrType'] = X.loc[mask, 'MasVnrType'].fillna(mapped_mode)
        return X

# ======================================================
# 4. Structural Default Imputer
# ======================================================
class StructuralImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_cols = [
            'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
        ]
        self.num_cols = [
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea', 
            'MasVnrArea'
        ]
        
    def fit(self, X, y=None):
        self.is_fitted_ = True 
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