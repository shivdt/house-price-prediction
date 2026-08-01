import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

# IMPORTANT: Importing your custom modularized code!
from src.features.custom_transformers import (
    AmesFeatureEngineer, 
    NeighborhoodLotFrontageImputer, 
    Exterior1stMasVnrTypeImputer, 
    StructuralImputer
)

def build_model_pipeline():
    """Constructs the complete ML pipeline architecture."""
    
    # 1. Feature Lists (Your Milestone 1 Metadata)
    ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'GarageFinish', 'LotShape', 'LandSlope', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Utilities', 'Functional', 'PavedDrive']
    nominal_low_card_features = ['MSZoning', 'Street', 'Alley', 'BldgType', 'HouseStyle', 'RoofStyle', 'Foundation', 'GarageType', 'SaleCondition', 'LotConfig', 'LandContour', 'MasVnrType', 'Heating', 'Electrical', 'MiscFeature', 'Condition1', 'Condition2', 'RoofMatl', 'SaleType']
    nominal_high_card_features = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'MSSubClass']
    numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtUnfSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch', 'PoolArea', 'MiscVal', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'LowQualFinSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalUsableSF', 'BsmtFinishedRatio', 'OverallQual_x_TotalUsableSF', 'TotalBathrooms', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'BedroomAbvGr', 'MoSold', 'GarageCars', 'HouseAge', 'YearsSinceRemodel', 'GarageAge', 'OverallQual', 'OverallCond']
    binary_features = ['CentralAir', 'HasGarage', 'HasBsmt']

    # 2. Ordinal Mappings Setup
    quality_scale = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
    garage_finish_scale = ["None", "Unf", "RFn", "Fin"]
    basement_finish_scale = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
    exposure_scale = ["None", "No", "Mn", "Av", "Gd"]
    
    ordinal_mappings = {
        'ExterQual': quality_scale, 'ExterCond': quality_scale, 'BsmtQual': quality_scale, 'BsmtCond': quality_scale, 'HeatingQC': quality_scale, 'KitchenQual': quality_scale, 'FireplaceQu': quality_scale, 'GarageQual': quality_scale, 'GarageCond': quality_scale, 'PoolQC': quality_scale, 'Fence': ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"], 'GarageFinish': garage_finish_scale, 'BsmtExposure': exposure_scale, 'LotShape': ["IR3", "IR2", "IR1", "Reg"], 'Utilities': ["ELO", "NoSeWa", "NoSewr", "AllPub"], 'LandSlope': ["Sev", "Mod", "Gtl"], 'BsmtFinType1': basement_finish_scale, 'BsmtFinType2': basement_finish_scale, 'Functional': ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], 'PavedDrive': ["N", "P", "Y"]
    }
    ordered_categories = [ordinal_mappings[feat] for feat in ordinal_features]

    # 3. Branch Pipelines
    ordinal_encoder = OrdinalEncoder(categories=ordered_categories, handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
    nominal_low_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    nominal_high_encoder = TargetEncoder(target_type='continuous', smooth='auto', cv=5)
    
    numeric_pipeline = Pipeline([('safety_imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    binary_pipeline = Pipeline([('safety_imputer', SimpleImputer(strategy='most_frequent'))])

    # 4. Column Transformer
    branch_preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_encoder, ordinal_features),
            ('nominal_low', nominal_low_encoder, nominal_low_card_features),
            ('nominal_high', nominal_high_encoder, nominal_high_card_features),
            ('numeric', numeric_pipeline, numeric_features),
            ('binary', binary_pipeline, binary_features)
        ],
        remainder='drop'
    )

    # 5. Master Preprocessor
    base_processing = Pipeline([
        ('structural_imputer', StructuralImputer()),
        ('lot_frontage_imputer', NeighborhoodLotFrontageImputer()),
        ('masvnr_imputer', Exterior1stMasVnrTypeImputer()),
        ('feature_engineer', AmesFeatureEngineer())
    ])
    
    preprocessor = Pipeline([('base', base_processing), ('branch', branch_preprocessor)])

    # 6. Final Model Pipeline
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    return TransformedTargetRegressor(regressor=model_pipeline, func=np.log1p, inverse_func=np.expm1)

def main():
    print("Loading raw training data...")
    # Assuming script is run from the root directory of the project
    data_path = os.path.join("data", "raw", "train.csv")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['SalePrice', 'Id'])
    y = df['SalePrice']
    
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building and compiling the pipeline architecture...")
    model = build_model_pipeline()
    
    print("Training the master pipeline... (This may take a moment)")
    model.fit(X_train, y_train)
    
    print("Training complete. Serializing the model...")
    model_dir = os.path.join("models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "xgboost_baseline_v1.joblib")
    
    joblib.dump(model, save_path)
    print(f"Success! Production model saved to: {save_path}")

if __name__ == "__main__":
    main()