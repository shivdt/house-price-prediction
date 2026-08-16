import os
import yaml
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
from sklearn.metrics import root_mean_squared_log_error


from src.features.preprocessing import (
    AmesFeatureEngineer, 
    NeighborhoodLotFrontageImputer, 
    MasVnrImputer, 
    StructuralImputer,
    remove_training_outliers
)

def load_config(config_path = "src/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Load the dictionary
config = load_config()



def build_model_pipeline():
    
    # Feature Lists 
    ordinal_features = config['feature_groups']['ordinal']

    nominal_low_card_features = config['feature_groups']['nominal_low_card']

    nominal_high_card_features = config['feature_groups']['nominal_high_card']

    numeric_features = config['feature_groups']['continuous_numerical'] + config['feature_groups']['discrete_numerical'] + config['feature_groups']['elapsed_time'] + config['feature_groups']['passthrough_numeric']

    binary_features = config['feature_groups']['binary_flag']

   
    
    ordinal_mappings = {
        'ExterQual': config['mappings']['ordinal']['ExterQual'], 
        'ExterCond': config['mappings']['ordinal']['ExterCond'], 
        'BsmtQual': config['mappings']['ordinal']['BsmtQual'], 
        'BsmtCond': config['mappings']['ordinal']['BsmtCond'], 
        'HeatingQC': config['mappings']['ordinal']['HeatingQC'], 
        'KitchenQual': config['mappings']['ordinal']['KitchenQual'], 
        'FireplaceQu': config['mappings']['ordinal']['FireplaceQu'], 
        'GarageQual': config['mappings']['ordinal']['GarageQual'], 
        'GarageCond': config['mappings']['ordinal']['GarageCond'], 
        'PoolQC': config['mappings']['ordinal']['PoolQC'], 
        'Fence': config['mappings']['ordinal']['Fence'], 
        'GarageFinish': config['mappings']['ordinal']['GarageFinish'], 
        'BsmtExposure': config['mappings']['ordinal']['BsmtExposure'], 
        'LotShape': config['mappings']['ordinal']['LotShape'], 
        'Utilities': config['mappings']['ordinal']['Utilities'], 
        'LandSlope': config['mappings']['ordinal']['LandSlope'], 
        'BsmtFinType1': config['mappings']['ordinal']['BsmtFinType1'], 
        'BsmtFinType2': config['mappings']['ordinal']['BsmtFinType2'], 
        'Functional': config['mappings']['ordinal']['Functional'], 
        'PavedDrive': config['mappings']['ordinal']['PavedDrive']
    }


    ordered_categories = [ordinal_mappings[feat] for feat in ordinal_features]

    # Branch Pipelines
    ordinal_encoder = OrdinalEncoder(
        categories=ordered_categories, 
        handle_unknown='use_encoded_value', 
        unknown_value=-1, 
        encoded_missing_value=-1
        )
    
    nominal_low_encoder = OneHotEncoder(
        handle_unknown='ignore', 
        sparse_output=False
        )
    
    nominal_high_encoder = TargetEncoder(
        target_type='continuous', 
        smooth='auto', 
        cv=5
        )
    
    numeric_pipeline = Pipeline(
        [
            ('safety_imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]
    )
    binary_pipeline = Pipeline(
        [
            ('safety_imputer', SimpleImputer(strategy='most_frequent'))
        ]
    )

    # Column Transformer
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

    # Master Preprocessor
    base_processing = Pipeline(
        [
            ('structural_imputer', StructuralImputer()),
            ('lot_frontage_imputer', NeighborhoodLotFrontageImputer()),
            ('masvnr_imputer', MasVnrImputer()),
            ('feature_engineer', AmesFeatureEngineer(
                drop_columns=config['feature_groups']['candidates_for_removal'], 
                central_air_map=config['mappings']['binary']['CentralAir'])
                )
        ]
    )
    
    preprocessor = Pipeline(
        [
            ('base', base_processing), 
            ('branch', branch_preprocessor)
        ]
    )

    # Final Model Pipeline
    model_pipeline = Pipeline(
        [
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]
    )

    return TransformedTargetRegressor(regressor=model_pipeline, func=np.log1p, inverse_func=np.expm1)

def main():
    print("Loading raw training data...")
    # Run script from the root directory of the project
    data_path = os.path.join("data", "raw", "train.csv")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['SalePrice', 'Id'])
    y = df['SalePrice']
    
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, y_train = remove_training_outliers(X_train, y_train)

    print("Building and compiling the pipeline architecture...")
    model = build_model_pipeline()
    
    print("Training the master pipeline...")
    model.fit(X_train, y_train)
    
    print("Training complete. Serializing the model...")
    model_dir = os.path.join("models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "xgboost_baseline_v1.joblib")
    
    joblib.dump(model, save_path)
    print(f"Success! Production model saved to: {save_path}")

    print("Evaluating model on validation set...")
    valid_preds = model.predict(X_valid)

    rmsle = root_mean_squared_log_error(y_valid, valid_preds)
    print(f"Validation RMSLE: {rmsle:.5f}")


if __name__ == "__main__":
    main() # Run script via: python -m src.models.train_model