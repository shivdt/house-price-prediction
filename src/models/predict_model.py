import os
import pandas as pd
import joblib

def load_model(model_path):
    """Loads a serialized machine learning model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

def generate_submission(predictions, test_ids, output_path):
    """Formats predictions into a Kaggle-ready CSV and saves it."""
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to: {output_path}")

def main():
    # 1. Define Paths (Assuming script is run from the project root)
    model_path = os.path.join("models", "xgboost_baseline_v1.joblib")
    data_path = os.path.join("data", "raw", "test.csv")
    output_dir = os.path.join("data", "processed")
    output_path = os.path.join(output_dir, "baseline_submission.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load the Data
    print("Loading test data...")
    test_df = pd.read_csv(data_path)
    
    # 3. Load the Model
    # This automatically includes the preprocessor, encoders, imputers, and the XGBoost algorithm
    model = load_model(model_path)
    
    # 4. Generate Predictions
    print("Generating predictions...")
    # Because of our robust pipeline, we just pass the raw dataframe in!
    predictions = model.predict(test_df)
    
    # 5. Save the Output
    generate_submission(predictions, test_df['Id'], output_path)

if __name__ == "__main__":
    main()