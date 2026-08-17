import os
import argparse
import pandas as pd
import joblib

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

def generate_submission(predictions, test_ids, output_path):
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to: {output_path}")

def main():

    # AI Generated:

    # Initializes the parser
    parser = argparse.ArgumentParser(description="Generate predictions using the trained Ames Housing model.") 
    
    # Argument for the input CSV
    parser.add_argument( # Looks out for --input flag
        "--input", # Custom name
        type=str, # Proided path must be read as string
        default=os.path.join("data", "raw", "test.csv"),
        help="Path to the input CSV file containing raw housing data."
    )
    
    # Argument for the output CSV
    parser.add_argument( # Looks out for --output flag
        "--output", 
        type=str, 
        default=os.path.join("data", "processed", "baseline_submission.csv"),
        help="Path where the output predictions CSV will be saved."
    )
    
    # Parse the arguments provided by the user
    args = parser.parse_args() # Reads what ever is written in terminal and stores it once the script is run


    model_path = os.path.join("models", "xgboost_baseline_v1.joblib")
    data_path = args.input # Remove the -- and put the name
    output_path = args.output 



    print("Loading test data...")
    test_df = pd.read_csv(data_path)
    
    model = load_model(model_path)

    print("Generating predictions...")
  
    predictions = model.predict(test_df)

    generate_submission(predictions, test_df['Id'], output_path)

if __name__ == "__main__": 
    main()

# Run this script in any of the following way:
    # python -m src.models.predict_model --input "data/new_client_data.csv" --output "data/client_predictions.csv"
    # python -m src.models.predict_model --help
    # python -m src.models.predict_model
    # python -m src.models.predict_model --input "data/new_client_data.csv"