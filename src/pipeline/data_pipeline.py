import pandas as pd
from src.data.data_preprocessing import preprocess_data, split_data, scale_data
from src.data.data_augmentation import augment_data

def run_data_pipeline(raw_filepath):
    """
    Runs the data pipeline: loads data, preprocesses it, scales it, and saves the processed data.
    
    Args:
    - raw_filepath (str): Path to the raw data file (CSV).
    """
    # Load raw data
    df = pd.read_csv(raw_filepath)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, 'fraud')
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Save processed data
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_train_scaled_df.to_csv('../data/processed/X_train.csv', index=False)
    X_test_scaled_df.to_csv('../data/processed/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('../data/processed/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('../data/processed/y_test.csv', index=False)
    
    # Augment data
    augment_data()

if __name__ == "__main__":
    run_data_pipeline('../data/raw/example_transactions.csv')
