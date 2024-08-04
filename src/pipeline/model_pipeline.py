import pandas as pd
from src.models.anomaly_detection import train_isolation_forest, train_one_class_svm
from src.models.autoencoder import train_autoencoder

def run_model_pipeline():
    """
    Runs the model pipeline: trains and saves anomaly detection models.
    """
    # Load processed data
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')

    # Train and save Isolation Forest model
    isolation_forest_model = train_isolation_forest(X_train, y_train)
    joblib.dump(isolation_forest_model, '../models/isolation_forest_model.pkl')

    # Train and save One-Class SVM model
    one_class_svm_model = train_one_class_svm(X_train, y_train)
    joblib.dump(one_class_svm_model, '../models/one_class_svm_model.pkl')

    # Train and save Autoencoder model
    train_autoencoder(X_train)

if __name__ == "__main__":
    run_model_pipeline()
