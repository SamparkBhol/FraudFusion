import mlflow
import mlflow.keras
from src.models.autoencoder import build_autoencoder
import pandas as pd

def log_model_training():
    """
    Logs model training parameters and metrics using MLflow.
    """
    mlflow.set_experiment('fraud_detection_experiment')

    with mlflow.start_run():
        # Example configuration
        input_dim = 30  # Example input dimension
        autoencoder = build_autoencoder(input_dim)
        mlflow.keras.autolog()
        
        # Example training
        X_train = pd.read_csv('../data/processed/X_train.csv')
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.1)
        
        # Log parameters and metrics
        mlflow.log_params({'batch_size': 256, 'epochs': 50})

if __name__ == "__main__":
    log_model_training()
