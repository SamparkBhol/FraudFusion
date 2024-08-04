import mlflow

def log_metrics(metrics):
    with mlflow.start_run():
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
