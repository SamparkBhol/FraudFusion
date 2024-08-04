import subprocess

def main():
    """
    Orchestrates the entire pipeline: data preparation, model training, and monitoring.
    """

    subprocess.run(['python', 'src/pipeline/data_pipeline.py'])

    subprocess.run(['python', 'src/pipeline/model_pipeline.py'])

    subprocess.run(['python', 'src/pipeline/monitoring.py'])

if __name__ == "__main__":
    main()
