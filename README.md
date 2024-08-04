# 🚀 FraudFusion

Welcome to the **FraudFusion** project! This comprehensive system uses Generative Adversarial Networks (GANs) for data augmentation, various anomaly detection models, and a complete end-to-end machine learning pipeline to tackle fraudulent financial transactions. 🎯

## 📦 Project Overview

I've designed this project to detect fraudulent financial transactions using advanced machine learning techniques and data augmentation. It includes:

- **Data Augmentation with GANs**: Enhances training data using synthetic data.
- **Anomaly Detection Models**: Implements Isolation Forest, One-Class SVM, and Autoencoders.
- **End-to-End ML Pipeline**: Automates data preprocessing, model training, deployment, and monitoring.

## 🚀 Getting Started

_Prepare Your Data_

We need to prepare a financial transaction data in CSV format. The dataset should include both genuine and fraudulent transactions.

Data Format: Ensure your CSV file contains relevant features and a column indicating whether a transaction is fraudulent.
File Location: Place your raw CSV files in the data/raw/ directory.
_
Run Data Pipeline_

Process your data and generate the augmented dataset:

bash
Copy code
python src/pipeline/data_pipeline.py
This script will read your raw data, preprocess it, and save the processed data in the data/processed/ directory.

_Next steps:_

Train and deploy the model

nteract with the Web Interface

Upload your data CSV file via the web interface to get predictions on whether transactions are fraudulent or not. The web interface is styled with interactive CSS and JavaScript for a seamless experience. 🎨

## 💡 Example CSV Format
Ensure your CSV file is structured with columns like:

transaction_id
amount
transaction_date
feature1
feature2
is_fraud (1 for fraudulent, 0 for genuine)
Advanced-Fraud-Detection-System/

1. data/raw/example_transactions.csv
2. data/ processed/
      - X_train.csv
      - X_test.csv
      - y_train.csv
      - y_test.csv

**The above is a format for explination of placement of csv files**__

## 🌟 Features
Data Augmentation: Uses GANs to generate synthetic data.
Multiple Models: Includes Isolation Forest, One-Class SVM, and Autoencoders.
Automated Pipeline: Streamlines data processing and model training.
Web Deployment: Interactive web interface for real-time fraud detection.

## 🤔 Troubleshooting
Issues with Data: Ensure your CSV file is properly formatted and placed in the correct directory.
Dependencies: Make sure all required packages are installed.
Model Loading: Verify that the models are correctly saved and loaded.

##### It's a beginner's 🙃 project so there might be bugs and issues feel free to let me know.
