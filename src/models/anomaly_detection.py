import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

def train_isolation_forest(X_train, y_train):
    model = IsolationForest(contamination=0.01)
    model.fit(X_train)
    return model

def train_one_class_svm(X_train, y_train):
    model = OneClassSVM(gamma='auto', nu=0.01)
    model.fit(X_train)
    return model

def load_autoencoder_model(filepath):
    return load_model(filepath)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    if hasattr(model, 'predict'):
        predictions = (predictions > 0.5).astype(int)
    else:
        predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv')

    isolation_forest_model = train_isolation_forest(X_train, y_train)
    isolation_forest_model.save('../models/isolation_forest_model.pkl')

    one_class_svm_model = train_one_class_svm(X_train, y_train)
    one_class_svm_model.save('../models/one_class_svm_model.pkl')

    autoencoder_model = load_autoencoder_model('../models/autoencoder_model.h5')
    
    evaluate_model(isolation_forest_model, X_test, y_test)
    evaluate_model(one_class_svm_model, X_test, y_test)
    evaluate_model(autoencoder_model, X_test, y_test)
