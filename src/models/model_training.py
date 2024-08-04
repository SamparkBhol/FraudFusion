# src/models/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from gan import GAN
from autoencoder import Autoencoder

def load_data():
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    
    return X_train, X_test, y_train, y_test

def train_isolation_forest(X_train, y_train):
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=0.01)
    model.fit(X_train)
    return model

def train_one_class_svm(X_train, y_train):
    print("Training One-Class SVM model...")
    model = OneClassSVM(gamma='auto', nu=0.01)
    model.fit(X_train)
    return model

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder(X_train):
    print("Training Autoencoder model...")
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
    return autoencoder

def train_gan(X_train):
    print("Training GAN model...")
    gan = GAN(input_dim=X_train.shape[1])
    gan.train(X_train, epochs=10000, batch_size=64)
    return gan

def save_models(isolation_forest, one_class_svm, autoencoder, gan):
    print("Saving models...")
    import joblib
    joblib.dump(isolation_forest, 'models/isolation_forest.pkl')
    joblib.dump(one_class_svm, 'models/one_class_svm.pkl')
    autoencoder.save('models/autoencoder.h5')
    gan.save('models/gan.h5')

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    isolation_forest = train_isolation_forest(X_train, y_train)
    one_class_svm = train_one_class_svm(X_train, y_train)
    autoencoder = train_autoencoder(X_train)
    gan = train_gan(X_train)
    
    # Save models
    save_models(isolation_forest, one_class_svm, autoencoder, gan)
    
    print("Model training complete.")

if __name__ == "__main__":
    main()
