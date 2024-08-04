import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.dropna()  # Drop missing values
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
    return df

def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df = load_data('../data/raw/example_transactions.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, 'fraud')
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_train_scaled_df.to_csv('../data/processed/X_train.csv', index=False)
    X_test_scaled_df.to_csv('../data/processed/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('../data/processed/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('../data/processed/y_test.csv', index=False)
