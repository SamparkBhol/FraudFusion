import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(df):
    # Example feature extraction
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(df['transaction_description'])
    df = pd.concat([df, pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)
    df.drop('transaction_description', axis=1, inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/example_transactions.csv')
    df = extract_features(df)
    df.to_csv('../data/processed/feature_engineered_transactions.csv', index=False)
