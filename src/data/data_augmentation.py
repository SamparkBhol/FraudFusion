import pandas as pd
from sklearn.utils import shuffle
from gan import GAN

def augment_data():
    df = pd.read_csv('../data/processed/X_train.csv')
    gan = GAN(data_dim=df.shape[1])
    gan.train(df.values, epochs=10000, batch_size=32)
    synthetic_data = gan.generate_data(num_samples=1000)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    augmented_df = pd.concat([df, synthetic_df])
    augmented_df = shuffle(augmented_df)
    augmented_df.to_csv('../data/augmented/augmented_transactions.csv', index=False)

if __name__ == "__main__":
    augment_data()
