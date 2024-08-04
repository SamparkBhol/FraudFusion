import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class GAN:
    def __init__(self, data_dim):
        self.data_dim = data_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.data_dim, activation='tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.data_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(loss='binary_crossentropy', optimizer=Adam())
        return model

    def train(self, data, epochs=10000, batch_size=32):
        for epoch in range(epochs):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]
            noise = np.random.normal(0, 1, (batch_size, 100))
            generated_data = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 1000 == 0:
                print(f"{epoch} [D loss: {d_loss_real[0]}, acc.: {100*d_loss_real[1]}%] [G loss: {g_loss}]")

    def generate_data(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, 100))
        return self.generator.predict(noise)

if __name__ == "__main__":
    df = pd.read_csv('../data/processed/X_train.csv')
    data = df.values
    gan = GAN(data_dim=data.shape[1])
    gan.train(data, epochs=10000, batch_size=32)
    synthetic_data = gan.generate_data(num_samples=1000)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    synthetic_df.to_csv('../data/augmented/augmented_transactions.csv', index=False)
