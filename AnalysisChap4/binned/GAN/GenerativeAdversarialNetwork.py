import numpy as np
import tensorflow as tf
from keras import layers, Sequential, Model
import matplotlib.pyplot as plt

from keras.layers import Layer, Dense

EPOCHS = 10000
BATCH_SIZE = 332  # Number of eigenvalue sets
START_EV = 0
END_EV = 70
NUM_EIGENVALUES = END_EV - START_EV


@tf.keras.utils.register_keras_serializable()
class SpectralNormalization(Layer):
    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(**kwargs)
        # Check if 'layer' is a serialized dictionary
        if isinstance(layer, dict):
            self.layer = tf.keras.layers.deserialize(layer)
        else:
            self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape)

        # Initializing 'u' with the correct shape
        self.u = self.add_weight(
            shape=(1, self.layer.kernel.shape[1]),
            initializer=tf.initializers.RandomNormal(),
            trainable=False,
            name="sn_u",
        )
        super(SpectralNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        W_shape = self.layer.kernel.shape
        W_reshaped = tf.reshape(self.layer.kernel, [-1, W_shape[-1]])
        u_hat = self.u

        # Power iteration for approximating the spectral norm
        v_hat = tf.math.l2_normalize(tf.matmul(u_hat, W_reshaped, transpose_b=True))
        u_hat = tf.math.l2_normalize(tf.matmul(v_hat, W_reshaped))

        # u_hat = u_hat.numpy()
        sigma = tf.matmul(tf.matmul(v_hat, W_reshaped), u_hat, transpose_b=True)

        self.u.assign(u_hat)
        self.layer.kernel.assign(self.layer.kernel / sigma)

        return self.layer(inputs)

    def get_config(self):
        config = super(SpectralNormalization, self).get_config()
        config["layer"] = tf.keras.layers.serialize(self.layer)
        return config

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)


def generator():
    model = Sequential()
    model.add(layers.Dense(128, activation="relu", input_dim=NUM_EIGENVALUES))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(NUM_EIGENVALUES, activation="tanh"))
    return model


# def discriminator():
#     model = Sequential()
#     model.add(layers.Dense(256, activation="relu", input_shape=(100,)))
#     model.add(layers.Dense(128, activation="relu"))
#     model.add(layers.Dense(1, activation="sigmoid"))
#     return model


def discriminator():
    model = Sequential()
    model.add(
        SpectralNormalization(
            Dense(256, activation="relu"), input_shape=(NUM_EIGENVALUES,)
        )
    )
    model.add(SpectralNormalization(Dense(128, activation="relu")))
    model.add(SpectralNormalization(Dense(1, activation="sigmoid")))
    return model


def build_and_train(data):
    discr = discriminator()
    gen = generator()
    discr.compile(optimizer="adam", loss="binary_crossentropy")

    discr.trainable = False
    gan_input = layers.Input(shape=(NUM_EIGENVALUES,))
    fake_eigenvalues = gen(gan_input)
    gan_output = discr(fake_eigenvalues)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer="adam", loss="binary_crossentropy")

    # Training loop
    d_loss_list = []
    g_loss_list = []

    for epoch in range(EPOCHS):
        # Generate fake data
        noise = np.random.normal(0, 1, size=[BATCH_SIZE, NUM_EIGENVALUES])
        fake_data = gen.predict(noise)

        # Train discriminator
        d_loss_real = discr.train_on_batch(data, np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discr.train_on_batch(fake_data, np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)

        print(
            f"Epoch: {epoch} \t Discriminator Loss: {d_loss} \t Generator Loss: {g_loss}"
        )
    gen.save("path/to/generator.h5")  # Salva il modello del generatore
    discr.save("path/to/discriminator.h5")  # Salva il modello del discriminatore
    np.savetxt("path/to/g_loss.txt", g_loss_list)
    np.savetxt("path/to/d_loss.txt", d_loss_list)


# if __name__ == "__main__":
#     # Make sure to reshape or prepare your 'data' correctly before calling build_and_train
#     # For example: data = data.reshape(-1, 100) if 'data' is a flat array of 33200 elements
#     # build_and_train(data)
