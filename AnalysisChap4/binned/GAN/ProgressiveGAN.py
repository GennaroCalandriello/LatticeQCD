import numpy as np
import tensorflow as tf
from keras import layers, Sequential, Model
import matplotlib.pyplot as plt

from keras.layers import Layer, Dense, Embedding, Flatten, Concatenate, Input

EPOCHS = 100
BATCH_SIZE = 5  # Number of eigenvalue sets
START_EV = 0
END_EV = 100
NUM_EV_FINAL = END_EV - START_EV
NUM_EV_INITIAL = 100
# NUM_EV_FINAL = 100
INCREMENT_EV = 0  # number of eigenvalues to add at each increment
INCREMENT_INTERVAL = 10  # int(EPOCHS / 50)  # number of epochs between each increment
NUM_CONFIGURATIONS = 330
EMBEDDING_DIM = 40  # dimensione dello spazio di embedding


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


def generator_model(num_eigenvalues):
    # Eigenvalues input
    eigenvalues_input = Input(shape=(num_eigenvalues,))

    # Configuration input and embedding
    config_input = Input(shape=(1,))
    config_embedding = Embedding(NUM_CONFIGURATIONS + 1, EMBEDDING_DIM)(config_input)
    config_embedding = Flatten()(config_embedding)

    # Combine the inputs
    combined_input = Concatenate()([eigenvalues_input, config_embedding])

    # Ensure the first Dense layer after concatenation has the correct input dimension
    x = layers.Dense(128, activation="relu")(combined_input)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    output = layers.Dense(num_eigenvalues, activation="tanh")(x)
    return Model(inputs=[eigenvalues_input, config_input], outputs=output)


def discriminator_model(num_eigenvalues):
    # input per gli autovalori
    eigenvalues_input = Input(shape=(num_eigenvalues,))

    # input e layer di embedding per la configurazione
    config_input = Input(shape=(1,))
    config_embedding = Embedding(NUM_CONFIGURATIONS + 1, EMBEDDING_DIM)(config_input)
    config_embedding = Flatten()(config_embedding)

    # combina gli input
    combined_input = Concatenate()([eigenvalues_input, config_embedding])

    # resto del modellololooottololo
    x = SpectralNormalization(Dense(256, activation="relu"))(combined_input)
    x = SpectralNormalization(Dense(128, activation="relu"))(x)
    # x = SpectralNormalization(Dense(64, activation = "relu"))(x)
    output = SpectralNormalization(Dense(1, activation="sigmoid"))(x)
    return Model([eigenvalues_input, config_input], output)


def train(data):
    num_eigenvalues = NUM_EV_INITIAL  # Usa un numero fisso di autovalori
    config_indices = np.arange(1, NUM_CONFIGURATIONS + 2, 1)
    # Crea i modelli di generatore e discriminatore
    generator = generator_model(num_eigenvalues)
    discriminator = discriminator_model(num_eigenvalues)

    # Compila il discriminatore
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")

    # Crea il modello GAN combinato
    discriminator.trainable = False  # Imposta il discriminatore come non addestrabile nel modello GAN combinato
    gan_input = Input(shape=(num_eigenvalues,))
    config_input = Input(shape=(1,))  # Input aggiuntivo per le configurazioni
    gen_output = generator([gan_input, config_input])
    gan_output = discriminator([gen_output, config_input])
    gan_model = Model([gan_input, config_input], gan_output)
    gan_model.compile(optimizer="adam", loss="binary_crossentropy")
    gan_model.summary()

    for epoch in range(EPOCHS):

        for start_idx in range(0, NUM_CONFIGURATIONS, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, NUM_CONFIGURATIONS + 1)
            real_data = data[start_idx:end_idx, :num_eigenvalues]
            batch_indices = config_indices[start_idx:end_idx]
            batch_indices = batch_indices.reshape(-1, 1)

            # Selezione casuale dei batch di dati reali e degli indici di configurazione corrispondenti
            # real_data = data[indices, :num_eigenvalues]
            # batch_config_indices = config_indices[indices]

            # Generazione di dati falsi
            noise = np.random.normal(0, 1, (BATCH_SIZE, num_eigenvalues))

            generated_data = generator.predict([noise, batch_indices])

            # Addestramento del discriminatore
            real_labels = np.ones((end_idx - start_idx, 1))
            fake_labels = np.zeros((end_idx - start_idx, 1))
            d_loss_real = discriminator.train_on_batch(
                [real_data, batch_indices], real_labels
            )
            d_loss_fake = discriminator.train_on_batch(
                [generated_data, batch_indices], fake_labels
            )

            # Addestramento del generatore (attraverso il modello GAN combinato)
            g_loss = gan_model.train_on_batch([noise, batch_indices], real_labels)

        # Stampa le statistiche di addestramento ogni epoca
        print(
            f"Epoch: {epoch}, D_Loss: {(d_loss_real + d_loss_fake)/2}, G_Loss: {g_loss}"
        )

    # Salva i modelli
    generator.save("generator.h5")
    discriminator.save("discriminator.h5")


# Nota: Assicurati di definire le funzioni generator_model e discriminator_model,
# che dovrebbero creare e restituire i modelli del generatore e del discriminatore, rispettivamente.
# Inoltre, ricorda di adattare le parti di codice come la reinizializzazione dei modelli per trasferire i pesi
# e l'aggiustamento delle dimensioni di input/output per i tuoi specifici casi d'uso.


# if __name__ == "__main__":
#     # Make sure to reshape or prepare your 'data' correctly before calling build_and_train
#     # For example: data = data.reshape(-1, 100) if 'data' is a flat array of 33200 elements
#     # build_and_train(data)
