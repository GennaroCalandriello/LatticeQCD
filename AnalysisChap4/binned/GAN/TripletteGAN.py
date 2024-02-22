import numpy as np
import tensorflow as tf
from keras import layers, Sequential, Model
import matplotlib.pyplot as plt

from keras.layers import Layer, Dense, Embedding, Flatten, Concatenate, Input

EPOCHS = 20
BATCH_SIZE = 5  # Number of eigenvalue sets
START_EV = 0
END_EV = 80
NUM_EV_FINAL = END_EV - START_EV
NUM_EV_INITIAL = 80
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


def build_generator(noise_dim):
    noise_input = Input(shape=(noise_dim,))
    config_input = Input(shape=(1,))
    position_input = Input(shape=(1,))
    merged_input = Concatenate()([noise_input, config_input, position_input])

    # Potresti voler aggiungere layer aggiuntivi o modificare le dimensioni a seconda della complessità desiderata
    x = Dense(128, activation="relu")(noise_input)
    # x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    # x = Dense(16, activation="relu")(x)

    # Output layer: si adatta per generare triplette
    # Assumi che ogni componente della tripletta sia un singolo valore scalare
    output = Dense(3, activation="linear")(x)  # 'linear' per un output numerico diretto

    model = Model(inputs=[noise_input, config_input, position_input], outputs=output)
    return model


def build_discriminator():
    input_triplet = Input(shape=(3,))  # Triplette di input: lambda, conf, posizione

    x = SpectralNormalization(Dense(128, activation="relu"))(input_triplet)
    # x = SpectralNormalization(Dense(64, activation="relu"))(x)
    x = SpectralNormalization(Dense(64, activation="relu"))(x)
    # x = SpectralNormalization(Dense(16, activation="relu"))(x)

    # Output layer: classifica la tripletta come reale o falsa
    output = SpectralNormalization(Dense(1, activation="sigmoid"))(x)

    model = Model(inputs=input_triplet, outputs=output)

    return model


def custom_loss(y_true, y_pred):
    weights = tf.constant([0.8, 0.9, 0.3], dtype=tf.float32)

    # calcola la perdita ponderata
    squared_difference = tf.square(y_true - y_pred)
    weighted_squared_difference = squared_difference * weights
    loss = tf.reduce_mean(
        weighted_squared_difference, axis=-1
    )  # media sull'ultimo asse

    return loss


def organize_and_sort_data(data):
    # Ordina i dati in base al valore di lambda
    sorted_data = []
    for i in range(NUM_CONFIGURATIONS):
        for j in range(NUM_EV_INITIAL):
            sorted_data.append([i, data[i, j]])

    sorted_data = np.array(sorted_data)
    sorted_data = sorted(sorted_data, key=lambda x: x[1])

    structured_triplette = []
    for i in range(len(sorted_data)):
        structured_triplette.append([sorted_data[i][1], sorted_data[i][0], i])
    return structured_triplette


def train(data):
    data = data[:, 0:END_EV]
    triplette = organize_and_sort_data(data)
    triplette = np.array(triplette)
    print("triplette", triplette.shape)
    print("triplette", triplette[0:5])

    num_eigenvalues = NUM_EV_INITIAL  # Usa un numero fisso di autovalori
    config_indices = np.arange(1, NUM_CONFIGURATIONS + 2, 1)

    # Crea i modelli di generatore e discriminatore
    generator = build_generator(noise_dim=num_eigenvalues)
    discriminator = build_discriminator()

    # Compila il discriminatore
    discriminator.compile(optimizer="adam", loss=custom_loss)

    # Crea il modello GAN combinato------------------------------------------------
    discriminator.trainable = False  # Imposta il discriminatore come non addestrabile nel modello GAN combinato
    gan_input = Input(shape=(num_eigenvalues,))
    config_input = Input(shape=(1,))  # Input aggiuntivo per le configurazioni
    position_input = Input(shape=(1,))  # Posizione come input aggiuntivo
    # ---------------------------------------------------------------

    gen_output = generator([gan_input, config_input, position_input])
    gan_output = discriminator(gen_output)

    gan_model = Model([gan_input, config_input, position_input], gan_output)
    gan_model.compile(optimizer="adam", loss=custom_loss)
    gan_model.summary()

    d_loss_list = []
    g_loss_list = []

    for epoch in range(EPOCHS):
        i = 0
        for start_idx in range(0, NUM_CONFIGURATIONS, BATCH_SIZE):
            i += 1
            end_idx = min(start_idx + BATCH_SIZE, NUM_CONFIGURATIONS + 1)
            # real_data = data[start_idx:end_idx, :num_eigenvalues]
            batch_data = triplette[start_idx:end_idx]
            batch_data = np.array(batch_data)

            real_labels = np.ones((batch_data.shape[0], 1))

            # Selezione casuale dei batch di dati reali e degli indici di configurazione corrispondenti
            # real_data = data[indices, :num_eigenvalues]
            # batch_config_indices = config_indices[indices]

            # Generazione di dati falsi
            noise = np.random.normal(0, 1, (batch_data.shape[0], num_eigenvalues))
            batch_configs = batch_data[:, 1].reshape(-1, 1)
            batch_positions = batch_data[:, 2].reshape(-1, 1)
            # print("config", batch_configs)
            # print("pos", batch_positions)
            fake_labels = np.zeros((batch_data.shape[0], 1))

            generated_data = generator.predict([noise, batch_configs, batch_positions])

            # Addestramento del discriminatore

            d_loss_real = discriminator.train_on_batch(batch_data, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            # Addestramento del generatore (attraverso il modello GAN combinato)
            g_loss = gan_model.train_on_batch(
                [noise, batch_configs, batch_positions], real_labels
            )
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)
        # Stampa le statistiche di addestramento ogni epoca
        print(
            f"Epoch: {epoch}, D_Loss: {(d_loss_real + d_loss_fake)/2}, G_Loss: {g_loss}"
        )

    # Salva i modelli
    generator.save("generator.h5")
    discriminator.save("discriminator.h5")

    def show_loss():
        plt.figure()
        plt.plot(g_loss_list, label="g_loss", marker="^")
        plt.plot(d_loss_list, label="d_loss", marker="+")
        plt.show()

    show_loss()


# Nota: Assicurati di definire le funzioni generator_model e discriminator_model,
# che dovrebbero creare e restituire i modelli del generatore e del discriminatore, rispettivamente.
# Inoltre, ricorda di adattare le parti di codice come la reinizializzazione dei modelli per trasferire i pesi
# e l'aggiustamento delle dimensioni di input/output per i tuoi specifici casi d'uso.


# if __name__ == "__main__":
