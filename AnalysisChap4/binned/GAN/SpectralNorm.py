import tensorflow as tf
from keras.layers import Layer, Dense


class SpectralNormalization(Layer):
    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(SpectralNormalization, self).build()

    def call(self, inputs):
        weights = self.layer.kernel
        u, v = None, None  # You can initialize u and v here based on your needs

        # Power iteration for approximation of spectral norm
        for _ in range(1):  # Number of power iterations, 1 is often enough
            v = tf.math.l2_normalize(tf.matmul(weights, u, transpose_a=True))
            u = tf.math.l2_normalize(tf.matmul(weights, v))

        sigma = tf.matmul(tf.matmul(v, weights), u, transpose_a=True)
        self.layer.kernel = weights / sigma

        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
