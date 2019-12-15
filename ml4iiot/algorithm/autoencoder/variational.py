from keras import Input, Model
from keras.layers import Dense, Lambda
from keras import backend as K
from keras.losses import binary_crossentropy

from ml4iiot.algorithm.autoencoder.abstractautoencoder import AbstractAutoencoder


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(AbstractAutoencoder):

    def create_autoencoder_model(self) -> Model:
        input_size = self.get_config('input_dimension', default=90)
        bottleneck_size = self.get_config('bottleneck_size', default=60)
        latent_size = self.get_config('latent_size', default=2)
        input_shape = (input_size,)

        input_layer = Input(shape=input_shape)
        encoder_bottleneck_layer = Dense(bottleneck_size, activation='relu')(input_layer)

        z_mean = Dense(latent_size)(encoder_bottleneck_layer)
        z_log_var = Dense(latent_size)(encoder_bottleneck_layer)
        z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_log_var])

        encoder = Model(input_layer, [z_mean, z_log_var, z])

        latent_input_layer = Input(shape=(latent_size,))
        decoder_bottleneck_layer = Dense(bottleneck_size, activation='relu')(latent_input_layer)
        output_layer = Dense(input_size, activation='sigmoid')(decoder_bottleneck_layer)

        decoder = Model(latent_input_layer, output_layer)
        output_layer = decoder(encoder(input_layer)[2])

        vae = Model(input_layer, output_layer)

        def loss(x, x_decoded_mean):
            xent_loss = binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        vae.compile(
            optimizer=self.get_config('optimizer', default='adam'),
            loss=loss
        )

        return vae

    def compile_autoencoder(self):
        pass  # Already compiled in create_autoencoder_model
