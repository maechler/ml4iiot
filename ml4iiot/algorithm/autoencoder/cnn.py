from keras import Input, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from pandas import DataFrame
import numpy as np
from ml4iiot.algorithm.autoencoder.abstractautoencoder import AbstractAutoencoder


class AbstractCNNAutoencoder(AbstractAutoencoder):

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dimension = self.get_config('input_dimension', default=36)

    def data_frame_to_input_data(self, data_frame: DataFrame):
        input_data = np.array([data_frame[self.get_config('input')].values])

        return input_data.reshape((1, self.input_dimension, 1))


class CNNAutoencoder(AbstractCNNAutoencoder):

    def create_autoencoder_model(self):
        input_layer = Input(shape=(self.input_dimension, 1))

        layer = Conv1D(16, 4, activation='relu', padding='same')(input_layer)
        layer = MaxPooling1D(4, padding='same')(layer)
        layer = Conv1D(16, 4, activation='relu', padding='same')(layer)

        encoder = Model(input_layer, layer)

        layer = Conv1D(16, 4, activation='relu', padding='same')(layer)
        layer = UpSampling1D(4)(layer)
        layer = Conv1D(16, 4, activation='relu', padding='same')(layer)

        decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(layer)

        return Model(input_layer, decoded)


class BottleneckCNNAutoencoder(AbstractCNNAutoencoder):

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dimension = self.get_config('input_dimension', default=128)

    def create_autoencoder_model(self):
        input_layer = Input(shape=(self.input_dimension, 1))

        layer = Conv1D(16, 4, activation='relu', padding='same')(input_layer)
        layer = MaxPooling1D(2, padding='same')(layer)
        layer = Conv1D(32, 4, activation='relu', padding='same')(layer)

        layer = Flatten()(layer)
        layer = Dense(1536)(layer)
        layer = Dense(2048)(layer)

        encoder = Model(input_layer, layer)

        layer = Reshape((64, 32))(layer)

        layer = Conv1D(32, 4, activation='relu', padding='same')(layer)
        layer = UpSampling1D(2)(layer)
        layer = Conv1D(16, 4, activation='relu', padding='same')(layer)

        decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(layer)

        return Model(input_layer, decoded)