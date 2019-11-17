from keras import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from pandas import DataFrame
import numpy as np
from ml4iiot.algorithm.autoencoder.abstractautoencoder import AbstractAutoencoder


class LSTMAutoencoder(AbstractAutoencoder):

    def __init__(self, config: dict):
        super().__init__(config)

        self.features = self.get_config('features')
        self.features_len = len(self.features)
        self.time_steps = self.get_config('time_steps')

    def create_autoencoder_model(self):
        model = Sequential()

        model.add(LSTM(units=90, activation='relu', input_shape=(self.time_steps, self.features_len), return_sequences=True))
        model.add(LSTM(60, activation='relu', return_sequences=False))
        model.add(RepeatVector(self.time_steps))
        model.add(LSTM(60, activation='relu', return_sequences=True))
        model.add(LSTM(90, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))

        return model

    def data_frame_to_input_data(self, data_frame: DataFrame):
        input_data = np.array([])

        for feature in self.features:
            input_data = np.append(input_data, data_frame[feature].values)

        return input_data.reshape(1, self.time_steps, self.features_len)
