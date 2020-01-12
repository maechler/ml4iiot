from keras import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from pandas import DataFrame
import numpy as np
from ml4iiot.algorithm.autoencoder.abstractautoencoder import AbstractAutoencoder


class AbstractLSTMAutoencoder(AbstractAutoencoder):

    def __init__(self, config: dict):
        super().__init__(config)

        self.features = self.get_config('features')
        self.features_len = len(self.features)
        self.time_steps = self.get_config('time_steps', default=10)

    def data_frame_to_input_data(self, data_frame: DataFrame):
        input_data = np.array([])

        for feature in self.features:
            input_data = np.append(input_data, data_frame[feature].values)

        return input_data.reshape((1, self.time_steps, self.features_len))


class ReconstructionLSTMAutoencoder(AbstractLSTMAutoencoder):

    def create_autoencoder_model(self):
        model = Sequential()

        model.add(LSTM(units=50, activation='relu', input_shape=(self.time_steps, self.features_len), return_sequences=True))
        model.add(LSTM(35, activation='relu', return_sequences=False))
        model.add(RepeatVector(self.time_steps))
        model.add(LSTM(35, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(self.features_len)))

        return model


class PredictionLSTMAutoencoder(AbstractLSTMAutoencoder):

    def __init__(self, config: dict):
        super().__init__(config)

        self.number_of_units = self.get_config('number_of_units', default=50)
        self.number_of_layers = self.get_config('number_of_layers', default=11)

    def create_autoencoder_model(self):
        model = Sequential()

        for _ in range(self.number_of_layers):
            model.add(LSTM(self.number_of_units, input_shape=(self.time_steps, self.features_len), return_sequences=True))

        model.add(Dense(self.features_len))

        return model

    def fit(self):
        source = np.array(self.next_batch)
        target = np.array(self.next_batch)

        source = np.delete(source, -1, axis=0)  # Remove last sample as the target for this would be in the next batch
        target = np.delete(target, 0, axis=0)  # Remove first sample as the first value of the source should predict its next value

        return self.autoencoder.fit(source, target, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=self.shuffle)
