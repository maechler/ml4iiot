from abc import abstractmethod
import numpy as np
from ml4iiot.algorithms.abstractalgorithm import AbstractAlgorithm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ml4iiot.utility import str2bool


class AbstractAutoencoder(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        pd.options.mode.chained_assignment = None

        self.verbose = str2bool(self.get_config('verbose', default=False))
        self.autoencoder = self.create_autoencoder_model()
        self.scaler = self.create_scaler()

        self.autoencoder.compile(
            optimizer=self.get_config('optimizer', default='adam'),
            loss=self.get_config('loss', default='binary_crossentropy')
        )

        if self.verbose:
            self.autoencoder.summary()

    @abstractmethod
    def create_autoencoder_model(self):
        pass

    def create_scaler(self):
        return MinMaxScaler(feature_range=(0, 1))

    def do_fit(self, input_frame, reconstruction, reconstruction_error):
        return True

    def compute(self, input_frame):
        input = input_frame[[self.get_config('column')]]
        normalized_input = np.array([self.scaler.fit_transform(input)[:, 0]])

        normalized_reconstruction = self.autoencoder.predict(normalized_input)
        reconstruction_difference = normalized_reconstruction - normalized_input
        reconstruction_error = sum(abs(reconstruction_difference[0]))
        reconstruction = self.scaler.inverse_transform(normalized_reconstruction)[0]

        data_frame = pd.DataFrame(index=input_frame.index, columns=['reconstruction', 'absolute_reconstruction_error', 'relative_reconstruction_error', 'do_fit'])
        data_frame['absolute_reconstruction_error'][input_frame.index[-1]] = reconstruction_error
        data_frame['relative_reconstruction_error'][input_frame.index[-1]] = reconstruction_error / len(input_frame.index)
        data_frame['reconstruction'] = reconstruction

        if self.do_fit(input_frame, reconstruction, reconstruction_error):
            data_frame['do_fit'][input_frame.index[-1]] = 1
            self.autoencoder.fit(normalized_input, normalized_input, batch_size=1, epochs=1, verbose=self.verbose)
        else:
            data_frame['do_fit'][input_frame.index[-1]] = 0

        return data_frame
