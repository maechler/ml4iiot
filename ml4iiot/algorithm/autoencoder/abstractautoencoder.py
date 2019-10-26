from abc import abstractmethod
import numpy as np
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm
import pandas as pd
from ml4iiot.utility import str2bool


class AbstractAutoencoder(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        pd.options.mode.chained_assignment = None

        self.verbose = str2bool(self.get_config('verbose', default=False))
        self.autoencoder = self.create_autoencoder_model()

        self.autoencoder.compile(
            optimizer=self.get_config('optimizer', default='adam'),
            loss=self.get_config('loss', default='binary_crossentropy')
        )

        if self.verbose:
            self.autoencoder.summary()

    @abstractmethod
    def create_autoencoder_model(self):
        pass

    def do_fit(self, data_frame):
        return True

    def process(self, data_frame):
        normalized_input = np.array([data_frame[self.get_config('input')]])

        normalized_reconstruction = self.autoencoder.predict(normalized_input)
        normalized_reconstruction_difference = normalized_reconstruction - normalized_input
        normalized_reconstruction_error = sum(abs(normalized_reconstruction_difference[0]))

        data_frame['reconstruction'] = normalized_reconstruction[0]
        data_frame['absolute_reconstruction_error'] = float('nan')
        data_frame['relative_reconstruction_error'] = float('nan')
        data_frame['do_fit'] = float('nan')

        data_frame['absolute_reconstruction_error'].iloc[-1] = normalized_reconstruction_error
        data_frame['relative_reconstruction_error'].iloc[-1] = normalized_reconstruction_error / len(data_frame.index)

        if self.do_fit(data_frame):
            data_frame['do_fit'].iloc[-1] = 1
            self.autoencoder.fit(normalized_input, normalized_input, batch_size=1, epochs=1, verbose=self.verbose)
        else:
            data_frame['do_fit'].iloc[-1] = 0
