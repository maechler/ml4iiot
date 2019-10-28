from abc import abstractmethod
import numpy as np
from keras import Model
from pandas import DataFrame
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm
from ml4iiot.utility import str2bool


class AbstractAutoencoder(AbstractAlgorithm):

    def __init__(self, config: dict):
        super().__init__(config)

        self.verbose = str2bool(self.get_config('verbose', default=False))
        self.batch_size = self.get_config('batch_size', default=1)
        self.epochs = self.get_config('epochs', default=1)
        self.next_batch = []
        self.autoencoder = self.create_autoencoder_model()

        self.autoencoder.compile(
            optimizer=self.get_config('optimizer', default='adam'),
            loss=self.get_config('loss', default='binary_crossentropy')
        )

        if self.verbose:
            self.autoencoder.summary()

    @abstractmethod
    def create_autoencoder_model(self) -> Model:
        pass

    def process(self, data_frame: DataFrame) -> None:
        normalized_input = np.array([data_frame[self.get_config('input')]])
        data_frame['do_fit'] = float('nan')
        data_frame['do_predict'] = float('nan')

        if self.do_predict(data_frame):
            data_frame['do_predict'].iloc[-1] = 1

            normalized_reconstruction = self.autoencoder.predict(normalized_input)
            normalized_reconstruction_difference = normalized_reconstruction - normalized_input
            normalized_reconstruction_error = sum(abs(normalized_reconstruction_difference[0]))

            data_frame['reconstruction'] = normalized_reconstruction[0]
            data_frame['absolute_reconstruction_error'] = float('nan')
            data_frame['relative_reconstruction_error'] = float('nan')

            data_frame['absolute_reconstruction_error'].iloc[-1] = normalized_reconstruction_error
            data_frame['relative_reconstruction_error'].iloc[-1] = normalized_reconstruction_error / len(data_frame.index)
        else:
            data_frame['do_predict'].iloc[-1] = 0

        if self.do_fit(data_frame):
            data_frame['do_fit'].iloc[-1] = 1
            self.next_batch.append(normalized_input[0])

            if len(self.next_batch) >= self.batch_size:
                arr = np.array(self.next_batch)
                self.autoencoder.fit(arr, arr, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
                self.next_batch = []
        else:
            data_frame['do_fit'].iloc[-1] = 0
