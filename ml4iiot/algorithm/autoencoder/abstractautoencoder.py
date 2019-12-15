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
        self.autoencoder = None

    def init(self) -> None:
        self.autoencoder = self.create_autoencoder_model()

        self.compile_autoencoder()

        if self.verbose:
            self.autoencoder.summary()

    @abstractmethod
    def create_autoencoder_model(self) -> Model:
        pass

    def compile_autoencoder(self):
        self.autoencoder.compile(
             optimizer=self.get_config('optimizer', default='adam'),
             loss=self.get_config('loss', default='binary_crossentropy')
        )

    def data_frame_to_input_data(self, data_frame: DataFrame):
        return np.array([data_frame[self.get_config('input')]])

    def process(self, data_frame: DataFrame) -> None:
        data_frame['do_fit'] = float('nan')
        data_frame['do_predict'] = float('nan')
        data_frame['loss'] = float('nan')
        input_data = self.data_frame_to_input_data(data_frame)

        if self.do_predict(data_frame):
            data_frame['do_predict'].iloc[-1] = 1

            reconstruction = self.autoencoder.predict(input_data)
            reconstruction_difference = reconstruction - input_data
            reconstruction_error = sum(abs(reconstruction_difference[0]))

            data_frame['reconstruction'] = reconstruction[0]
            data_frame['absolute_reconstruction_error'] = float('nan')
            data_frame['relative_reconstruction_error'] = float('nan')

            data_frame['absolute_reconstruction_error'].iloc[-1] = reconstruction_error
            data_frame['relative_reconstruction_error'].iloc[-1] = reconstruction_error / len(data_frame.index)
        else:
            data_frame['do_predict'].iloc[-1] = 0

        if self.do_fit(data_frame):
            data_frame['do_fit'].iloc[-1] = 1
            self.next_batch.append(input_data[0])

            if len(self.next_batch) >= self.batch_size:
                next_batch = np.array(self.next_batch)
                history = self.autoencoder.fit(next_batch, next_batch, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
                data_frame['loss'] = np.average(history.history['loss'])
                self.next_batch = []
        else:
            data_frame['do_fit'].iloc[-1] = 0
