from keras import Input, Model
from keras.layers import Dense
from ml4iiot.algorithm.autoencoder.abstractautoencoder import AbstractAutoencoder


class FullyConnectedAutoencoder(AbstractAutoencoder):

    def create_autoencoder_model(self) -> Model:
        layer_config = self.get_config('layer')
        input_layer_config = layer_config[0]
        output_layer_config = layer_config[-1]
        input_layer = Input(shape=(input_layer_config['dimension'],), name='input_layer')
        layer = input_layer
        layer_len = len(layer_config)

        for index, layer_config in enumerate(layer_config):
            if index == 0 or index == layer_len - 1:
                continue  # Skip first and last layer

            layer = Dense(layer_config['dimension'], activation='relu', name='hidden_layer_' + str(index))(layer)

        output_layer = Dense(output_layer_config['dimension'], activation='sigmoid', name='output_layer')(layer)
        autoencoder = Model(input_layer, output_layer)

        return autoencoder
