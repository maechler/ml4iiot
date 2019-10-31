from ml4iiot.preprocessing.normalization.minmaxscaler import MinMaxScaler as PrePorcessingMinMaxScaler


class MinMaxScaler(PrePorcessingMinMaxScaler):
    def __init__(self, config: dict):
        super().__init__(config)

        self.mode = 'denormalize'
