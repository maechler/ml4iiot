from pandas import DataFrame
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm


class VoidAlgorithm(AbstractAlgorithm):
    def process(self, data_frame: DataFrame) -> None:
        pass
