from ml4iiot.pipeline.step import AbstractStep
import pandas as pd


class AbstractAlgorithm(AbstractStep):

    def __init__(self, config):
        super().__init__(config)

        pd.options.mode.chained_assignment = None
