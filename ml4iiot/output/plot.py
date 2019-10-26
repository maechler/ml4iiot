from datetime import datetime
from matplotlib.path import Path
from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pandas.plotting import register_matplotlib_converters
from ml4iiot.utility import str2bool, get_absolute_path, get_recursive_config


class PlotOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.format = self.get_config('format', default='svg')
        self.show_plots = str2bool(self.get_config('show_plots', default=True))
        self.save_to_image = str2bool(self.get_config('save_to_image', default=False))
        self.save_to_pickle = str2bool(self.get_config('save_to_pickle', default=False))
        self.save_path = self.get_config('save_path', default='./out/')

        self.columns_to_plot = []
        self.accumulated_data_frame = None

        for figure_config in self.get_config('figures'):
            for plot_config in figure_config['plots']:
                self.columns_to_plot.append(plot_config['column'])

    def process(self, data_frame: DataFrame) -> None:
        data_frame_copy = data_frame.copy()

        if self.accumulated_data_frame is None:
            self.accumulated_data_frame = pd.DataFrame(index=data_frame_copy.index)

        for column_name, column_data in data_frame_copy.iteritems():
            if column_name not in self.columns_to_plot:
                del data_frame_copy[column_name]

        self.accumulated_data_frame = self.accumulated_data_frame.combine_first(data_frame_copy)

    def destroy(self) -> None:
        super().destroy()
        register_matplotlib_converters()

        for figure_config in self.get_config('figures'):
            fig, ax = plt.subplots()

            for plot_config in figure_config['plots']:
                sanitized_column = self.accumulated_data_frame[plot_config['column']].dropna()

                ax.plot(
                    sanitized_column.index,
                    sanitized_column.values,
                    color=get_recursive_config(plot_config, 'color', default='#2A638C'),
                    label=plot_config['column'],
                    linestyle=get_recursive_config(plot_config, 'linestyle', default='solid'),
                    marker=get_recursive_config(plot_config, 'marker', default=None)
                )

            plt.legend(loc='best')
            fig.autofmt_xdate()

            if 'xlabel' in figure_config:
                ax.set_xlabel(figure_config['xlabel'])

            if 'ylabel' in figure_config:
                ax.set_ylabel(figure_config['ylabel'])

            if 'title' in figure_config:
                ax.set_title(figure_config['title'])

            if self.save_to_image:
                image_save_path = self.get_save_path_from_figure_config(figure_config, self.format)
                fig.savefig(image_save_path, format=self.format)

            if self.save_to_pickle:
                pickle_save_path = self.get_save_path_from_figure_config(figure_config, 'pickle')

                with open(str(pickle_save_path), 'wb') as pickle_file:
                    pickle.dump(fig, pickle_file)

        if self.show_plots:
            plt.show()

    def get_save_path_from_figure_config(self, figure_config: dict, file_extension: str = '') -> Path:
        file_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        for plot_config in figure_config['plots']:
            file_name += '_' + plot_config['column']

        figure_path = self.save_path + file_name + '.' + file_extension

        return get_absolute_path(figure_path)
