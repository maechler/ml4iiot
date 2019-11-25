import os
from datetime import datetime
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
            x_axis_formatter = get_recursive_config(figure_config, 'x_axis_formatter', default='datetime')
            start_datetime = get_recursive_config(figure_config, 'start_datetime', default=None)
            end_datetime = get_recursive_config(figure_config, 'end_datetime', default=None)

            for plot_config in figure_config['plots']:
                if not plot_config['column'] in self.accumulated_data_frame:
                    continue

                sanitized_column = self.accumulated_data_frame.loc[start_datetime:end_datetime][plot_config['column']].dropna()
                plot_type = get_recursive_config(plot_config, 'type', default='line')

                if plot_type == 'line':
                    ax.plot(
                        sanitized_column.index if x_axis_formatter == 'datetime' else range(0, len(sanitized_column.index)),
                        sanitized_column.values,
                        color=get_recursive_config(plot_config, 'color', default='#2A638C'),
                        label=plot_config['label'] if 'label' in plot_config else plot_config['column'],
                        linestyle=get_recursive_config(plot_config, 'linestyle', default='solid'),
                        marker=get_recursive_config(plot_config, 'marker', default=None)
                    )
                elif plot_type == 'histogram':
                    ax.hist(
                        sanitized_column.values,
                        color=get_recursive_config(plot_config, 'color', default='#2A638C'),
                        bins=get_recursive_config(plot_config, 'bins', default=40),
                        range=[get_recursive_config(plot_config, 'range', 'min', default=None), get_recursive_config(plot_config, 'range', 'max', default=None)]
                    )

            if 'xlabel' in figure_config:
                ax.set_xlabel(figure_config['xlabel'], labelpad=10)

            if 'ylabel' in figure_config:
                ax.set_ylabel(figure_config['ylabel'], labelpad=10)

            if 'title' in figure_config:
                ax.set_title(figure_config['title'])

            if 'ylim' in figure_config:
                ax.set_ylim([figure_config['ylim']['min'], figure_config['ylim']['max']])

            if 'xlim' in figure_config:
                ax.set_xlim([figure_config['xlim']['min'], figure_config['xlim']['max']])

            plt.tight_layout(0)
            plt.legend(loc='best')
            fig.autofmt_xdate()

            if self.save_to_image:
                image_save_path = self.get_save_path_from_figure_config(figure_config, self.format)
                fig.savefig(image_save_path, format=self.format)

            if self.save_to_pickle:
                pickle_save_path = self.get_save_path_from_figure_config(figure_config, 'pickle')

                with open(str(pickle_save_path), 'wb') as pickle_file:
                    pickle.dump(fig, pickle_file)

        if self.show_plots:
            plt.show()

    def get_save_path_from_figure_config(self, figure_config: dict, file_extension: str = '') -> str:
        folder_name = datetime.now().strftime('%Y_%m_%d')
        file_name = datetime.now().strftime('%Y_%m_%d_%H_%M')
        save_path = os.path.join(str(get_absolute_path(self.save_path)), folder_name)
        i = 1

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for plot_config in figure_config['plots']:
            file_name += '_' + plot_config['column']

        file_path = os.path.join(save_path, file_name + '.' + file_extension)

        while os.path.exists(file_path):
            file_path = os.path.join(save_path, file_name + '_' + str(i) + '.' + file_extension)
            i = i + 1

        return file_path
