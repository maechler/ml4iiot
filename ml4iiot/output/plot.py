from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pandas.plotting import register_matplotlib_converters
from ml4iiot.utility import str2bool, get_recursive_config, get_cli_arguments, get_current_out_path

colors = {
    'red': '#D01431',
    'yellow': '#F2C430',
    'green': '#5DB64D',
    'grey': '#A6A5A1',
    'blue': '#2A638C',
}


class PlotOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.format = self.get_config('format', default='svg')
        self.dpi = self.get_config('dpi', default=None)
        self.show_plots = str2bool(self.get_config('show_plots', default=True))
        self.save_to_image = str2bool(self.get_config('save_to_image', default=False))
        self.save_to_pickle = str2bool(self.get_config('save_to_pickle', default=False))
        self.save_path = self.get_config('save_path', default='./out/')

        self.columns_to_plot = []
        self.accumulated_data_frame = None
        self.cli_arguments = get_cli_arguments()

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
            plt.rcParams.update({'font.size': get_recursive_config(figure_config, 'font_size', default=12)})

            fig, ax = plt.subplots()
            x_axis_formatter = get_recursive_config(figure_config, 'x_axis_formatter', default='datetime')
            start_datetime = get_recursive_config(figure_config, 'start_datetime', default=None)
            end_datetime = get_recursive_config(figure_config, 'end_datetime', default=None)

            for plot_config in figure_config['plots']:
                if not plot_config['column'] in self.accumulated_data_frame:
                    continue

                sanitized_column = self.accumulated_data_frame.loc[start_datetime:end_datetime][plot_config['column']].dropna()
                plot_type = get_recursive_config(plot_config, 'type', default='line')

                if 'label' in plot_config:
                    label = plot_config['label'] if plot_config['label'] != 'None' else None
                else:
                    label = plot_config['column']

                if plot_type == 'line':
                    ax.plot(
                        sanitized_column.index if x_axis_formatter == 'datetime' else list(map(lambda x: x.value, sanitized_column.index)),
                        sanitized_column.values,
                        color=self.get_color(get_recursive_config(plot_config, 'color', default=colors['blue'])),
                        label=label,
                        linestyle=get_recursive_config(plot_config, 'linestyle', default='solid'),
                        alpha=get_recursive_config(plot_config, 'alpha', default=1),
                        marker=get_recursive_config(plot_config, 'marker', default=None)
                    )
                elif plot_type == 'histogram':
                    histogram_range = None
                    range_min = get_recursive_config(plot_config, 'range', 'min', default=None)
                    range_max = get_recursive_config(plot_config, 'range', 'max', default=None)

                    if range_min or range_max:
                        histogram_range = [range_min, range_max]

                    ax.hist(
                        sanitized_column.values,
                        color=self.get_color(get_recursive_config(plot_config, 'color', default=colors['blue'])),
                        bins=get_recursive_config(plot_config, 'bins', default=40),
                        label=label,
                        histtype=get_recursive_config(plot_config, 'histtype', default='bar'),
                        alpha=get_recursive_config(plot_config, 'alpha', default=1),
                        range=histogram_range,
                    )

            if 'vline' in figure_config:
                for vline_config in figure_config['vline']:
                    color = self.get_color(get_recursive_config(vline_config, 'color', default=colors['red']))
                    linestyle = get_recursive_config(vline_config, 'linestyle', default='solid')
                    label = get_recursive_config(vline_config, 'label', default=None)

                    plt.axvline(x=vline_config['x'], color=color, linestyle=linestyle, label=label)

            if 'hline' in figure_config:
                for hline_config in figure_config['hline']:
                    color = self.get_color(get_recursive_config(hline_config, 'color', default=colors['red']))
                    linestyle = get_recursive_config(hline_config, 'linestyle', default='solid')
                    label = get_recursive_config(hline_config, 'label', default=None)

                    plt.axvline(y=hline_config['x'], color=color, linestyle=linestyle, label=label)

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

            plt.rcParams.update({'font.size': get_recursive_config(figure_config, 'font_size', default=12)})
            plt.tight_layout(0)
            plt.legend(loc=get_recursive_config(figure_config, 'legend_location', default='best'))
            fig.autofmt_xdate()

            if self.save_to_image:
                image_save_path = self.get_save_path_from_figure_config(figure_config, self.format)
                fig.savefig(image_save_path, format=self.format, dpi=self.dpi)

            if self.save_to_pickle:
                pickle_save_path = self.get_save_path_from_figure_config(figure_config, 'pickle')

                with open(str(pickle_save_path), 'wb') as pickle_file:
                    pickle.dump(fig, pickle_file)

        if self.show_plots:
            plt.show()

    def get_color(self, color: str):
        return color if color not in colors else colors[color]

    def get_save_path_from_figure_config(self, figure_config: dict, file_extension: str = '') -> str:
        file_name = ''

        for plot_config in figure_config['plots']:
            file_name += plot_config['column'] + '_'

        file_name = file_name.rstrip('_') + '.' + file_extension

        return get_current_out_path(file_name)
