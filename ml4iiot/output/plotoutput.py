from datetime import datetime
from ml4iiot.output.abstractoutput import AbstractOutput
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from ml4iiot.utility import str2bool, get_absolute_path


class PlotOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.figures = []

        for figure_config in self.get_config('figures'):
            figure = {'config': figure_config, 'plots': []}

            for plot_config in figure_config['plots']:
                figure['plots'].append({'config': plot_config, 'data': {}})

            self.figures.append(figure)

    def open(self):
        super().open()

    def emit(self, input_frame, output_frame):
        for figure in self.figures:
            for plot in figure['plots']:
                source = input_frame if plot['config']['source'] == 'input' else output_frame

                plot['data'].update(source[plot['config']['column']].dropna().to_dict())

    def close(self):
        super().close()
        register_matplotlib_converters()

        for figure in self.figures:
            fig = plt.figure()

            for plot in figure['plots']:
                plt.plot(
                    plot['data'].keys(),
                    plot['data'].values(),
                    color=plot['config']['color'],
                    label=plot['config']['column'],
                    linestyle=plot['config']['linestyle']
                )

            plt.legend(loc='best')
            plt.xticks(rotation=15)

            if str2bool(figure['config']['save_figure']):
                fig.savefig(self.get_save_path_from_figure_config(figure['config']), format=self.get_config('format'))

        if str2bool(self.get_config('show_plot')):
            plt.show()

    def get_save_path_from_figure_config(self, figure_config):
        file_name = ''

        for plot_config in figure_config['plots']:
            file_name += plot_config['column'] + '_'

        figure_path = self.get_config('save_path') + file_name + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.' + self.get_config('format')

        return get_absolute_path(figure_path)
