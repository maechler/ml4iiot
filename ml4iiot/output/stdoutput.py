from ml4iiot.output.abstractoutput import AbstractOutput


class StdOutput(AbstractOutput):

    def emit(self, input_frame, output_frame):
        print(output_frame)
