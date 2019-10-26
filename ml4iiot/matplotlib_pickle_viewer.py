import pickle
import argparse
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


def run(pickle_file_path: str) -> None:
    register_matplotlib_converters()

    with open(pickle_file_path, 'rb') as pickle_file:
        figure = pickle.load(pickle_file)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pickle_file_path', help='Path to a pickle file.', type=str, required=True)

    args = parser.parse_args()

    run(args.pickle_file_path)
