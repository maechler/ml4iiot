import pandas as pd
import argparse
import matplotlib.pyplot as plt


def run(csv_file_path: str, index_column: str, columns) -> None:
    ax = plt.gca()

    df = pd.read_csv(csv_file_path)
    df.set_index(index_column)

    for column in columns:
        df[column].fillna(method='ffill', inplace=True)
        df.plot(kind='line', y=column, ax=ax)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--csv_file_path', help='Path to a csv file.', type=str, required=True)
    parser.add_argument('-i', '--index_column', help='The index column used for the x axis.', type=str, default='time')
    parser.add_argument('columns', nargs='+')

    args = parser.parse_args()

    run(args.csv_file_path, args.index_column, args.columns)
