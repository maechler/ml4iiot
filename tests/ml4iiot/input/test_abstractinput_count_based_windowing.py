from ml4iiot.input.abstractinput import AbstractInput
import unittest
import pandas as pd
from datetime import datetime


class TestInput(AbstractInput):
    test_data_iter = iter([])

    def __init__(self, config):
        super().__init__(config)

        self.test_data = self.get_config('test_data')
        self.test_data_iter = iter(self.test_data)

    def next_data_frame(self, batch_size=None):
        data_frame = self.data_row_to_data_frame(next(self.test_data_iter))

        for i in range(0, batch_size - 1):
            data_frame = data_frame.append(self.data_row_to_data_frame(next(self.test_data_iter)))

        return data_frame

    def data_row_to_data_frame(self, data_row):
        datetime_column = self.get_config('index_column')
        datetime_column_value = data_row[datetime_column]
        datetime_column_object = datetime.utcfromtimestamp(datetime_column_value)

        return pd.DataFrame(
            {datetime_column: [datetime_column_object], 'value': [data_row['value']]},
            columns=[datetime_column, 'value']
        ).set_index(datetime_column)


class TestAbstractInputWithoutResample(unittest.TestCase):

    def create_test_input(self, window_size, stride_size, test_data, batch_size=1):
        config = {
            'windowing_strategy': {
                'class': 'ml4iiot.input.windowing.CountBasedWindowingStrategy',
                'config': {
                    'batch_size': batch_size,
                    'window_size': window_size,
                    'stride_size': stride_size,
                }
            },
            'test_data': test_data,
            'index_column': 'datetime',
        }

        return TestInput(config)

    def test_window_size(self):
        test_data = [
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
            {'datetime': 80, 'value': 8},
            {'datetime': 90, 'value': 9},
        ]

        test_input_1 = self.create_test_input(1, 1, test_data)
        batch_1 = next(test_input_1)
        test_input_2 = self.create_test_input(2, 1, test_data)
        batch_2 = next(test_input_2)
        test_input_9 = self.create_test_input(9, 1, test_data)
        batch_9 = next(test_input_9)

        self.assertEqual(1, len(batch_1))
        self.assertEqual(2, len(batch_2))
        self.assertEqual(9, len(batch_9))

    def test_stride_smaller_than_window(self):
        window_size = 2
        stride_size = 1
        test_data = [
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
            {'datetime': 80, 'value': 8},
            {'datetime': 90, 'value': 9},
        ]

        test_input = self.create_test_input(window_size, stride_size, test_data)

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1, 2], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([2, 3], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([3, 4], batch['value'].tolist())

    def test_stride_equals_window(self):
        window_size = 2
        stride_size = 2
        test_data = [
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
            {'datetime': 80, 'value': 8},
            {'datetime': 90, 'value': 9},
        ]

        test_input = self.create_test_input(window_size, stride_size, test_data)

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1, 2], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([3, 4], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([5, 6], batch['value'].tolist())

    def test_stride_bigger_than_window(self):
        window_size = 1
        stride_size = 3
        test_data = [
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
            {'datetime': 80, 'value': 8},
            {'datetime': 90, 'value': 9},
        ]

        test_input = self.create_test_input(window_size, stride_size, test_data)

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([4], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([7], batch['value'].tolist())

    def test_end_of_iter(self):
        window_size = 2
        stride_size = 1
        test_data = [
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
        ]
        test_input = self.create_test_input(window_size, stride_size, test_data)

        next(test_input)

        self.assertRaises(StopIteration, test_input.__next__)

    def test_bigger_batch_size(self):
        window_size = 1
        stride_size = 3
        batch_size = 2
        test_data = [
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
            {'datetime': 80, 'value': 8},
            {'datetime': 90, 'value': 9},
        ]

        test_input = self.create_test_input(window_size, stride_size, test_data, batch_size)

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([4], batch['value'].tolist())

        batch = next(test_input)

        self.assertEqual(window_size, len(batch))
        self.assertEqual([7], batch['value'].tolist())


if __name__ == '__main__':
    unittest.main()
