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

    def next_data_frame(self, recommended_frame_size=None, recommended_end_timestamp=None):
        next_data_frame = next(self.test_data_iter)
        datetime_column = self.get_config('datetime_column')
        datetime_column_value = next_data_frame[datetime_column]
        datetime_column_object = datetime.utcfromtimestamp(datetime_column_value)

        return pd.DataFrame({
            datetime_column: [datetime_column_object],
            'value': [next_data_frame['value']]},
            columns=[datetime_column, 'value']
        ).set_index(datetime_column)


class TestAbstractInputWithoutResample(unittest.TestCase):

    def create_test_input(self, window_size, stride_size, test_data):
        config = {
            'window_size': window_size,
            'stride_size': stride_size,
            'test_data': test_data,
            'datetime_column': 'datetime',
            'resample': {
                'enabled': False
            }
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
        batch_1 = test_input_1.next_batch()
        test_input_2 = self.create_test_input(2, 1, test_data)
        batch_2 = test_input_2.next_batch()
        test_input_9 = self.create_test_input(9, 1, test_data)
        batch_9 = test_input_9.next_batch()

        self.assertEqual(1, len(batch_1))
        self.assertEqual(2, len(batch_2))
        self.assertEqual(9, len(batch_9))

    def test_stride_small_than_window(self):
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

        batch = test_input.next_batch()

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1, 2], batch['value'].tolist())

        batch = test_input.next_batch()

        self.assertEqual(window_size, len(batch))
        self.assertEqual([2, 3], batch['value'].tolist())

        batch = test_input.next_batch()

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

        batch = test_input.next_batch()

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1, 2], batch['value'].tolist())

        batch = test_input.next_batch()

        self.assertEqual(window_size, len(batch))
        self.assertEqual([3, 4], batch['value'].tolist())

        batch = test_input.next_batch()

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

        batch = test_input.next_batch()

        self.assertEqual(window_size, len(batch))
        self.assertEqual([1], batch['value'].tolist())

        batch = test_input.next_batch()

        self.assertEqual(window_size, len(batch))
        self.assertEqual([4], batch['value'].tolist())

        batch = test_input.next_batch()

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

        test_input.next_batch()

        self.assertRaises(StopIteration, test_input.next_batch)


if __name__ == '__main__':
    unittest.main()
