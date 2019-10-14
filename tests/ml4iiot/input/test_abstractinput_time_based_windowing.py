import unittest
from tests.ml4iiot.input.test_abstractinput_count_based_windowing import TestInput


class TestAbstractInputWithResample(unittest.TestCase):

    def create_test_input(self, window_size, stride_size, resample, test_data, batch_size=1):
        config = {
            'windowing_strategy': {
                'class': 'ml4iiot.input.windowing.timebased.TimeBasedWindowingStrategy',
                'config': {
                    'batch_size': batch_size,
                    'window_size': window_size,
                    'stride_size': stride_size,
                    'resample': resample,
                }
            },
            'datetime_column': 'datetime',
            'test_data': test_data,
        }

        return TestInput(config)

    def test_window_size(self):
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
        ]
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'ffill',
        }
        test_input_1 = self.create_test_input('10s', '1s', resample, test_data)
        batch_1 = test_input_1.next_window()
        test_input_2 = self.create_test_input('20s', '1s', resample, test_data)
        batch_2 = test_input_2.next_window()
        test_input_3 = self.create_test_input('40s', '1s', resample, test_data)
        batch_3 = test_input_3.next_window()

        self.assertEqual(2, len(batch_1))
        self.assertEqual(3, len(batch_2))
        self.assertEqual(5, len(batch_3))

    def test_end_of_iter(self):
        window_size = '10s'
        stride_size = '10s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'ffill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
        ]
        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        test_input.next_window()

        self.assertRaises(StopIteration, test_input.next_window)

    def test_stride_smaller_than_window(self):
        window_size = '30s'
        stride_size = '10s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'ffill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(4, len(batch))
        self.assertEqual([0, 1, 2, 3], batch['value'].tolist())

        batch = test_input.next_window()

        self.assertEqual(4, len(batch))
        self.assertEqual([1, 2, 3, 4], batch['value'].tolist())

        batch = test_input.next_window()

        self.assertEqual(4, len(batch))
        self.assertEqual([2, 3, 4, 5], batch['value'].tolist())

    def test_stride_equals_window(self):
        window_size = '20s'
        stride_size = '20s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'ffill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(3, len(batch))
        self.assertEqual([0, 1, 2], batch['value'].tolist())

        batch = test_input.next_window()

        self.assertEqual(3, len(batch))
        self.assertEqual([2, 3, 4], batch['value'].tolist())

        batch = test_input.next_window()

        self.assertEqual(3, len(batch))
        self.assertEqual([4, 5, 6], batch['value'].tolist())

    def test_stride_bigger_than_window(self):
        window_size = '10s'
        stride_size = '30s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'ffill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 50, 'value': 5},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(2, len(batch))
        self.assertEqual([0, 1], batch['value'].tolist())

        batch = test_input.next_window()

        self.assertEqual(2, len(batch))
        self.assertEqual([3, 4], batch['value'].tolist())

        batch = test_input.next_window()

        self.assertEqual(2, len(batch))
        self.assertEqual([6, 7], batch['value'].tolist())

    def test_resample_ffill(self):
        window_size = '50s'
        stride_size = '20s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'ffill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 30, 'value': 3},
            {'datetime': 50, 'value': 5},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(6, len(batch))
        self.assertEqual([0, 0, 0, 3, 3, 5], batch['value'].tolist())

    def test_resample_bfill(self):
        window_size = '50s'
        stride_size = '20s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'bfill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 30, 'value': 3},
            {'datetime': 50, 'value': 5},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(6, len(batch))
        self.assertEqual([0, 3, 3, 3, 5, 5], batch['value'].tolist())

    def test_resample_interpolate_linear(self):
        window_size = '100s'
        stride_size = '10s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'interpolate',
            'interpolation_method': 'linear',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 40, 'value': 4},
            {'datetime': 100, 'value': 7},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(11, len(batch))
        self.assertEqual([0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7], batch['value'].tolist())

    def test_resample_fill_value(self):
        window_size = '100s'
        stride_size = '10s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'fill_value',
            'fill_value': -1,
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 40, 'value': 4},
            {'datetime': 100, 'value': 7},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(11, len(batch))
        self.assertEqual([0, -1, -1, -1, 4, -1, -1, -1, -1, -1, 7], batch['value'].tolist())

    def test_upsampling(self):
        window_size = '50s'
        stride_size = '10s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '5s',
            'method': 'ffill',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 50, 'value': 5},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(11, len(batch))
        self.assertEqual([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], list(map(lambda element: element.timestamp(), batch.index.tolist())))
        self.assertEqual([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 5], batch['value'].tolist())

    def test_downsampling(self):
        window_size = '60s'
        stride_size = '10s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '20s',
            'method': 'interpolate',
            'interpolation_method': 'linear',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 60, 'value': 6},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch = test_input.next_window()

        self.assertEqual(4, len(batch))
        self.assertEqual([0, 20, 40, 60], list(map(lambda element: element.timestamp(), batch.index.tolist())))
        self.assertEqual([0, 2, 4, 6], batch['value'].tolist())

    def test_long_gap(self):
        window_size = '40s'
        stride_size = '20s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '10s',
            'method': 'interpolate',
            'interpolation_method': 'linear',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 100, 'value': 10},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch_1 = test_input.next_window()

        self.assertEqual(5, len(batch_1))
        self.assertEqual([0, 10, 20, 30, 40], list(map(lambda element: element.timestamp(), batch_1.index.tolist())))
        self.assertEqual([0, 1, 2, 3, 4], batch_1['value'].tolist())

        batch_2 = test_input.next_window()

        self.assertEqual(5, len(batch_2))
        self.assertEqual([20, 30, 40, 50, 60], list(map(lambda element: element.timestamp(), batch_2.index.tolist())))
        self.assertEqual([2, 3, 4, 5, 6], batch_2['value'].tolist())

        batch_3 = test_input.next_window()

        self.assertEqual(5, len(batch_3))
        self.assertEqual([40, 50, 60, 70, 80], list(map(lambda element: element.timestamp(), batch_3.index.tolist())))
        self.assertEqual([4, 5, 6, 7, 8], batch_3['value'].tolist())

        batch_4 = test_input.next_window()

        self.assertEqual(5, len(batch_4))
        self.assertEqual([60, 70, 80, 90, 100], list(map(lambda element: element.timestamp(), batch_4.index.tolist())))
        self.assertEqual([6, 7, 8, 9, 10], batch_4['value'].tolist())

    def test_full_example(self):
        window_size = '40s'
        stride_size = '20s'
        resample = {
            'enabled': True,
            'target_sampling_rate': '5s',
            'method': 'interpolate',
            'interpolation_method': 'linear',
        }
        test_data = [
            {'datetime': 0, 'value': 0},
            {'datetime': 10, 'value': 1},
            {'datetime': 20, 'value': 2},
            {'datetime': 30, 'value': 3},
            {'datetime': 40, 'value': 4},
            {'datetime': 60, 'value': 6},
            {'datetime': 70, 'value': 7},
            {'datetime': 100, 'value': 10},
        ]

        test_input = self.create_test_input(window_size, stride_size, resample, test_data)

        batch_1 = test_input.next_window()

        self.assertEqual(9, len(batch_1))
        self.assertEqual([0, 5, 10, 15, 20, 25, 30, 35, 40], list(map(lambda element: element.timestamp(), batch_1.index.tolist())))
        self.assertEqual([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], batch_1['value'].tolist())

        batch_2 = test_input.next_window()

        self.assertEqual(9, len(batch_2))
        self.assertEqual([20, 25, 30, 35, 40, 45, 50, 55, 60], list(map(lambda element: element.timestamp(), batch_2.index.tolist())))
        self.assertEqual([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], batch_2['value'].tolist())

        batch_3 = test_input.next_window()

        self.assertEqual(9, len(batch_3))
        self.assertEqual([40, 45, 50, 55, 60, 65, 70, 75, 80], list(map(lambda element: element.timestamp(), batch_3.index.tolist())))
        self.assertEqual([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8], batch_3['value'].tolist())

        batch_4 = test_input.next_window()

        self.assertEqual(9, len(batch_4))
        self.assertEqual([60, 65, 70, 75, 80, 85, 90, 95, 100], list(map(lambda element: element.timestamp(), batch_4.index.tolist())))
        self.assertEqual([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10], batch_4['value'].tolist())

        self.assertRaises(StopIteration, test_input.next_window)


if __name__ == '__main__':
    unittest.main()
