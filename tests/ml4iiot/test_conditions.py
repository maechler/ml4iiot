from datetime import datetime
import unittest
import pandas as pd
from ml4iiot.utility import instance_from_config


class TestConditions(unittest.TestCase):
    seconds_in_a_day = 60 * 60 * 24
    seconds_in_an_hour = 60 * 60

    def create_data_frame(self, timestamp):
        datetime_column_object = datetime.utcfromtimestamp(timestamp)

        return pd.DataFrame(
            {'index': [datetime_column_object]},
            columns=['index']
        ).set_index('index')

    def test_true_condition(self):
        condition = instance_from_config({
            'class': 'ml4iiot.conditions.TrueCondition'
        })

        self.assertTrue(condition.evaluate(None))

    def test_false_condition(self):
        condition = instance_from_config({
            'class': 'ml4iiot.conditions.FalseCondition'
        })

        self.assertFalse(condition.evaluate(None))

    def test_inverted_condition(self):
        condition_true = instance_from_config({
            'class': 'ml4iiot.conditions.FalseCondition',
            'config': {
                'inverted': True
            }
        })
        condition_false = instance_from_config({
            'class': 'ml4iiot.conditions.FalseCondition',
            'config': {
                'inverted': False
            }
        })

        self.assertTrue(condition_true.evaluate(None))
        self.assertFalse(condition_false.evaluate(None))

    def test_composite_condition(self):
        condition_and_true = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'and',
                'conditions': [
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                ]
            }
        })

        condition_and_false_1 = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'and',
                'conditions': [
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                ]
            }
        })

        condition_and_false_2 = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'and',
                'conditions': [
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                ]
            }
        })

        condition_and_false_3 = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'and',
                'conditions': [
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                ]
            }
        })

        condition_or_false = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'or',
                'conditions': [
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                ]
            }
        })

        condition_or_true_1 = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'or',
                'conditions': [
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                ]
            }
        })

        condition_or_true_2 = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'or',
                'conditions': [
                    {'class': 'ml4iiot.conditions.FalseCondition'},
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                ]
            }
        })

        condition_or_true_3 = instance_from_config({
            'class': 'ml4iiot.conditions.CompositeCondition',
            'config': {
                'operator': 'or',
                'conditions': [
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                    {'class': 'ml4iiot.conditions.TrueCondition'},
                ]
            }
        })

        self.assertTrue(condition_and_true.evaluate(None))
        self.assertFalse(condition_and_false_1.evaluate(None))
        self.assertFalse(condition_and_false_2.evaluate(None))
        self.assertFalse(condition_and_false_3.evaluate(None))

        self.assertFalse(condition_or_false.evaluate(None))
        self.assertTrue(condition_or_true_1.evaluate(None))
        self.assertTrue(condition_or_true_2.evaluate(None))
        self.assertTrue(condition_or_true_3.evaluate(None))

    def test_weekday_condition(self):
        weekday_condition = instance_from_config({
            'class': 'ml4iiot.conditions.WeekdayCondition',
            'config': {
                'weekdays': ['Monday', 'Tuesday']
            }
        })

        thursday = self.create_data_frame(self.seconds_in_a_day * 0)
        friday = self.create_data_frame(self.seconds_in_a_day * 1)
        saturday = self.create_data_frame(self.seconds_in_a_day * 2)
        sunday = self.create_data_frame(self.seconds_in_a_day * 3)
        monday = self.create_data_frame(self.seconds_in_a_day * 4)
        tuesday = self.create_data_frame(self.seconds_in_a_day * 5)
        wednesday = self.create_data_frame(self.seconds_in_a_day * 6)

        self.assertFalse(weekday_condition.evaluate(thursday))
        self.assertFalse(weekday_condition.evaluate(friday))
        self.assertFalse(weekday_condition.evaluate(saturday))
        self.assertFalse(weekday_condition.evaluate(sunday))
        self.assertFalse(weekday_condition.evaluate(wednesday))

        self.assertTrue(weekday_condition.evaluate(monday))
        self.assertTrue(weekday_condition.evaluate(tuesday))

    def test_daytime_condition(self):
        daytime_condition = instance_from_config({
            'class': 'ml4iiot.conditions.DaytimeCondition',
            'config': {
                'start_time': '08:00:00',
                'end_time': '20:00:00'
            }
        })

        before_time_window = self.create_data_frame(self.seconds_in_an_hour * 7)
        in_time_window_1 = self.create_data_frame(self.seconds_in_an_hour * 9)
        in_time_window_2 = self.create_data_frame(self.seconds_in_an_hour * 14)
        after_time_window = self.create_data_frame(self.seconds_in_an_hour * 22)

        self.assertFalse(daytime_condition.evaluate(before_time_window))
        self.assertTrue(daytime_condition.evaluate(in_time_window_1))
        self.assertTrue(daytime_condition.evaluate(in_time_window_2))
        self.assertFalse(daytime_condition.evaluate(after_time_window))


if __name__ == '__main__':
    unittest.main()
