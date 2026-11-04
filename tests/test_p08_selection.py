"""Amended trial gate and the strict five-percent selection boundary."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gpu/scripts'))
from p08_selection import select


def trial(lr, farm, deaths, early=0):
    return dict(learning_rate=lr, farm=farm, death_matches=deaths, early_deaths=early)


class SelectionRuleTests(unittest.TestCase):
    def test_observed_first_trial_is_now_eligible_but_selection_waits(self):
        decision = select([trial(1e-5, 895, 5)])
        self.assertTrue(decision['trials'][0]['trial_eligible'])
        self.assertFalse(decision['complete'])
        self.assertIsNone(decision['selected'])

    def test_one_eligible(self):
        decision = select([trial(1e-5, 895, 5), trial(3e-5, 1000, 7)])
        self.assertEqual(decision['selected']['learning_rate'], 1e-5)

    def test_close_farm_prefers_survival(self):
        decision = select([trial(1e-5, 960, 4), trial(3e-5, 1000, 6)])
        self.assertEqual(decision['selected']['learning_rate'], 1e-5)

    def test_exact_five_percent_prefers_farm(self):
        decision = select([trial(1e-5, 950, 4), trial(3e-5, 1000, 6)])
        self.assertEqual(decision['selected']['learning_rate'], 3e-5)

    def test_neither_requests_more_data_instead_of_stopping(self):
        decision = select([trial(1e-5, 895, 7), trial(3e-5, 1000, 6, early=2)])
        self.assertTrue(decision['needs_additional_64'])
        self.assertFalse(decision['stop_required'])
        self.assertIsNone(decision['selected'])


if __name__ == '__main__':
    unittest.main()
