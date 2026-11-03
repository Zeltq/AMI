import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'reports/P03/analyze.py'
spec=importlib.util.spec_from_file_location('p03_analysis',path)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


class P03MetricsTests(unittest.TestCase):
    def test_checkpoints_after_death_are_censored_even_if_later_telemetry_exists(self):
        samples=[dict(time=t,lh=t//10,deny=1,health=100,mana=50)for t in (59,60,120,300,600)]
        points=analysis.checkpoints(samples,100)
        self.assertEqual(points['60']['lh'],6)
        self.assertIsNone(points['120'])
        self.assertIsNone(points['600'])
