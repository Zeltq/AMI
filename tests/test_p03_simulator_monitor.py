import importlib.util
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'reports/P03/run_simulator.py'
spec=importlib.util.spec_from_file_location('p03_simulator',path)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


class SimulatorMonitorTests(unittest.TestCase):
    def test_adjacent_seeds_collide_but_selected_plan_has_200_distinct_starts(self):
        initial=monitor.initial_rng_states(20261103,100,20)
        adjacent=monitor.initial_rng_states(20261104,100,20)
        self.assertEqual(len(set(initial+adjacent)),150)
        self.assertEqual(len(set(monitor.validate_seed_plan(20,100))),200)

    def test_shared_status_is_utf8_even_on_windows_codepage(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'status.json'
            path.write_bytes('{"этап":"готово ⚙"}'.encode('utf-8'))
            self.assertEqual(monitor.read_json(path),{'этап':'готово ⚙'})
