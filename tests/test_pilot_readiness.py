import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('followthrough', Path(__file__).resolve().parents[1] / 'scripts/run_pilot_followthrough.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
evaluation_spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).resolve().parents[1] / 'scripts/evaluate_native_policy.py')
evaluation = importlib.util.module_from_spec(evaluation_spec); evaluation_spec.loader.exec_module(evaluation)


class ReadinessTests(unittest.TestCase):
    def test_raw_evaluation_is_terminal_without_teacher_dataset_eligibility(self):
        evaluation.validate_result({'eligible': False, 'reason': 'death', 'outcome': {'winner': 2}}, True)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            evaluation.validate_result({'reason': 'diagnostic_cut', 'outcome': None}, True)
        with self.assertRaisesRegex(ValueError, 'Native evaluation error'):
            evaluation.validate_result({'error': 'EOF'}, False)

    def test_whole_match_split_cache_attempt_and_teacher_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset = root / 'dataset'; cache = root / 'cache'
            dataset.mkdir(); cache.mkdir()
            matches = []
            for i in range(192):
                run = str(dataset / f'match-{i:04d}')
                matches.append(dict(index=i, run=run, split='train' if i < 128 else 'validation', eligible=True,
                                    model_observation_adapter_validated=True, reason='death', teacher_policy_sha256='frozen'))
                (cache / f'match-{i:04d}.json').write_text(json.dumps({'run': run}))
                for team in (2, 3):
                    (cache / f'match-{i:04d}-team-{team}.npz').write_bytes(b'fixture')
            manifest = dict(matches=matches, teacher_policy_sha256='frozen')
            for path in (dataset, cache):
                (path / 'dataset.json').write_text(json.dumps(manifest))
            self.assertEqual(len(module.validate_dataset(dataset, cache, 'frozen')['matches']), 192)
            with self.assertRaisesRegex(ValueError, 'Teacher hash'):
                module.validate_dataset(dataset, cache, 'changed')
            (cache / 'match-0000.json').write_text(json.dumps({'run': 'wrong-attempt'}))
            with self.assertRaisesRegex(ValueError, 'different match attempt'):
                module.validate_dataset(dataset, cache, 'frozen')
            matches[-1]['split'] = 'train'
            (cache / 'dataset.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'split 128/64'):
                module.validate_dataset(dataset, cache, 'frozen')
