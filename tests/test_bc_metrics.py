import tempfile
import unittest
from pathlib import Path

import torch
from sf_duel.bc_metrics import ImitationMetrics, write_progress


class MetricsTests(unittest.TestCase):
    def test_continue_collapse_does_not_hide_missed_attacks_and_razes(self):
        labels = {k: torch.zeros(5, dtype=torch.long) for k in ('action', 'buy', 'item', 'target', 'x', 'y')}
        labels['action'] = torch.tensor([0, 0, 3, 6, 3])
        predictions = {k: torch.zeros_like(v) for k, v in labels.items()}
        meter = ImitationMetrics(); meter.update(predictions, labels, torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool))
        result = meter.summary()
        self.assertEqual(result['action_accuracy']['value'], .5)
        self.assertEqual(result['balanced_action_recall']['value'], 1 / 3)
        self.assertEqual(result['attack_with_target_recall'], {'value': 0, 'count': 1})
        self.assertEqual(result['raze_recall']['value'], 0)
        self.assertIsNone(result['raze_precision']['value'])

    def test_joint_target_spatial_error_and_false_positive(self):
        labels = {k: torch.zeros(4, dtype=torch.long) for k in ('action', 'buy', 'item', 'target', 'x', 'y')}
        labels['action'] = torch.tensor([3, 1, 6, 0])
        predictions = {k: v.clone() for k, v in labels.items()}
        predictions['target'][0] = 1
        predictions['x'][1] = 3; predictions['y'][1] = 4
        predictions['action'][3] = 6
        meter = ImitationMetrics(); meter.update(predictions, labels, torch.ones(4, dtype=torch.bool))
        result = meter.summary()
        self.assertEqual(result['attack_recall']['value'], 1)
        self.assertEqual(result['attack_with_target_recall']['value'], 0)
        self.assertEqual(result['move_distance_mean_given_teacher_action']['value'], 80)
        self.assertEqual(result['move_within_64_units']['value'], 0)
        self.assertEqual(result['move_within_128_units']['value'], 1)
        self.assertEqual(result['raze_recall']['value'], 1)
        self.assertEqual(result['raze_precision']['value'], .5)
        self.assertEqual(result['active_full_command_accuracy']['value'], 1/3)
        with tempfile.TemporaryDirectory() as directory:
            write_progress(directory, {'metrics': result}, [])
            self.assertIn('80.0', (Path(directory) / 'metrics.html').read_text(encoding='utf-8'))
