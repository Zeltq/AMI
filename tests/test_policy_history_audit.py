import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('policy_history_audit',
    Path(__file__).resolve().parents[1] / 'gpu/scripts/policy_history_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PolicyHistoryAuditTests(unittest.TestCase):
    def test_variable_targets_keep_creep_tower_tree_and_base_identity(self):
        arrays = {key: np.zeros(shape, np.float32) for key, shape in audit.LAYOUT.items()}
        arrays['creeps'] = np.zeros((2, 96), np.float32)
        arrays['trees'] = np.zeros((1, 8), np.float32)
        arrays['projectiles'] = np.zeros((0, 12), np.float32)
        arrays['creep_valid'] = np.ones(2, bool)
        arrays['creep_ally'] = np.array([True, False])
        arrays['tree_valid'] = np.ones(1, bool)
        arrays['projectile_valid'] = np.zeros(0, bool)
        # hero0,hero1,creep0,creep1,tower0..3,tree0,base
        for key in audit.TARGET_MASKS:
            arrays[key] = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
        frame = audit.canonical_native(arrays)
        for key in audit.TARGET_MASKS:
            self.assertEqual(np.flatnonzero(frame[key]).tolist(), [1, 2, 259, 261, 262, 518])
        self.assertEqual(frame['creep_valid'].sum(), 2)
        self.assertEqual(frame['creep_ally'].sum(), 1)

    def test_numeric_tolerance_does_not_get_reported_as_bit_identity(self):
        result = audit.comparison(np.array([1., -0.], np.float32), np.array([1.00001, 0.], np.float32))
        self.assertTrue(result['within_tolerance'])
        self.assertFalse(result['bit_equal'])
        self.assertEqual(result['different_scalars'], 2)

    def test_greedy_respects_mask_first_tie_and_reports_small_margin(self):
        selected = audit.greedy_summary([1., 1., 100.], [True, True, False])
        self.assertEqual(selected['action'], 0)
        self.assertEqual(selected['top_two_logit_margin'], 0)
        self.assertEqual(selected['probabilities'][2], 0)

    def test_reset_chain_preserves_signed_zero_from_multiplication(self):
        actual = audit.expected_memory_chain(np.array([[-1., 2.], [3., -4.]], np.float32),
                                            np.array([True, True]))
        self.assertTrue(np.signbit(actual[1, 0]))
        self.assertFalse(np.signbit(actual[1, 1]))
        np.testing.assert_array_equal(actual, np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
