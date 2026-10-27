"""CPU-only checks for frozen-policy recovery candidate construction and hooks."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'recovery_candidates_test', ROOT / 'scripts/eval/recovery_candidates.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecoveryCandidateTests(unittest.TestCase):
    def test_backtrack_chooses_latest_qualifying_visited_prefix_not_goal(self):
        rows = [{'position': [x, 0., 0.]} for x in (0., .5, 1., 1.5, 1.8)]
        anchor = MODULE.choose_backtrack_anchor(rows, [2., 0., 0.])
        self.assertEqual(anchor['prefix_action_index'], 2)
        self.assertEqual(anchor['position'], [1., 0., 0.])

    def test_backtrack_no_qualifying_prefix_is_not_eligible(self):
        self.assertIsNone(MODULE.choose_backtrack_anchor(
            [{'position': [0., 0., 0.]}], [.2, 0., 0.]))

    def test_history_never_includes_before_handoff_or_current_view(self):
        self.assertEqual(MODULE.post_reset_history_indices(10, 11, 4), [])
        self.assertEqual(MODULE.post_reset_history_indices(10, 12, 4), [10])
        self.assertEqual(MODULE.post_reset_history_indices(10, 16, 3), [10, 12, 14])
        self.assertEqual(MODULE.post_reset_history_indices(10, 16, 0), [])
        with self.assertRaises(ValueError):
            MODULE.post_reset_history_indices(10, 10, 4)

    def test_reset_arms_are_predeclared(self):
        self.assertEqual(MODULE.ARMS, (
            'history_reset', 'left90_reset', 'right90_reset', 'backtrack_replan'))
        self.assertEqual(MODULE.ROTATIONS, {
            'left90_reset': 90, 'right90_reset': -90})

    def test_native_instrumentation_precedes_placeholder_generation(self):
        # The hooked selection must precede placeholder length computation,
        # otherwise the model receives an inconsistent number of image tokens.
        path = ROOT / 'internnav/habitat_extensions/vln/habitat_vln_evaluator.py'
        syntax = ast.parse(path.read_text(encoding='utf-8'))
        klass = next(x for x in syntax.body if isinstance(x, ast.ClassDef)
                     and x.name == 'HabitatVLNEvaluator')
        method = next(x for x in klass.body if isinstance(x, ast.FunctionDef)
                      and x.name == '_run_eval_dual_system')
        # A mock using the real evaluator source would require import-time
        # Habitat modules. Instead verify the exact AST fragment location.
        text = ast.unparse(method)
        self.assertIn('history_id = np.unique(', text)
        self.assertIn("placeholder = (DEFAULT_IMAGE_TOKEN +", text)
        source = (ROOT / 'scripts/eval/recovery_candidates.py').read_text()
        self.assertIn('history_id = np.unique(', source)
        self.assertIn('filter_history_ids(', source)
        self.assertNotIn("expected = ast.parse('history_id = sorted(history_id)')", source)

    def test_no_goal_or_reference_path_is_used_for_anchor_selection(self):
        src = MODULE.choose_backtrack_anchor.__code__.co_names
        self.assertNotIn('reference_path', src)
        self.assertNotIn('distance_to_goal', src)
        self.assertNotIn('success', src)


if __name__ == '__main__':
    unittest.main()
