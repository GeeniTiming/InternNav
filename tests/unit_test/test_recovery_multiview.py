"""CPU-only sanity tests for multiview factorial and experiment safeguards."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "scripts/eval/recovery_multiview_compare.py"
spec = importlib.util.spec_from_file_location("multiview_compare", path)
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)


class FactorialMathTests(unittest.TestCase):
    def test_episode10_genuine_joint_only_rescue(self):
        result = mv.effects({0: {0: 0, 1: 0}, -90: {0: 0, 1: 1}})
        self.assertEqual(result[-90]["gamma_vs_no_turn"], 1)
        self.assertTrue(result[-90]["joint_only_rescue"])
        self.assertEqual(result[-90]["relative_reset"], 1)

    def test_episode9_joint_can_erase_reset_benefit(self):
        result = mv.effects({0: {0: 0, 1: 1}, -90: {0: 0, 1: 0}})
        self.assertEqual(result[-90]["gamma_vs_no_turn"], -1)
        self.assertFalse(result[-90]["joint_only_rescue"])

    def test_positive_gamma_does_not_imply_rescue(self):
        result = mv.effects({0: {0: 1, 1: 0}, -90: {0: 1, 1: 1}})
        self.assertEqual(result[-90]["gamma_vs_no_turn"], 1)
        self.assertFalse(result[-90]["joint_only_rescue"])
        self.assertEqual(result[-90]["relative_reset"], 0)

    def test_missing_factor_pair_refused(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            mv.effects({0: {0: 0, 1: 0}, 45: {1: 1}})

    def test_nonbinary_sr_refused(self):
        with self.assertRaisesRegex(ValueError, "nonbinary"):
            mv.effects({0: {0: 0, 1: 0}, 45: {0: 1, 1: 0.5}})

    def test_fixed_grid_is_symmetric(self):
        self.assertEqual(mv.ANGLES_FULL, (-90, -45, 0, 45, 90))
        self.assertEqual(mv.ANGLES_BASE, (-90, 0, 90))

    def test_new_runner_has_only_four_three_step_arms(self):
        src = (ROOT / "scripts/eval/recovery_multiview_45.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        self.assertIn("'left45_keep'", src)
        self.assertIn("'right45_keep'", src)
        self.assertIn("'left45_reset'", src)
        self.assertIn("'right45_reset'", src)
        self.assertIn("budget=3", src)
        self.assertIn("self.external_actions != 3", src)
        self.assertIn("preserve_policy_rng()", src)
        self.assertIn("turn unexpectedly translated", src)
        self.assertNotIn("expected six actual native turns", src)


if __name__ == "__main__":
    unittest.main()
