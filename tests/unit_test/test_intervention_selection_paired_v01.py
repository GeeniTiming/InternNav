"""CPU-only v0.1 paired rollout protocol and audit regression tests."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_paired_v01 as paired


def context_sample():
    gate = {
        "history_sha256": "H_native",
        "rng_sha256": "R_native",
    }
    saved = {
        "context_id": "ep_123_perturbed",
        "current_pixel_sha256": "RGB_actual",
        "historical_rgb_history_sha256": "H_native",
        "perturbation_actions_before_context": [2, 2, 2],
    }
    actual = {
        "context_id": saved["context_id"],
        "current_rgb_pixel_sha256": "RGB_actual",
        "history_sha256": "H_native",
        "rng_sha256": "R_native",
        "perturbation_actions": [2, 2, 2],
    }
    return actual, saved, gate


class ActionContractTests(unittest.TestCase):
    def test_six_explicit_registered_arms_only(self):
        self.assertEqual(len(paired.ARMS), 6)
        self.assertEqual(set(paired.ARMS), {
            "natural_seek_left45", "natural_seek_right45",
            "natural_history_reset",
            "perturbed_seek_left45", "perturbed_seek_right45",
            "perturbed_history_reset",
        })
        for arm in paired.ARMS:
            mode, action = paired.split_arm(arm)
            self.assertIn(mode, paired.MODES)
            self.assertIn(action, paired.NEW_ACTIONS)

    def test_disallow_unknown_action(self):
        for arm in ("natural_keep", "perturbed_oracle",
                    "natural_seek_left90", "natural_random"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                paired.split_arm(arm)

    def test_left_right_seek_are_symmetric_real_six_turns(self):
        left = paired.scan_action_codes("SEEK_LEFT45")
        right = paired.scan_action_codes("SEEK_RIGHT45")
        self.assertEqual(left, [2, 2, 2, 3, 3, 3])
        self.assertEqual(right, [3, 3, 3, 2, 2, 2])
        self.assertEqual(len(left), len(right))
        self.assertRaises(ValueError, paired.scan_action_codes, "HISTORY_RESET")


class FingerprintTests(unittest.TestCase):
    def test_exact_prearm_context_fingerprints_pass(self):
        actual, saved, gate = context_sample()
        self.assertTrue(paired.assert_prearm_matches_keep(
            actual, saved, gate, [2, 2, 2]))

    def test_prearm_pixel_history_rng_and_yaw_changes_fail(self):
        actual, saved, gate = context_sample()
        mutations = (
            ("actual", "current_rgb_pixel_sha256", "changed"),
            ("actual", "history_sha256", "changed"),
            ("actual", "rng_sha256", "changed"),
            ("actual", "context_id", "changed"),
            ("actual", "perturbation_actions", [3, 3, 3]),
            ("saved", "current_pixel_sha256", "changed"),
            ("saved", "historical_rgb_history_sha256", "changed"),
            ("saved", "perturbation_actions_before_context", [3, 3, 3]),
            ("gate", "history_sha256", "changed"),
            ("gate", "rng_sha256", "changed"),
        )
        for location, key, value in mutations:
            a, b, c = copy.deepcopy((actual, saved, gate))
            item = {"actual": a, "saved": b, "gate": c}[location]
            item[key] = value
            with self.subTest(location=location, key=key):
                with self.assertRaisesRegex(RuntimeError, "NOT_COMPARABLE"):
                    paired.assert_prearm_matches_keep(a, b, c, [2, 2, 2])

    def test_natural_prearm_without_physical_yaw(self):
        actual, saved, gate = context_sample()
        actual["context_id"] = "ep_123_natural"
        saved["context_id"] = "ep_123_natural"
        actual["perturbation_actions"] = []
        saved["perturbation_actions_before_context"] = []
        self.assertTrue(paired.assert_prearm_matches_keep(
            actual, saved, gate, []))


class PredeclaredCohortTests(unittest.TestCase):
    @staticmethod
    def mock_case(i):
        key = [f"scene_{i:03d}", str(i + 10)]
        return {"key": key, "case_id": f"ep_{i:03d}",
                "checkpoint": {"step": 30},
                "perturbation_plan": {"yaw_deg": 45}}

    def test_first_two_then_all_ten_and_no_outcome_filtering(self):
        cases = [self.mock_case(i) for i in range(10)]
        report = {
            "schema": paired.pert.SCHEMA,
            "status": "PERTURBED_KEEP_AUDITED",
            "N_original_cases": 12,
            "N_eligible_cases": 10,
            "N_perturbed_keep_outcomes": 10,
            # Every other case is a failure. Selection MUST ignore this.
            "cases": [
                {"episode_key": c["key"], "success": i % 2}
                for i, c in enumerate(cases)
            ],
        }
        natural_report = {
            "schema": paired.natural.SCHEMA,
            "N_cases_predeclared": 12,
            "N_captured_gates": 10,
            "N_ineligible_retained": 2,
        }
        def fake_read(path):
            n = str(path)
            if n.endswith("perturbed_keep_report.json"):
                return report
            if n.endswith("natural_checkpoint_report.json"):
                return natural_report
            raise AssertionError("Unexpected read " + n)
        with (
            patch.object(paired.pert, "read_run", return_value=(
                Path("/frozen/nat"), Path("/frozen/native"),
                {"cases": cases})),
            patch.object(paired.pilot, "read_json", side_effect=fake_read),
        ):
            pilot = paired.eligible_sources("/frozen/pert", 2)
            full = paired.eligible_sources("/frozen/pert", 10)
            self.assertEqual([c["key"] for c in pilot[-1]],
                             [c["key"] for c in cases[:2]])
            self.assertEqual(len(full[-1]), 10)
            with self.assertRaisesRegex(ValueError, "Only preregistered"):
                paired.eligible_sources("/frozen/pert", 3)

    def test_tampered_or_incomplete_keep_cohort_rejected(self):
        cases = [self.mock_case(i) for i in range(10)]
        outcome = {
            "schema": paired.pert.SCHEMA,
            "status": "PERTURBED_KEEP_AUDITED",
            "N_original_cases": 12,
            "N_eligible_cases": 10,
            "N_perturbed_keep_outcomes": 9,
            "cases": [{"episode_key": c["key"]} for c in cases],
        }
        with (
            patch.object(paired.pert, "read_run", return_value=(
                Path("/frozen/nat"), Path("/frozen/native"),
                {"cases": cases})),
            patch.object(paired.pilot, "read_json", return_value=outcome),
        ):
            with self.assertRaisesRegex(ValueError, "Require audited"):
                paired.eligible_sources("/frozen/pert", 2)


if __name__ == "__main__":
    unittest.main()
