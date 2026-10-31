"""CPU-only regression tests for v0.1 native gates + physical perturb plan."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import intervention_selection_native_gates_v01 as nat
import intervention_selection_perturbed_keep_v01 as pert


def case(tier=40, yaw=45, episode=("example_scene", "5")):
    return {
        "key": list(episode), "case_id": "ep_test",
        "checkpoint_rule": {
            "minimum_main_step": tier,
            "minimum_prefix_translation_m": 1.0,
            "minimum_remaining_main_loop_steps": 30,
            "camera_level_required": True,
            "no_future_baseline_length": True,
        },
        "perturbation_plan": {
            "yaw_deg": yaw,
            "actual_native_yaw_actions": (
                [2, 2, 2] if yaw == 45 else [3, 3, 3]),
            "predecision_only": True,
            "preserve_policy_rng": True,
        },
    }


def eligible(c, step=40, translation=1.3, limit=100,
             done=False, action_seq=None, pixel_goal=None,
             local_actions=None, action=0, camera_balance=0, pitch=0.0):
    return nat.per_case_eligibility(
        c, step, limit, 4, translation, done,
        action_seq or [], pixel_goal, local_actions or [],
        action, camera_balance, pitch,
    )


class PerCaseEligibilityTests(unittest.TestCase):
    def test_each_episode_uses_locked_tier(self):
        self.assertTrue(eligible(case(20), step=20)["eligible"])
        self.assertFalse(eligible(case(40), step=20)["eligible"])
        self.assertTrue(eligible(case(40), step=40)["eligible"])
        self.assertFalse(eligible(case(60), step=40)["eligible"])
        self.assertTrue(eligible(case(60), step=60)["eligible"])

    def test_checks_prior_motion_and_remaining_budget(self):
        self.assertFalse(eligible(case(), translation=.4)["eligible"])
        self.assertFalse(eligible(case(), step=75, limit=100)["eligible"])
        self.assertTrue(eligible(case(), step=60, limit=100)["eligible"])

    def test_native_idle_and_camera_level_are_required(self):
        c = case()
        for kwargs in (
            {"done": True}, {"action_seq": [2]},
            {"pixel_goal": [4, 5]}, {"local_actions": [1]},
            {"action": 5}, {"camera_balance": 1},
            {"pitch": .0015}, {"pitch": None},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(eligible(c, **kwargs)["eligible"])

    def test_no_future_baseline_or_outcome_in_eligibility(self):
        c = case()
        a = eligible(c)
        c["native_success"] = 0
        c["future_actions"] = [2, 2]
        c["oracle_remaining_m"] = 50
        self.assertEqual(a, eligible(c))
        self.assertEqual(set(a), {
            "checks", "eligible", "remaining_native_main_steps",
        })


class PerturbationTests(unittest.TestCase):
    def test_right_left_45_three_actual_habitat_yaws(self):
        self.assertEqual(pert.expected_yaw_actions(case(yaw=45)[
            "perturbation_plan"]), [2, 2, 2])
        self.assertEqual(pert.expected_yaw_actions(case(yaw=-45)[
            "perturbation_plan"]), [3, 3, 3])

    def test_disallow_unregistered_action_or_yaw(self):
        for change in (
            {"yaw_deg": 30},
            {"actual_native_yaw_actions": [2, 2]},
            {"actual_native_yaw_actions": [3, 3, 3]},
            {"predecision_only": False},
            {"preserve_policy_rng": False},
        ):
            c = case()
            c["perturbation_plan"].update(change)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    pert.expected_yaw_actions(c["perturbation_plan"])

    def test_ineligible_natural_gate_retained_not_replaced(self):
        c0 = case(20, 45, ("scene0", "10"))
        c1 = case(60, -45, ("scene1", "11"))
        cases = [c0, c1]
        report = {
            "schema": nat.SCHEMA,
            "status": "NATURAL_GATES_AUDITED_WITH_INELIGIBLE",
            "cases": [
                {"episode_key": c0["key"], "case_id": "ep_test",
                 "natural_checkpoint": None,
                 "status": "INELIGIBLE_RETAINED_NO_REPLACEMENT"},
                {"episode_key": c1["key"], "case_id": "ep_test",
                 "natural_checkpoint": {"step": 60},
                 "status": "NATURAL_GATE_CAPTURED"},
            ],
        }
        old_root = Path("/frozen/natural")
        saved = {
            str(old_root / "natural_checkpoint_report.json"): report,
            str(old_root / "control/checkpoints/scene1_11/checkpoint.json"): {
                "step": 60},
        }
        with (patch.object(pert.natural, "read_run", return_value=(
                {}, Path("/frozen/native"), {"cases": cases},
                {}, {}, {})),
              patch.object(pert.pilot, "read_json",
                           side_effect=lambda p: saved[str(p)])):
            _, _, _, _, old, picked = pert.input_cases(old_root)
        self.assertEqual(len(old["cases"]), 2)
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0]["key"], c1["key"])

    def test_no_fake_post_perturb_rgb_in_cpu_protocol(self):
        self.assertIn("real", pert.SCOPE)
        self.assertIn("NOT", pert.SCOPE)
        self.assertNotIn("all 96 outcomes complete", pert.SCOPE)


if __name__ == "__main__":
    unittest.main()
