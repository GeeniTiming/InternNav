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



class NativeTerminationTests(unittest.TestCase):
    def test_done_true_is_normal_native_termination(self):
        self.assertEqual(
            nat.native_episode_termination(
                ("scene", "1"), [{"done": True}], {"steps": 45}, 100),
            "ENV_DONE",
        )

    def test_budget_exhaustion_done_false_is_valid_native_completion(self):
        # Native evaluator loops while step_id <= limit. Once it has
        # incremented beyond limit, last Habitat action may still be false.
        self.assertEqual(
            nat.native_episode_termination(
                ("scene", "1"), [{"done": False}], {"steps": 101}, 100),
            "NATIVE_LOOP_BUDGET_EXHAUSTED",
        )

    def test_done_false_without_exhausted_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without step budget"):
            nat.native_episode_termination(
                ("scene", "1"), [{"done": False}], {"steps": 40}, 100)

    def test_incomplete_native_progress_or_trace_is_rejected(self):
        for rows, progress, limit in (
            ([], {"steps": 101}, 100),
            ([{"done": False}], {"steps": None}, 100),
            ([{"done": False}], {"steps": 102}, 100),
            ([{"done": "false"}], {"steps": 101}, 100),
        ):
            with self.subTest(rows=rows, progress=progress):
                with self.assertRaises(ValueError):
                    nat.native_episode_termination(
                        ("scene", "1"), rows, progress, limit)

    def test_native_ready_accepts_complete_12_scene_budget_exhaustion(self):
        # This is the original failing prepare-path: all 12 controls are
        # accounted for but some last action rows have done=False.
        manifest = {
            "episodes": [
                {"scene_id": f"scene_{i}.glb", "episode_id": str(i)}
                for i in range(nat.ds.N_CASES)
            ]
        }
        keys = [nat.ds.key_of(ep) for ep in manifest["episodes"]]
        actions = [
            {"scene_id": ep["scene_id"], "episode_id": ep["episode_id"],
             "done": i % 2 == 0, "action": 1}
            for i, ep in enumerate(manifest["episodes"])
        ]
        progress = {
            key: {"steps": 45 if i % 2 == 0 else 101}
            for i, key in enumerate(keys)
        }
        summary = {
            "status": "complete", "actual_action_count": len(actions)
        }
        captured = {
            "status": "TRAIN_NATIVE_DECISION_DATASET_READY",
            "N_episodes": nat.ds.N_CASES,
            "N_action_records": len(actions),
        }
        def data(path):
            path = str(path)
            if path.endswith("pilot_summary.json"):
                return summary
            if path.endswith("decision_report.json"):
                return captured
            raise AssertionError("Unexpected source report path: " + path)

        with (
            patch.object(nat.pilot, "verify_manifest", return_value=manifest),
            patch.object(nat.ds.train, "validate_selection"),
            patch.object(nat.pilot, "read_json", side_effect=data),
            patch.object(nat.capture, "read_progress", return_value=progress),
            patch.object(nat.physical, "validate_trace", side_effect=lambda x, _: x),
            patch.object(nat.physical, "read_lines", return_value=actions),
            patch.object(nat.pilot, "load_config", return_value=SimpleNamespace(
                eval_settings={"max_steps_per_episode": 100})),
        ):
            _, _, actual, references = nat.native_ready(Path("/mock/new"))
        self.assertEqual(set(actual), set(keys))
        self.assertEqual(sum(len(r) for r in references.values()),
                         nat.ds.N_CASES)



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
        # Production input_cases MUST retain all 12 preregistered cases.
        # Only one gate is captured; the 11 unavailable cases stay recorded.
        cases = [
            case((20, 40, 60)[i // 4],
                 45 if i % 4 < 2 else -45,
                 (f"scene{i}", str(10 + i)))
            for i in range(pert.ds.N_CASES)
        ]
        for i, c in enumerate(cases):
            c["case_id"] = f"ep_{i:02d}"
        eligible_case = cases[1]
        report = {
            "schema": nat.SCHEMA,
            "status": "NATURAL_GATES_AUDITED_WITH_INELIGIBLE",
            "cases": [
                {
                    "episode_key": c["key"],
                    "case_id": c["case_id"],
                    "natural_checkpoint": (
                        {"step": 20} if c is eligible_case else None
                    ),
                    "status": (
                        "NATURAL_GATE_CAPTURED" if c is eligible_case else
                        "INELIGIBLE_RETAINED_NO_REPLACEMENT"
                    ),
                }
                for c in cases
            ],
        }
        old_root = Path("/frozen/natural")
        checkpoint_path = (old_root / "control/checkpoints" /
                           "_".join(eligible_case["key"]) / "checkpoint.json")
        saved = {
            str(old_root / "natural_checkpoint_report.json"): report,
            str(checkpoint_path): {"step": 20},
        }
        with (patch.object(pert.natural, "read_run", return_value=(
                {}, Path("/frozen/native"), {"cases": cases},
                {}, {}, {})),
              patch.object(pert.pilot, "read_json",
                           side_effect=lambda p: saved[str(p)])):
            _, _, _, _, original, picked = pert.input_cases(old_root)
            self.assertEqual(len(original["cases"]), pert.ds.N_CASES)
            self.assertEqual(len(picked), 1)
            self.assertEqual(picked[0]["key"], eligible_case["key"])
            self.assertEqual(
                sum(r["natural_checkpoint"] is None
                    for r in original["cases"]), 11)
            # An incomplete report must still be rejected, not silently
            # accepted by weakening the production 12-case invariant.
            saved[str(old_root / "natural_checkpoint_report.json")] = {
                **report, "cases": report["cases"][:2]
            }
            with self.assertRaisesRegex(ValueError, "fully audited"):
                pert.input_cases(old_root)

    def test_no_fake_post_perturb_rgb_in_cpu_protocol(self):
        self.assertIn("real", pert.SCOPE)
        self.assertIn("NOT", pert.SCOPE)
        self.assertNotIn("all 96 outcomes complete", pert.SCOPE)


if __name__ == "__main__":
    unittest.main()
