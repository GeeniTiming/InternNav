"""CPU tests only: no model/Habitat loading or GPU rollouts."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_intervention_choice_pilot as choice


class ChoicePilotTests(TestCase):
    def test_six_arm_mapping_and_binary_contrasts(self):
        self.assertEqual(set(choice.LABELS), {
            "keep", "reset", "placebo_keep", "seek_keep",
            "placebo_reset", "seek_reset",
        })
        # Synthetic values, NOT empirical outcomes.
        success = {
            "keep": 0, "reset": 1, "placebo_keep": 0,
            "seek_keep": 1, "placebo_reset": 1, "seek_reset": 1,
        }
        result = choice.compare_6_conditions({
            ("scene", "1"): {arm: {"success": y, "ne": 1.0 + i}
                             for i, (arm, y) in enumerate(success.items())}
        })[0]
        c = result["binary_success_contrasts"]
        self.assertEqual(c["reset_minus_keep"], 1)
        self.assertEqual(c["extra_rgb_minus_placebo_keep"], 1)
        self.assertEqual(c["extra_rgb_minus_placebo_reset"], 0)
        self.assertEqual(c["rgb_reset_interaction"], -1)

    def test_six_arm_requires_complete_continuations(self):
        incomplete = {a: {"success": 0, "ne": 2.0}
                      for a in choice.LABELS if a != "seek_reset"}
        with self.assertRaisesRegex(ValueError, "Missing six"):
            choice.compare_6_conditions({("scene", "1"): incomplete})

    def test_reject_nonfinite_navigation_error(self):
        arms = {a: {"success": 0, "ne": 1.0} for a in choice.LABELS}
        arms["seek_reset"]["ne"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            choice.compare_6_conditions({("scene", "1"): arms})

    def test_joint_reset_occurs_after_scan_before_s2(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "lookback_right45_evidence" / "checkpoints" / "scene_1"
            folder.mkdir(parents=True)
            probe = choice.JointProbe.__new__(choice.JointProbe)
            # Actual first callback occurs before TurnProbe.begin()
            # creates gate; emulate the attribute's absence.
            probe.current = ("scene", "1")
            probe.directory = Path(tmp) / "lookback_right45_evidence"
            probe.history_cutoff = None
            probe.reset_enabled = False
            probe.handoff = {
                "external_actions": 6,
                "history_reset": False, "history_cutoff": None,
            }

            def fake_scan(self, *a):
                self.gate = {"step": 20}
                return {"rgb": "pretend"}, False

            with patch.object(choice.iso.MechanismProbe, "on_boundary", fake_scan):
                obs, done = probe.on_boundary(
                    20, {}, False, ["r0", "r1"], [], None, [], 3)
            self.assertFalse(done)
            self.assertEqual(obs["rgb"], "pretend")
            self.assertTrue(probe.reset_enabled)
            self.assertEqual(probe.history_cutoff, 2)
            handoff = choice.pilot.read_json(folder / "handoff.json")
            self.assertTrue(handoff["history_reset"])
            self.assertEqual(handoff["history_cutoff"], 2)

    def test_reset_audit_rejects_leaked_prefix_history(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            gate = {"step": 20, "history_length": 3, "history_sha256": "aaa"}
            detail = {
                "gate": gate, "prefix_comparison": {"actions_and_poses_match": True},
                "handoff": {"history_reset": True, "history_cutoff": 3,
                            "external_actions": 6},
                "history_prompt_events": 1,
            }
            choice.pilot.write_json(folder / "history_inputs.json", [{
                "main_step": 20, "cutoff": 3, "rgb_count": 4,
                "selected_history_ids": [],
            }])
            choice.pilot.write_json(folder / "evidence_input.json", {
                "events": [{"main_step": 20, "prompt": choice.iso.EXTRA_PROMPT,
                            "source": "duplicate_current",
                            "supplementary_pixel_sha256": "c",
                            "same_image_count": True}],
                "scan_view_sha256": "v", "post_return_current_sha256": "c",
            })
            self.assertEqual(choice.validate_joint_case(
                detail, folder, "lookback_right45_placebo", gate
            )["scan_view_sha256"], "v")
            choice.pilot.write_json(folder / "history_inputs.json", [{
                "main_step": 20, "cutoff": 3, "rgb_count": 4,
                "selected_history_ids": [0],
            }])
            with self.assertRaisesRegex(ValueError, "pre-gate history leaked"):
                choice.validate_joint_case(
                    detail, folder, "lookback_right45_placebo", gate
                )

    def test_scan_sequence_and_pose_identity(self):
        key = ("scene", "1")
        def make(a):
            return {
                "scene_id": "s/scene.glb", "episode_id": "1",
                "phase": "external_recovery", "action": a, "done": False,
                "position": [0.0, 0.0, 0.0],
                "orientation_wxyz": [1., 0., 0., 0.],
            }
        rows = [make(a) for a in (3, 3, 3, 2, 2, 2)]
        self.assertEqual(len(choice.verify_scan_pair(rows, rows, key)), 6)
        tampered = [dict(r) for r in rows]
        tampered[0]["action"] = 2
        with self.assertRaisesRegex(ValueError, "scan actions"):
            choice.verify_scan_pair(rows, tampered, key)


if __name__ == "__main__":
    main()
