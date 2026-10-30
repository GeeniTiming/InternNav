"""CPU-only, no CUDA/Habitat model loading, no navigation rollouts."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_train_six_arm as pilot


def gate(step=25):
    return {"step": step, "history_length": step, "prefix_action_count": 100,
            "state": {"position": [0., 0., 0.],
                      "orientation_wxyz": [1., 0., 0., 0.]},
            "history_sha256": "lock"}


def cases(n=6):
    return [{"key": [f"Scene{i}", str(i)],
             "checkpoint": gate(step=20+i)} for i in range(n)]


class ProtocolTest(TestCase):
    def test_predeclared_six_by_six_without_native_keep_rerun(self):
        self.assertEqual(len(pilot.ARMS), 5)
        self.assertEqual(set(pilot.COMPARISONS), {
            "keep", "reset", "placebo_keep", "seek_keep",
            "placebo_reset", "seek_reset",
        })
        p = pilot.build_protocol(cases(), native_limit=500)
        self.assertEqual(p["native_main_step_limit"], 500)
        self.assertEqual(p["charged_native_main_step_limit"], 494)
        self.assertEqual(p["forced_native_actions"], 6)
        self.assertEqual(p["split"], "train_development_only")
        self.assertEqual(len(p["cases"]) * len(p["arms"]), 30)

    def test_reject_non_six_or_insufficient_budget(self):
        with self.assertRaisesRegex(ValueError, "six captured"):
            pilot.build_protocol(cases(5), native_limit=500)
        with self.assertRaisesRegex(ValueError, "Insufficient charged"):
            pilot.build_protocol(cases(), native_limit=27)

    def test_two_yaw_scans_keep_and_reset_have_same_internal_view(self):
        self.assertEqual(pilot.VIEW[pilot.ARMS[1]], pilot.VIEW[pilot.ARMS[3]])
        self.assertEqual(pilot.VIEW[pilot.ARMS[2]], pilot.VIEW[pilot.ARMS[4]])
        self.assertTrue(pilot.VIEW[pilot.ARMS[1]].endswith("_placebo"))
        self.assertTrue(pilot.VIEW[pilot.ARMS[2]].endswith("_evidence"))

    def test_scan_probe_initializer_uses_actual_directory_then_alias(self):
        for cls, base, arm in (
            (pilot.ScanKeepProbe, pilot.iso.MechanismProbe, pilot.ARMS[1]),
            (pilot.ScanResetProbe, pilot.choice.JointProbe, pilot.ARMS[4]),
        ):
            captured = {}
            def init(self, evaluator, root, supplied, protocol, references):
                captured["supplied"] = supplied
                self.directory = Path(root) / supplied
                self.arm = supplied
            with patch.object(base, "__init__", init):
                obj = cls(object(), Path("/tmp/native"), arm, {}, {})
            self.assertEqual(captured["supplied"], arm)
            self.assertEqual(obj.directory.name, arm)
            self.assertEqual(obj.arm, pilot.VIEW[arm])


class HistoryGateTest(TestCase):
    def make_detail(self, cutoff=25, reset=True):
        return {
            "history_prompt_events": 1 if reset else 0,
            "handoff": {
                "history_reset": reset,
                "history_cutoff": cutoff if reset else None,
            },
        }

    def test_no_old_history_after_reset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = ("scene", "1")
            folder = root / pilot.ARMS[0] / "checkpoints" / "scene_1"
            folder.mkdir(parents=True)
            checkpoint = gate()
            pilot.pilot.write_json(folder / "history_inputs.json", [{
                "main_step": 25, "cutoff": 25, "rgb_count": 26,
                "selected_history_ids": [],
            }])
            out = pilot.validate_histories(
                root, pilot.ARMS[0], key,
                self.make_detail(), checkpoint, True)
            self.assertEqual(len(out), 1)
            pilot.pilot.write_json(folder / "history_inputs.json", [{
                "main_step": 25, "cutoff": 25, "rgb_count": 28,
                "selected_history_ids": [3],
            }])
            with self.assertRaisesRegex(ValueError, "leaked"):
                pilot.validate_histories(
                    root, pilot.ARMS[0], key,
                    self.make_detail(), checkpoint, True)

    def test_reset_first_decision_must_have_no_history_images(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / pilot.ARMS[0] / "checkpoints" / "scene_1"
            folder.mkdir(parents=True)
            pilot.pilot.write_json(folder / "history_inputs.json", [{
                "main_step": 25, "cutoff": 25, "rgb_count": 27,
                "selected_history_ids": [25],
            }])
            with self.assertRaisesRegex(ValueError, "First RESET"):
                pilot.validate_histories(
                    root, pilot.ARMS[0], ("scene", "1"),
                    self.make_detail(), gate(), True)

    def test_no_fake_reset_or_history_change_in_keep(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / pilot.ARMS[1] / "checkpoints" / "scene_1"
            folder.mkdir(parents=True)
            pilot.pilot.write_json(folder / "history_inputs.json", [])
            self.assertEqual(pilot.validate_histories(
                root, pilot.ARMS[1], ("scene", "1"),
                self.make_detail(reset=False), gate(), False), [])
            pilot.pilot.write_json(folder / "history_inputs.json", [{"x": 1}])
            with self.assertRaisesRegex(ValueError, "count differs"):
                pilot.validate_histories(
                    root, pilot.ARMS[1], ("scene", "1"),
                    self.make_detail(reset=False), gate(), False)

    def test_runner_does_not_treat_preliminary_labels_as_gold(self):
        self.assertIn("preliminary", pilot.SCOPE.lower())
        self.assertNotIn("val_unseen", pilot.SCOPE)
        self.assertIn("native success=1", pilot.SCOPE)


if __name__ == "__main__":
    import unittest
    unittest.main()
