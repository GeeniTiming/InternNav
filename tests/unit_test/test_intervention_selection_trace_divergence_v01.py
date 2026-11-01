"""CPU-only regression for evidence-safe continuation action divergence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_trace_divergence_v01 as d


def row(action, phase=None, position=None):
    z = {
        "action": action, "done": False,
        "scene_id": "some/path/scene.glb", "episode_id": "70",
        "position": position or [0.0, 0.0, 0.0],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    if phase is not None:
        z["phase"] = phase
    return z


class SegmentationTests(unittest.TestCase):
    def test_episode_key_uses_scene_stem(self):
        self.assertEqual(d.key_of(row(1)), ("scene", "70"))

    def test_natural_keep_untagged_native_actions(self):
        seq = [row(1), row(2), row(3), row(0)]
        part = d.phase_segments(seq, 2, "natural", "KEEP")
        self.assertEqual([r["action"] for r in part["continuation"]], [3, 0])
        self.assertEqual(len(part["alignment"]), 0)
        self.assertEqual(len(part["external_recovery"]), 0)

    def test_perturbed_right_seek_excludes_three_plus_six_real_yaw(self):
        seq = ([row(1, "prefix"), row(2, "prefix")]
               + [row(2, "alignment")] * 3
               + [row(a, "external_recovery") for a in (3,3,3,2,2,2)]
               + [row(1, "continuation"), row(0, "continuation")])
        parts = d.phase_segments(seq, 2, "perturbed", "SEEK_RIGHT45")
        self.assertEqual([len(parts[x]) for x in (
            "alignment", "external_recovery", "continuation")], [3, 6, 2])
        self.assertEqual([r["action"] for r in parts["continuation"]], [1, 0])

    def test_wrong_scan_direction_or_wrong_order_rejected(self):
        prefix = [row(1, "prefix"), row(1, "prefix")]
        wrong = prefix + [row(2, "external_recovery")] * 6 + [
            row(0, "continuation")]
        with self.assertRaisesRegex(ValueError, "scan did not follow"):
            d.phase_segments(wrong, 2, "natural", "SEEK_RIGHT45")
        wrong = (prefix + [row(0, "continuation")]
                 + [row(3, "external_recovery")] * 6)
        with self.assertRaisesRegex(ValueError, "causal order"):
            d.phase_segments(wrong, 2, "natural", "SEEK_RIGHT45")

    def test_prefix_matching_detects_pose_and_action_changes(self):
        base = [row(1), row(2), row(3)]
        candidate = copy.deepcopy(base)
        self.assertTrue(d.compare_prefix(base, candidate, 2))
        candidate[1]["position"] = [0.3, 0, 0]
        with self.assertRaisesRegex(ValueError, "prefix pose"):
            d.compare_prefix(base, candidate, 2)
        candidate = copy.deepcopy(base)
        candidate[0]["action"] = 3
        with self.assertRaisesRegex(ValueError, "prefix action"):
            d.compare_prefix(base, candidate, 2)


class FirstDivergenceTests(unittest.TestCase):
    def test_first_differing_policy_action_after_shared_prefix(self):
        out = d.first_action_divergence([1,1,2,0], [1,1,3,0])
        self.assertEqual(out["first_different_continuation_index"], 2)
        self.assertEqual(out["keep_action_at_divergence"], 2)
        self.assertEqual(out["treated_action_at_divergence"], 3)
        self.assertEqual(out["reason"], "ACTION_CODE_DIFFERS")

    def test_prefix_and_identical_sequences_are_not_misread(self):
        out = d.first_action_divergence([1,2], [1,2])
        self.assertIsNone(out["first_different_continuation_index"])
        self.assertEqual(out["reason"], "IDENTICAL_ACTION_CODES")
        out = d.first_action_divergence([1,2], [1,2,3])
        self.assertEqual(out["reason"], "ONE_SEQUENCE_TERMINATES")
        self.assertIsNone(out["keep_action_at_divergence"])

    def test_no_extra_actions_after_gate_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "continuation"):
            d.phase_segments([row(1), row(2)], 2, "natural", "KEEP")


class ExportTests(unittest.TestCase):
    def test_create_only_and_audit_rejects_corrupt_trace(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "new"
            info = {
                "schema": d.SCHEMA,
                "status": "POSTHOC_CONTINUATION_TRACES_EXPORTED_NOT_CAUSALLY_IDENTIFIED",
                "N_scene_groups": 4,
                "N_contexts": 8,
                "N_real_arm_traces": 32,
                "N_action_rows": 11,
                "N_effects_vs_KEEP": 24,
                "N_first_S2_pair_comparisons": 16,
                "source_review_dir": td,
                "original_trace_sha256": {},
            }
            outputs = {n:(n+"\n").encode() for n in d.OUTPUTS}
            with patch.object(
                d, "collect",
                side_effect=lambda *_: (copy.deepcopy(info), dict(outputs))
            ):
                d.build(td, folder)
                d.audit(folder)
                with self.assertRaises(FileExistsError):
                    d.build(td, folder)
                with (folder / d.OUTPUTS[0]).open("ab") as f:
                    f.write(b"tampering\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    d.audit(folder)


if __name__ == "__main__":
    unittest.main()
