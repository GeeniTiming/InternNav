"""CPU regression tests for task-evidence v2 per-landmark labels."""
import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_task_evidence_annotation_v2 as v2
import recovery_task_evidence_acquire as acq


class EvidenceV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "capture/images").mkdir(parents=True)
        refs = []
        for color in (10, 30, 30):
            image = Image.new("RGB", (12, 12), (color, 0, 0))
            digest = acq.pixel_digest(image)
            image.save(self.root / "capture/images" / (digest + ".png"))
            refs.append({"path": "images/" + digest + ".png",
                         "pixel_sha256": digest, "width": 12, "height": 12,
                         "mode": "RGB"})
        self.native = {
            "schema": acq.SCHEMA, "call_id": 3, "episode_key": ["sceneB", "1"],
            "instruction": "Go around the center unit then stop at the doorway with a mirror.",
            "main_step": 2, "call_type": "ordinary",
            "native_action_prefix_count": 4, "history_native_indices": [0, 1],
            "images": refs, "current_image_ref": refs[-1]["path"],
            "remaining_main_step_budget": 498,
        }
        self.actions = [
            {"episode_key": ["sceneA", "1"], "action": 1},
            {"episode_key": ["sceneA", "1"], "action": 3},
            {"episode_key": ["sceneB", "1"], "action": 3},
            {"episode_key": ["sceneB", "1"], "action": 3},
        ]
        self.row = v2.make_template(self.native, self.root, self.actions)

    def annotated(self):
        row = copy.deepcopy(self.row)
        row["annotation"].update({
            "status": "complete", "annotator_id": "reviewer_1",
            "progress": "unknown", "progress_basis": "unknown",
            "stage_span": "", "action_evidence_indices": [],
            "history_consistency": "insufficient",
            "reason": "No reliable evidence of current navigation stage.",
            "confidence": "low",
            "landmarks": [
                {"name": "center unit", "instruction_span": "center unit",
                 "role": "route_anchor", "visibility": "history_only",
                 "support_image_slots": [0], "regions": [],
                 "reason": "The unit is visible in the first history frame."},
                {"name": "mirror", "instruction_span": "mirror",
                 "role": "goal_disambiguator", "visibility": "uncertain",
                 "support_image_slots": [], "regions": [],
                 "reason": "No definite mirror can be identified."},
            ],
        })
        return row

    def test_unknown_progress_and_distinct_landmarks(self):
        self.assertEqual(self.row["duplicate_current_history_slots"], [1])
        self.assertEqual(self.row["past_action_codes"], [3, 3])
        self.assertEqual(
            v2.check_row(self.annotated(), self.native, self.actions, self.root),
            "complete")

    def test_current_only_does_not_accept_history_slots(self):
        row = self.annotated()
        landmark = row["annotation"]["landmarks"][0]
        landmark["visibility"] = "current"
        landmark["support_image_slots"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "Current-only"):
            v2.check_row(row, self.native, self.actions, self.root)

    def test_progress_from_actions_requires_citations(self):
        row = self.annotated()
        a = row["annotation"]
        a["progress"] = "before_subgoal"
        a["progress_basis"] = "actions"
        a["stage_span"] = "Go around the center unit"
        with self.assertRaisesRegex(ValueError, "cited earlier"):
            v2.check_row(row, self.native, self.actions, self.root)
        a["action_evidence_indices"] = [0, 1]
        self.assertEqual(v2.check_row(row, self.native, self.actions, self.root), "complete")

    def test_conflict_is_not_supported_by_identical_rgb(self):
        row = self.annotated()
        row["annotation"]["history_consistency"] = "conflicting"
        row["annotation"]["conflict_slot_pairs"] = [[1, 2]]
        with self.assertRaisesRegex(ValueError, "nonidentical"):
            v2.check_row(row, self.native, self.actions, self.root)

    def test_future_image_index_and_provenance_rejected(self):
        row = self.annotated()
        row["annotation"]["landmarks"][0]["support_image_slots"] = [10]
        with self.assertRaisesRegex(ValueError, "slots"):
            v2.check_row(row, self.native, self.actions, self.root)
        row = self.annotated()
        row["episode_action_prefix_count"] = 123
        with self.assertRaisesRegex(ValueError, "provenance"):
            v2.check_row(row, self.native, self.actions, self.root)


if __name__ == "__main__":
    unittest.main()
