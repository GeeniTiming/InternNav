"""CPU-only regression checks for scene-disjoint acquisition and predecision labels."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_train_disjoint_acquire as train
import recovery_train_checkpoint_evidence as evidence


def episode(scene, eid, ins="Go to the doorway"):
    return {"scene_id": "assets/" + scene + "/" + scene + ".glb",
            "episode_id": str(eid), "instruction": {"instruction_text": ins}}


def blank_record():
    return {
        "schema": evidence.SCHEMA, "episode_key": ["SceneC", "2"],
        "main_step": 30,
        "instruction": "Go around the table and stop beside the doorway.",
        "historical_rgb": ["h0", "h1"],
        "current_rgb": "current",
        "image_pixel_hashes_by_slot": ["sha1", "sha2", "sha2"],
        "past_action_codes": [1, 2, 2],
        "annotation": {
            "status": "complete", "annotator_id": "reviewer",
            "stage_span": "", "progress": "unknown", "progress_basis": "unknown",
            "action_evidence_indices": [], "history_consistency": "insufficient",
            "conflict_slot_pairs": [],
            "landmarks": [
                {"name": "table", "instruction_span": "table",
                 "role": "route_anchor", "visibility": "history_only",
                 "support_image_slots": [0], "regions": [],
                 "reason": "Table appears in an earlier native view."},
                {"name": "doorway", "instruction_span": "doorway",
                 "role": "stop_reference", "visibility": "uncertain",
                 "support_image_slots": [], "regions": [],
                 "reason": "No task-matching doorway confirmed."},
            ],
            "reason": "Cannot determine the current subinstruction reliably.",
            "confidence": "low",
        },
    }


class TrainCohortTests(TestCase):
    def test_inspected_train_scenes_excluded_and_order_preserved(self):
        src = {"episodes": [
            episode("5LpN3gDmAk7", 4),
            episode("SceneA", 1),
            episode("SceneA", 2),
            episode("X7HyMhZNoso", 46),
            episode("SceneB", 1),
            episode("SceneC", 1),
        ]}
        original = copy.deepcopy(src)
        out = train.select_outcome_blind_new_scenes(src, 2, train.INSPECTED)
        self.assertEqual([train.capture.episode_key(x) for x in out["episodes"]],
                         [("SceneA", "1"), ("SceneB", "1")])
        self.assertEqual(original, src)

    def test_disjoint_selector_needs_enough_distinct_scenes(self):
        src = {"episodes": [episode("SceneA", 1), episode("SceneA", 2),
                             episode("5LpN3gDmAk7", 4)]}
        with self.assertRaisesRegex(ValueError, "only 1"):
            train.select_outcome_blind_new_scenes(src, 2, train.INSPECTED)

    def test_manifest_rejects_inspected_overlap(self):
        manifest = {
            "episodes": [episode("SceneA", 1), episode("SceneB", 1)],
            "task_evidence_capture": {
                "schema": train.acquire.SCHEMA, "split": "train", "episodes": 2},
            "train_scene_disjoint": {
                "schema": train.SCHEMA, "outcome_blind": True,
                "prior_scenes_excluded": True,
                "excluded_scene_stems": ["SceneX"],
            },
        }
        self.assertEqual(len(train.validate_selection(manifest)), 2)
        manifest["train_scene_disjoint"]["excluded_scene_stems"] = ["SceneA"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            train.validate_selection(manifest)

    def test_selection_does_not_recurse_through_patched_global(self):
        src = {"episodes": [episode("SceneA", 1), episode("SceneB", 2)]}
        original = train.acquire.select_train_scenes
        try:
            def installed(dataset, count):
                return train.select_outcome_blind_new_scenes(dataset, count, ())
            train.acquire.select_train_scenes = installed
            result = train.acquire.select_train_scenes(src, 1)
            self.assertEqual(len(result["episodes"]), 1)
        finally:
            train.acquire.select_train_scenes = original

    def test_exclusion_source_uses_verified_manifest(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            native = {"episodes": [episode("SceneA", 4), episode("SceneB", 4)]}
            original = train.pilot.verify_manifest
            try:
                train.pilot.verify_manifest = lambda r: native
                self.assertEqual(train.scene_stems_from_locked_run(root),
                                 {"SceneA", "SceneB"})
            finally:
                train.pilot.verify_manifest = original


class CheckpointEvidenceTests(TestCase):
    def test_unknown_stage_and_landmark_roles(self):
        self.assertEqual(evidence.validate_review(blank_record()), "complete")

    def test_no_citation_of_future_action(self):
        x = blank_record()
        x["annotation"]["action_evidence_indices"] = [3]
        with self.assertRaisesRegex(ValueError, "earlier actions"):
            evidence.validate_review(x)

    def test_reject_conflict_of_duplicate_rgb(self):
        x = blank_record()
        x["annotation"]["history_consistency"] = "conflicting"
        x["annotation"]["conflict_slot_pairs"] = [[1, 2]]
        with self.assertRaisesRegex(ValueError, "identical"):
            evidence.validate_review(x)

    def test_reject_progress_without_action_evidence(self):
        x = blank_record()
        x["annotation"].update({
            "progress": "before_subgoal", "progress_basis": "actions",
            "stage_span": "Go around the table",
        })
        with self.assertRaisesRegex(ValueError, "needs predecision action"):
            evidence.validate_review(x)
        x["annotation"]["action_evidence_indices"] = [0, 1]
        self.assertEqual(evidence.validate_review(x), "complete")

    def test_reject_invalid_landmark_slot(self):
        x = blank_record()
        x["annotation"]["landmarks"][0]["support_image_slots"] = [9]
        with self.assertRaisesRegex(ValueError, "image slots"):
            evidence.validate_review(x)

    def test_allows_distinct_rgb_conflict_evidence_as_structure_only(self):
        x = blank_record()
        x["annotation"]["history_consistency"] = "conflicting"
        x["annotation"]["conflict_slot_pairs"] = [[0, 2]]
        self.assertEqual(evidence.validate_review(x), "complete")
        # Passing structural validation is NOT evidence that human-visible
        # semantic historical contradiction actually exists.


if __name__ == "__main__":
    main()
