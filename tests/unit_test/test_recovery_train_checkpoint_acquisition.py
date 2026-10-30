"""CPU-only regression checks for scene-disjoint acquisition and predecision labels."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
from PIL import Image

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
    def test_checkpoint_context_has_exact_native_history_and_prefix(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "native"
            source.mkdir()
            root = Path(tmp) / "checkpoint"
            folder = root / "control/checkpoints/SceneC_2"
            hist_dir = folder / "historical_rgb"
            hist_dir.mkdir(parents=True)
            native_ep = episode("SceneC", 2,
                "Go around the table and stop beside the doorway.")
            image = Image.new("RGB", (12, 12), (10, 25, 30))
            current = Image.new("RGB", (12, 12), (40, 50, 60))
            hpath = hist_dir / "history_0000.png"
            cpath = folder / "current_model_rgb.png"
            image.save(hpath)
            current.save(cpath)
            instructions = native_ep["instruction"]["instruction_text"]
            actions = []
            for act in (3, 4):
                actions.append({
                    "scene_id": native_ep["scene_id"], "episode_id": "2",
                    "action": act, "done": False, "position": [0., 0., 0.],
                    "orientation_wxyz": [1., 0., 0., 0.]
                })
            (root / "control").mkdir(exist_ok=True)
            with (root / "control/actions.jsonl").open("w") as stream:
                for action in actions:
                    stream.write(json.dumps(action) + "\n")
            gate = {
                "step": 30, "history_sha256": "frozen_history",
                "prefix_action_count": 2,
                "prefix_trace_sha256": evidence.cp.cohort.digest(actions),
            }
            evidence.pilot.write_json(folder / "context_inputs.json", {
                "schema": "independent_recovery_context_inputs.v1",
                "key": ["SceneC", "2"], "main_step": 30,
                "history_sha256": "frozen_history",
                "sampled_history": [{
                    "path": "historical_rgb/history_0000.png",
                    "pixel_sha256": evidence.physical.history_hash([image]),
                    "png_sha256": evidence.pilot.sha256(hpath),
                }],
                "current_model_rgb_path": "current_model_rgb.png",
                "current_model_rgb_pixel_sha256":
                    evidence.physical.history_hash([current]),
                "current_model_rgb_png_sha256": evidence.pilot.sha256(cpath),
            })
            evidence.pilot.write_json(root / "checkpoint_report.json", {
                "schema": "independent_recovery_checkpoints.v1",
                "status": "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION",
                "cases": [{
                    "key": ["SceneC", "2"],
                    "status": "COMPARABLE_CHECKPOINT_CAPTURED",
                    "checkpoint": gate,
                }],
            })
            plan = {"cases": [{
                "key": ["SceneC", "2"], "instruction": instructions,
            }]}
            data = (plan, None, source, None,
                    {"episodes": [native_ep]}, None, None)
            with patch.object(evidence.cp, "read_run", return_value=(None, data)), \
                 patch.object(evidence.pilot, "verify_manifest", return_value={}), \
                 patch.object(evidence.train, "validate_selection",
                              return_value=[("SceneC", "2")]):
                records = evidence.native_contexts(root)
            self.assertEqual(len(records), 1)
            row = records[0]
            self.assertEqual(row["main_step"], 30)
            self.assertEqual(row["past_action_codes"], [3, 4])
            self.assertEqual(row["episode_action_prefix_count"], 2)
            self.assertEqual(row["instruction"], instructions)
            self.assertEqual(row["annotation"]["status"], "unreviewed")
            # Extra future/offline state must never be exported.
            self.assertNotIn("distance_to_goal", row)
            self.assertNotIn("sim_position", row)

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
