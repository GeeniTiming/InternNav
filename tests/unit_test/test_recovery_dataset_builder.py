"""CPU-only tests for strict 2x2 recovery label export (no Habitat/GPU)."""
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "recovery_dataset_builder", ROOT / "scripts/eval/recovery_dataset_builder.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def write_rows(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.cp, self.turn, self.cand, self.output = (
            root / "checkpoint", root / "turn", root / "candidates", root / "dataset")
        self.keys = [("F", "9"), ("S", "16")]
        self.cases = []
        self.values = {
            "00": [0., 1.], "10": [0., 1.],
            "01": [0., 0.], "11": [1., 1.],
        }
        progress = {code: [] for code in BUILDER.ARMS}
        turn_rows, candidate_rows, controls = [], [], []
        for idx, (scene, episode) in enumerate(self.keys):
            gate = {
                "step": 15, "prefix_action_count": 3,
                "history_length": 15, "history_sha256": "old-history",
                "rng_sha256": "rng", "parameters_sha256": "weights",
                "prefix_trace_sha256": "prefix",
                "observation_hashes": {"rgb": "rgb"},
                "state": {"position": [1., 2., 3.], "distance_to_goal_OFFLINE_ONLY": 5.},
            }
            self.cases.append({
                "key": [scene, episode], "status": "COMPARABLE_CHECKPOINT_CAPTURED",
                "checkpoint": gate, "native_trace_comparison": {"actions_and_poses_match": True},
            })
            folder = scene + "_" + episode
            write_json(self.cp / "control/checkpoints" / folder / "checkpoint.json", gate)
            (self.cp / "control/checkpoints" / folder / "rgb.png").write_bytes(b"FAKE_RGB")
            for base, arm in ((self.turn, "right90"), (self.cand, "history_reset"),
                              (self.cand, "right90_reset")):
                write_json(base / arm / "checkpoints" / folder / "boundary.json", gate)
            for arm in ("history_reset", "right90_reset"):
                write_json(self.cand / arm / "checkpoints" / folder / "history_inputs.json", [
                    {"cutoff": 15, "rgb_count": 16, "selected_history_ids": []},
                    {"cutoff": 15, "rgb_count": 17, "selected_history_ids": [15]},
                ])
            for code in BUILDER.ARMS:
                progress[code].append({
                    "scene_id": scene, "episode_id": episode,
                    "episode_instruction": "go to door " + episode,
                    "success": self.values[code][idx],
                    "spl": 0.5 * self.values[code][idx],
                    "os": self.values[code][idx], "ne": 1. if self.values[code][idx] else 9.,
                    "steps": 40,
                })
            turn_rows.append({
                "key": [scene, episode], "forced_turns": 6,
                "success": self.values["10"][idx],
                "control_success": self.values["00"][idx],
                "control_ne": 9. if not self.values["00"][idx] else 1.,
            })
            for arm, code, cost in (("history_reset", "01", 0), ("right90_reset", "11", 6)):
                candidate_rows.append({
                    "key": [scene, episode], "arm": arm,
                    "success": self.values[code][idx],
                    "ne": 1. if self.values[code][idx] else 9.,
                    "external_actions": cost, "history_prompt_events": 2,
                    "baseline_success": self.values["00"][idx],
                    "baseline_os": self.values["00"][idx],
                    "baseline_ne": 9. if not self.values["00"][idx] else 1.,
                })
            for action in (1, 2, 3, 0):
                controls.append({
                    "scene_id": "mp3d/" + scene + "/" + scene + ".glb",
                    "episode_id": episode, "action": action,
                    "position": [5., 6., 7.], "orientation_wxyz": [1., 0., 0., 0.]
                })
        write_json(self.cp / "checkpoint_report.json", {
            "schema": BUILDER.SCHEMAS["checkpoint"],
            "status": "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION",
            "cases": self.cases
        })
        write_json(self.turn / "turn_report.json", {
            "schema": BUILDER.SCHEMAS["turn"],
            "status": "READY_FOR_REVIEW", "outcomes": {"right90": turn_rows}
        })
        write_json(self.cand / "candidate_report.json", {
            "schema": BUILDER.SCHEMAS["candidates"],
            "status": "READY_FOR_INTERPRETATION",
            "source_control_reused": True,
            "outcomes": candidate_rows,
        })
        write_json(self.turn / "turn_protocol.json", {
            "cases": self.cases, "forced_turn_actions": 6,
            "native_main_step_limit": 500, "charged_main_step_limit": 494
        })
        write_json(self.cand / "candidate_protocol.json", {
            "cases": self.cases, "native_main_step_limit": 500,
            "arms": ["history_reset", "left90_reset", "right90_reset", "backtrack_replan"],
            "release_control_source": str(self.cp / "control"),
        })
        write_json(self.turn / "manifest.json", {
            "orientation_turn": {"source_run": str(self.cp)}
        })
        write_json(self.cand / "manifest.json", {
            "recovery_candidates": {"source_run": str(self.cp)}
        })
        write_rows(self.cp / "control/progress.json", progress["00"])
        write_rows(self.turn / "right90/progress.json", progress["10"])
        write_rows(self.cand / "history_reset/progress.json", progress["01"])
        write_rows(self.cand / "right90_reset/progress.json", progress["11"])
        write_rows(self.cp / "control/actions.jsonl", controls)

    def build(self):
        return BUILDER.make_dataset(self.cp, self.turn, self.cand, self.output)

    def test_paired_labels_and_interaction_gamma(self):
        report = self.build()
        self.assertEqual(report["checkpoint_count"], 2)
        self.assertEqual(report["candidate_count"], 8)
        self.assertEqual(report["interaction_gamma_counts"]["1"], 2)
        self.assertFalse(report["ready_for_full_multimodal_training"])
        labels = list(BUILDER.lines(self.output / "paired_labels.jsonl"))
        self.assertEqual([l["interaction_gamma"] for l in labels], [1, 1])
        self.assertEqual([l["Y11_right90_reset"] for l in labels], [1, 1])
        self.assertEqual(report["metrics_by_arm"]["right90_reset"]["rescued"], 1)
        self.assertEqual(report["metrics_by_arm"]["history_reset"]["harmed"], 1)

    def test_exports_action_ids_not_privileged_positions(self):
        self.build()
        text = (self.output / "contexts/F_9/prefix_actions.jsonl").read_text()
        self.assertNotIn("position", text)
        self.assertNotIn("orientation", text)
        self.assertNotIn("distance_to_goal", text)
        self.assertEqual([r["action"] for r in BUILDER.lines(
            self.output / "contexts/F_9/prefix_actions.jsonl")], [1, 2, 3])
        contexts = list(BUILDER.lines(self.output / "contexts.jsonl"))
        self.assertTrue(all(not c["history_rgb_available"] for c in contexts))
        self.assertEqual(contexts[0]["history_rgb_paths"], [])

    def test_reject_changed_checkpoint_before_creating_output(self):
        path = self.cand / "right90_reset/checkpoints/F_9/boundary.json"
        gate = BUILDER.load_json(path)
        gate["history_sha256"] = "wrong"
        write_json(path, gate)
        with self.assertRaisesRegex(ValueError, "NOT_COMPARABLE"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_reject_changed_episode_success(self):
        path = self.cand / "right90_reset/progress.json"
        rows = list(BUILDER.lines(path))
        rows[0]["success"] = 0.
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "report/progress SR mismatch"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_reject_pre_handoff_reset_history_image(self):
        path = self.cand / "history_reset/checkpoints/F_9/history_inputs.json"
        entries = BUILDER.load_json(path)
        entries[1]["selected_history_ids"] = [14]
        write_json(path, entries)
        with self.assertRaisesRegex(ValueError, "history leaked"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_refuse_overwrite_existing_dataset(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()


if __name__ == "__main__":
    unittest.main()
