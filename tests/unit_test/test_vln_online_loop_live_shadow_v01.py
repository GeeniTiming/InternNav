"""CPU tests: live passive shadow does not alter native S2 capture rows."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))

import recovery_task_evidence_acquire as native
import vln_online_loop_live_shadow_v01 as live
from vln_online_loop_monitor_v01 import STATIC_KIND


class FakeEvaluator:
    def __init__(self):
        self.max_steps_per_episode = 100
        self.parse_calls = 0
        self.env = SimpleNamespace(
            step=lambda action: (None, 0.0, False, {}),
            get_current_episode=lambda: SimpleNamespace(
                scene_id="data/scene_data/mp3d/scene1/scene1.glb",
                episode_id="20",
                instruction=SimpleNamespace(instruction_text="Walk past the door.")),
            _env=SimpleNamespace(sim=SimpleNamespace(
                get_agent_state=lambda: SimpleNamespace(
                    position=[1., 0., 2.],
                    rotation=SimpleNamespace(w=1., x=0., y=0., z=0.))))
        )

    def parse_actions(self, output):
        self.parse_calls += 1
        return [5] if output == "↓" else [3]


def record_simulated_native_run(root, *, shadow):
    root.mkdir()
    evaluator = FakeEvaluator()
    manifest = {"episodes": [{
        "scene_id": "data/scene_data/mp3d/scene1/scene1.glb",
        "episode_id": "20",
    }]}
    recorder = native.DecisionRecorder(
        evaluator, root, manifest, shadow_monitor=shadow)
    control = root / "control"
    control.mkdir()
    actions = []
    repeated = Image.new("RGB", (10, 10), (15, 25, 35))
    try:
        for i in range(6):
            recorder.step(1)
            actions.append({
                "scene_id": "data/scene_data/mp3d/scene1/scene1.glb",
                "episode_id": "20",
                "action": 1, "done": False,
            })
            recorder.before(
                step=i, action=None,
                instruction="Walk past the door.",
                images=[repeated], history_ids=[], rgb_count=1,
                prompt="Walk past the door.",
                rendered_prompt="Walk past the door. <image>")
            recorder.after(i, "↓")
    finally:
        recorder.close()
    (control / "actions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in actions), encoding="utf-8")
    return evaluator


class ShadowIntegrationTests(unittest.TestCase):
    def test_explicit_opt_in_preserves_original_s2_rows_and_actions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            no_shadow = root / "baseline"
            with_shadow = root / "shadow"
            unmodified = record_simulated_native_run(no_shadow, shadow=False)
            observed = record_simulated_native_run(with_shadow, shadow=True)
            self.assertEqual(unmodified.parse_calls, observed.parse_calls)
            self.assertEqual(unmodified.parse_calls, 6)
            base_rows = (no_shadow / "capture/online_decisions.jsonl").read_bytes()
            shadow_rows = (with_shadow / "capture/online_decisions.jsonl").read_bytes()
            self.assertEqual(base_rows, shadow_rows)
            self.assertEqual(
                (no_shadow / "control/actions.jsonl").read_bytes(),
                (with_shadow / "control/actions.jsonl").read_bytes())
            self.assertFalse((no_shadow / "capture" / live.TIMELINE).exists())
            self.assertFalse((no_shadow / "capture" / live.EVENTS).exists())

            report = live.verify_run(with_shadow)
            self.assertEqual(report["status"], "LIVE_SHADOW_EXACT_PREFIX_REPLAY_MATCH")
            self.assertEqual(report["N_episodes"], 1)
            self.assertEqual(report["N_native_actions"], 6)
            self.assertEqual(report["N_decisions"], 6)
            self.assertEqual(report["N_candidate_events"], 1)
            self.assertEqual(report["N_triggered_episodes"], 1)
            events = list(live.read_jsonl(with_shadow / "capture" / live.EVENTS))
            self.assertEqual(events[0]["kind"], STATIC_KIND)
            self.assertEqual(events[0]["trigger_global_action_prefix"], 6)
            timeline = list(live.read_jsonl(with_shadow / "capture" / live.TIMELINE))
            self.assertEqual(len(timeline), 6)
            self.assertEqual([x["candidate_types_emitted"] for x in timeline[:5]],
                             [[], [], [], [], []])
            self.assertEqual(timeline[-1]["candidate_types_emitted"], [STATIC_KIND])
            self.assertTrue(all("sim_position" not in x
                                and "rendered_prompt" not in x for x in timeline))
            self.assertEqual(live.read_json(with_shadow / "capture" / live.SUMMARY)
                             ["status"], "complete")

    def test_modified_live_decision_record_fails_exact_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            record_simulated_native_run(root, shadow=True)
            file = root / "capture" / live.TIMELINE
            rows = list(live.read_jsonl(file))
            rows[-1]["candidate_types_emitted"] = []
            file.write_text("".join(json.dumps(r) + "\n" for r in rows),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LIVE_SHADOW_NOT_COMPARABLE"):
                live.verify_run(root)

    def test_mismatched_native_action_prefix_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            record_simulated_native_run(root, shadow=True)
            file = root / "control/actions.jsonl"
            rows = list(live.read_jsonl(file))
            file.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]),
                            encoding="utf-8")
            with self.assertRaises((ValueError, FileNotFoundError)):
                live.verify_run(root)

    def test_sink_cannot_overwrite_existing_shadow_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            record_simulated_native_run(root, shadow=True)
            with self.assertRaises(FileExistsError):
                live.LiveShadowSink(root / "capture")

    def test_failed_capture_marked_failed_not_successful(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            root.mkdir()
            capture = root / "capture"
            capture.mkdir()
            sink = live.LiveShadowSink(capture)
            result = sink.close("Interrupted")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"], "Interrupted")
            with self.assertRaises(RuntimeError):
                sink.observe({})


if __name__ == "__main__":
    unittest.main()
