"""CPU-only tests for frozen native decision capture and evidence annotations."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_task_evidence_acquire as acq
import recovery_task_evidence_annotation as ann


class FakeProcessor:
    def __call__(self, text, images, return_tensors):
        return SimpleNamespace(
            to=lambda device: SimpleNamespace(input_ids=SimpleNamespace(shape=[1, 0]))
        )
    tokenizer = None


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens):
        return "→→"


FakeProcessor.tokenizer = FakeTokenizer()


def native_fake(self):
    done, step_id, action = False, 1, getattr(self, "test_action", 3)
    input_images = [Image.new("RGB", (10, 10), (1, 2, 3))]
    history_id = [0]
    rgb_list = ["previous", "current"]
    episode_instruction = "Walk to the bed."
    sources = [{"value": "Walk to the bed. <image>"}]
    text = "Walk to the bed. <image>"
    output_ids = [["x"]]
    while (not done) and (step_id <= self.max_steps_per_episode):
        inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)
        llm_outputs = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        done = True
    return llm_outputs


class FakeHook:
    def __init__(self):
        self.events = []
        self.history_ids = []

    def before(self, *args):
        self.events.append("input")
        self.history_ids.append(args[4])
    def after(self, *args):
        self.events.append("output")


class FakeEval:
    max_steps_per_episode = 50
    processor = FakeProcessor()
    model = SimpleNamespace(device="cpu")

    def __init__(self):
        self._evidence_recorder = FakeHook()


class FakeRawEpisode:
    scene_id = "data/scene_data/mp3d_ce/a/a.glb"
    episode_id = "28"
    instruction = SimpleNamespace(instruction_text="Walk to the bed.")


class FakeState:
    position = [1., 0., 2.]
    rotation = SimpleNamespace(w=1., x=0., y=0., z=0.)


class FakeSim:
    def get_agent_state(self):
        return FakeState()


class FakeEnv:
    _env = SimpleNamespace(sim=FakeSim())

    def get_current_episode(self):
        return FakeRawEpisode()

    def step(self, action):
        return None, 0., False, {}


class RecorderEval(FakeEval):
    def __init__(self):
        self.env = FakeEnv()
        self.max_steps_per_episode = 500

    @staticmethod
    def parse_actions(message):
        return [3] if message == "→" else []


def decision(call_id, scene, step):
    return {
        "schema": acq.SCHEMA, "call_id": call_id,
        "episode_key": [scene, "1"],
        "instruction": "Go to the bed then the closet.",
        "main_step": step, "call_type": "ordinary",
        "native_action_prefix_count": step * 3,
        "history_native_indices": [],
        "images": [{"path": "images/current.png", "pixel_sha256": "abc"}],
        "current_image_ref": "images/current.png",
        "remaining_main_step_budget": 500 - step,
    }


class AcquisitionTests(unittest.TestCase):
    def test_scene_order_outcome_blind(self):
        dataset = {"episodes": [
            {"scene_id": "a/a.glb", "episode_id": "1",
             "instruction": {"instruction_text": "First"}},
            {"scene_id": "a/a.glb", "episode_id": "2",
             "instruction": {"instruction_text": "Different"}},
            {"scene_id": "b/b.glb", "episode_id": "1",
             "instruction": {"instruction_text": "Second"}},
        ]}
        subset = acq.select_train_scenes(dataset, 2)
        self.assertEqual([x["instruction"]["instruction_text"]
                          for x in subset["episodes"]], ["First", "Second"])
        self.assertEqual(len(dataset["episodes"]), 3)
        with self.assertRaisesRegex(ValueError, "only 2"):
            acq.select_train_scenes(dataset, 3)

    def test_process_local_ast_hooks_no_policy_mutation(self):
        wrapped = acq.instrument_loop(native_fake)
        e = FakeEval()
        self.assertEqual(wrapped(e), "→→")
        self.assertEqual(e._evidence_recorder.events, ["input", "output"])
        self.assertEqual(e._evidence_recorder.history_ids, [[0]])

    def test_native_first_step_action_none(self):
        # Actual DualVLN begins each episode with action=None; it must not
        # coerce this to int before the first native System-2 generation.
        wrapped = acq.instrument_loop(native_fake)
        e = FakeEval()
        e.test_action = None
        self.assertEqual(wrapped(e), "→→")
        self.assertEqual(e._evidence_recorder.events, ["input", "output"])
        self.assertEqual(e._evidence_recorder.history_ids, [[0]])

    def test_ast_hook_lookdown_does_not_reference_foreign_globals(self):
        wrapped = acq.instrument_loop(native_fake)
        e = FakeEval()
        e.test_action = 5
        self.assertEqual(wrapped(e), "→→")
        self.assertEqual(e._evidence_recorder.events, ["input", "output"])
        self.assertEqual(e._evidence_recorder.history_ids, [[]])

    def test_ast_source_drift_rejected(self):
        def drift(self):
            done, step_id = False, 0
            while not done and step_id <= self.max_steps_per_episode:
                done = True
        with self.assertRaisesRegex(RuntimeError, "input/output points changed"):
            acq.instrument_loop(drift)

    def test_images_logged_without_privileged_fields(self):
        image = Image.new("RGB", (10, 10), (1, 2, 3))
        with TemporaryDirectory() as t:
            root = Path(t)
            ep = FakeRawEpisode()
            evaluator = RecorderEval()
            rec = acq.DecisionRecorder(evaluator, root, {
                "episodes": [{"scene_id": ep.scene_id, "episode_id": ep.episode_id}]
            })
            rec.before(
                step=10, action=3, instruction=ep.instruction.instruction_text,
                images=[image], history_ids=[], rgb_count=1,
                prompt="Walk to the bed. <image>",
                rendered_prompt="TOKEN: Walk to the bed. <image>",
            )
            rec.after(10, "→")
            rec.close()
            data = acq.read_records(root / "capture/online_decisions.jsonl")
            self.assertEqual(len(data), 1)
            self.assertTrue(acq.validate_online_row(data[0]))
            self.assertEqual(data[0]["native_parsed_actions"], [3])
            self.assertNotIn("sim_position", data[0])
            self.assertIn("sim_position", acq.read_records(
                root / "capture/OFFLINE_ONLY_poses.jsonl")[0])
            self.assertTrue((root / "capture" / data[0]["current_image_ref"]).is_file())
            self.assertEqual(acq.pixel_digest(image), data[0]["images"][0]["pixel_sha256"])
            self.assertEqual(
                acq.pilot.read_json(root / "capture/capture_summary.json")["N_decisions"], 1)

    def test_round_robin_not_outcome_conditioned(self):
        records = ([decision(i, "a", i * 10) for i in range(6)]
                   + [decision(i + 6, "b", i * 10) for i in range(6)])
        first = ann.choose_rows(records, 6, 123, 4)
        second = ann.choose_rows(records, 6, 123, 4)
        self.assertEqual([r["call_id"] for r in first], [r["call_id"] for r in second])
        self.assertEqual(sum(r["episode_key"][0] == "a" for r in first), 3)
        self.assertEqual(sum(r["episode_key"][0] == "b" for r in first), 3)

    def test_annotation_source_validation(self):
        native = decision(5, "scene", 22)
        row = {
            "schema": ann.SCHEMA, "call_id": native["call_id"],
            "episode_key": native["episode_key"], "main_step": native["main_step"],
            "instruction": native["instruction"],
            "current_rgb": "images/current.png", "historical_rgb": [],
            "native_action_prefix_count": native["native_action_prefix_count"],
            "past_action_codes": [],
            "annotation": {
                "status": "complete", "annotator_id": "reviewer1",
                "current_subinstruction_span": "Go to the bed",
                "progress": "before_subgoal",
                "target_landmarks": ["bed"], "visibility": "current",
                "ambiguity": "one_candidate", "history_consistency": "insufficient",
                "support_image_refs": ["images/current.png"],
                "support_image_slots": [0],
                "evidence_regions": [{"slot": 0, "landmark": "bed",
                    "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5]}],
                "reason": "Visible bed supports this target", "confidence": "medium",
            }
        }
        self.assertEqual(ann.check_row(row, native), "complete")
        row["annotation"]["support_image_refs"] = ["images/future.png"]
        with self.assertRaisesRegex(ValueError, "pre-decision"):
            ann.check_row(row, native)
        row["annotation"]["support_image_refs"] = ["images/current.png"]
        row["annotation"]["support_image_slots"] = [1]
        with self.assertRaisesRegex(ValueError, "valid native input indices"):
            ann.check_row(row, native)
        row["annotation"]["support_image_slots"] = [0]
        row["annotation"]["current_subinstruction_span"] = "invented goal"
        with self.assertRaisesRegex(ValueError, "literal span"):
            ann.check_row(row, native)


if __name__ == "__main__":
    unittest.main()
