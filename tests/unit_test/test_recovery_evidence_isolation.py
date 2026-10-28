"""CPU-only evidence-isolation tests; no Habitat model or GPU required."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_evidence_isolation as mech


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens):
        return "MOVE_LEFT"


class FakeProcessor:
    tokenizer = FakeTokenizer()


class FakeCandidate:
    def __init__(self):
        self.calls = []

    def on_boundary(self, step, observations, done, rgb_list, actions, pixel, local, action):
        self.calls.append("boundary")
        return observations, done

    def filter_history_ids(self, step, ids, length):
        self.calls.append("history")
        return ids

    def inject_extra_view(self, step, sources, images):
        self.calls.append("image")
        sources[0]["value"] += mech.EXTRA_PROMPT
        images.insert(-1, images[-1])

    def record_system2_output(self, step, output):
        self.calls.append("response")
        if output != "MOVE_LEFT":
            raise RuntimeError("Incorrect decoded System-2 response")


class FakeEvaluator:
    max_steps_per_episode = 10
    num_history = 1
    processor = FakeProcessor()

    def __init__(self):
        self._recovery_candidate = FakeCandidate()


def fake_native_loop(self):
    observations, done = {"rgb": "not_used"}, False
    rgb_list = ["previous", "current"]
    action_seq, pixel_goal, local_actions, action = [], None, [], 1
    step_id = 1
    output_ids = [["token"]]
    sources = [{"value": "An instruction."}]
    cur_images = ["current"]
    inputs = type("InputStub", (), {"input_ids": type("InputIdStub", (), {"shape": [1, 0]})()})()
    while (not done) and (step_id <= self.max_steps_per_episode):
        history_id = np.unique(
            np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
        ).tolist()
        input_images = [rgb_list[i] for i in history_id] + cur_images
        llm_outputs = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        self.last_prompt = sources[0]["value"]
        self.last_image_count = len(input_images)
        done = True
    return llm_outputs


class MechanismTests(unittest.TestCase):
    def test_symmetric_turn_return_arms(self):
        self.assertEqual(len(mech.ARMS), 4)
        self.assertEqual(set(mech.ARM_ANGLES.values()), {-45, 45})
        self.assertEqual(mech.SCAN_ACTIONS, 6)
        for arm in mech.ARMS:
            self.assertEqual(mech.arm_pair(arm) + (
                "_evidence" if arm.endswith("_evidence") else "_placebo"
            ), arm[len("lookback_"):])

    def test_ast_hooks_order_once_only(self):
        wrapped = mech.instrument_loop(fake_native_loop)
        evaluator = FakeEvaluator()
        self.assertEqual(wrapped(evaluator), "MOVE_LEFT")
        self.assertEqual(
            evaluator._recovery_candidate.calls,
            ["boundary", "history", "image", "response"],
        )
        self.assertEqual(evaluator.last_image_count, 3)
        self.assertEqual(evaluator.last_prompt.count("<image>"), 1)

    def test_native_shape_drift_rejected(self):
        def altered(self):
            done, step_id = False, 1
            while not done and step_id <= self.max_steps_per_episode:
                done = True
        with self.assertRaisesRegex(RuntimeError, "native history/image/response"):
            mech.instrument_loop(altered)

    def test_outbound_and_return_cost(self):
        for arm in mech.ARMS:
            angle = mech.ARM_ANGLES[arm]
            self.assertEqual(abs(angle) // 15 * 2, mech.SCAN_ACTIONS)
            self.assertEqual(
                ([2, 2, 2, 3, 3, 3] if angle > 0 else [3, 3, 3, 2, 2, 2]),
                ([2] * 3 + [3] * 3 if angle == 45 else [3] * 3 + [2] * 3),
            )


if __name__ == "__main__":
    unittest.main()
