"""CPU tests for the frozen single-episode mechanism diagnosis (no GPU)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_episode13_ablation as ab


def fake_case():
    return {
        "key": list(ab.TARGET),
        "checkpoint": {"step": 21, "prefix_action_count": 108}
    }


class ProtocolTest(unittest.TestCase):
    def test_exact_predeclared_context_and_four_conditions(self):
        self.assertEqual(ab.TARGET, ("cV4RVeZvu5T", "13"))
        self.assertEqual(len(ab.ARMS), 4)
        self.assertEqual(set(ab.OLD_MAP), {
            "replay_placebo_reset", "replay_seek_keep"
        })
        p = ab.make_protocol(fake_case(), 500)
        self.assertEqual(p["cases"][0]["key"], list(ab.TARGET))
        self.assertEqual(p["native_main_step_limit"], 500)
        self.assertEqual(p["charged_native_main_step_limit"], 494)
        self.assertEqual(p["per_arm_extra_actions"][ab.ARMS[0]], 0)
        self.assertEqual(p["per_arm_extra_actions"][ab.ARMS[1]], 6)
        self.assertTrue(p["per_arm_reset"][ab.ARMS[0]])
        self.assertFalse(p["per_arm_reset"][ab.ARMS[3]])

    def test_protocol_budget_gate(self):
        with self.assertRaises(ValueError):
            ab.make_protocol(fake_case(), 25)

    def test_single_context_scope_is_not_efficacy(self):
        self.assertIn("no reranking", ab.SCOPE)
        self.assertIn("one", ab.SCOPE.lower())
        self.assertNotIn("generalization proved", ab.SCOPE)


class InjectionTest(unittest.TestCase):
    def test_no_injection_does_not_mutate_images_or_prompt(self):
        probe = object.__new__(ab.ScanNoInjection)
        probe.inject_pending = True
        probe.inject_count = 0
        probe.gate = {"step": 21}
        probe.phase = "continuation"
        probe.external_actions = 6
        probe.injection_diagnostic = []
        sources = [{"value": "native prompt"}]
        images = ["image_0", "current_image"]
        ab.ScanNoInjection.inject_extra_view(probe, 21, sources, images)
        self.assertEqual(sources, [{"value": "native prompt"}])
        self.assertEqual(images, ["image_0", "current_image"])
        self.assertFalse(probe.inject_pending)
        self.assertFalse(probe.injection_diagnostic[0]["injected"])
        self.assertEqual(probe.injection_diagnostic[0]["external_actions"], 6)

    def test_virtual_placebo_duplicates_current_slot_only(self):
        from PIL import Image
        probe = object.__new__(ab.VirtualPlacebo)
        probe.pending_virtual = True
        probe.injected_virtual = 0
        probe.gate = {"step": 21}
        probe.phase = "continuation"
        probe.injection_diagnostic = []
        im = Image.new("RGB", (4, 4), (20, 30, 40))
        sources = [{"value": "native prompt"}]
        images = [im]
        ab.VirtualPlacebo.inject_extra_view(probe, 21, sources, images)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].tobytes(), images[1].tobytes())
        self.assertIsNot(images[0], images[1])
        self.assertIn(ab.iso.EXTRA_PROMPT, sources[0]["value"])
        self.assertEqual(probe.injected_virtual, 1)
        self.assertFalse(probe.pending_virtual)
        self.assertTrue(probe.injection_diagnostic[0]["injected"])

    def test_duplicate_virtual_injection_is_not_repeated(self):
        probe = object.__new__(ab.VirtualPlacebo)
        probe.pending_virtual = False
        sources = [{"value": "native"}]
        images = ["unchanged"]
        ab.VirtualPlacebo.inject_extra_view(probe, 99, sources, images)
        self.assertEqual(images, ["unchanged"])
        self.assertEqual(sources[0]["value"], "native")


class DecisionCaptureTest(unittest.TestCase):
    def test_step33_capture_uses_generated_text_not_hand_labeled_belief(self):
        class DummyParent:
            def begin(self, key):
                self.current = key
                self.gate = {"step": 21}
                self.phase = "continuation"
                self.records = [1, 2, 3]

            def filter_history_ids(self, step, ids, rgb_count):
                return ids[-2:]

        class Probe(ab.DecisionLogger, DummyParent):
            pass

        probe = Probe()
        probe.begin(ab.TARGET)
        selected = probe.filter_history_ids(33, [0, 2, 4, 8], 34)
        probe.record_system2_output(33, "turn left")
        self.assertEqual(selected, [4, 8])
        self.assertEqual(probe.decisions[0]["step"], 33)
        self.assertEqual(probe.decisions[0]["text"], "turn left")
        self.assertEqual(probe.decisions[0]["action_count_at_generation"], 3)
        self.assertEqual(
            probe.history_selections[0]["selected_history_indices"], [4, 8])


if __name__ == "__main__":
    unittest.main()
