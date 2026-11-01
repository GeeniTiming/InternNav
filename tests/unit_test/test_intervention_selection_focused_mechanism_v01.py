"""CPU-only contract tests for the posthoc three-scene ablation plan."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_focused_mechanism_v01 as f


def fixture_cases():
    return [
        {"key": [scene, episode], "case_id": "case_" + episode,
         "checkpoint": {"step": 30, "prefix_action_count": 42}}
        for scene, episode, _ in reversed(f.FOCUS)
    ]


class FocusSelectionTests(unittest.TestCase):
    def test_exact_three_independent_scenes_and_predeclared_sides(self):
        result = f.select_cases(fixture_cases())
        self.assertEqual([tuple(c["key"]) for c in result],
                         [(scene, episode) for scene, episode, _ in f.FOCUS])
        self.assertEqual([c["focus_scan_side"] for c in result],
                         ["right", "right", "left"])
        self.assertEqual(len(result), 3)

    def test_does_not_edit_source_cases(self):
        source = fixture_cases()
        original = copy.deepcopy(source)
        f.select_cases(source)
        self.assertEqual(source, original)

    def test_missing_focus_fails_closed_no_replacement(self):
        cases = fixture_cases()[:-1]
        with self.assertRaisesRegex(ValueError, "missing"):
            f.select_cases(cases)

    def test_duplicate_original_keys_rejected(self):
        cases = fixture_cases()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            f.select_cases(cases + [copy.deepcopy(cases[0])])

    def test_missing_checkpoint_rejected(self):
        cases = fixture_cases()
        cases[0]["checkpoint"] = None
        with self.assertRaisesRegex(ValueError, "Uncaptured"):
            f.select_cases(cases)

    def test_arm_vocabulary_and_posthoc_scope(self):
        self.assertEqual(f.ARMS, (
            "perturbed_keep", "perturbed_scan_no_injection",
            "perturbed_placebo", "perturbed_real_seek",
        ))
        self.assertEqual(len(f.FOCUS) * len(f.ARMS), 12)
        self.assertIn("Posthoc", f.SCOPE)


class NoInjectionTests(unittest.TestCase):
    def fake_probe(self):
        probe = object.__new__(f.FocusScanProbe)
        probe.focus_role = f.ARMS[1]
        probe.gate = None
        probe.inject_pending = True
        probe.inject_count = 0
        return probe

    def test_no_injection_disarmed_at_first_handoff(self):
        probe = self.fake_probe()
        def original(_probe, *_args):
            _probe.gate = {"step": 21}
            return "observation", False
        with patch.object(f.paired.PairProbe, "on_boundary", new=original):
            result = probe.on_boundary(21, None)
        self.assertEqual(result, ("observation", False))
        self.assertFalse(probe.inject_pending)

    def test_no_injection_disarmed_after_episode_changes(self):
        probe = self.fake_probe()
        def original(_probe, *_args):
            _probe.gate = {"step": 31}
            _probe.inject_pending = True
            return "observation", False
        with patch.object(f.paired.PairProbe, "on_boundary", new=original):
            probe.on_boundary(31)
            probe.gate = None
            probe.on_boundary(31)
        self.assertFalse(probe.inject_pending)

    def test_no_injection_does_not_touch_prompt_or_images(self):
        probe = self.fake_probe()
        probe.inject_pending = False
        sources = [{"value": "original prompt"}]
        images = ["old_image", "current_image"]
        probe.inject_extra_view(21, sources, images)
        self.assertEqual(sources, [{"value": "original prompt"}])
        self.assertEqual(images, ["old_image", "current_image"])

    def test_no_injection_rejects_pending_image(self):
        probe = self.fake_probe()
        with self.assertRaisesRegex(RuntimeError, "changed S2 prompt"):
            probe.inject_extra_view(21, [{"value": "text"}], ["image"])

    def test_placebo_and_real_use_native_image_hook(self):
        for role in (f.ARMS[2], f.ARMS[3]):
            probe = object.__new__(f.FocusScanProbe)
            probe.focus_role = role
            with patch.object(f.paired.PairProbe, "inject_extra_view") as inherited:
                probe.inject_extra_view(33, [{"value": "text"}], ["image"])
                inherited.assert_called_once()


if __name__ == "__main__":
    unittest.main()
