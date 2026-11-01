"""CPU-only tests for immutable posthoc evidence review generation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_mechanism_review_v01 as review


def frozen_cases():
    return [
        {"key": [scene, ep], "case_id": f"case_{i}"}
        for i, (scene, ep, _) in enumerate(review.TARGETS)
    ] + [
        {"key": ["different_scene", "99"], "case_id": "extra"}
    ]


class ReviewSelectionTests(unittest.TestCase):
    def test_only_four_explicit_posthoc_cases(self):
        picked = review.expected_targets(frozen_cases())
        self.assertEqual(len(picked), 4)
        self.assertEqual(
            [tuple(c["key"]) for c, _ in picked],
            [(s, e) for s, e, _ in review.TARGETS],
        )
        self.assertTrue(all(purpose.startswith("POSTHOC") for _, purpose in picked))

    def test_missing_frozen_case_rejected_not_replaced(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            review.expected_targets(frozen_cases()[:-2])


class EvidenceContractTests(unittest.TestCase):
    def test_real_png_hash_and_history_frozen(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "current.png").write_bytes(b"known-current")
            (folder / "hist.png").write_bytes(b"known-history")
            row = {
                "schema": "intervention_selection_predecision.v0.1",
                "context_id": "case_1_perturbed",
                "episode_key": ["scene", "12"],
                "source_type": "perturbed",
                "no_privileged_simulator_metrics": True,
                "main_step": 40,
                "instruction": "go to the last door",
                "past_action_codes": [1, 1, 2, 2, 2],
                "perturbation_actions_before_context": [2, 2, 2],
                "current_rgb": "current.png",
                "current_png_sha256": review.labels.sha256(folder / "current.png"),
                "current_pixel_sha256": "pixel",
                "historical_rgb": [{
                    "path": "hist.png",
                    "png_sha256": review.labels.sha256(folder / "hist.png"),
                    "pixel_sha256": "hist-pixel",
                    "native_history_index": 0,
                }],
                "historical_rgb_history_sha256": "history-digest",
            }
            got = review.validate_predecision(
                row, folder, ("scene", "12"), "perturbed")
            self.assertEqual(len(got["historical_rgb"]), 1)
            self.assertEqual(got["past_action_codes"][-3:], [2, 2, 2])
            self.assertNotIn("success", got)
            self.assertIsNone(got["mechanism_annotation"])
            bad = copy.deepcopy(row)
            bad["current_png_sha256"] = "tampered"
            with self.assertRaisesRegex(ValueError, "provenance"):
                review.validate_predecision(
                    bad, folder, ("scene", "12"), "perturbed")
            bad = copy.deepcopy(row)
            bad["perturbation_actions_before_context"] = [3, 3, 3]
            with self.assertRaisesRegex(ValueError, "action history"):
                review.validate_predecision(
                    bad, folder, ("scene", "12"), "perturbed")

    def test_not_a_fake_natural_instruction_or_perturbation(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "current.png").write_bytes(b"img")
            row = {
                "schema": "intervention_selection_predecision.v0.1",
                "context_id": "case_0_natural",
                "episode_key": ["scene", "1"],
                "source_type": "natural",
                "no_privileged_simulator_metrics": True,
                "main_step": 20,
                "instruction": " ",
                "past_action_codes": [],
                "perturbation_actions_before_context": [],
                "current_rgb": "current.png",
                "current_png_sha256": review.labels.sha256(folder / "current.png"),
                "current_pixel_sha256": "pixels", "historical_rgb": [],
                "historical_rgb_history_sha256": "h",
            }
            with self.assertRaisesRegex(ValueError, "instruction"):
                review.validate_predecision(
                    row, folder, ("scene", "1"), "natural")

    def test_stop_codes_are_observations_not_intended_semantics(self):
        rows = [
            {"phase": "prefix", "action": 0, "main_step": 3, "done": False},
            {"phase": "prefix", "action": 1, "main_step": 4, "done": False},
            {"phase": "alignment", "action": 2, "main_step": 4, "done": False},
            {"phase": "continuation", "action": 1, "main_step": 5, "done": False},
            {"phase": "continuation", "action": 0, "main_step": 6, "done": True},
        ]
        summary = review.summarize_stop_actions(rows, 2)
        self.assertEqual(summary["continuation_stop_action_count"], 1)
        self.assertEqual(summary["first_continuation_stop_action"]["main_step"], 6)
        self.assertFalse(summary["stop_intention_inferred"])
        self.assertFalse(summary["stop_success_causality_inferred"])

    def test_perturbed_keep_s2_remains_unknown(self):
        unknown = review.first_system2(
            None, ("scene", "8"), "KEEP", "perturbed")
        self.assertEqual(unknown["status"], "NOT_CAPTURED_IN_PERTURBED_KEEP")
        self.assertIsNone(unknown["text"])
        self.assertTrue(unknown["not_assumed_identical_to_other_mode"])
        natural = review.native_source_first_decision([], ("scene", "8"), 40, 75)
        self.assertEqual(natural["status"],
                         "NATIVE_REFERENCE_DECISION_NOT_PRESENT")
        self.assertFalse(natural["use_as_perturbed_keep_output"])

    def test_actual_seek_first_system2_strict_identity(self):
        record = {"first_system2_response": {
            "key": ["scene", "8"], "step": 40, "text": "Turn right"
        }}
        ok = review.first_system2(
            record, ("scene", "8"), "SEEK_RIGHT45", "perturbed")
        self.assertEqual(ok["status"], "ACTUAL_RECORDED_POST_INTERVENTION")
        self.assertEqual(ok["text"], "Turn right")
        with self.assertRaisesRegex(ValueError, "Missing/invalid"):
            review.first_system2(
                record, ("other_scene", "8"), "SEEK_RIGHT45", "perturbed")

    def test_native_source_lookups_require_exact_step_and_prefix(self):
        calls = [
            {"episode_key": ["scene", "8"], "main_step": 40,
             "native_action_prefix_count": 77, "call_type": "ordinary",
             "model_output": "native turn left", "call_id": 5},
        ]
        ok = review.native_source_first_decision(
            calls, ("scene", "8"), 40, 77)
        self.assertEqual(
            ok["status"], "ORIGINAL_NATIVE_SOURCE_DECISION_AT_NATURAL_GATE")
        self.assertFalse(ok["use_as_perturbed_keep_output"])
        missing = review.native_source_first_decision(
            calls, ("scene", "8"), 40, 78)
        self.assertIsNone(missing["text"])


class ReproducibleExportTests(unittest.TestCase):
    def test_immutable_create_only_and_audit_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "review"
            source = Path(td) / "source"
            source.mkdir()
            expected = {
                "schema": review.SCHEMA,
                "source_label_run": str(source),
                "status": "POSTHOC_MECHANISM_EVIDENCE_REVIEW_READY_NOT_ANNOTATED",
                "N_review_scenes": 4, "N_review_contexts": 8,
                "N_review_action_outcomes": 32,
                "source_hashes": {},
            }
            files = {name: (name+"\n").encode() for name in review.OUTPUT_FILES}
            with patch.object(
                review, "build_packet",
                side_effect=lambda *_: (copy.deepcopy(expected), dict(files))
            ):
                review.build(source, out)
                review.audit(out)
                with self.assertRaises(FileExistsError):
                    review.build(source, out)
                with (out / review.OUTPUT_FILES[0]).open("ab") as f:
                    f.write(b"mutated\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    review.audit(out)


if __name__ == "__main__":
    unittest.main()
