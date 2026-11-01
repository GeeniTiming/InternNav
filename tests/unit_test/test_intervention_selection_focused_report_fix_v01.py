"""Regression tests for the source-hash-safe, report-only frozen gate hotfix."""
from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_focused_report_fix_v01 as fix


def paired_gates():
    replayed = {name: "value:" + name for name in fix.RUNTIME_GATE_FIELDS}
    frozen = copy.deepcopy(replayed)
    frozen.update({
        "checks": {name: True for name in fix.EXPECTED_CHECKS},
        "eligible": True, "key": ["scene", "70"],
        "prefix_translation_m": 2.1,
        "remaining_native_main_steps": 80,
    })
    return replayed, frozen


class GateSchemaTests(unittest.TestCase):
    def setUp(self):
        self.label = sorted(fix.GATE_LABELS)[0]
        self.original = fix.focus.common.same_fields

    def test_original_frozen_metadata_no_longer_raises_keyerror(self):
        actual, frozen = paired_gates()
        result = fix.verify_replay_gate(actual, frozen, self.label, self.original)
        self.assertIs(result, True)
        self.assertNotIn("checks", actual)

    def test_wrong_pose_history_rng_or_model_still_fails(self):
        for changed in ("state", "history_sha256", "rng_sha256",
                        "observation_hashes", "parameters_sha256",
                        "prefix_trace_sha256"):
            actual, frozen = paired_gates()
            actual[changed] = "wrong"
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, changed):
                    fix.verify_replay_gate(actual, frozen, self.label, self.original)

    def test_ineligible_or_bad_source_metadata_rejected(self):
        actual, frozen = paired_gates()
        frozen["eligible"] = False
        with self.assertRaisesRegex(ValueError, "not an eligible"):
            fix.verify_replay_gate(actual, frozen, self.label, self.original)
        actual, frozen = paired_gates()
        frozen["checks"]["idle_s2"] = False
        with self.assertRaisesRegex(ValueError, "not an eligible"):
            fix.verify_replay_gate(actual, frozen, self.label, self.original)
        actual, frozen = paired_gates()
        del frozen["checks"]["threshold_reached"]
        with self.assertRaisesRegex(ValueError, "not an eligible"):
            fix.verify_replay_gate(actual, frozen, self.label, self.original)

    def test_missing_replay_field_never_silently_ignored(self):
        actual, frozen = paired_gates()
        del actual["history_sha256"]
        with self.assertRaisesRegex(ValueError, "Unexpected replay gate schema"):
            fix.verify_replay_gate(actual, frozen, self.label, self.original)

    def test_unknown_labels_delegate_to_original(self):
        actual, frozen = paired_gates()
        with patch.object(fix.focus.common, "same_fields", wraps=self.original) as mocked:
            f = fix.audited_comparator(mocked)
            f(actual, frozen, ("step",), "other audit")
            mocked.assert_called_once_with(actual, frozen, ("step",), "other audit")

    def test_wrong_call_shape_rejected(self):
        actual, frozen = paired_gates()
        f = fix.audited_comparator(self.original)
        with self.assertRaisesRegex(ValueError, "Unexpected frozen-gate comparison"):
            f(actual, frozen, ("step",), self.label)


class OutputSafetyTests(unittest.TestCase):
    def test_report_uses_existing_driver_and_restores_comparator(self):
        with TemporaryDirectory() as temp:
            original = fix.focus.common.same_fields
            with patch.object(fix.focus, "report", return_value={
                    "N_actual_rollouts": 12}) as mocked:
                result = fix.report(temp)
            self.assertEqual(result["N_actual_rollouts"], 12)
            self.assertIs(fix.focus.common.same_fields, original)
            mocked.assert_called_once()

    def test_report_restores_comparator_on_failure(self):
        with TemporaryDirectory() as temp:
            original = fix.focus.common.same_fields
            with patch.object(fix.focus, "report", side_effect=RuntimeError("replay mismatch")):
                with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
                    fix.report(temp)
            self.assertIs(fix.focus.common.same_fields, original)

    def test_refuses_to_overwrite_existing_report(self):
        with TemporaryDirectory() as temp:
            (Path(temp) / "focused_mechanism_report.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                fix.report(temp)


if __name__ == "__main__":
    unittest.main()
