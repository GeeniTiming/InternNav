"""CPU-only tests for audit-only frozen paired-run compatibility (no GPU)."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_paired_audit_v01 as fix


def gate_pair():
    actual = {
        "step": 40,
        "prefix_action_count": 116,
        "state": {"position": [1.0, 0.0, 3.0],
                  "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                  "distance_to_goal_OFFLINE_ONLY": 5.7},
        "observation_hashes": {"rgb": "rr", "gps": "nn"},
        "history_length": 40,
        "history_sha256": "history",
        "rng_sha256": "rng",
        "parameters_sha256": "weights",
        "prefix_trace_sha256": "prefix",
    }
    expected = {
        **copy.deepcopy(actual),
        "checks": {"translated_enough": True, "idle_s2": True},
        "eligible": True,
        "key": ["scene", "episode"],
        "prefix_body_actions": 40,
        "remaining_native_main_steps": 60,
    }
    return actual, expected


def basic_same_fields(actual, expected, fields, label):
    for field in fields:
        if actual[field] != expected[field]:
            raise RuntimeError("NOT_COMPARABLE: " + label + " differs: " + field)


class GateCompatibilityTests(unittest.TestCase):
    def test_frozen_eligibility_enriched_checkpoint_accepts_exact_core(self):
        actual, expected = gate_pair()
        self.assertTrue(fix.compare_native_gate(
            actual, expected, tuple(expected), "paired native gate",
            basic_same_fields) is None)

    def test_each_mismatched_core_field_rejected(self):
        for field in fix.GATE_CORE_FIELDS:
            a, e = gate_pair()
            a[field] = "INVALID"
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "NOT_COMPARABLE"):
                    fix.compare_native_gate(a, e, tuple(e),
                                            "paired native gate",
                                            basic_same_fields)

    def test_absent_or_extra_runtime_core_fields_rejected(self):
        a, e = gate_pair()
        a.pop("history_sha256")
        with self.assertRaisesRegex(RuntimeError, "core schema"):
            fix.compare_native_gate(a, e, tuple(e), "paired native gate",
                                    basic_same_fields)
        a, e = gate_pair()
        a["checks"] = e["checks"]
        with self.assertRaisesRegex(RuntimeError, "core schema"):
            fix.compare_native_gate(a, e, tuple(e), "paired native gate",
                                    basic_same_fields)

    def test_source_checkpoint_missing_core_rejected(self):
        a, e = gate_pair()
        e.pop("rng_sha256")
        with self.assertRaisesRegex(RuntimeError, "missing core"):
            fix.compare_native_gate(a, e, tuple(e), "paired native gate",
                                    basic_same_fields)

    def test_other_comparisons_are_not_weakened(self):
        a, e = gate_pair()
        a["rng_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "NOT_COMPARABLE"):
            fix.compare_native_gate(
                a, e, ("rng_sha256",), "other label",
                basic_same_fields,
            )

    def test_wrong_audit_field_contract_rejected(self):
        a, e = gate_pair()
        with self.assertRaisesRegex(RuntimeError, "caller unexpectedly"):
            fix.compare_native_gate(
                a, e, tuple(fix.GATE_CORE_FIELDS), "paired native gate",
                basic_same_fields)

    def test_wrapper_patch_restored_after_audit(self):
        old = fix.paired.common.same_fields
        with fix.audited_compatibility():
            self.assertIsNot(fix.paired.common.same_fields, old)
            a, e = gate_pair()
            fix.paired.common.same_fields(
                a, e, tuple(e), "paired native gate")
        self.assertIs(fix.paired.common.same_fields, old)

    def test_audit_arm_invokes_existing_full_audit_without_altering_source(self):
        old = fix.paired.common.same_fields
        with patch.object(fix.paired, "audit_single",
                          return_value={"status": "ONE_ARM_PAIRED_AUDITED"}) as audit:
            result = fix.audit_arm(SimpleNamespace(
                run_dir="/frozen/pilot", arm="natural_seek_left45"))
        self.assertEqual(result["status"], "ONE_ARM_PAIRED_AUDITED")
        audit.assert_called_once()
        self.assertIs(fix.paired.common.same_fields, old)


if __name__ == "__main__":
    unittest.main()
