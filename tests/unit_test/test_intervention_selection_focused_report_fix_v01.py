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
            with (patch.object(fix, "pinned_focus_module", return_value=fix.focus),
                  patch.object(fix, "focused_report_with_camera_audit",
                               side_effect=lambda original: original),
                  patch.object(fix.focus, "report", return_value={
                      "N_actual_rollouts": 12}) as mocked):
                result = fix.report(temp)
            self.assertEqual(result["N_actual_rollouts"], 12)
            self.assertIs(fix.focus.common.same_fields, original)
            mocked.assert_called_once()

    def test_report_restores_comparator_on_failure(self):
        with TemporaryDirectory() as temp:
            original = fix.focus.common.same_fields
            with (patch.object(fix, "pinned_focus_module", return_value=fix.focus),
                  patch.object(fix, "focused_report_with_camera_audit",
                               side_effect=lambda original: original),
                  patch.object(fix.focus, "report",
                               side_effect=RuntimeError("replay mismatch"))):
                with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
                    fix.report(temp)
            self.assertIs(fix.focus.common.same_fields, original)

    def test_refuses_to_overwrite_existing_report(self):
        with TemporaryDirectory() as temp:
            (Path(temp) / "focused_mechanism_report.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                fix.report(temp)

    def test_pinned_loader_rejects_changed_source_sha(self):
        import hashlib
        import json
        with TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "intervention_selection_focused_mechanism_v01.py"
            src.write_text("ARMS = ()\n", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "source_hashes": {str(src): hashlib.sha256(b"different").hexdigest()}
            }))
            with self.assertRaisesRegex(RuntimeError, "FROZEN_SOURCE_SHA_MISMATCH"):
                fix.pinned_focus_module(root)

    def test_pinned_loader_reads_exact_manifest_source(self):
        import hashlib
        import json
        with TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "intervention_selection_focused_mechanism_v01.py"
            source = "ARMS = " + repr(fix.focus.ARMS) + "\n"
            src.write_text(source, encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "source_hashes": {
                    str(src): hashlib.sha256(src.read_bytes()).hexdigest()
                }
            }))
            loaded = fix.pinned_focus_module(root)
            self.assertEqual(tuple(loaded.ARMS), tuple(fix.focus.ARMS))
            self.assertEqual(Path(loaded.__file__).resolve(), src.resolve())



class NativeCameraBoundaryTests(unittest.TestCase):
    def setup_input(self):
        key = ("scene", "70")
        gate = {"step": 40, "prefix_action_count": 12}
        tagged, decisions = {}, {}
        for arm in fix.focus.ARMS:
            count = 0 if arm == fix.focus.ARMS[0] else 6
            phases = (["prefix"] * 12
                      + ["alignment"] * 3
                      + ["external_recovery"] * count
                      + ["continuation"] * 4)
            actions = [
                {"phase": phase, "action": (
                    [5, 5, 4, 4][i-(len(phases)-4)]
                    if i >= len(phases)-4 else 2),
                 "main_step": 40} for i, phase in enumerate(phases)
            ]
            tagged[arm] = {key: actions}
            decisions[arm] = {key: [
                {"phase": "continuation",
                 "main_step": 40,
                 "episode_call_id": 6,
                 "native_action_prefix_count": len(phases)}
            ]}
        return key, gate, tagged, decisions

    def test_exact_real_camera_steps_accepted(self):
        key, gate, tagged, decisions = self.setup_input()
        self.assertTrue(fix.verify_first_post_gate_s2(
            key, gate, tagged, decisions, fix.focus.ARMS))

    def test_extra_non_camera_action_fails(self):
        key, gate, tagged, decisions = self.setup_input()
        tagged[fix.focus.ARMS[1]][key][-4]["action"] = 1
        with self.assertRaisesRegex(ValueError, "camera"):
            fix.verify_first_post_gate_s2(
                key, gate, tagged, decisions, fix.focus.ARMS)

    def test_different_first_s2_offset_fails(self):
        key, gate, tagged, decisions = self.setup_input()
        decisions[fix.focus.ARMS[2]][key][0]["native_action_prefix_count"] -= 1
        with self.assertRaisesRegex(ValueError, "exact native camera"):
            fix.verify_first_post_gate_s2(
                key, gate, tagged, decisions, fix.focus.ARMS)

    def test_wrong_camera_main_step_fails(self):
        key, gate, tagged, decisions = self.setup_input()
        tagged[fix.focus.ARMS[0]][key][-2]["main_step"] = 41
        with self.assertRaisesRegex(ValueError, "camera"):
            fix.verify_first_post_gate_s2(
                key, gate, tagged, decisions, fix.focus.ARMS)

    def test_missing_scan_action_fails(self):
        key, gate, tagged, decisions = self.setup_input()
        tagged[fix.focus.ARMS[3]][key][16]["phase"] = "continuation"
        with self.assertRaisesRegex(ValueError, "phase"):
            fix.verify_first_post_gate_s2(
                key, gate, tagged, decisions, fix.focus.ARMS)

    def test_first_s2_episode_call_ordinal_must_match(self):
        key, gate, tagged, decisions = self.setup_input()
        decisions[fix.focus.ARMS[2]][key][0]["episode_call_id"] = 7
        with self.assertRaisesRegex(ValueError, "ordinal"):
            fix.verify_first_post_gate_s2(
                key, gate, tagged, decisions, fix.focus.ARMS)

    def test_ast_frozen_report_is_matched_exactly(self):
        def frozen_report(args):
            for arm in ARMS:
                cont = [e for e in decisions[arm][key]
                        if e["phase"] == "continuation"]
                if not cont or cont[0]["native_action_prefix_count"] != (
                        gate["prefix_action_count"] + 3
                        + (6 if arm != ARMS[0] else 0)):
                    raise ValueError("No first post-gate S2 call at exact action offset")
            return True
        patched = fix.focused_report_with_camera_audit(frozen_report)
        self.assertEqual(patched.__name__, "frozen_report")

    def test_ast_matches_checked_in_original_focused_report(self):
        # Check the real frozen report structure, not just a mini fixture.
        patched = fix.focused_report_with_camera_audit(fix.focus.report)
        self.assertEqual(patched.__name__, "report")

    def test_ast_rejects_source_drift(self):
        def changed_report(args):
            for arm in ARMS:
                if bad:
                    raise ValueError("No first post-gate S2 call at exact action offset")
            return True
        with self.assertRaisesRegex(RuntimeError, "assertion shape"):
            fix.focused_report_with_camera_audit(changed_report)


if __name__ == "__main__":
    unittest.main()
