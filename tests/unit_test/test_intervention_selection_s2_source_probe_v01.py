"""Standard-library-only tests for immutable source-log/S2-readiness probe."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "eval"
sys.path.insert(0, str(SCRIPT_DIR if SCRIPT_DIR.is_dir() else Path(__file__).resolve().parent))
import intervention_selection_s2_source_probe_v01 as probe


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "raw"
        self.a = self.source / "native" / "control" / "actions.jsonl"
        self.b = self.source / "treated" / "natural_seek_right45" / "turn_actions.jsonl"
        self.a.parent.mkdir(parents=True)
        self.b.parent.mkdir(parents=True)
        self.a.write_bytes(encode({"action": 1, "episode_id": "70"}))
        self.b.write_bytes(encode({"action": 2, "episode_id": "70"}))
        self.paths = [self.a, self.b]
        self.zip = self.root / "trace.zip"
        self.build_bundle()

    def build_bundle(self, corrupt=None):
        items = {
            "continuation_action_traces.jsonl": encode({"action_code": 1}),
            "trace_divergence_vs_keep.jsonl": encode({"first_different_continuation_index": 4}),
            "decision_comparisons.jsonl": encode({"KEEP_first_S2_available": False}),
        }
        report = {
            "schema": "intervention_selection_trace_divergence.v0.1",
            "original_trace_sha256": {str(p): probe.sha256_file(p) for p in self.paths},
            "output_hashes": {k: digest(v) for k, v in items.items()},
        }
        with ZipFile(self.zip, "w") as output:
            output.writestr("trace_divergence_report.json", encode(report))
            for name, blob in items.items():
                output.writestr(name, b"CORRUPTED\n" if corrupt == name else blob)
        return report

    def test_missing_original_server_logs(self):
        summary, records, _ = probe.probe(self.zip, self.root / "nonexistent")
        self.assertEqual(summary["N_original_action_logs"], 2)
        self.assertEqual(summary["N_SYSTEM2_call_aligned_pairs"], 0)
        self.assertEqual(summary["status"], "BLOCKED_ORIGINAL_ACTION_LOGS_MISSING")
        self.assertEqual(len(records), 2)

    def test_verified_raw_action_logs_do_not_imply_s2_alignment(self):
        summary, raw, candidates = probe.probe(self.zip)
        self.assertEqual(summary["N_matched_action_logs"], 2)
        self.assertEqual(summary["status"], "RAW_ACTION_LOGS_VERIFIED_NO_LINKABLE_S2_CALL_ANCHORS")
        self.assertFalse(summary["can_execute_paired_ablation"])
        self.assertEqual(len(candidates), 0)
        self.assertTrue(all(r["status"] == "MATCHED" for r in raw))

    def test_valid_candidate_still_requires_replay_binding(self):
        dest = self.b.parent / "system2_calls.jsonl"
        dest.write_bytes(encode({
            "call_id": 0, "episode_key": ["scene", "70"], "main_step": 41,
            "native_action_prefix_count": 12,
            "call_type": "ordinary", "model_output": "turn right",
            "prompt_text": "PRIVATE PROMPT: do not export",
            "sim_position": [1, 2, 3],
        }))
        summary, _, entries = probe.probe(self.zip)
        self.assertEqual(summary["N_candidate_anchor_files"], 1)
        self.assertEqual(summary["status"],
                         "RAW_ACTION_LOGS_VERIFIED_S2_CANDIDATES_REQUIRE_EXACT_REPLAY_BINDING")
        self.assertEqual(entries[0]["status"], "CALL_ANCHOR_CANDIDATE_UNVERIFIED")
        self.assertNotIn("PRIVATE PROMPT", json.dumps(entries))
        self.assertNotIn("sim_position", json.dumps(entries))
        self.assertEqual(summary["N_SYSTEM2_call_aligned_pairs"], 0)

    def test_reject_bad_call_record_duplicate_id(self):
        file = self.b.parent / "s2_events.jsonl"
        row = {"call_id": 4, "episode_key": ["scene", "70"],
               "native_action_prefix_count": 11, "main_step": 12,
               "call_type": "ordinary", "model_output": "forward"}
        file.write_bytes(encode(row) + encode(dict(row, native_action_prefix_count=12)))
        entry = probe.inspect_candidate(file)
        self.assertEqual(entry["status"], "NOT_USABLE_AS_CALL_ANCHOR")
        self.assertIn("DUPLICATE_CALL_ID", entry["failure_reasons"])

    def test_reject_main_step_only_records(self):
        file = self.b.parent / "s2_output.jsonl"
        file.write_bytes(encode({"main_step": 23, "model_output": "right"}))
        entry = probe.inspect_candidate(file)
        self.assertEqual(entry["status"], "NOT_USABLE_AS_CALL_ANCHOR")
        self.assertTrue(any(x.startswith("MISSING_REQUIRED_FIELDS") for x in entry["failure_reasons"]))

    def test_detect_mutated_original_action_file(self):
        self.a.write_bytes(b"mutated\n")
        summary, raw, _ = probe.probe(self.zip)
        self.assertEqual(summary["status"], "BLOCKED_SOURCE_HASH_MISMATCH")
        self.assertEqual(summary["N_changed_action_logs"], 1)
        self.assertIn("HASH_MISMATCH", [r["status"] for r in raw])

    def test_root_remapping_of_source_data_tree(self):
        registered = self.build_bundle()
        relocated = self.root / "moved"
        for old in self.paths:
            dest = relocated / old.relative_to(self.source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(old.read_bytes())
        summary, raw, _ = probe.probe(self.zip, source_root=relocated)
        self.assertEqual(summary["N_matched_action_logs"], 2)
        self.assertTrue(all(str(relocated) in r["resolved_source_path"] for r in raw))
        self.assertEqual(len(registered["original_trace_sha256"]), 2)

    def test_reject_tampered_archive_member(self):
        self.build_bundle(corrupt="decision_comparisons.jsonl")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            probe.load_bundle(self.zip)

    def test_create_only_does_not_overwrite(self):
        result, raw, candidates = probe.probe(self.zip)
        target = self.root / "out"
        probe.write_create_only(target, result, raw, candidates)
        original = (target / "s2_readiness_report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            probe.write_create_only(target, {}, [], [])
        self.assertEqual(original, (target / "s2_readiness_report.json").read_bytes())

    def test_report_retains_no_online_trajectory_or_gt(self):
        result, raw, candidates = probe.probe(self.zip)
        self.assertFalse(result["can_execute_paired_ablation"])
        self.assertEqual(result["N_SYSTEM2_call_aligned_pairs"], 0)
        self.assertNotIn("success", json.dumps([result, raw, candidates]))
        self.assertNotIn("sim_position", json.dumps([result, raw, candidates]))


if __name__ == "__main__":
    unittest.main()