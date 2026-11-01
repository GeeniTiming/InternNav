"""Small synthetic CPU tests for read-only S2 offset diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_s2_offset_diagnostic_v01 as d


def jsonl(path, lines):
    raw = "".join(json.dumps(line) + "\n" for line in lines)
    Path(path).write_text(raw, encoding="utf-8")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DiagnosticTests(unittest.TestCase):
    def make_run(self, path, shift=None):
        root = Path(path)
        cases = [{"key": [f"scene{i}", str(i)], "checkpoint": {
            "step": 10, "prefix_action_count": 4}} for i in range(3)]
        protocol = {"arms": list(d.ARMS), "cases": cases}
        file = root / "focused_protocol.json"
        file.write_text(json.dumps(protocol))
        (root / "manifest.json").write_text(json.dumps({
            "generated_hashes": {str(file): d.sha256(file)}
        }))
        for arm in d.ARMS:
            directory = root / arm
            directory.mkdir()
            action_lines = []
            event_lines = []
            cid = 0
            for i, c in enumerate(cases):
                expected = 7 + (0 if arm == d.ARMS[0] else 6)
                observed = expected + (shift or 0)
                for j in range(max(15, observed+2)):
                    phase = ("prefix" if j < 4 else
                             "alignment" if j < 7 else
                             "external_recovery" if j < 13 and arm != d.ARMS[0]
                             else "continuation")
                    action_lines.append({
                        "scene_id": c["key"][0]+".glb",
                        "episode_id": c["key"][1], "phase": phase,
                        "action": j % 4})
                for epcall, (phase, offset) in enumerate(
                        [("prefix", 2), ("continuation", observed)]):
                    fields = {"episode_key": c["key"], "phase": phase,
                              "main_step": 10, "native_action_prefix_count": offset,
                              "call_id": cid, "episode_call_id": epcall}
                    event_lines.append(dict(fields, event="enter"))
                    event_lines.append(dict(fields, event="exit"))
                    cid += 1
            eh = jsonl(directory / "s2_call_events.jsonl", event_lines)
            ah = jsonl(directory / "turn_actions.jsonl", action_lines)
            (directory / "s2_call_events_audit.json").write_text(json.dumps({
                "call_event_sha256": eh, "same_arm_action_sha256": ah
            }))
        return root

    def test_all_expected_offsets(self):
        with TemporaryDirectory() as t:
            root = self.make_run(t)
            result = d.diagnose(root)
            self.assertEqual(result["N_focus_case_arms"], 12)
            self.assertEqual(result["N_mismatching_legacy_offsets"], 0)
            self.assertFalse(result["files_modified"])

    def test_shift_reported_not_silently_reinterpreted(self):
        with TemporaryDirectory() as t:
            root = self.make_run(t, shift=1)
            result = d.diagnose(root)
            self.assertEqual(result["N_mismatching_legacy_offsets"], 12)
            self.assertEqual({x["offset_delta"] for x in result["cases"]}, {1})

    def test_manifest_mismatch_fails(self):
        with TemporaryDirectory() as t:
            root = self.make_run(t)
            (root / "focused_protocol.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                d.diagnose(root)

    def test_event_hash_mismatch_fails(self):
        with TemporaryDirectory() as t:
            root = self.make_run(t)
            p = root / d.ARMS[1] / "s2_call_events.jsonl"
            p.write_bytes(p.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                d.diagnose(root)


if __name__ == "__main__":
    unittest.main()
