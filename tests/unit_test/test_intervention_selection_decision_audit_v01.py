"""No Habitat/GPU: evidence-only decision audit validation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import intervention_selection_decision_audit_v01 as audit


class MathAndContractTests(unittest.TestCase):
    def test_rotation_quaternion_sign_invariance(self):
        a = {"physical_orientation_OFFLINE_ONLY": [1, 0, 0, 0]}
        b = {"physical_orientation_OFFLINE_ONLY": [-1, 0, 0, 0]}
        self.assertAlmostEqual(audit.orientation_distance(a, b), 0.0)

    def test_rotation_known_angle(self):
        a = {"physical_orientation_OFFLINE_ONLY": [1, 0, 0, 0]}
        b = {"physical_orientation_OFFLINE_ONLY": [0.7071067811865476, 0, 0.7071067811865475, 0]}
        self.assertAlmostEqual(audit.orientation_distance(a, b), 90.0, places=6)

    def test_pose_reject_nonfinite_and_zero_quaternion(self):
        with self.assertRaises(ValueError):
            audit.position_distance({"physical_position_OFFLINE_ONLY": [float("nan"), 1, 0]},
                                    {"physical_position_OFFLINE_ONLY": [1, 0, 0]})
        with self.assertRaises(ValueError):
            audit.orientation_distance({"physical_orientation_OFFLINE_ONLY": [0, 0, 0, 0]},
                                       {"physical_orientation_OFFLINE_ONLY": [1, 0, 0, 0]})

    def test_relative_index_not_system2_boundary(self):
        x = [{"action_code": 1}, {"action_code": 2}]
        y = [{"action_code": 1}, {"action_code": 3}]
        out = audit.first_difference(x, y, lambda a, b: (a["action_code"], b["action_code"])
                                     if a["action_code"] != b["action_code"] else None)
        self.assertEqual(out["continuation_index"], 1)
        self.assertNotIn("system2_call_id", out)

    def test_unique_keys_fail_closed(self):
        with self.assertRaises(ValueError):
            audit.unique_index([{"x": 1}, {"x": 1}], "test", lambda r: r["x"])

    def test_actions_pair_spec(self):
        self.assertEqual(set(audit.ARMS), {"KEEP", "SEEK_LEFT45", "SEEK_RIGHT45", "HISTORY_RESET"})
        self.assertEqual(len(audit.FOCUS), 3)


class RealBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = os.environ.get("TRACE_AUDIT_BUNDLE")
        if not cls.bundle:
            raise unittest.SkipTest("Set TRACE_AUDIT_BUNDLE to the supplied four-file ZIP for integration tests")
        cls.report, cls.data, cls.hashes = audit.load_input(cls.bundle)

    def test_actual_bundle_expected_scope_and_no_s2_invention(self):
        events, dossiers, plan, result = audit.collect(self.report, self.data, self.hashes)
        self.assertEqual((len(events), len(dossiers), len(plan)), (24, 3, 12))
        self.assertEqual(result["N_scene_groups"], 4)
        self.assertEqual(result["N_SYSTEM2_call_aligned_pairs"], 0)
        self.assertTrue(all(e["system2_call_alignment"] == "NOT_OBSERVABLE" for e in events))
        self.assertTrue(all(e["first_system2_decision_difference"] is None for e in events))
        self.assertTrue(all(a["gpu_rollout_executed"] is False and a["result"] is None for a in plan))
        self.assertTrue(all(a["execution_status"].startswith("BLOCKED_") for a in plan))
        self.assertEqual(result["success_change_by_arm"]["SEEK_RIGHT45"]["RESCUE"], 2)
        self.assertEqual(result["success_change_by_arm"]["SEEK_LEFT45"]["HARM"], 1)
        self.assertEqual(result["success_change_by_arm"]["HISTORY_RESET"]["HARM"], 5)
        self.assertEqual({d["case_role"] for d in dossiers}, {"RESCUE", "HARM"})

    def test_actual_bundle_read_only_and_create_only(self):
        before = Path(self.bundle).read_bytes() if Path(self.bundle).is_file() else None
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new_run"
            out = audit.audit(self.bundle, output)
            self.assertEqual(len(list(output.iterdir())), 4)
            self.assertTrue(out["status"].startswith("CPU_POSTHOC_AUDITED"))
            with self.assertRaises(FileExistsError):
                audit.audit(self.bundle, output)
            if before is not None:
                self.assertEqual(before, Path(self.bundle).read_bytes())

    def test_modifying_raw_member_fails_hash_audit(self):
        with zipfile.ZipFile(self.bundle) as z:
            payload = {name: z.read(name) for name in audit.MEMBERS}
        payload["decision_comparisons.jsonl"] += b"\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "altered.zip"
            with zipfile.ZipFile(p, "w") as z:
                for name, data in payload.items():
                    z.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                audit.load_input(p)

    def test_duplicate_action_index_rejected(self):
        data = copy.deepcopy(self.data)
        data["continuation_action_traces.jsonl"][1]["continuation_index"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate continuation"):
            audit.collect(self.report, data, self.hashes)

    def test_missing_trajectory_rejected(self):
        data = copy.deepcopy(self.data)
        data["continuation_action_traces.jsonl"] = data["continuation_action_traces.jsonl"][:-1]
        with self.assertRaises(ValueError):
            audit.collect(self.report, data, self.hashes)

    def test_no_online_privileged_values_in_ablation_manifest(self):
        _, _, plan, _ = audit.collect(self.report, self.data, self.hashes)
        keys = set(plan[0]["design"])
        self.assertFalse(keys & {"distance_to_goal", "gps", "depth", "pathfinder", "oracle_success", "goal_position"})
        self.assertEqual(set(a["ablation_arm"] for a in plan),
                         {"KEEP", "SCAN_NO_INJECTION", "PLACEBO", "REAL_SEEK"})
        self.assertEqual(len({a["design"]["context_id"] for a in plan}), 3)
        for ctx in {a["design"]["context_id"] for a in plan}:
            group = [a for a in plan if a["design"]["context_id"] == ctx]
            self.assertEqual({tuple(a["design"]["scan_actions_candidate_unverified"]) for a in group}.__len__(), 1)


if __name__ == "__main__":
    unittest.main()