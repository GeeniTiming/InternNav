"""CPU-only tests for frozen H1/native anomaly cross audit."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import vln_anomaly_h1_cross_audit_v1 as audit


def outcome(case_id, key, mode, arm, sr, ne=1.5, spl=0.2):
    return {
        "context_id": case_id + "_" + mode,
        "episode_key": list(key),
        "source_type": mode,
        "action": arm,
        "success": int(sr),
        "ne": float(ne),
        "spl": float(spl),
        "os": 1.0,
        "extra_yaw_actions": 6 if arm.startswith("SEEK_") else 0,
        "prefix_perturbation_actions": 3 if mode == "perturbed" else 0,
        "audit": "REUSED_AUDITED_KEEP" if arm == "KEEP" else "PAIRED_PREARM_VERIFIED",
    }


def sample_episode(key, name="case"):
    selected = [{"key": list(key), "case_id": name,
                 "checkpoint": {"step": 30, "prefix_action_count": 42}}]
    protocol = {"selected_cases": selected, "max_cases": 1}
    outcomes = [
        outcome(name, key, mode, arm, sr=int(mode == "perturbed" and arm == "SEEK_RIGHT45"))
        for mode in audit.MODES for arm in audit.ARMS
    ]
    report = {"N_four_arm_outcomes": 8, "N_contexts": 2}
    return protocol, outcomes, report


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TableValidationTests(unittest.TestCase):
    def test_full_four_arm_cross_mode_results_are_audited(self):
        key = ("scene_a", "1")
        protocol, outcomes, report = sample_episode(key)
        selected, by_token = audit.verify_table(protocol, outcomes, report, "development")
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(by_token), 8)
        self.assertEqual(
            by_token[(key, "perturbed", "SEEK_RIGHT45")]["success"], 1)
        self.assertEqual(
            by_token[(key, "natural", "SEEK_RIGHT45")]["success"], 0)

    def test_duplicate_or_missing_arm_fails_closed(self):
        key = ("scene_a", "1")
        protocol, outcomes, report = sample_episode(key)
        for bad in (outcomes[:-1], outcomes[:-1] + [outcomes[0]]):
            with self.subTest(length=len(bad)):
                with self.assertRaisesRegex(ValueError, "incomplete|duplicate"):
                    audit.verify_table(protocol, bad, report, "development")

    def test_context_identity_and_audit_provenance_cannot_be_swapped(self):
        key = ("scene_a", "1")
        protocol, outcomes, report = sample_episode(key)
        for field, value, message in (
            ("context_id", "another_perturbed", "context identity mismatch"),
            ("audit", "NOT_AUDITED", "outcome provenance not audited"),
            ("prefix_perturbation_actions", 0, "perturbed prefix action budget changed"),
        ):
            mutated = [dict(row) for row in outcomes]
            mutated[4][field] = value  # first perturbed arm
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    audit.verify_table(protocol, mutated, report, "development")

    def test_gate_interval_is_retrospective_not_an_online_trigger(self):
        key = ("scene_a", "1")
        protocol, outcomes, report = sample_episode(key)
        _, indexed = audit.verify_table(protocol, outcomes, report, "development")
        mode_rows = {arm: indexed[(key, "perturbed", arm)] for arm in audit.ARMS}
        gate = {"step": 30, "prefix_action_count": 42}
        summary = {"native_success": 0}
        events = [
            {"type": "in_place_turning_candidate", "start_action": 25,
             "end_action": 99},
            {"type": "near_goal_then_departed_candidate", "start_action": 65,
             "end_action": 100},
        ]
        rows = audit.summarize_pair(
            "development", key, "perturbed", mode_rows, summary, events, gate)
        right = next(x for x in rows if x["arm"] == "SEEK_RIGHT45")
        self.assertEqual(right["sr_change"], "RESCUE")
        self.assertEqual(right["gate_native_action_prefix"], 42)
        self.assertEqual(
            right["retrospective_candidate_labels_overlapping_gate"],
            "in_place_turning_candidate")
        self.assertEqual(
            right["gate_precedes_nearest_detectable_visual_loop"],
            "NOT_ESTABLISHED_BY_RETROSPECTIVE_INTERVAL")


class CompleteCohortTests(unittest.TestCase):
    def make_complete_sources(self, temp):
        base = Path(temp)
        audit_root = base / "native_anomaly_audit_v1_02"
        audit_root.mkdir()
        records = []
        native_summary = []
        source_meta = {}
        for cohort, paired_name in audit.COHORTS.items():
            paired = base / paired_name
            paired.mkdir()
            gate_root = base / (cohort + "_natural_gate")
            gate_root.mkdir()
            native_root = base / (cohort + "_native")
            native_root.mkdir()
            selected_n = 10 if cohort == "development" else 11
            special = (("VLzqgDo317F", "58"),
                       ("i5noydFURQK", "70")) if cohort == "development" else ()
            keys = list(special) + [
                (cohort + "_scene" + str(i), str(i)) for i in range(12 - len(special))]
            selected, outcomes, gates = [], [], []
            for i, key in enumerate(keys):
                checkpoint = {"step": 20, "prefix_action_count": 50}
                gates.append({"episode_key": list(key),
                              "natural_checkpoint": checkpoint if i < selected_n else None})
                native_summary.append({
                    "cohort": cohort, "scene": key[0], "episode": key[1],
                    "native_success": 0 if key in special else 1,
                    "native_termination": "ENV_DONE",
                    "native_action_count": 120,
                    "original_gate_action_prefix": "50" if i < selected_n else "",
                })
                if i >= selected_n:
                    continue
                case_id = "case_{}_{}".format(cohort, i)
                selected.append({"key": list(key), "case_id": case_id, "checkpoint": checkpoint})
                for mode in audit.MODES:
                    for arm in audit.ARMS:
                        success = 0 if key in special else 1
                        if key == ("i5noydFURQK", "70") and mode == "perturbed" and arm == "SEEK_RIGHT45":
                            success = 1
                        outcomes.append(outcome(case_id, key, mode, arm, success))
            write_json(gate_root / "natural_checkpoint_report.json", {"cases": gates})
            write_json(paired / "paired_protocol.json", {
                "max_cases": selected_n, "selected_cases": selected,
                "source_natural_gate_run": str(gate_root),
                "source_native_run": str(native_root),
            })
            write_jsonl(paired / "intervention_outcomes.jsonl", outcomes)
            write_json(paired / "paired_report.json", {
                "N_four_arm_outcomes": len(outcomes), "N_contexts": selected_n * 2})
            source_meta[cohort] = {
                "paired": str(paired), "gate": str(gate_root), "native": str(native_root),
            }
        write_json(audit_root / "audit_manifest.json", {
            "schema": "vln_native_anomaly_audit.v1",
            "N_episodes": 24, "N_event_candidates": 1,
            "sources": source_meta,
        })
        with (audit_root / "episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(native_summary[0]))
            writer.writeheader()
            writer.writerows(native_summary)
        write_jsonl(audit_root / "event_candidates.jsonl", [{
            "cohort": "development", "scene": "VLzqgDo317F", "episode": "58",
            "type": "forward_without_translation_candidate",
            "start_action": 30, "end_action": 100,
        }])
        return base, audit_root

    def test_21_cases_168_outcomes_and_2_failure_case_dossiers(self):
        with tempfile.TemporaryDirectory() as temp:
            base, anomaly = self.make_complete_sources(temp)
            all_rows, prov, native, events = audit.build(base, anomaly)
            self.assertEqual(len(all_rows), 168)
            targets, cases, stats = audit.summarize(all_rows, native, events)
            self.assertEqual(len(targets), 16)
            self.assertEqual(len(cases), 2)
            self.assertEqual(len(stats), 12)
            ep70 = next(c for c in cases if c["episode_key"][0] == "i5noydFURQK")
            pert = ep70["arms_by_context"]["perturbed"]
            right = next(row for row in pert if row["arm"] == "SEEK_RIGHT45")
            self.assertEqual(right["vs_KEEP"], "RESCUE")
            output = base / "new_cross_audit"
            manifest = audit.write_report(
                output, all_rows, targets, cases, stats, prov, anomaly)
            self.assertEqual(manifest["N_h1_outcomes"], 168)
            self.assertEqual(len(list(csv.DictReader(
                (output / "failure_episodes_16_arms.csv").open()))), 16)
            with self.assertRaises(FileExistsError):
                audit.write_report(output, all_rows, targets, cases, stats, prov, anomaly)

    def test_missing_failure_case_or_mismatched_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            base, anomaly = self.make_complete_sources(temp)
            manifest_path = anomaly / "audit_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sources"]["development"]["paired"] = "/tmp/other_pair"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "paired root differs"):
                audit.build(base, anomaly)


if __name__ == "__main__":
    unittest.main()
