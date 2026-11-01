"""CPU-only regression tests for conservative offline v0.1 outcome labels."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_outcome_labels_v01 as lab


def outcome(context_id, action, success, spl, ne, mode="natural"):
    return {
        "context_id": context_id,
        "episode_key": ["scene_00", "10"],
        "source_type": mode, "action": action, "success": success,
        "spl": spl, "ne": ne, "os": 1.0,
        "extra_yaw_actions": lab.EXPECTED_COST[action],
        "prefix_perturbation_actions": 3 if mode == "perturbed" else 0,
        "audit": lab.EXPECTED_AUDIT[action],
    }


def context(suffix, mode, success_vector, spl_vector=None, ne_vector=None):
    cid = "ep_" + suffix + "_" + mode
    if spl_vector is None:
        spl_vector = [.9 if s else 0 for s in success_vector]
    if ne_vector is None:
        ne_vector = [.2 if s else 10.0 for s in success_vector]
    return cid, {
        a: outcome(cid, a, s, sp, ne, mode)
        for a, s, sp, ne in zip(
            lab.ACTIONS, success_vector, spl_vector, ne_vector)
    }


class LabelContractTests(unittest.TestCase):
    def test_rescue_harm_and_ne_better_without_success(self):
        _, rescued = context("r", "natural", [0, 0, 1, 0],
                             ne_vector=[10, 2, .3, 1])
        k = rescued["KEEP"]
        e = lab.effect_vs_keep(
            k["context_id"], "natural", "SEEK_RIGHT45", k,
            rescued["SEEK_RIGHT45"])
        self.assertEqual(e["outcome_relation"], "RESCUE")
        self.assertEqual(e["delta_success"], 1)
        e = lab.effect_vs_keep(
            k["context_id"], "natural", "SEEK_LEFT45", k,
            rescued["SEEK_LEFT45"])
        self.assertEqual(e["outcome_relation"], "BOTH_FAILURE")
        self.assertEqual(e["delta_success"], 0)
        self.assertGreater(e["delta_ne_improvement_m"], 0)
        self.assertTrue(e["ne_is_diagnostic_not_rescue_label"])
        _, harmed = context("h", "natural", [1, 1, 1, 0])
        k = harmed["KEEP"]
        e = lab.effect_vs_keep(
            k["context_id"], "natural", "HISTORY_RESET", k,
            harmed["HISTORY_RESET"])
        self.assertEqual(e["outcome_relation"], "HARM")
        self.assertEqual(e["delta_success"], -1)

    def test_both_failure_ne_gain_must_remain_unresolved(self):
        _, d = context("fail", "natural", [0, 0, 0, 0],
                       ne_vector=[10, 1, 3, 5])
        p = lab.preference(d["KEEP"], d["SEEK_LEFT45"], .02)
        self.assertEqual(p["relation"], "UNRESOLVED_BOTH_FAILED")
        self.assertIsNone(p["preferred"])

    def test_rescue_hard_preference_even_when_costs_six_yaw(self):
        _, d = context("rescue", "natural", [0, 0, 1, 0])
        p = lab.preference(d["KEEP"], d["SEEK_RIGHT45"], .02)
        self.assertEqual(p["relation"], "SUCCESS_DOMINANCE")
        self.assertEqual(p["preferred"], "SEEK_RIGHT45")

    def test_both_success_tolerance_cost_and_tradeoff(self):
        cid, d = context("trade", "natural", [1, 1, 1, 1],
                         spl_vector=[.9, .95, .89, .9])
        # Larger SPL comes at six extra yaw => do not invent a scalar Q win.
        p = lab.preference(d["KEEP"], d["SEEK_LEFT45"], .02)
        self.assertEqual(p["relation"], "SPL_COST_TRADEOFF")
        self.assertIsNone(p["preferred"])
        p = lab.preference(d["KEEP"], d["SEEK_RIGHT45"], .02)
        self.assertEqual(p["relation"], "COST_DOMINANCE_WITHIN_SPL_TOLERANCE")
        self.assertEqual(p["preferred"], "KEEP")
        p = lab.preference(d["KEEP"], d["HISTORY_RESET"], .02)
        self.assertEqual(p["relation"], "TIE_WITHIN_SPL_TOLERANCE")
        self.assertIsNone(p["preferred"])
        d["HISTORY_RESET"]["spl"] = .95
        p = lab.preference(d["KEEP"], d["HISTORY_RESET"], .02)
        self.assertEqual(p["relation"], "SPL_AND_COST_DOMINANCE")
        self.assertEqual(p["preferred"], "HISTORY_RESET")

    def test_invalid_metrics_cost_or_audit_provenance_rejected(self):
        cid, d = context("normal", "perturbed", [1, 1, 1, 1])
        z = d["SEEK_LEFT45"]
        self.assertTrue(lab.check_outcome(z))
        changes = [
            ("extra_yaw_actions", 0),
            ("prefix_perturbation_actions", 0),
            ("audit", "REUSED_AUDITED_KEEP"),
            ("spl", math.nan),
            ("success", 2),
            ("ne", -1),
            ("os", float("inf")),
        ]
        for field, value in changes:
            w = dict(z)
            w[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                lab.check_outcome(w)

    def test_reject_cross_context_comparison_or_bad_tolerance(self):
        _, a = context("x", "natural", [1, 1, 1, 1])
        _, b = context("y", "natural", [1, 1, 1, 1])
        self.assertRaises(ValueError, lab.preference,
                          a["KEEP"], b["KEEP"], .02)
        for bad in (-.1, 1.0, float("nan"), None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                lab.verify_tolerance(bad)

    def test_frozen_20_context_80_outcome_shape_no_best_action(self):
        expected, records = [], {}
        # Same categories as the provided, audited 80-outcome pilot;
        # purely synthetic keys/cases and metrics, no hidden future inputs.
        types = (
            [[1, 1, 1, 1]] * 10
            + [[1, 1, 1, 0]] * 5
            + [[0, 0, 1, 0]] * 2
            + [[0, 0, 0, 0]] * 3
        )
        for i, status in enumerate(types):
            mode = ("natural", "perturbed")[i % 2]
            cid, data = context(str(i), mode, status)
            key = (f"scene_{i//2:02d}", str(i//2))
            for row in data.values():
                row["episode_key"] = list(key)
            expected.append((cid, key, mode))
            records[cid] = data
        ctx, effects, pairs = lab.build_labels(expected, records, .02)
        self.assertEqual((len(ctx), len(effects), len(pairs)), (20, 60, 120))
        report = lab.audit_summaries(ctx, effects, pairs, n_cases=10)
        self.assertEqual(report["N_rescue_pairs"], 2)
        self.assertEqual(report["N_harm_pairs"], 5)
        self.assertTrue(report["no_best_action_fabricated"])
        self.assertEqual(sum(x["context_type"].startswith("KEEP_FAILS")
                             for x in ctx), 5)
        self.assertEqual(sum(x["relation"] == "UNRESOLVED_BOTH_FAILED"
                             for x in pairs), 24)

    def test_immutable_create_only_output_and_recomputed_audit(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "source", Path(td) / "labels"
            source.mkdir()
            for item in lab.SOURCE_FILES:
                (source / item).write_text("frozen", encoding="utf-8")
            package = {
                "schema": lab.SCHEMA,
                "status": "OFFLINE_LABELS_PREPARED_NOT_TRAINED",
                "source_run": str(source),
                "source_hashes": {
                    str(source / p): lab.sha256(source / p)
                    for p in lab.SOURCE_FILES
                },
                "spl_tolerance": .02,
                "N_scene_groups": 10,
                "N_contexts": 20,
                "N_effects_vs_keep": 60,
                "N_pairwise_comparisons": 120,
                "N_rescue_pairs": 2,
                "N_harm_pairs": 6,
                "context_type_counts": {"KEEP_SUCCEEDS": 15},
            }
            data = {p: (p + "\n").encode() for p in lab.OUTPUT_FILES}
            with patch.object(
                lab, "make_label_package",
                side_effect=lambda *_: (copy.deepcopy(package), dict(data))
            ):
                written = lab.write_package(source, output, .02)
                lab.audit_package(output)
                self.assertEqual(written["N_contexts"], 20)
                with self.assertRaises(FileExistsError):
                    lab.write_package(source, output, .02)
                with (output / lab.OUTPUT_FILES[0]).open("ab") as stream:
                    stream.write(b"tamper\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    lab.audit_package(output)


if __name__ == "__main__":
    unittest.main()
