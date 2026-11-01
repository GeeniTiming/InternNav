"""CPU-only synthetic regression suite for the frozen H1 development audit."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_h1_development_audit_v01 as h1


def make_frozen_data():
    expected = []
    by_context = {}
    for i in range(10):
        scene, episode = "newscene_%02d" % i, str(i)
        for mode in h1.MODES:
            cid = "ep%02d_%s" % (i, mode)
            expected.append((cid, (scene, episode), mode))
            keep = 1 if i % 3 == 0 else 0
            outcomes = {
                "KEEP": keep,
                "HISTORY_RESET": 1 if i % 3 == 1 else 0,
                "SEEK_LEFT45": 1 if i % 3 == 2 else 0,
                "SEEK_RIGHT45": 1 if i % 5 == 0 else 0,
            }
            # In perturbed states, some episode action effects differ, but both
            # contexts of the SAME scene remain one correlated scene group.
            if mode == "perturbed" and i % 2 == 0:
                outcomes["SEEK_RIGHT45"] = 1 - outcomes["SEEK_RIGHT45"]
            by_context[cid] = {
                act: {
                    "context_id": cid,
                    "episode_key": [scene, episode],
                    "source_type": mode,
                    "action": act,
                    "success": success,
                    "spl": .5 if success else 0.0,
                    "ne": 1.0 if success else 10.0,
                }
                for act, success in outcomes.items()
            }
    return expected, by_context


class H1CoreTests(unittest.TestCase):
    def test_counts_and_valid_descriptive_upper_bound(self):
        expected, by_context = make_frozen_data()
        r = h1.analyze(expected, by_context, bootstrap_reps=10)
        self.assertEqual((r["N_scene_groups"], r["N_contexts"],
                          r["N_actual_rollouts"], r["N_actual_effects_vs_keep"]),
                         (10, 20, 80, 60))
        self.assertGreater(r["overall"]["descriptive_oracle_success_rate"],
                           r["overall"]["best_fixed_success_rate_in_sample"])
        self.assertEqual(len(r["operation_effects_vs_keep"]), 3)
        self.assertTrue(r["not_a_preregistered_confirmatory_test"])
        for act, stats in r["operation_effects_vs_keep"].items():
            self.assertEqual(sum(stats["context_relation_counts"].values()), 20)
            self.assertEqual(
                sum(sum(c.values()) for c in stats["context_relation_by_mode"].values()),
                20
            )
        self.assertEqual(r["by_context_mode"]["natural"]["N_contexts"], 10)
        self.assertEqual(r["by_context_mode"]["perturbed"]["N_contexts"], 10)

    def test_bootstrap_is_deterministic_and_groups_by_scene(self):
        expected, data = make_frozen_data()
        a = h1.analyze(expected, data, bootstrap_reps=50, seed=4)
        b = h1.analyze(expected, data, bootstrap_reps=50, seed=4)
        self.assertEqual(a, b)
        self.assertEqual(a["exploratory_scene_cluster_bootstrap"][
            "N_scene_cluster_resamples"], 50)

    def test_no_successful_action_is_not_mislabeled_rescue(self):
        expected, data = make_frozen_data()
        for outcomes in data.values():
            for value in outcomes.values():
                value["success"], value["spl"] = 0, 0.0
        r = h1.analyze(expected, data, bootstrap_reps=0)
        self.assertEqual(r["overall"]["descriptive_oracle_success_rate"], 0)
        self.assertEqual(r["overall"]["best_fixed_action_in_sample"], "KEEP")
        for effect in r["operation_effects_vs_keep"].values():
            self.assertEqual(effect["context_relation_counts"]["BOTH_FAILURE"], 20)
            self.assertEqual(effect["context_relation_counts"]["RESCUE"], 0)

    def test_missing_action_rejected(self):
        expected, data = make_frozen_data()
        del data[expected[0][0]]["SEEK_LEFT45"]
        with self.assertRaisesRegex(ValueError, "four actual"):
            h1.analyze(expected, data, bootstrap_reps=0)

    def test_wrong_context_identity_rejected(self):
        expected, data = make_frozen_data()
        data[expected[0][0]]["KEEP"]["episode_key"] = ["wrong", "0"]
        with self.assertRaisesRegex(ValueError, "identity"):
            h1.analyze(expected, data, bootstrap_reps=0)

    def test_duplicate_or_missing_mode_rejected(self):
        expected, data = make_frozen_data()
        tampered = list(expected)
        cid, key, _ = tampered[0]
        tampered[0] = (cid, key, "perturbed")
        with self.assertRaisesRegex(ValueError, "pairing"):
            h1.analyze(tampered, data, bootstrap_reps=0)

    def test_nonfinite_outcome_rejected(self):
        expected, data = make_frozen_data()
        data[expected[0][0]]["KEEP"]["ne"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            h1.analyze(expected, data, bootstrap_reps=0)

    def test_positive_spl_on_failure_rejected(self):
        expected, data = make_frozen_data()
        cid = expected[1][0]
        data[cid]["KEEP"]["success"] = 0
        data[cid]["KEEP"]["spl"] = .1
        with self.assertRaisesRegex(ValueError, "Invalid SPL"):
            h1.analyze(expected, data, bootstrap_reps=0)

    def test_single_mode_contexts_not_independent_scenes(self):
        expected, data = make_frozen_data()
        expected = expected[:10]
        with self.assertRaisesRegex(ValueError, "20-context"):
            h1.analyze(expected, data, bootstrap_reps=0)

    def test_output_is_create_only(self):
        expected, data = make_frozen_data()
        report = h1.analyze(expected, data, bootstrap_reps=0)
        with TemporaryDirectory() as temp:
            output = Path(temp) / "h1.json"
            with patch.object(h1, "make_report", return_value=report):
                h1.main(["--paired-run", temp, "--bootstrap-reps", "0",
                         "--output", str(output)])
                original = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    h1.main(["--paired-run", temp, "--bootstrap-reps", "0",
                             "--output", str(output)])
            self.assertEqual(output.read_bytes(), original)

    def test_analysis_cannot_mutate_audited_rows(self):
        expected, data = make_frozen_data()
        original = copy.deepcopy((expected, data))
        h1.analyze(expected, data, bootstrap_reps=10)
        self.assertEqual((expected, data), original)


if __name__ == "__main__":
    unittest.main()
