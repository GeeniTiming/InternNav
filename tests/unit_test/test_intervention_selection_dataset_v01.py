"""CPU-only preregistration tests: no GPU, no simulation, no results."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import intervention_selection_dataset_v01 as v


def manifest(n=12):
    return {
        "episodes": [
            {"scene_id": f"dataset/scene_{i:02d}.glb", "episode_id": str(2*i + 1)}
            for i in range(n)
        ]
    }


class AllocationTests(unittest.TestCase):
    def test_balanced_tiers_and_perturbation_directions(self):
        p = v.build_plan(manifest(), v.DEFAULT_SEED, 500)
        self.assertEqual(p["status"], "PRE_REGISTERED_NO_ROLLOUTS")
        self.assertEqual(len(p["cases"]), 12)
        self.assertEqual(len(p["contexts"]), 24)
        self.assertEqual(len(p["evaluations"]), 96)
        for tier in (20, 40, 60):
            rows = [c for c in p["cases"]
                    if c["checkpoint_rule"]["minimum_main_step"] == tier]
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                sorted(c["perturbation_plan"]["yaw_deg"] for c in rows),
                [-45, -45, 45, 45],
            )

    def test_same_episode_has_natural_and_physical_perturbed_state(self):
        p = v.build_plan(manifest(), 17, 500)
        for c in p["cases"]:
            contexts = [r for r in p["contexts"]
                        if r["case_id"] == c["case_id"]]
            self.assertEqual({r["source_type"] for r in contexts},
                             {"natural", "perturbed"})
            self.assertEqual(
                sorted(r["perturbation_yaw_deg"] for r in contexts),
                sorted([0, c["perturbation_plan"]["yaw_deg"]]),
            )
            for row in contexts:
                arms = [a for a in p["evaluations"]
                        if a["context_id"] == row["context_id"]]
                self.assertEqual({x["action"] for x in arms},
                                 set(v.ACTION_SET))
                self.assertTrue(all(x["status"] == "PLANNED_NO_ROLLOUT"
                                    and x["outcome"] is None for x in arms))
                self.assertTrue(row["same_state_all_arms_required"])

    def test_preselection_is_outcome_blind_and_order_independent(self):
        first = manifest()
        second = copy.deepcopy(first)
        second["episodes"].reverse()
        for i, ep in enumerate(second["episodes"]):
            ep["success"] = i % 2
            ep["final_ne"] = i * 100.0
            ep["future_actions"] = [3, 2, 1]
        p, q = (v.build_plan(m, 20261031, 500)
                for m in (first, second))
        def assignment(plan):
            return {
                tuple(c["episode_key"]): (
                    c["checkpoint_rule"]["minimum_main_step"],
                    c["perturbation_plan"]["yaw_deg"],
                ) for c in plan["cases"]
            }
        self.assertEqual(assignment(p), assignment(q))

    def test_reject_non_scene_disjoint_and_too_small_source(self):
        source = manifest()
        source["episodes"][1]["scene_id"] = source["episodes"][0]["scene_id"]
        with self.assertRaisesRegex(ValueError, "distinct scene"):
            v.build_plan(source, 5, 500)
        with self.assertRaisesRegex(ValueError, "Exactly 12"):
            v.build_plan(manifest(11), 5, 500)

    def test_ineligible_gate_stays_ineligible_no_replacement(self):
        p = v.build_plan(manifest(), 1, 500)
        self.assertTrue(all(c["checkpoint_rule"]["if_no_eligible_boundary"]
                            == "INELIGIBLE_RETAINED_NO_REPLACEMENT"
                            for c in p["cases"]))

    def test_no_invented_labels_or_privileged_selector_features(self):
        p = v.build_plan(manifest(), 5, 500)
        v.verify_plan(p, manifest(), 5, 500)
        self.assertFalse(p["training_labels_available"])
        self.assertFalse(p["training_features_available"])
        self.assertIn("distance_to_goal", p["online_forbidden_inputs"])
        self.assertTrue(all(e["outcome"] is None for e in p["evaluations"]))
        mutated = copy.deepcopy(p)
        mutated["evaluations"][0]["outcome"] = {"sr": 1}
        with self.assertRaises(ValueError):
            v.verify_plan(mutated, manifest(), 5, 500)

    def test_no_seed_bool_or_inadequate_remaining_budget(self):
        with self.assertRaises(ValueError):
            v.build_plan(manifest(), True, 500)
        with self.assertRaises(ValueError):
            v.build_plan(manifest(), 3, 60)


class SourceTests(unittest.TestCase):
    def test_reject_old_scene_overlap_and_missing_exclusion(self):
        source = Path("/mock/new")
        old = Path("/mock/old")
        m = manifest()
        m["task_evidence_capture"] = {"no_intervention": True}
        m["train_scene_disjoint"] = {
            "outcome_blind": True,
            "excluded_scene_stems": sorted(v.train.INSPECTED | {"old_scene"}),
        }
        keys = [v.key_of(x) for x in m["episodes"]]
        with (
            patch.object(v.pilot, "verify_manifest", return_value=m),
            patch.object(v.train, "validate_selection", return_value=keys),
            patch.object(v.train, "scene_stems_from_locked_run",
                         return_value={"old_scene"}),
        ):
            native, selected, prior = v.source_episodes(source, [old])
            self.assertEqual(len(selected), 12)
            self.assertEqual(prior, [old])
            with self.assertRaisesRegex(ValueError, "mandatory"):
                v.source_episodes(source, [])
            bad = copy.deepcopy(m)
            bad["train_scene_disjoint"]["excluded_scene_stems"] = []
            with patch.object(v.pilot, "verify_manifest", return_value=bad):
                with self.assertRaisesRegex(ValueError, "excluding"):
                    v.source_episodes(source, [old])


if __name__ == "__main__":
    unittest.main()
