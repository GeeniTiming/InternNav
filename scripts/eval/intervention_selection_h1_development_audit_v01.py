"""H1 development-only evaluation of already-audited multi-operation VLN rollouts.

Uses the original ten TRAIN scenes / 20 correlated contexts / 80 four-arm
rollouts; DOES NOT execute Habitat, select examples by outcome, train a model,
modify pinned source files, or claim held-out efficacy. Stdlib core is testable
without Torch/Habitat. All integrity checks remain in the original v0.1
outcome-label reader, imported only by the CLI.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

SCHEMA = "intervention_selection_h1_development_audit.v0.1"
ACTIONS = ("KEEP", "HISTORY_RESET", "SEEK_LEFT45", "SEEK_RIGHT45")
MODES = ("natural", "perturbed")
SEED = 20261101
BOOTSTRAPS = 2000
# Tie priority is descriptive only, NOT a trained/predeclared selector.
# Favour avoiding additional yaw when in-sample success is indistinguishable.
TIE_PRIORITY = ACTIONS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _check_row(row, key, mode, action):
    if (row.get("episode_key") != list(key)
            or row.get("source_type") != mode or row.get("action") != action):
        raise ValueError("Outcome identity or arm mismatch: " + repr((key, mode, action)))
    if type(row.get("success")) is not int or row["success"] not in (0, 1):
        raise ValueError("Expected completed binary actual rollout outcome")
    for field in ("spl", "ne"):
        value = row.get(field)
        if (type(value) not in (int, float)
                or not math.isfinite(value) or value < 0):
            raise ValueError("Nonfinite/negative navigation outcome: " + field)
    if row["spl"] > 1 or (row["spl"] > 0 and row["success"] == 0):
        raise ValueError("Invalid SPL or failed rollout with nonzero SPL")


def _summarize(sample):
    """Sample is a list of complete scene groups, never individual contexts."""
    contexts = [ctx for group in sample for ctx in group]
    if not contexts:
        raise ValueError("Cannot summarize empty H1 cohort")
    n = len(contexts)
    scores = {
        arm: sum(c["success"][arm] for c in contexts) / n
        for arm in ACTIONS
    }
    best_rate = max(scores.values())
    best_tied = [arm for arm in TIE_PRIORITY if scores[arm] == best_rate]
    best_fixed = best_tied[0]
    oracle_rate = sum(max(c["success"].values()) for c in contexts) / n
    return {
        "N_contexts": n,
        "success_rate_by_fixed_action": scores,
        "best_fixed_action_in_sample": best_fixed,
        "tied_best_fixed_actions": best_tied,
        "best_fixed_success_rate_in_sample": best_rate,
        "descriptive_oracle_success_rate": oracle_rate,
        "oracle_minus_best_fixed_success_rate": oracle_rate - best_rate,
    }


def analyze(expected, by_context, *, bootstrap_reps=BOOTSTRAPS, seed=SEED):
    """Check scene correlation and measure H1 without using online features.

    expected: original frozen [(context_id, (scene, episode), mode), ...]
    by_context: original frozen {context_id: {action: measured outcome row}}.
    """
    if type(seed) is not int or type(bootstrap_reps) is not int or bootstrap_reps < 0:
        raise ValueError("Invalid deterministic bootstrap configuration")
    if len(expected) != 20 or len(by_context) != 20:
        raise ValueError("Only the completed 20-context development cohort is supported")
    scenes, seen = defaultdict(list), set()
    for cid, key, mode in expected:
        if not isinstance(cid, str) or mode not in MODES or len(key) != 2:
            raise ValueError("Invalid predeclared context identity")
        key = tuple(key)
        if not all(type(v) is str and v for v in key):
            raise ValueError("Noncanonical episode identity")
        if (cid, mode) in seen or cid not in by_context:
            raise ValueError("Duplicate or missing frozen context")
        seen.add((cid, mode))
        outcomes = by_context[cid]
        if set(outcomes) != set(ACTIONS):
            raise ValueError("Not all four actual operation outcomes are available")
        for arm in ACTIONS:
            _check_row(outcomes[arm], key, mode, arm)
        successes = {arm: outcomes[arm]["success"] for arm in ACTIONS}
        contextspl = {arm: float(outcomes[arm]["spl"]) for arm in ACTIONS}
        scenes[key[0]].append({
            "context_id": cid,
            "episode_key": list(key),
            "mode": mode,
            "success": successes,
            "spl": contextspl,
        })
    if set(by_context) != {x[0] for x in expected} or len(scenes) != 10:
        raise ValueError("Unexpected outcome context/scene denominator")
    for scene, group in scenes.items():
        if (len(group) != 2 or {c["mode"] for c in group} != set(MODES)
                or len({tuple(c["episode_key"]) for c in group}) != 1):
            raise ValueError("Missing natural/perturbed pairing in scene " + scene)
    groups = [sorted(scenes[scene], key=lambda x: MODES.index(x["mode"]))
              for scene in sorted(scenes)]
    overall = _summarize(groups)
    by_mode = {
        mode: _summarize([[c] for group in groups for c in group
                          if c["mode"] == mode])
        for mode in MODES
    }
    comparisons = {}
    for arm in ACTIONS[1:]:
        relation_counts = {"RESCUE": 0, "HARM": 0,
                           "BOTH_SUCCESS": 0, "BOTH_FAILURE": 0}
        scene_effects = []
        by_mode_counts = {
            mode: {"RESCUE": 0, "HARM": 0,
                   "BOTH_SUCCESS": 0, "BOTH_FAILURE": 0}
            for mode in MODES
        }
        for group in groups:
            deltas = []
            for c in group:
                keep, treated = c["success"]["KEEP"], c["success"][arm]
                relation = (
                    "RESCUE" if keep == 0 and treated == 1 else
                    "HARM" if keep == 1 and treated == 0 else
                    "BOTH_SUCCESS" if keep == 1 else "BOTH_FAILURE"
                )
                relation_counts[relation] += 1
                by_mode_counts[c["mode"]][relation] += 1
                deltas.append(treated - keep)
            scene_effects.append(sum(deltas) / len(deltas))
        comparisons[arm] = {
            "context_relation_counts": relation_counts,
            "context_relation_by_mode": by_mode_counts,
            "mean_success_delta_vs_keep": sum(scene_effects) / len(scene_effects),
            "N_scenes_with_positive_mean_delta": sum(x > 0 for x in scene_effects),
            "N_scenes_with_negative_mean_delta": sum(x < 0 for x in scene_effects),
            "both_rescue_and_harm_observed": (
                relation_counts["RESCUE"] > 0 and relation_counts["HARM"] > 0
            ),
        }
    # Exploratory scene-cluster bootstrap ONLY; the chosen "best fixed" is
    # re-selected within each resample. Not a confirmatory CI or p-value.
    bootstrap = None
    if bootstrap_reps:
        rng = random.Random(seed)
        diffs = []
        for _ in range(bootstrap_reps):
            resampled = [groups[rng.randrange(len(groups))]
                         for _ in range(len(groups))]
            diffs.append(_summarize(resampled)[
                "oracle_minus_best_fixed_success_rate"])
        diffs.sort()
        def percentile(p):
            return diffs[min(len(diffs) - 1, int(p * (len(diffs) - 1)))]
        bootstrap = {
            "N_scene_cluster_resamples": bootstrap_reps, "seed": seed,
            "exploratory_percentile_2_5": percentile(.025),
            "exploratory_percentile_97_5": percentile(.975),
            "not_confirmatory_inference": True,
        }
    return {
        "schema": SCHEMA,
        "status": "EXPLORATORY_TRAIN_DEVELOPMENT_H1_NOT_HELD_OUT",
        "N_scene_groups": len(groups),
        "N_contexts": len(expected),
        "N_actual_rollouts": len(expected) * len(ACTIONS),
        "N_actual_effects_vs_keep": len(expected) * (len(ACTIONS)-1),
        "action_set": list(ACTIONS),
        "data_contexts": list(MODES),
        "primary_outcome": "actual_binary_success",
        "no_online_features_used": True,
        "in_sample_best_fixed_is_optimistically_selected": True,
        "oracle_is_infeasible_uses_future_outcome": True,
        "outcome_costs_not_combined_into_arbitrary_scalar": True,
        "additional_scan_motion_6_yaw_actions_reset_0": True,
        "RESET_only_filters_pre_gate_history_not_true_model_memory_erasure": True,
        "SEEK_combines_physical_scan_and_supplementary_RGB_no_placebo_control": True,
        "two_contexts_within_scene_are_correlated": True,
        "single_rollout_per_case_action_no_stochastic_repeats": True,
        "excludes_additional_posthoc_three_case_rollouts": True,
        "posthoc_focus_scenes_overlap_original_ten": True,
        "not_a_preregistered_confirmatory_test": True,
        "overall": overall,
        "by_context_mode": by_mode,
        "operation_effects_vs_keep": comparisons,
        "exploratory_scene_cluster_bootstrap": bootstrap,
        "interpretation_guardrails": [
            "Oracle-best-fixed gap is a descriptive upper bound, not deployable policy gain",
            "A positive development gap does not prove predictability from predecision data",
            "10 independent scenes are insufficient to conclude robust generalization",
            "Three posthoc-inspected scenes are contained in the original 10; only their extra posthoc arms are excluded",
            "Do not select a new confirmatory cohort or operation set based on individual outcomes",
        ],
    }


def make_report(paired_run, label_dir=None, *, reps=BOOTSTRAPS, seed=SEED):
    # Import all original validation logic, but never import it in unit tests
    # evaluating pure H1 analytics.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import intervention_selection_outcome_labels_v01 as frozen
    root, paired_report, protocol, expected, by_context = (
        frozen.load_audited_source(Path(paired_run)))
    if label_dir is not None:
        audited = frozen.audit_package(Path(label_dir))
        if Path(audited["source_run"]).resolve() != root:
            raise ValueError("Audited offline label package is from another paired run")
    result = analyze(expected, by_context, bootstrap_reps=reps, seed=seed)
    result["provenance"] = {
        "original_paired_run": str(root),
        "paired_report_status": paired_report["status"],
        "paired_source_manifest_sha256": sha256(root / "manifest.json"),
        "paired_protocol_sha256": sha256(root / "paired_protocol.json"),
        "paired_outcomes_sha256": sha256(root / "intervention_outcomes.jsonl"),
        "audited_label_dir": str(Path(label_dir).resolve()) if label_dir else None,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-run", required=True, type=Path)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Create-only JSON output path; stdout if omitted")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    result = make_report(args.paired_run, args.label_dir,
                         reps=args.bootstrap_reps, seed=args.seed)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print("H1_DEVELOPMENT_AUDIT=PASS", result["N_scene_groups"],
          "scenes", result["N_contexts"], "contexts",
          result["N_actual_rollouts"], "rollouts")
    print(payload)


if __name__ == "__main__":
    main()
