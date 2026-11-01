"""Offline, create-only v0.1 paired intervention *outcome* label builder.

Input: audited paired run with 20 context records and 80 actual outcomes.
Output: 20 context summaries, 60 action-vs-KEEP effects, 120 unordered
pairwise comparisons, and a source-hashed provenance report.

This deliberately DOES NOT train Q, invent continuous rewards, use outcome
labels as online features, assign a best action in all-failure contexts,
select episodes using hindsight, or run Habitat. The SPL/cost preference
layer is an EXPLICIT heuristic, not preregistered scientific ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_selection_paired_v01 as paired

ACTIONS = ("KEEP",) + paired.NEW_ACTIONS
MODES = paired.MODES
SCHEMA = "intervention_selection_outcome_labels.v0.1"
DEFAULT_SPL_TOLERANCE = 0.02
EXPECTED_COST = {
    "KEEP": 0, "SEEK_LEFT45": 6,
    "SEEK_RIGHT45": 6, "HISTORY_RESET": 0,
}
EXPECTED_AUDIT = {
    "KEEP": "REUSED_AUDITED_KEEP",
    "SEEK_LEFT45": "PAIRED_PREARM_VERIFIED",
    "SEEK_RIGHT45": "PAIRED_PREARM_VERIFIED",
    "HISTORY_RESET": "PAIRED_PREARM_VERIFIED",
}
REQUIRED_METRICS = (
    "context_id", "episode_key", "source_type", "action",
    "success", "ne", "spl", "os", "extra_yaw_actions",
    "prefix_perturbation_actions", "audit",
)
SOURCE_FILES = (
    "manifest.json", "paired_protocol.json", "paired_report.json",
    "contexts_predecision.jsonl", "intervention_outcomes.jsonl",
)
OUTPUT_FILES = (
    "context_label_summary.jsonl",
    "action_effects_vs_keep.jsonl",
    "pairwise_preferences.jsonl",
)
SCOPE = (
    "TRAIN development data only; deterministic post-outcome OFFLINE labels. "
    "No privileged results are used as online selector inputs; per-scene "
    "natural/perturbed contexts are correlated. No trained Q-values, "
    "oracle explanation of failure mechanisms, or held-out claim. "
    "The SPL tolerance and cost proxy are heuristic and must be frozen "
    "before training; evaluate sensitivity before any scientific claim."
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record at {path}:{i}")
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Non-object record at {path}:{i}")
            rows.append(obj)
    return rows


def verify_tolerance(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("SPL tolerance must be finite")
    if not 0.0 <= value < 1.0:
        raise ValueError("SPL tolerance outside [0,1)")
    return float(value)


def check_outcome(row):
    if not isinstance(row, dict):
        raise ValueError("Non-object outcome row")
    if not all(k in row for k in REQUIRED_METRICS):
        raise ValueError("Missing required outcome field")
    act = row["action"]
    if act not in ACTIONS or row["source_type"] not in MODES:
        raise ValueError("Unknown frozen action/context mode")
    key = row["episode_key"]
    if (not isinstance(key, list) or len(key) != 2 or
            any(not isinstance(v, str) or not v for v in key)):
        raise ValueError("Invalid scene/episode identity")
    if not isinstance(row["context_id"], str) or not row["context_id"].endswith(
            "_" + row["source_type"]):
        raise ValueError("Context identifier/mode mismatch")
    if (row["audit"] != EXPECTED_AUDIT[act]
            or type(row["success"]) is not int or row["success"] not in (0, 1)
            or type(row["extra_yaw_actions"]) is not int
            or row["extra_yaw_actions"] != EXPECTED_COST[act]
            or type(row["prefix_perturbation_actions"]) is not int
            or row["prefix_perturbation_actions"] !=
               (3 if row["source_type"] == "perturbed" else 0)):
        raise ValueError("Action cost, physical prefix or audit tag changed")
    for key, lo, hi in (("spl", 0.0, 1.0), ("os", 0.0, 1.0),
                        ("ne", 0.0, math.inf)):
        val = row[key]
        if (type(val) not in (float, int) or not math.isfinite(val)
                or not lo <= val <= hi):
            raise ValueError("Invalid navigation metric: " + key)
    if float(row["spl"]) > 0 and row["success"] == 0:
        raise ValueError("Failure cannot have positive reported SPL")
    return True


def load_audited_source(root):
    root = Path(root).resolve()
    # The saved manifest records all original evaluator, model, and script
    # hashes. This call never changes or re-runs the source.
    manifest = paired.pilot.verify_manifest(root)
    report = read_json(root / "paired_report.json")
    protocol = read_json(root / "paired_protocol.json")
    if (manifest.get("schema") != paired.SCHEMA
            or report.get("schema") != paired.SCHEMA
            or protocol.get("schema") != paired.SCHEMA
            or report.get("status") != "TEN_CASE_PAIRED_AUDITED"
            or report.get("N_original_cases") != 12
            or report.get("N_eligible_cases") != 10
            or report.get("N_selected_cases") != 10
            or report.get("N_contexts") != 20
            or report.get("N_new_intervention_results") != 60
            or report.get("N_four_arm_outcomes") != 80
            or report.get("N_hindsight_preference_labels") != 0
            or report.get("outcome_labels_used_as_online_features") is not False
            or protocol.get("max_cases") != 10
            or len(protocol.get("selected_cases", [])) != 10):
        raise ValueError("Require original fully audited 10-scene/80-outcome dataset")

    expected = []
    for c in protocol["selected_cases"]:
        key = c["key"]
        if (not isinstance(key, list) or len(key) != 2 or
                not isinstance(c["case_id"], str)):
            raise ValueError("Malformed source predeclared case")
        for mode in MODES:
            expected.append((c["case_id"] + "_" + mode, tuple(key), mode))
    if len(set((cid, key) for cid, key, _ in expected)) != 20:
        raise ValueError("Duplicate predeclared contexts")
    if len({key[0] for _, key, _ in expected}) != 10:
        raise ValueError("Source scenes no longer disjoint")

    inputs = read_jsonl(root / "contexts_predecision.jsonl")
    outcomes = read_jsonl(root / "intervention_outcomes.jsonl")
    expected_ids = {cid for cid, _, _ in expected}
    context_index = {}
    for row in inputs:
        cid = row.get("context_id")
        if cid not in expected_ids or cid in context_index:
            raise ValueError("Unexpected/duplicate online context")
        if (row.get("schema") != "intervention_selection_predecision.v0.1"
                or row.get("no_privileged_simulator_metrics") is not True):
            raise ValueError("Online context schema/safety status changed")
        # Fail closed if any offline target field is slipped into the saved
        # online feature row. This does not duplicate online feature content.
        forbidden = {
            "success", "ne", "spl", "os", "goal_position",
            "distance_to_goal", "sim_position", "sim_orientation",
            "gps", "depth", "oracle_success", "intervention_outcomes",
            "best_action", "q_value", "oracle_best_action",
        }
        if forbidden & set(row):
            raise ValueError("Privileged or target field leaked into online context")
        context_index[cid] = row
    if set(context_index) != expected_ids or len(inputs) != 20:
        raise ValueError("Missing predecision contexts")

    by_context = defaultdict(dict)
    for row in outcomes:
        check_outcome(row)
        cid, action = row["context_id"], row["action"]
        if cid not in expected_ids or action in by_context[cid]:
            raise ValueError("Unknown context or duplicated candidate outcome")
        by_context[cid][action] = row
    if len(outcomes) != 80 or set(by_context) != expected_ids:
        raise ValueError("Missing/extra actual arm outcomes")
    for cid, key, mode in expected:
        input_row = context_index[cid]
        if (input_row.get("episode_key") != list(key)
                or input_row.get("source_type") != mode):
            raise ValueError("Online context identity mismatched")
        rows = by_context[cid]
        if set(rows) != set(ACTIONS):
            raise ValueError("Incomplete real four-arm outcomes")
        for action in ACTIONS:
            if rows[action]["episode_key"] != list(key) or rows[action]["source_type"] != mode:
                raise ValueError("Cross-scene/mode arm outcome mix")
    return root, report, protocol, expected, by_context


def effect_vs_keep(context_id, mode, action, keep, treated):
    if action == "KEEP":
        raise ValueError("Action effects require a non-KEEP action")
    ks, ts = keep["success"], treated["success"]
    label = (
        "RESCUE" if ks == 0 and ts == 1
        else "HARM" if ks == 1 and ts == 0
        else "BOTH_SUCCESS" if ks == 1 and ts == 1
        else "BOTH_FAILURE"
    )
    return {
        "schema": SCHEMA, "context_id": context_id,
        "episode_key": keep["episode_key"], "source_type": mode,
        "action": action, "baseline": "KEEP", "outcome_relation": label,
        "keep_success": ks, "action_success": ts,
        "delta_success": ts - ks,
        "delta_spl": float(treated["spl"]) - float(keep["spl"]),
        # Positive delta_ne_improvement_m indicates CLOSER final position,
        # not necessarily a successful recovered trajectory.
        "delta_ne_improvement_m": float(keep["ne"]) - float(treated["ne"]),
        "extra_yaw_actions": treated["extra_yaw_actions"],
        "shared_prefix_perturb_yaw_actions":
            keep["prefix_perturbation_actions"],
        "ne_is_diagnostic_not_rescue_label": True,
        "offline_only": True,
    }


def preference(first, second, spl_tolerance):
    """Return provisional Pareto-like winner, tie, tradeoff, or unresolved.

    Prioritize actual SR. If BOTH failed there is no success-based label and
    no action is crowned the best even when one final NE is smaller.
    When BOTH succeed, SPL within tolerance is treated as equivalent and
    *additional* physical yaw is minimized; a significant SPL gain obtained
    using more yaw is an incomparable tradeoff, not a fake scalar Q target.
    """
    t = verify_tolerance(spl_tolerance)
    a, b = first, second
    if a["context_id"] != b["context_id"]:
        raise ValueError("Cannot compare outcomes from different contexts")
    ac, bc = a["extra_yaw_actions"], b["extra_yaw_actions"]
    dspl = float(a["spl"]) - float(b["spl"])
    if a["success"] != b["success"]:
        return {
            "relation": "SUCCESS_DOMINANCE",
            "preferred": a["action"] if a["success"] == 1 else b["action"],
            "training_strength": "HARD_OBSERVED_SINGLE_ROLLOUT",
        }
    if a["success"] == 0:
        return {
            "relation": "UNRESOLVED_BOTH_FAILED",
            "preferred": None,
            "training_strength": "NO_SUCCESS_SUPERVISION",
        }
    if abs(dspl) <= t:
        if ac == bc:
            return {
                "relation": "TIE_WITHIN_SPL_TOLERANCE",
                "preferred": None,
                "training_strength": "AMBIGUOUS",
            }
        return {
            "relation": "COST_DOMINANCE_WITHIN_SPL_TOLERANCE",
            "preferred": a["action"] if ac < bc else b["action"],
            "training_strength": "HEURISTIC_TOLERANCE",
        }
    better_spl = a if dspl > 0 else b
    worse_spl = b if dspl > 0 else a
    if better_spl["extra_yaw_actions"] <= worse_spl["extra_yaw_actions"]:
        return {
            "relation": "SPL_AND_COST_DOMINANCE",
            "preferred": better_spl["action"],
            "training_strength": "HEURISTIC_TOLERANCE",
        }
    return {
        "relation": "SPL_COST_TRADEOFF",
        "preferred": None,
        "training_strength": "AMBIGUOUS",
    }


def build_labels(expected, by_context, spl_tolerance):
    tol = verify_tolerance(spl_tolerance)
    context_summaries, effects, pairs = [], [], []
    for cid, key, mode in expected:
        action_rows = by_context[cid]
        keep = action_rows["KEEP"]
        success_set = [a for a in ACTIONS if action_rows[a]["success"] == 1]
        changes = [
            effect_vs_keep(cid, mode, action, keep, action_rows[action])
            for action in paired.NEW_ACTIONS
        ]
        effects.extend(changes)
        rescue = [x["action"] for x in changes if x["outcome_relation"] == "RESCUE"]
        harm = [x["action"] for x in changes if x["outcome_relation"] == "HARM"]
        context_type = (
            "KEEP_FAILS_RESCUABLE" if keep["success"] == 0 and rescue
            else "KEEP_FAILS_NO_SUCCESSFUL_CANDIDATE"
                 if keep["success"] == 0 else
            "KEEP_SUCCEEDS_BUT_HARM_POSSIBLE" if harm else
            "KEEP_SUCCEEDS_ALL_CANDIDATES_SUCCESSFUL"
        )
        context_summaries.append({
            "schema": SCHEMA, "context_id": cid,
            "episode_key": list(key), "source_type": mode,
            "context_type": context_type,
            "keep_success": keep["success"],
            "successful_actions_OFFLINE_ONLY": success_set,
            "rescue_actions_OFFLINE_ONLY": rescue,
            "harm_actions_OFFLINE_ONLY": harm,
            "no_successful_candidate": not success_set,
            "best_action": None,
            "best_action_not_assigned_due_to_cost_and_ambiguity": True,
            "offline_only": True,
        })
        for aa, bb in itertools.combinations(ACTIONS, 2):
            a, b = action_rows[aa], action_rows[bb]
            pref = preference(a, b, tol)
            pairs.append({
                "schema": SCHEMA, "context_id": cid,
                "episode_key": list(key), "source_type": mode,
                "action_a": aa, "action_b": bb,
                "a_success": a["success"], "b_success": b["success"],
                "delta_sr_a_minus_b": a["success"] - b["success"],
                "delta_spl_a_minus_b": float(a["spl"]) - float(b["spl"]),
                "ne_improvement_a_over_b_m": float(b["ne"]) - float(a["ne"]),
                "additional_yaw_a": a["extra_yaw_actions"],
                "additional_yaw_b": b["extra_yaw_actions"],
                "preferred_action_OFFLINE_ONLY": pref["preferred"],
                "relation": pref["relation"],
                "training_strength": pref["training_strength"],
                "offline_only": True,
            })
    n = len(expected)
    if (len(context_summaries) != n
            or len(effects) != 3*n
            or len(pairs) != 6*n):
        raise RuntimeError("Incomplete context/effect/preference cardinality")
    return context_summaries, effects, pairs


def audit_summaries(summaries, effects, pairs, n_cases):
    counts = Counter(s["context_type"] for s in summaries)
    effects_counts = Counter(x["outcome_relation"] for x in effects)
    pair_counts = Counter(x["relation"] for x in pairs)
    rescue = sum(x["outcome_relation"] == "RESCUE" for x in effects)
    harm = sum(x["outcome_relation"] == "HARM" for x in effects)
    if sum(counts.values()) != 2*n_cases:
        raise ValueError("Context summary size changed")
    if rescue != sum(len(s["rescue_actions_OFFLINE_ONLY"]) for s in summaries):
        raise ValueError("Rescue summary and effects disagree")
    if harm != sum(len(s["harm_actions_OFFLINE_ONLY"]) for s in summaries):
        raise ValueError("Harm summary and effects disagree")
    return {
        "N_scene_groups": n_cases,
        "N_contexts": len(summaries),
        "N_effects_vs_keep": len(effects),
        "N_pairwise_comparisons": len(pairs),
        "context_type_counts": dict(sorted(counts.items())),
        "action_effect_counts": dict(sorted(effects_counts.items())),
        "pairwise_relation_counts": dict(sorted(pair_counts.items())),
        "N_rescue_pairs": rescue, "N_harm_pairs": harm,
        "no_best_action_fabricated": all(s["best_action"] is None for s in summaries),
    }


def jsonl_bytes(records):
    return "".join(
        json.dumps(r, ensure_ascii=False, allow_nan=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
        for r in records
    ).encode("utf-8")


def make_label_package(root, spl_tolerance):
    root, source_report, protocol, expected, by_context = load_audited_source(root)
    contexts, effects, pairs = build_labels(expected, by_context, spl_tolerance)
    audit = audit_summaries(contexts, effects, pairs, len(protocol["selected_cases"]))
    provenance = {str(p): sha256(root / p) for p in SOURCE_FILES}
    provenance[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    package = {
        "schema": SCHEMA, "status": "OFFLINE_LABELS_PREPARED_NOT_TRAINED",
        "source_run": str(root),
        "source_status": source_report["status"],
        "source_hashes": provenance,
        "spl_tolerance": verify_tolerance(spl_tolerance),
        "cost_metric": "additional_real_yaw_actions_per_context",
        "cost_scope": "pre-context_perturbation_actions_shared_not_ranked",
        "binary_success_priority": True,
        "both_failure_positive_target": False,
        "all_preference_labels_offline_only": True,
        "hard_success_labels_are_single_rollout_not_robust_expectations": True,
        "context_summaries_have_no_best_action": True,
        "split": "train_development_only",
        "no_held_out_generalization_claim": True,
        "scope": SCOPE,
        **audit,
    }
    outputs = dict(zip(OUTPUT_FILES, [
        jsonl_bytes(contexts), jsonl_bytes(effects), jsonl_bytes(pairs),
    ]))
    return package, outputs


def write_package(root, label_dir, spl_tolerance):
    dest = Path(label_dir).resolve()
    if dest.exists():
        raise FileExistsError("Labels are create-only; never overwrite")
    package, outputs = make_label_package(root, spl_tolerance)
    dest.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        with (dest / name).open("xb") as f:
            f.write(data)
    package["output_hashes"] = {
        name: sha256(dest / name) for name in OUTPUT_FILES
    }
    (dest / "label_report.json").write_text(
        json.dumps(package, ensure_ascii=False, sort_keys=True, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        k: package[k] for k in (
            "status", "N_scene_groups", "N_contexts", "N_effects_vs_keep",
            "N_pairwise_comparisons", "N_rescue_pairs", "N_harm_pairs",
            "context_type_counts",
        )
    }, ensure_ascii=False, indent=2))
    return package


def audit_package(root):
    dest = Path(root).resolve()
    report = read_json(dest / "label_report.json")
    if (report.get("schema") != SCHEMA or
            report.get("status") != "OFFLINE_LABELS_PREPARED_NOT_TRAINED"):
        raise ValueError("Unsupported/incomplete label report")
    package, outputs = make_label_package(
        report["source_run"], report["spl_tolerance"])
    expected_hashes = {name: hashlib.sha256(data).hexdigest()
                       for name, data in outputs.items()}
    if (report.get("output_hashes") != expected_hashes or
            package != {k: v for k, v in report.items()
                        if k != "output_hashes"}):
        raise ValueError("Offline labels, provenance or source changed")
    for name, expected in expected_hashes.items():
        if sha256(dest / name) != expected:
            raise ValueError("Saved offline label file changed: " + name)
    print("OFFLINE_LABELS_AUDITED_NO_Q_TRAINING", report["N_contexts"],
          "contexts", report["N_effects_vs_keep"], "effects",
          report["N_pairwise_comparisons"], "pairs")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("build", help="CPU only, immutable offline label export")
    a.add_argument("--run-dir", required=True,
                   help="Full 10-scene TEN_CASE_PAIRED_AUDITED source")
    a.add_argument("--out-dir", required=True,
                   help="NEW empty output directory, not under original run")
    a.add_argument("--spl-tolerance", type=float,
                   default=DEFAULT_SPL_TOLERANCE)
    a.set_defaults(fn=lambda x: write_package(
        x.run_dir, x.out_dir, x.spl_tolerance))
    a = sub.add_parser("audit", help="CPU-only recompute and compare every file")
    a.add_argument("--label-dir", required=True)
    a.set_defaults(fn=lambda x: audit_package(x.label_dir))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
