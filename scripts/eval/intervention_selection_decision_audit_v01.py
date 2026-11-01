"""CPU-only, fail-closed posthoc audit of v0.1 intervention continuation traces.

Only policy-continuation indices are present in the current share bundle;
main_step is NOT a System-2 event. This script never infers a System-2
boundary from action codes, pose changes, or main_step transitions.

Offline GT pose and outcome fields are used ONLY for posthoc auditing. None
are exported as online features or training labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

SCHEMA = "intervention_selection_decision_audit.v0.1"
SOURCE_SCHEMA = "intervention_selection_trace_divergence.v0.1"
MEMBERS = (
    "trace_divergence_report.json",
    "continuation_action_traces.jsonl",
    "trace_divergence_vs_keep.jsonl",
    "decision_comparisons.jsonl",
)
ARMS = ("KEEP", "SEEK_LEFT45", "SEEK_RIGHT45", "HISTORY_RESET")
FOCUS = (
    ("i5noydFURQK", "70", "perturbed", "SEEK_RIGHT45", "RESCUE"),
    ("r1Q1Z4BcV1o", "106", "perturbed", "SEEK_RIGHT45", "RESCUE"),
    ("PX4nDJXEHrG", "79", "perturbed", "SEEK_LEFT45", "HARM"),
)
POSITION_THRESHOLD_M = 0.05
ORIENTATION_THRESHOLD_DEG = 1.0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jsonl(rows) -> str:
    return "".join(stable_json(row) + "\n" for row in rows)


def load_input(input_path):
    """Read exactly four named files, no archive extraction or input writes."""
    input_path = Path(input_path)
    if input_path.is_dir():
        raw = {name: (input_path / name).read_bytes() for name in MEMBERS}
    elif zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as bundle:
            names = bundle.namelist()
            if sorted(names) != sorted(MEMBERS) or len(names) != len(set(names)):
                raise ValueError("Archive must contain exactly four unique expected members")
            if any(bundle.getinfo(n).file_size > 100_000_000 for n in names):
                raise ValueError("Input member exceeds size limit")
            raw = {name: bundle.read(name) for name in MEMBERS}
    else:
        raise ValueError("Input must be a directory or a trace_divergence_share ZIP")
    report = json.loads(raw[MEMBERS[0]])
    expected = report.get("output_hashes", {})
    for name in MEMBERS[1:]:
        if expected.get(name) != sha256(raw[name]):
            raise ValueError(f"Original bundle SHA256 mismatch: {name}")
    if report.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Wrong source report schema")
    if report.get("status") != "POSTHOC_CONTINUATION_TRACES_EXPORTED_NOT_CAUSALLY_IDENTIFIED":
        raise ValueError("Unexpected source status; inspect before auditing")
    parsed = {}
    for name in MEMBERS[1:]:
        try:
            parsed[name] = [json.loads(s) for s in raw[name].decode("utf-8").splitlines()]
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSONL {name}") from exc
    return report, parsed, {n: sha256(raw[n]) for n in MEMBERS}


def key(row):
    return (str(row["context_id"]), str(row["action"]))


def unique_index(rows, name, key_fn):
    result = {}
    for row in rows:
        k = key_fn(row)
        if k in result:
            raise ValueError(f"Duplicate {name}: {k}")
        result[k] = row
    return result


def validate_identity(row, context_info):
    episode = row.get("episode_key")
    if not isinstance(episode, list) or len(episode) != 2:
        raise ValueError("Missing episode identity")
    identity = (tuple(str(x) for x in episode), row.get("source_type"))
    ctx = str(row["context_id"])
    if ctx in context_info and context_info[ctx] != identity:
        raise ValueError(f"Context metadata differs: {ctx}")
    context_info[ctx] = identity


def normalized_vec(row, field, size):
    value = row.get(field)
    if (not isinstance(value, list) or len(value) != size
            or not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                       and math.isfinite(v) for v in value)):
        raise ValueError(f"Invalid offline pose field {field}")
    return [float(x) for x in value]


def position_distance(a, b):
    x = normalized_vec(a, "physical_position_OFFLINE_ONLY", 3)
    y = normalized_vec(b, "physical_position_OFFLINE_ONLY", 3)
    return math.dist(x, y)


def orientation_distance(a, b):
    """Quaternion geodesic angle, independent of xyzw vs wxyz convention."""
    x = normalized_vec(a, "physical_orientation_OFFLINE_ONLY", 4)
    y = normalized_vec(b, "physical_orientation_OFFLINE_ONLY", 4)
    nx, ny = math.sqrt(sum(v*v for v in x)), math.sqrt(sum(v*v for v in y))
    if nx < 1e-8 or ny < 1e-8:
        raise ValueError("Zero-norm quaternion")
    dot = abs(sum(p*q for p, q in zip(x, y)) / (nx*ny))
    return math.degrees(2 * math.acos(min(1.0, max(0.0, dot))))


def first_difference(keep, treated, predicate):
    """Compare matched *relative* continuation action indices, never time."""
    n = min(len(keep), len(treated))
    for i in range(n):
        distance = predicate(keep[i], treated[i])
        if distance is not None:
            return {"continuation_index": i, "observed": distance}
    return None


def collect(report, data, hashes):
    actions = data["continuation_action_traces.jsonl"]
    effects = data["trace_divergence_vs_keep.jsonl"]
    comparisons = data["decision_comparisons.jsonl"]
    if (len(actions) != report.get("N_action_rows")
            or len(effects) != report.get("N_effects_vs_KEEP")
            or len(comparisons) != report.get("N_first_S2_pair_comparisons")):
        raise ValueError("Report row counts disagree with source data")

    context_info, trace_groups = {}, defaultdict(list)
    row_ids = set()
    for row in actions:
        if row.get("schema") != SOURCE_SCHEMA or row.get("action") not in ARMS:
            raise ValueError("Unexpected trace schema/action")
        validate_identity(row, context_info)
        if row.get("ground_truth_trajectory_must_never_be_online_input") is not True:
            raise ValueError("Offline pose isolation flag missing")
        idx = row.get("continuation_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError("Invalid relative continuation index")
        rid = (str(row["context_id"]), row["action"], idx)
        if rid in row_ids:
            raise ValueError("Duplicate continuation trace action index")
        row_ids.add(rid)
        normalized_vec(row, "physical_position_OFFLINE_ONLY", 3)
        normalized_vec(row, "physical_orientation_OFFLINE_ONLY", 4)
        trace_groups[key(row)].append(row)
    for k, rows in trace_groups.items():
        rows.sort(key=lambda r: r["continuation_index"])
        if [r["continuation_index"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous continuation action indices: {k}")

    effect_idx = unique_index(effects, "effect", key)
    for row in effects:
        if row.get("schema") != SOURCE_SCHEMA or row["action"] not in ARMS[1:]:
            raise ValueError("Unexpected effect row")
        validate_identity(row, context_info)
        if (row.get("action_index_not_a_System2_decision_time") is not True
                or row.get("true_scan_actions_removed_from_policy_comparison") is not True
                or row.get("true_perturbation_actions_removed_from_policy_comparison") is not True):
            raise ValueError("Missing index-alignment disclaimers")

    if len(trace_groups) != report.get("N_real_arm_traces"):
        raise ValueError("Missing or duplicate arm trajectories")
    if len(context_info) != report.get("N_contexts"):
        raise ValueError("Context count mismatch")
    for ctx in context_info:
        if {arm for c, arm in trace_groups if c == ctx} != set(ARMS):
            raise ValueError(f"Incomplete four-arm context: {ctx}")
        if {arm for c, arm in effect_idx if c == ctx} != set(ARMS[1:]):
            raise ValueError(f"Incomplete KEEP comparison: {ctx}")

    comp_idx = unique_index(comparisons, "first S2 textual comparison",
                            lambda r: (str(r["context_id"]), r["action_a"], r["action_b"]))
    for row in comparisons:
        if row.get("schema") != SOURCE_SCHEMA or row.get("KEEP_first_S2_available") is not False:
            raise ValueError("Unexpected first S2 evidence; re-audit required")
        validate_identity(row, context_info)
        if (str(row["context_id"]), row["action_a"]) not in trace_groups:
            raise ValueError("Missing S2 comparison trajectory")
        if (str(row["context_id"]), row["action_b"]) not in trace_groups:
            raise ValueError("Missing S2 comparison trajectory")
        if row["identical_first_S2_text"] != (row["first_S2_a"] == row["first_S2_b"]):
            raise ValueError("First-S2 text comparison contradicts raw strings")
    if len(comp_idx) != report.get("N_first_S2_pair_comparisons"):
        raise ValueError("S2 comparison count mismatch")

    audited, dossiers = [], []
    for ctx in sorted(context_info):
        episode, source_type = context_info[ctx]
        keep = trace_groups[(ctx, "KEEP")]
        for arm in ARMS[1:]:
            treated = trace_groups[(ctx, arm)]
            effect = effect_idx[(ctx, arm)]
            first_action = first_difference(
                keep, treated, lambda a, b: ({"KEEP_action_code": int(a["action_code"]),
                    "treated_action_code": int(b["action_code"])}
                    if a["action_code"] != b["action_code"] else None))
            first_pos = first_difference(
                keep, treated, lambda a, b: (d if (d := position_distance(a, b)) >= POSITION_THRESHOLD_M else None))
            first_yaw = first_difference(
                keep, treated, lambda a, b: (d if (d := orientation_distance(a, b)) >= ORIENTATION_THRESHOLD_DEG else None))
            actual_index = first_action["continuation_index"] if first_action else None
            if actual_index != effect.get("first_different_continuation_index"):
                raise ValueError(f"Action divergence differs from original report: {(ctx, arm)}")
            if len(keep) != effect["keep_continuation_action_count"] or len(treated) != effect["action_continuation_action_count"]:
                raise ValueError("Effect action count mismatch")
            keep_ok, treated_ok = effect["keep_success_OFFLINE_ONLY"], effect["action_success_OFFLINE_ONLY"]
            if keep_ok not in (0, 1) or treated_ok not in (0, 1):
                raise ValueError("Outcome must be binary")
            status = "RESCUE" if not keep_ok and treated_ok else ("HARM" if keep_ok and not treated_ok else "NO_SR_CHANGE")
            event = {
                "schema": SCHEMA,
                "context_id": ctx, "episode_key": list(episode), "source_type": source_type,
                "reference": "KEEP", "action": arm,
                "pair_type": "OFFLINE_RELATIVE_CONTINUATION_INDEX",
                "first_action_code_difference": first_action,
                "first_orientation_difference_ge_1deg_OFFLINE_ONLY": first_yaw,
                "first_position_difference_ge_0_05m_OFFLINE_ONLY": first_pos,
                "different_continuation_lengths": len(keep) != len(treated),
                "lengths_KEEP_action": [len(keep), len(treated)],
                "system2_call_alignment": "NOT_OBSERVABLE",
                "first_system2_decision_difference": None,
                "main_step_is_not_system2_call_id": True,
                "pre_scan_actions_excluded": True,
                "relative_index_is_not_elapsed_time": True,
                "outcome_change_OFFLINE_ONLY": status,
                "success_KEEP_action_OFFLINE_ONLY": [keep_ok, treated_ok],
                "ne_KEEP_action_OFFLINE_ONLY": [effect["keep_ne_OFFLINE_ONLY"], effect["action_ne_OFFLINE_ONLY"]],
                "causal_recovery_mechanism_identified": False,
            }
            audited.append(event)
            if (episode[0], episode[1], source_type, arm, status) in FOCUS:
                dossiers.append({
                    "schema": SCHEMA, "context_id": ctx, "episode_key": list(episode),
                    "source_type": source_type, "action": arm,
                    "case_role": status, "event": event,
                    "observed": "Paired outcomes and posthoc same-index action/pose divergence only",
                    "not_established": [
                        "First matching System-2 call boundary",
                        "Whether physical scan, input token format or newly observed pixels caused the outcome",
                        "Matched pre-arm RNG/checkpoint and return-pose proof from original run",
                    ],
                    "next_step": "Pre-registered 4-arm physically matched ablation, not yet executed",
                })
    if len(dossiers) != len(FOCUS):
        raise ValueError("Missing one of three predeclared mechanism cases")

    manifest = []
    for d in sorted(dossiers, key=lambda x: (x["episode_key"][0], x["episode_key"][1])):
        arm = d["action"]
        direction = "LEFT45" if arm == "SEEK_LEFT45" else "RIGHT45"
        out_code = 3 if direction == "LEFT45" else 2
        return_code = 2 if direction == "LEFT45" else 3
        scan_actions = [out_code] * 3 + [return_code] * 3
        common = {
            "episode_key": d["episode_key"], "context_id": d["context_id"],
            "source_type": d["source_type"],
            "scan_direction": direction,
            "scan_actions_candidate_unverified": scan_actions,
            "fixed_scan_action_budget": 6,
            "checkpoint_hash": None,
            "identical_gate_state_verified": False,
            "same_rng_weights_history_verified": False,
            "return_pose_current_rgb_verified": False,
            "no_online_privileged_state": True,
            "pre_registered_from_previously_observed_posthoc_cases": True,
            "for_independent_efficacy_evaluation": False,
        }
        for ablation_arm, description, injected in (
            ("KEEP", "No scan and no extra image; original policy continues", "NONE"),
            ("SCAN_NO_INJECTION", "Six identical scan actions; no supplementary image", "NONE"),
            ("PLACEBO", "Same scan; same image token structure; pre-gate current RGB", "DUPLICATE_PRE_GATE_RGB"),
            ("REAL_SEEK", "Same scan; outward-view RGB via same image token interface", "ACTUAL_OUTWARD_SCAN_RGB"),
        ):
            manifest.append({
                "schema": SCHEMA, "design": dict(common), "ablation_arm": ablation_arm,
                "scan_performed": ablation_arm != "KEEP",
                "extra_observation_content": injected,
                "protocol": description,
                "execution_status": "BLOCKED_UNTIL_MATCHING_CHECKPOINT_AND_EVALUATOR_HOOKS_AUDITED",
                "gpu_rollout_executed": False,
                "result": None,
            })
    report_out = {
        "schema": SCHEMA,
        "status": "CPU_POSTHOC_AUDITED_S2_BOUNDARIES_NOT_OBSERVABLE_ABLATION_BLOCKED",
        "source_schema": SOURCE_SCHEMA,
        "input_member_sha256": hashes,
        "source_raw_log_sha256_NOT_RECHECKED": report.get("original_trace_sha256", {}),
        "source_review_report_sha256_NOT_RECHECKED": report.get("review_report_sha256"),
        "N_scene_groups": len({a[0][0] for a in context_info.values()}),
        "N_contexts": len(context_info),
        "N_real_arm_traces": len(trace_groups),
        "N_continuation_action_rows": len(actions),
        "N_vs_KEEP_pairs": len(audited),
        "N_first_S2_text_comparisons_NOT_CALL_ALIGNED": len(comparisons),
        "N_SYSTEM2_call_aligned_pairs": 0,
        "N_focus_cases": len(dossiers),
        "N_planned_ablation_arms": len(manifest),
        "success_change_by_arm": {arm: {
            outcome: sum(a["action"] == arm and a["outcome_change_OFFLINE_ONLY"] == outcome for a in audited)
            for outcome in ("RESCUE", "HARM", "NO_SR_CHANGE")
        } for arm in ARMS[1:]},
        "limitations": [
            "No native System-2 call ID/timestamp for KEEP: cannot establish first matching S2 divergence",
            "main_step and relative continuation index are not System-2 decision boundaries",
            "Relative pose differences are indexed by action count, not synchronized simulation time",
            "Source raw trajectory hashes are attested by bundled report but raw source logs were not provided",
            "Historical intervention arms differ in physical motion/input history; they do not isolate new pixel evidence",
            "Case selection is posthoc; no held-out statistical or causal claim",
            "SCAN_NO_INJECTION controls physical scan but not extra-image-token format; PLACEBO controls the latter",
        ],
        "new_rollouts_executed": False,
        "model_training_executed": False,
        "online_features_exported": False,
        "original_outcomes_modified": False,
    }
    return audited, dossiers, manifest, report_out


def audit(input_path, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Create-only output directory already exists: {out_dir}")
    report, data, hashes = load_input(input_path)
    events, dossiers, manifest, audit_report = collect(report, data, hashes)
    files = {
        "decision_aligned_events.jsonl": jsonl(events),
        "case_mechanism_dossiers.json": json.dumps(dossiers, indent=2, ensure_ascii=False) + "\n",
        "paired_ablation_manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    }
    audit_report["generated_output_sha256"] = {
        name: sha256(content.encode("utf-8")) for name, content in files.items()
    }
    files["mechanism_audit_report.json"] = json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n"
    # All validations and serialization are completed before creation.
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, contents in files.items():
        (out_dir / name).write_text(contents, encoding="utf-8")
    return audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", nargs="?", default="audit", choices=["audit"])
    parser.add_argument("--input", required=True, help="Four-file ZIP or extracted directory")
    parser.add_argument("--run-dir", required=True, help="New, nonexistent output directory")
    args = parser.parse_args()
    summary = audit(args.input, args.run_dir)
    print("DECISION_AUDIT_V01=PASS")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("N_") or k == "status"}, indent=2))


if __name__ == "__main__":
    main()