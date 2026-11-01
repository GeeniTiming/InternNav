"""Create immutable, provenance-checked qualitative VLN evidence review packets.

READ ONLY: original audited TRAIN paired run, original offline labels,
true predecision RGB/history images, exact native and intervention traces,
supplementary SEEK RGB, and first post-treatment System-2 outputs.

This is an OUTCOME-SELECTED EXPLANATORY REVIEW. It cannot be used to train
or evaluate an online selector as if the cases were sampled blindly.
The program DOES NOT infer task progress, landmark validity, LOST, a reason
for STOP, causal effects of injected images, or any new reward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_selection_outcome_labels_v01 as labels

paired = labels.paired
SCHEMA = "intervention_selection_mechanism_review.v0.1"
TARGETS = (
    ("i5noydFURQK", "70", "POSTHOC_RESCUE_REVIEW"),
    ("r1Q1Z4BcV1o", "106", "POSTHOC_RESCUE_AND_HARM_REVIEW"),
    ("rPc6DW4iMge", "73", "POSTHOC_RESET_HARM_REVIEW"),
    ("PX4nDJXEHrG", "79", "POSTHOC_RESET_AND_SCAN_HARM_REVIEW"),
)
ACTIONS = labels.ACTIONS
OUTPUT_FILES = (
    "review_contexts.jsonl",
    "review_arm_evidence.jsonl",
    "human_annotation_templates.jsonl",
)
SCOPE = (
    "Posthoc qualitative evidence review of four TRAIN development scenes, "
    "chosen AFTER their outcome labels were known. Eight natural/perturbed "
    "contexts, 32 real candidate outcomes; no new rollouts or intervention "
    "labels. No automatic mechanism or causal explanation; no use of this "
    "outcome-selected review as independent train/test evidence."
)


def text_jsonl(records):
    return labels.jsonl_bytes(records)


def expected_targets(selected_cases):
    cases = {tuple(c["key"]): c for c in selected_cases}
    requested = {(s, e) for s, e, _ in TARGETS}
    if len(cases) != len(selected_cases) or not requested.issubset(cases):
        raise ValueError("Selected posthoc cases missing from frozen 10-scene cohort")
    if len({s for s, _, _ in TARGETS}) != len(TARGETS):
        raise ValueError("Duplicate review scene")
    return [(cases[(s, e)], tag) for s, e, tag in TARGETS]


def predecision_path(natural_root, pert_root, key, mode):
    name = "_".join(key)
    if mode == "natural":
        return natural_root / "control/checkpoints" / name
    if mode == "perturbed":
        return pert_root / "perturbed_keep/checkpoints" / name
    raise ValueError("Unregistered context mode")


def archived_image(folder, raw, sha, name):
    path = Path(raw)
    if not path.is_absolute():
        path = folder / path
    path = path.resolve(strict=True)
    if path.suffix.lower() != ".png":
        raise ValueError("Review image is not original PNG: " + name)
    observed_hash = labels.sha256(path)
    if observed_hash != sha:
        raise ValueError("Image provenance differs: " + name)
    return {"path": str(path), "sha256": observed_hash}


def validate_predecision(row, folder, expected_key, expected_mode):
    if (row["episode_key"] != list(expected_key)
            or row["source_type"] != expected_mode
            or row["schema"] != "intervention_selection_predecision.v0.1"
            or row["no_privileged_simulator_metrics"] is not True):
        raise ValueError("Predecision record identity/privilege changed")
    if not isinstance(row["instruction"], str) or not row["instruction"].strip():
        raise ValueError("Missing native instruction")
    current = archived_image(
        folder, row["current_rgb"], row["current_png_sha256"],
        "current RGB")
    history = []
    for i, f in enumerate(row["historical_rgb"]):
        history.append({
            **archived_image(folder, f["path"], f["png_sha256"],
                             "historical RGB " + str(i)),
            "native_history_index": f["native_history_index"],
            "pixel_sha256": f["pixel_sha256"],
        })
    codes = row["past_action_codes"]
    yaw = row["perturbation_actions_before_context"]
    if (not isinstance(codes, list)
            or not all(type(x) is int and 0 <= x <= 5 for x in codes)
            or not isinstance(yaw, list)
            or (expected_mode == "natural" and yaw)
            or (expected_mode == "perturbed" and
                (len(yaw) != 3 or codes[-3:] != yaw))):
        raise ValueError("Wrong observed physical action history")
    return {
        "schema": SCHEMA, "context_id": row["context_id"],
        "episode_key": row["episode_key"], "source_type": expected_mode,
        "main_step": row["main_step"],
        "instruction": row["instruction"],
        "past_action_codes": codes,
        "perturbation_actions_before_context": yaw,
        "current_rgb": current,
        "current_pixel_sha256": row["current_pixel_sha256"],
        "historical_rgb": history,
        "historical_rgb_history_sha256":
            row["historical_rgb_history_sha256"],
        "mechanism_annotation": None,
        "no_offline_outcome_fields_in_this_record": True,
        "online_legal_evidence_only": True,
    }


def summarize_stop_actions(rows, prefix_count):
    """Report *recorded* Habitat STOP codes, never inferred STOP rationale.

    A native STOP action has action_code 0. It need not mean the model
    deliberately chose task completion (the loop also uses action=0
    as a default); don't turn this into a confirmed semantic diagnosis.
    """
    if prefix_count < 0 or not isinstance(rows, list):
        raise ValueError("Invalid prefix trace")
    result = []
    for i, row in enumerate(rows):
        # Native KEEP control/actions.jsonl is untagged. Its post-gate
        # actions are identifiable by the audited frozen prefix length.
        # Intervened traces are phase-tagged; never count alignment yaws.
        continuation = (
            row.get("phase") == "continuation"
            or (row.get("phase") is None and i >= prefix_count)
        )
        if i >= prefix_count and continuation and row["action"] == 0:
            result.append({
                "episode_action_index": i,
                "main_step": row.get("main_step"),
                "env_done": row.get("done"),
            })
    return {
        "native_stop_action_code": 0,
        "continuation_stop_action_count": len(result),
        "first_continuation_stop_action": result[0] if result else None,
        "last_continuation_stop_action": result[-1] if result else None,
        "stop_intention_inferred": False,
        "stop_success_causality_inferred": False,
    }


def first_system2(record, key, action, mode):
    if action == "KEEP":
        return {
            "status": "NOT_CAPTURED_IN_PERTURBED_KEEP"
                      if mode == "perturbed"
                      else "NATIVE_REFERENCE_ONLY_SEPARATELY_RECORDED",
            "text": None,
            "not_assumed_identical_to_other_mode": True,
        }
    raw = record.get("first_system2_response")
    if (not isinstance(raw, dict) or raw.get("key") != list(key)
            or not isinstance(raw.get("step"), int)
            or not isinstance(raw.get("text"), str)):
        raise ValueError("Missing/invalid first actual System-2 response")
    return {
        "status": "ACTUAL_RECORDED_POST_INTERVENTION",
        "main_step": raw["step"], "text": raw["text"],
        "not_assumed_identical_to_other_mode": True,
    }


def native_source_first_decision(all_calls, key, step, prefix_count):
    match = [
        x for x in all_calls
        if x.get("episode_key") == list(key)
        and x.get("main_step") == step
        and x.get("call_type") == "ordinary"
        and x.get("native_action_prefix_count") == prefix_count
    ]
    if len(match) > 1:
        raise ValueError("Duplicated original native System-2 call at checkpoint")
    if not match:
        return {
            "status": "NATIVE_REFERENCE_DECISION_NOT_PRESENT",
            "text": None,
            "use_as_perturbed_keep_output": False,
        }
    x = match[0]
    if not isinstance(x.get("model_output"), str):
        raise ValueError("Native source decision text invalid")
    return {
        "status": "ORIGINAL_NATIVE_SOURCE_DECISION_AT_NATURAL_GATE",
        "text": x["model_output"],
        "parsed_actions": x.get("native_parsed_actions"),
        "call_id": x["call_id"],
        "use_as_perturbed_keep_output": False,
        "not_a_new_intervention_output": True,
    }


def extract_arm(trace_root, arm, episode_key, gate, details):
    mode, action = paired.split_arm(arm)
    k = tuple(episode_key)
    record = details[k]
    folder = trace_root / arm / "checkpoints" / "_".join(k)
    if record.get("key") != list(k) or record.get("operation") != action:
        raise ValueError("Wrong frozen arm/episode result provenance")
    if not (folder / "prearm.json").is_file():
        raise FileNotFoundError("Missing previously audited prearm record")
    prearm = labels.read_json(folder / "prearm.json")
    if prearm != record["prearm"]:
        raise ValueError("Frozen prearm fingerprints differ from runtime summary")
    first = first_system2(record, k, action, mode)
    evidence = {
        "first_system2": first,
        "prearm_rgb_pixel_sha256": prearm["current_rgb_pixel_sha256"],
        "historical_rgb_sha256": prearm["history_sha256"],
        "prearm_rng_sha256": prearm["rng_sha256"],
        "prearm_pose_OFFLINE_ONLY": prearm["state"],
        "scan_outward_rgb": None,
        "scan_injection": None,
        "history_filter_events": None,
        "mechanism_not_inferred": True,
    }
    if action.startswith("SEEK_"):
        ev = labels.read_json(folder / "evidence_input.json")
        if (len(ev["events"]) != 1 or
                ev["events"][0]["source"] != "outward_view"):
            raise ValueError("Real visual evidence injection not recorded")
        view = folder / "outward_view_model_rgb.png"
        if not view.is_file():
            raise FileNotFoundError(view)
        evidence["scan_outward_rgb"] = {
            "path": str(view.resolve()),
            "sha256": labels.sha256(view),
            "pixel_sha256": ev["scan_view_sha256"],
        }
        evidence["scan_injection"] = ev["events"][0]
    else:
        hist = labels.read_json(folder / "history_inputs.json")
        if (not hist or hist[0]["main_step"] != gate["step"]
                or hist[0]["selected_history_ids"]):
            raise ValueError("History RESET did not exclude old sampled frames")
        evidence["history_filter_events"] = hist
    return evidence


def build_packet(label_dir):
    source_label = labels.audit_package(label_dir)
    src = Path(source_label["source_run"]).resolve()
    root, _, protocol, expected, outcomes = labels.load_audited_source(src)
    targets = expected_targets(protocol["selected_cases"])
    natural = Path(protocol["source_natural_gate_run"]).resolve()
    perturb = Path(protocol["source_perturbed_keep_run"]).resolve()
    native = Path(protocol["source_native_run"]).resolve()
    online = {
        x["context_id"]: x for x in labels.read_jsonl(
            root / "contexts_predecision.jsonl")
    }
    summaries = {
        x["context_id"]: x for x in labels.read_jsonl(
            Path(label_dir) / "context_label_summary.jsonl")
    }
    if len(summaries) != 20:
        raise ValueError("Label summaries changed since audit")
    all_calls = labels.read_jsonl(native / "capture/online_decisions.jsonl")
    action_traces = {}
    arm_details = {}
    for mode in paired.MODES:
        for action in paired.NEW_ACTIONS:
            arm = mode + "_" + action.lower()
            grouped = defaultdict(list)
            for r in paired.physical.read_lines(
                    root / arm / "turn_actions.jsonl"):
                grouped[paired.capture.episode_key(r)].append(r)
            detail = {
                tuple(x["key"]): x for x in
                labels.read_json(root / arm / "turn_summary.json")["cases"]
            }
            action_traces[arm] = grouped
            arm_details[arm] = detail
    keep_traces = {
        "natural": defaultdict(list), "perturbed": defaultdict(list)
    }
    for r in paired.physical.read_lines(
            natural / "control/actions.jsonl"):
        keep_traces["natural"][paired.capture.episode_key(r)].append(r)
    for r in paired.physical.read_lines(
            perturb / "perturbed_keep/turn_actions.jsonl"):
        keep_traces["perturbed"][paired.capture.episode_key(r)].append(r)

    context_records, arm_records, annotations, photos = [], [], [], []
    for planned, purpose in targets:
        key = tuple(planned["key"])
        gate = planned["checkpoint"]
        for mode in paired.MODES:
            cid = planned["case_id"] + "_" + mode
            if cid not in summaries or cid not in online or cid not in outcomes:
                raise ValueError("Missing frozen label / online context")
            folder = predecision_path(natural, perturb, key, mode)
            ctx = validate_predecision(online[cid], folder, key, mode)
            if ctx["context_id"] != cid or ctx["main_step"] != gate["step"]:
                raise ValueError("Context does not match original natural checkpoint")
            ctx["purpose_POSTHOC_ONLY"] = purpose
            ctx["selected_after_outcomes_observed"] = True
            context_records.append(ctx)
            photos.append(ctx["current_rgb"])
            photos.extend(ctx["historical_rgb"])
            context_summary = summaries[cid]
            for action in ACTIONS:
                actual = outcomes[cid][action]
                if action == "KEEP":
                    trace = keep_traces[mode][key]
                    details = {
                        "first_system2": first_system2(None, key, "KEEP", mode),
                        "prearm_pose_OFFLINE_ONLY": None,
                        "scan_outward_rgb": None, "scan_injection": None,
                        "history_filter_events": None,
                        "mechanism_not_inferred": True,
                    }
                    if mode == "natural":
                        details["native_source_first_decision"] = (
                            native_source_first_decision(
                                all_calls, key, gate["step"],
                                gate["prefix_action_count"]))
                else:
                    arm = mode + "_" + action.lower()
                    trace = action_traces[arm][key]
                    details = extract_arm(
                        root, arm, key, gate, arm_details[arm])
                    if details["scan_outward_rgb"]:
                        photos.append(details["scan_outward_rgb"])
                if not trace:
                    raise ValueError("Missing real recorded action trace")
                cut = gate["prefix_action_count"]
                actual_yaw = (
                    actual["prefix_perturbation_actions"]
                    if mode == "perturbed" else 0)
                continuity = {
                    "context_id": cid, "action": action,
                    "episode_key": list(key), "source_type": mode,
                    "gate_main_step": gate["step"],
                    "prefix_native_action_count": cut,
                    "actions_since_native_gate": len(trace) - cut,
                    "perturbation_actions": actual_yaw,
                    "extra_yaw_actions": actual["extra_yaw_actions"],
                    "last_recorded_action_code": trace[-1]["action"],
                    "last_recorded_env_done": trace[-1]["done"],
                    "recorded_stops": summarize_stop_actions(trace, cut),
                    "trace_source_path": str((
                        root / (mode + "_" + action.lower()) /
                        "turn_actions.jsonl" if action != "KEEP"
                        else natural / "control/actions.jsonl"
                        if mode == "natural"
                        else perturb / "perturbed_keep/turn_actions.jsonl"
                    ).resolve()),
                }
                arm_records.append({
                    "schema": SCHEMA, "context_id": cid,
                    "episode_key": list(key), "source_type": mode,
                    "action": action,
                    "observed_success_OFFLINE_ONLY": actual["success"],
                    "observed_ne_OFFLINE_ONLY": actual["ne"],
                    "observed_spl_OFFLINE_ONLY": actual["spl"],
                    "rescue_or_harm_not_inferred_from_ne": True,
                    "selected_after_outcomes_observed": True,
                    "trace": continuity,
                    "evidence": details,
                })
            annotations.append({
                "schema": SCHEMA, "context_id": cid,
                "episode_key": list(key),
                "purpose_POSTHOC_ONLY": purpose,
                "observed_context_type_OFFLINE_ONLY":
                    context_summary["context_type"],
                "manual_observations": {
                    "instruction_segments": None,
                    "visible_current_landmarks": None,
                    "relevant_historical_landmarks": None,
                    "progress_before_action": "UNDETERMINED",
                    "history_current_consistency": "UNDETERMINED",
                    "first_s2_response_comparison": None,
                    "stop_action_interpretation": "UNDETERMINED",
                    "candidate_explanation": None,
                    "evidence_for_explanation": [],
                    "counterevidence": [],
                },
                "mechanism_status": "NOT_IDENTIFIED",
                "placebo_scan_without_new_image_observed": False,
                "scan_no_injection_control_observed": False,
                "training_feature_or_ground_truth_label": False,
            })
    if (len(context_records), len(arm_records), len(annotations)) != (8, 32, 8):
        raise ValueError("Review lacks eight contexts and 32 actual arms")
    included = {}
    for p in photos:
        if not Path(p["path"]).is_file():
            raise ValueError("Source RGB evidence disappeared")
        included[p["path"]] = labels.sha256(p["path"])
    for path in sorted(included):
        if included[path] != next(p["sha256"] for p in photos if p["path"] == path):
            raise ValueError("Provenance of saved evidence image changed")
    srcs = [
        root / n for n in labels.SOURCE_FILES
    ] + [
        Path(label_dir) / n for n in (
            "label_report.json",) + labels.OUTPUT_FILES
    ] + [
        Path(__file__).resolve(),
    ]
    snapshot = {str(p.resolve()): labels.sha256(p) for p in srcs}
    snapshot.update(included)
    output = {
        "schema": SCHEMA,
        "status": "POSTHOC_MECHANISM_EVIDENCE_REVIEW_READY_NOT_ANNOTATED",
        "source_full_run": str(root), "source_label_run": str(Path(label_dir).resolve()),
        "N_original_scene_groups": 10,
        "N_original_registered_cases": 12,
        "N_review_scenes": 4, "N_review_contexts": 8,
        "N_review_action_outcomes": 32,
        "selection": [list(x) for x in TARGETS],
        "selection_after_observing_results": True,
        "no_causal_mechanism_inferred": True,
        "no_online_selector_training_examples_created": True,
        "source_hashes": snapshot,
        "scope": SCOPE,
    }
    return output, dict(zip(OUTPUT_FILES, map(
        text_jsonl, (context_records, arm_records, annotations))))


def build(label_dir, out_dir):
    dest = Path(out_dir).resolve()
    if dest.exists():
        raise FileExistsError("Review output is create-only")
    report, files = build_packet(label_dir)
    dest.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (dest / name).write_bytes(content)
    report["output_hashes"] = {
        name: labels.sha256(dest / name) for name in OUTPUT_FILES
    }
    (dest / "review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "N_review_scenes": report["N_review_scenes"],
        "N_review_contexts": report["N_review_contexts"],
        "N_review_action_outcomes": report["N_review_action_outcomes"],
        "no_causal_mechanism_inferred": True,
    }, ensure_ascii=False, indent=2))


def audit(review_dir):
    dest = Path(review_dir).resolve()
    saved = labels.read_json(dest / "review_report.json")
    if saved.get("schema") != SCHEMA:
        raise ValueError("Wrong evidence review schema")
    report, contents = build_packet(saved["source_label_run"])
    hashes = {n: hashlib.sha256(contents[n]).hexdigest() for n in OUTPUT_FILES}
    if (saved.get("output_hashes") != hashes or
            {k:v for k,v in saved.items() if k != "output_hashes"} != report):
        raise ValueError("Source/evidence review provenance differs")
    for name in OUTPUT_FILES:
        if labels.sha256(dest / name) != hashes[name]:
            raise ValueError("Saved review evidence output changed: " + name)
    print("POSTHOC_EVIDENCE_REVIEW_AUDITED",
          report["N_review_scenes"], "scenes",
          report["N_review_contexts"], "contexts",
          report["N_review_action_outcomes"], "real arms")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    ss = p.add_subparsers(dest="command", required=True)
    s = ss.add_parser("build")
    s.add_argument("--label-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=lambda args: build(args.label_dir, args.out_dir))
    s = ss.add_parser("audit")
    s.add_argument("--review-dir", required=True)
    s.set_defaults(fn=lambda args: audit(args.review_dir))
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
