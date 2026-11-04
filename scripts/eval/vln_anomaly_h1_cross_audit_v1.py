#!/usr/bin/env python3
"""Read-only CPU cross-audit of Native anomaly intervals and frozen H1 outcomes.

This does NOT establish abnormality at gate time, a causal mechanism, online
LOST detection or recovery effectiveness in unseen scenes. Candidate intervals
come from an existing retrospective audit; the two failure cases are selected
post-hoc for mechanism inspection. All original H1 contexts (including success
controls) remain in the denominator; no new rollouts or labels are produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "vln_anomaly_h1_cross_audit.v1"
DEFAULT_BASE = Path("/mnt/data8t/geenitiming/internnav_data")
COHORTS = {
    "development": "intervention_v01_paired_all10_01",
    "validation": "h1_validation_paired_all11_01",
}
ARMS = ("KEEP", "SEEK_LEFT45", "SEEK_RIGHT45", "HISTORY_RESET")
MODES = ("natural", "perturbed")
FAILURE_TARGETS = (("development", "VLzqgDo317F", "58"),
                   ("development", "i5noydFURQK", "70"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def sha256(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            dig.update(block)
    return dig.hexdigest()


def case_key(row):
    return tuple(str(x) for x in row["episode_key"])


def verify_table(protocol, outcomes, report, cohort):
    """Fail closed on missing, duplicated, swapped, or incomplete paired arms."""
    selected = protocol["selected_cases"]
    if not selected or len(selected) != int(protocol["max_cases"]):
        raise ValueError(f"{cohort}: selected case count differs from protocol")
    keys = [tuple(str(x) for x in c["key"]) for c in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{cohort}: duplicate selected episode")
    if len({x[0] for x in keys}) != len(keys):
        raise ValueError(f"{cohort}: selected scenes are not independent")
    expected_contexts = {
        (tuple(str(x) for x in case["key"]), mode, arm)
        for case in selected for mode in MODES for arm in ARMS
    }
    if len(outcomes) != len(expected_contexts):
        raise ValueError(f"{cohort}: incomplete 8-arm-per-episode outcome table")
    observed = {}
    for row in outcomes:
        key = case_key(row)
        mode = row["source_type"]
        arm = row["action"]
        token = (key, mode, arm)
        if token not in expected_contexts:
            raise ValueError(f"{cohort}: unexpected episode/mode/arm: {token}")
        if token in observed:
            raise ValueError(f"{cohort}: duplicate episode/mode/arm: {token}")
        linked = next(c for c in selected if tuple(str(x) for x in c["key"]) == key)
        if row["context_id"] != linked["case_id"] + "_" + mode:
            raise ValueError(f"{cohort}: context identity mismatch: {token}")
        if row["audit"] != ("REUSED_AUDITED_KEEP" if arm == "KEEP"
                           else "PAIRED_PREARM_VERIFIED"):
            raise ValueError(f"{cohort}: outcome provenance not audited: {token}")
        if int(row["success"]) not in (0, 1):
            raise ValueError(f"{cohort}: success not binary: {token}")
        if not 0 <= float(row["spl"]) <= 1 or not 0 <= float(row["os"]) <= 1:
            raise ValueError(f"{cohort}: SPL/OS outside [0,1]: {token}")
        if float(row["ne"]) < 0:
            raise ValueError(f"{cohort}: negative navigation error: {token}")
        if mode == "perturbed" and int(row["prefix_perturbation_actions"]) != 3:
            raise ValueError(f"{cohort}: perturbed prefix action budget changed")
        if mode == "natural" and int(row["prefix_perturbation_actions"]) != 0:
            raise ValueError(f"{cohort}: natural prefix contains perturbation")
        if arm == "KEEP" and int(row["extra_yaw_actions"]) != 0:
            raise ValueError(f"{cohort}: KEEP contains extra yaw")
        if arm.startswith("SEEK_") and int(row["extra_yaw_actions"]) != 6:
            raise ValueError(f"{cohort}: SEEK scan does not have 6 yaw actions")
        if arm == "HISTORY_RESET" and int(row["extra_yaw_actions"]) != 0:
            raise ValueError(f"{cohort}: RESET contains extra yaw")
        observed[token] = row
    if set(observed) != expected_contexts:
        raise ValueError(f"{cohort}: incomplete outcome key set")
    if report.get("N_four_arm_outcomes") != len(outcomes):
        raise ValueError(f"{cohort}: paired report denominator mismatch")
    if int(report.get("N_contexts", -1)) != len(selected) * len(MODES):
        raise ValueError(f"{cohort}: paired report context count mismatch")
    return selected, observed


def load_evidence(audit):
    manifest = load_json(audit / "audit_manifest.json")
    if manifest.get("schema") != "vln_native_anomaly_audit.v1":
        raise ValueError("Unexpected source anomaly audit schema")
    with (audit / "episode_summary.csv").open(newline="", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    events = list(load_jsonl(audit / "event_candidates.jsonl"))
    by_episode = {}
    for row in summary:
        key = (row["cohort"], row["scene"], str(row["episode"]))
        if key in by_episode:
            raise ValueError("Duplicate anomaly summary episode: " + repr(key))
        by_episode[key] = row
    if (len(summary) != int(manifest["N_episodes"])
            or len(events) != int(manifest["N_event_candidates"])):
        raise ValueError("Source anomaly audit event/episode count mismatch")
    by_event = defaultdict(list)
    for row in events:
        key = (row["cohort"], row["scene"], str(row["episode"]))
        if key not in by_episode:
            raise ValueError("Anomaly event references unknown episode")
        start, end = int(row["start_action"]), int(row["end_action"])
        if not 0 <= start <= end <= int(by_episode[key]["native_action_count"]):
            raise ValueError("Anomaly event interval outside native actions")
        by_event[key].append(row)
    return manifest, by_episode, by_event


def summarize_pair(cohort, episode, mode, by_arm, summary, events, gate):
    keep = by_arm["KEEP"]
    prefix = int(gate["prefix_action_count"])
    distinct_intervals = sorted({
        (int(x["start_action"]), int(x["end_action"])) for x in events
    })
    labels_at_gate = sorted({
        x["type"] for x in events
        if int(x["start_action"]) <= prefix <= int(x["end_action"])
    })
    rows = []
    for arm in ARMS:
        treated = by_arm[arm]
        a, b = int(keep["success"]), int(treated["success"])
        state = ("RESCUE" if a == 0 and b == 1 else
                 "HARM" if a == 1 and b == 0 else "NO_SR_CHANGE")
        rows.append({
            "cohort": cohort,
            "scene": episode[0], "episode": episode[1],
            "mode": mode, "arm": arm,
            "native_success": int(summary["native_success"]),
            "gate_main_step": int(gate["step"]),
            "gate_native_action_prefix": prefix,
            "gate_precedes_nearest_detectable_visual_loop": "NOT_ESTABLISHED_BY_RETROSPECTIVE_INTERVAL",
            "n_candidate_labels": len(events),
            "n_distinct_candidate_intervals": len(distinct_intervals),
            "retrospective_candidate_labels_overlapping_gate": ";".join(labels_at_gate),
            "keep_sr": a, "treated_sr": b,
            "sr_change": state,
            "keep_ne_m": float(keep["ne"]),
            "treated_ne_m": float(treated["ne"]),
            "delta_ne_m": round(float(treated["ne"]) - float(keep["ne"]), 6),
            "keep_spl": float(keep["spl"]),
            "treated_spl": float(treated["spl"]),
            "delta_spl": round(float(treated["spl"]) - float(keep["spl"]), 6),
            "extra_yaw_actions": int(treated["extra_yaw_actions"]),
            "inference_guard": "POSTHOC_CONTEXT_OUTCOME_ONLY_NOT_LOOP_TRIGGERED_CAUSAL_MECHANISM",
        })
    return rows


def build(base, audit):
    anomaly_manifest, native_summary, native_events = load_evidence(audit)
    all_rows, report_stats, selected_keys = [], {}, set()
    for cohort, paired_folder in COHORTS.items():
        root = (base / paired_folder).resolve()
        path_protocol = root / "paired_protocol.json"
        path_outcome = root / "intervention_outcomes.jsonl"
        path_report = root / "paired_report.json"
        protocol, report = load_json(path_protocol), load_json(path_report)
        outcomes = list(load_jsonl(path_outcome))
        selected, indexed = verify_table(protocol, outcomes, report, cohort)
        source_audit = anomaly_manifest["sources"][cohort]
        if Path(source_audit["paired"]).resolve() != root:
            raise ValueError(f"{cohort}: source paired root differs from frozen anomaly audit")
        natural_root = Path(protocol["source_natural_gate_run"]).resolve()
        native_root = Path(protocol["source_native_run"]).resolve()
        if (natural_root != Path(source_audit["gate"]).resolve()
                or native_root != Path(source_audit["native"]).resolve()):
            raise ValueError(f"{cohort}: natural/native input roots differ from anomaly audit")
        gate_report = load_json(natural_root / "natural_checkpoint_report.json")
        if len(gate_report["cases"]) != 12:
            raise ValueError(f"{cohort}: original preregistration denominator changed")
        checked_gates = {case_key(c): c for c in gate_report["cases"]}
        if len(checked_gates) != 12:
            raise ValueError(f"{cohort}: gate report has duplicate/missing cases")
        if len(selected) != (10 if cohort == "development" else 11):
            raise ValueError(f"{cohort}: expected frozen all-eligible H1 cohort size")
        for c in selected:
            key = tuple(str(x) for x in c["key"])
            episode_id = (cohort, key[0], key[1])
            if episode_id not in native_summary:
                raise ValueError("Selected H1 episode is absent from Native anomaly audit")
            if key not in checked_gates:
                raise ValueError("Selected episode absent from gate report")
            natural = checked_gates[key]["natural_checkpoint"]
            if natural is None:
                raise ValueError("Selected H1 episode is ineligible in gate report")
            gate = c["checkpoint"]
            if (int(gate["step"]) != int(natural["step"])
                    or int(gate["prefix_action_count"]) != int(natural["prefix_action_count"])):
                raise ValueError("H1 checkpoint differs from frozen natural gate")
            if int(native_summary[episode_id]["original_gate_action_prefix"]) != int(gate["prefix_action_count"]):
                raise ValueError("Native anomaly audit differs from H1 checkpoint")
            selected_keys.add(episode_id)
            for mode in MODES:
                arm_rows = {arm: indexed[(key, mode, arm)] for arm in ARMS}
                all_rows.extend(summarize_pair(
                    cohort, key, mode, arm_rows, native_summary[episode_id],
                    native_events.get(episode_id, []), gate))
        report_stats[cohort] = {
            "N_selected_episodes": len(selected),
            "N_contexts": len(selected) * len(MODES),
            "N_outcomes": len(outcomes),
            "paired_protocol": str(path_protocol),
            "paired_protocol_sha256": sha256(path_protocol),
            "outcome_path": str(path_outcome),
            "outcome_sha256": sha256(path_outcome),
            "paired_report_sha256": sha256(path_report),
        }
    missing = set(FAILURE_TARGETS) - selected_keys
    if missing:
        raise ValueError("A targeted Native failure has no audited H1 paired outcome: " + repr(sorted(missing)))
    if len(selected_keys) != 21 or len(all_rows) != 168:
        raise ValueError("Frozen cohort denominator should be 21 episodes / 168 arm outcomes")
    return all_rows, report_stats, native_summary, native_events


def summarize(all_rows, source_summary, source_events):
    target_rows = [row for row in all_rows
                   if (row["cohort"], row["scene"], row["episode"]) in FAILURE_TARGETS]
    if len(target_rows) != 16:
        raise ValueError("Expected 16 arm outcomes for 2 targeted failures")
    counter = defaultdict(Counter)
    for row in all_rows:
        if row["arm"] != "KEEP":
            counter[(row["cohort"], row["mode"], row["arm"])][row["sr_change"]] += 1
    statistics = []
    for (cohort, mode, arm), counts in sorted(counter.items()):
        statistics.append({
            "cohort": cohort, "mode": mode, "arm": arm,
            "rescue": counts["RESCUE"], "harm": counts["HARM"],
            "no_sr_change": counts["NO_SR_CHANGE"],
            "N_contexts": counts["RESCUE"] + counts["HARM"] + counts["NO_SR_CHANGE"],
        })
    cases = []
    for key in FAILURE_TARGETS:
        rows = [r for r in target_rows if
                (r["cohort"], r["scene"], r["episode"]) == key]
        if len(rows) != 8:
            raise ValueError("Missing failure case's full 8-arm outcome set")
        events = source_events.get(key, [])
        gate = int(rows[0]["gate_native_action_prefix"])
        cases.append({
            "cohort": key[0], "episode_key": [key[1], key[2]],
            "original_native_success": int(source_summary[key]["native_success"]),
            "original_native_termination": source_summary[key]["native_termination"],
            "gate_action_prefix": gate,
            "gate_main_step": rows[0]["gate_main_step"],
            "retrospective_events": [
                {"type": e["type"], "start_action": int(e["start_action"]),
                 "end_action": int(e["end_action"]),
                 "gate_inside_retrospective_interval":
                     int(e["start_action"]) <= gate <= int(e["end_action"])}
                for e in events],
            "arms_by_context": {
                mode: [{
                    "arm": r["arm"], "sr": r["treated_sr"], "ne_m": r["treated_ne_m"],
                    "spl": r["treated_spl"], "vs_KEEP": r["sr_change"],
                    "delta_ne_m": r["delta_ne_m"],
                    "delta_spl": r["delta_spl"],
                } for r in rows if r["mode"] == mode]
                for mode in MODES},
            "causal_loop_repair_proven": False,
            "failure_selection_is_posthoc": True,
        })
    return target_rows, cases, statistics


def write_report(output, rows, targets, cases, stats, cohort_provenance, audit):
    if output.exists():
        raise FileExistsError("Create-only cross audit output exists: " + str(output))
    output.mkdir(parents=True, exist_ok=False)
    def csv_out(name, records):
        with (output / name).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    csv_out("full_h1_context_matrix.csv", rows)
    csv_out("failure_episodes_16_arms.csv", targets)
    csv_out("full_cohort_rescue_harm.csv", stats)
    (output / "failure_case_dossiers.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "status": "CPU_READ_ONLY_H1_NATIVE_ANOMALY_CROSS_AUDIT_PASS",
        "N_h1_eligible_episodes": 21,
        "N_h1_contexts": 42,
        "N_h1_outcomes": 168,
        "N_target_failure_episodes": 2,
        "N_target_failure_outcomes": len(targets),
        "source_anomaly_dir": str(audit),
        "source_anomaly_sha256": {
            file: sha256(audit / file) for file in
            ("audit_manifest.json", "episode_summary.csv", "event_candidates.jsonl")},
        "paired_cohort_provenance": cohort_provenance,
        "no_training": True, "no_new_rollout": True,
        "no_original_data_modified": True,
        "claims_not_established": [
            "Gate was selected by observed decision loop",
            "Retrospective anomaly was detectable at gate time",
            "Recovery arm repaired the detected loop",
            "S2 causal mechanism isolated",
            "Held-out treatment efficacy",
        ],
    }
    (output / "cross_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("NATIVE_ANOMALY_H1_CROSS_AUDIT=PASS")
    print("H1_EPISODES: 21 H1_CONTEXTS: 42 H1_OUTCOMES: 168")
    print("FAILURE_EPISODES: 2 FAILURE_ARM_ROWS:", len(targets))
    for case in cases:
        print("CASE:", "/".join(case["episode_key"]), "GATE_ACTION:", case["gate_action_prefix"])
        for mode in MODES:
            print(" ", mode, " ".join(
                "{}={}/NE{:.3f}".format(r["arm"], r["sr"], r["ne_m"])
                for r in case["arms_by_context"][mode]))
    print("OUTPUT_DIR:", output)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--anomaly-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = args.base.resolve()
    anomaly = (args.anomaly_dir or (base / "native_anomaly_audit_v1_02")).resolve()
    rows, provenance, native, events = build(base, anomaly)
    targets, cases, stats = summarize(rows, native, events)
    write_report(args.output.resolve(), rows, targets, cases, stats, provenance, anomaly)


if __name__ == "__main__":
    main()
