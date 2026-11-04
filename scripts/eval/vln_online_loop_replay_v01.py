#!/usr/bin/env python3
"""CPU-only prefix replay of passive VLN online loop monitor on 24 Native episodes.

Uses the frozen action/S2 capture for order/alignment only; the monitor gets
ordinary RGB pixel digests and S2 outputs, never success, pose, future actions,
goal distance or any outcome label. Original data is read-only. A separate
OFFLINE_ONLY episode table adds success labels *after* monitor processing.

This is exploratory TRAIN replay on cases used to motivate the thresholds.
It does not establish held-out accuracy, motion failure, or recovery efficacy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from vln_online_loop_monitor_v01 import (
    LoopConfig, OnlineDecisionVisualLoopMonitor, SCHEMA as MONITOR_SCHEMA,
    project_online_s2_call,
)
from vln_native_s2_visual_evidence_export_v1 import (
    group_action_rows, localize_call_records,
)

SCHEMA = "vln_online_loop_replay.v0.1"
DEFAULT_BASE = Path("/mnt/data8t/geenitiming/internnav_data")
COHORTS = ("development", "validation")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if raw.strip():
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON") from exc


def hash_file(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def load_offline_labels(anomaly_dir):
    manifest_path = anomaly_dir / "audit_manifest.json"
    summary_path = anomaly_dir / "episode_summary.csv"
    manifest = load_json(manifest_path)
    if (manifest.get("schema") != "vln_native_anomaly_audit.v1"
            or int(manifest["N_episodes"]) != 24):
        raise ValueError("Require complete 24-episode frozen Native anomaly audit")
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        summaries = list(csv.DictReader(f))
    result = {}
    for row in summaries:
        key = (row["cohort"], row["scene"], str(row["episode"]))
        if key in result or row["cohort"] not in COHORTS:
            raise ValueError("Duplicate or unexpected episode in offline summary")
        if row["native_success"] not in ("0", "1"):
            raise ValueError("Native outcome must be binary in offline-only summary")
        result[key] = row
    if len(result) != 24:
        raise ValueError("Native anomaly summary must cover exactly 24 episodes")
    return manifest, result, manifest_path, summary_path


def load_native_stream(cohort, source_meta):
    native = Path(source_meta["native"]).resolve()
    actions_path = native / "control" / "actions.jsonl"
    decisions_path = native / "capture" / "online_decisions.jsonl"
    action_rows = list(load_jsonl(actions_path))
    grouped, offsets, total_actions, episode_order = group_action_rows(action_rows)
    if len(episode_order) != 12:
        raise ValueError(f"{cohort}: expected exactly 12 native episodes")
    if int(source_meta["native_action_trace_rows"]) != total_actions:
        raise ValueError(f"{cohort}: total native action rows changed")
    audit_order = [tuple(str(x) for x in key)
                   for key in source_meta["native_action_episode_order"]]
    if audit_order != episode_order:
        raise ValueError(f"{cohort}: native action episode order changed")
    raw_s2 = list(load_jsonl(decisions_path))
    local = localize_call_records(
        raw_s2, offsets, {key: len(rows) for key, rows in grouped.items()},
        total_actions)
    if set(local) != set(episode_order):
        raise ValueError(f"{cohort}: S2 episode coverage differs from native action logs")
    # Order is the actual recorder call_id, not main_step or event-window order.
    ordered = sorted((row for calls in local.values() for row in calls),
                     key=lambda row: int(row["call_id"]))
    if len(ordered) != len(raw_s2):
        raise ValueError("S2 call rows lost during alignment")
    if [int(r["call_id"]) for r in ordered] != [int(r["call_id"]) for r in raw_s2]:
        raise ValueError("S2 source call order does not match action-aligned order")
    return {
        "native": native,
        "actions_path": actions_path,
        "decisions_path": decisions_path,
        "action_count": total_actions,
        "offsets": offsets,
        "lengths": {key: len(rows) for key, rows in grouped.items()},
        "episode_order": episode_order,
        "ordered": ordered,
    }


def replay_stream(cohort, source, offline_summary, config=None):
    monitor = OnlineDecisionVisualLoopMonitor(config)
    chronology, alarms = [], []
    episode_counters = defaultdict(Counter)
    first_by_episode = {}
    for raw in source["ordered"]:
        online = project_online_s2_call(raw)
        key = online["episode_key"]
        episode_key = (cohort, *key)
        if episode_key not in offline_summary:
            raise ValueError("Native S2 episode is missing from retrospective label set")
        local_prefix = int(raw["action_prefix_count_episode"])
        if local_prefix < 0 or local_prefix > int(source["lengths"][key]):
            raise ValueError("Episode-local action index outside Native action block")

        # All online observation fields are explicitly projected. Offline success
        # is accessed only after observe() and never passed into the monitor.
        result = monitor.observe(raw)
        episode_counters[episode_key]["all_s2_calls"] += 1
        if online["call_type"] == "ordinary":
            episode_counters[episode_key]["ordinary_s2_calls"] += 1
        else:
            episode_counters[episode_key]["lookdown_followup_calls"] += 1
        chronology.append({
            "cohort": cohort,
            "episode_key": list(key),
            "call_id": online["call_id"],
            "call_type": online["call_type"],
            "main_step": online["main_step"],
            "action_prefix_global": online["native_action_prefix_count"],
            # Offline alignment only; not a monitor feature.
            "action_prefix_episode_OFFLINE_ALIGNMENT": local_prefix,
            "model_output": online["model_output"],
            "current_rgb_pixel_sha256": online["current_rgb_pixel_sha256"],
            "same_output_streak": result["same_output_streak"],
            "candidate_types_emitted": [x["kind"] for x in result["events"]],
        })
        for alarm in result["events"]:
            if alarm["trigger_global_action_prefix"] != online["native_action_prefix_count"]:
                raise ValueError("Alarm was not emitted at the current action prefix")
            event = dict(alarm)
            event["cohort"] = cohort
            event["trigger_action_prefix_episode_OFFLINE_ALIGNMENT"] = local_prefix
            alarms.append(event)
            episode_counters[episode_key][alarm["kind"]] += 1
            first_by_episode.setdefault(episode_key, event)
    if len(episode_counters) != 12:
        raise ValueError(f"{cohort}: replay did not cover all 12 episodes")

    stats = []
    for key in source["episode_order"]:
        episode_key = (cohort, *key)
        c = episode_counters[episode_key]
        offline = offline_summary[episode_key]
        first = first_by_episode.get(episode_key)
        stats.append({
            "cohort": cohort,
            "scene": key[0],
            "episode": key[1],
            "ordinary_s2_calls": c["ordinary_s2_calls"],
            "lookdown_followup_calls": c["lookdown_followup_calls"],
            "total_s2_calls": c["all_s2_calls"],
            "candidate_triggered": int(first is not None),
            "n_static_candidates": c["STATIC_VISUAL_DECISION_REPEAT_CANDIDATE"],
            "n_periodic_candidates": c["PERIODIC_VISUAL_DECISION_REPEAT_CANDIDATE"],
            "first_candidate_kind": first["kind"] if first else "",
            "first_candidate_call_id": first["trigger_call_id"] if first else "",
            "first_candidate_action_prefix_episode": (
                first["trigger_action_prefix_episode_OFFLINE_ALIGNMENT"] if first else ""),
            # Only evaluated after replay; NEVER used in observe().
            "native_success_OFFLINE_ONLY": int(offline["native_success"]),
            "offline_label_role": "POSTHOC_EVALUATION_ONLY",
        })
    return chronology, alarms, stats


def run(base, anomaly_dir, config=None):
    config = config or LoopConfig()
    source_manifest, offline, path_manifest, path_summary = load_offline_labels(anomaly_dir)
    chronology, alarms, episodes = [], [], []
    provenance = {}
    all_keys = set()
    for cohort in COHORTS:
        source_meta = source_manifest["sources"][cohort]
        src = load_native_stream(cohort, source_meta)
        for key in src["episode_order"]:
            token = (cohort, *key)
            if token not in offline:
                raise ValueError(f"{cohort}: episode missing from anomaly summary: {key}")
            if token in all_keys:
                raise ValueError("Duplicate Native replay episode across cohorts")
            all_keys.add(token)
        calls, events, summary = replay_stream(cohort, src, offline, config)
        chronology.extend(calls)
        alarms.extend(events)
        episodes.extend(summary)
        provenance[cohort] = {
            "native": str(src["native"]),
            "native_action_count": src["action_count"],
            "native_episode_count": len(src["episode_order"]),
            "S2_call_count": len(src["ordered"]),
            "actions_sha256": hash_file(src["actions_path"]),
            "decisions_sha256": hash_file(src["decisions_path"]),
        }
    if all_keys != set(offline):
        raise ValueError("Replay cohort episodes differ from frozen offline outcome cohort")
    if len(episodes) != 24:
        raise ValueError("Expected 24 complete Native episodes")
    report = {
        "schema": SCHEMA,
        "monitor_schema": MONITOR_SCHEMA,
        "status": "CPU_PREFIX_ONLY_SHADOW_REPLAY_PASS",
        "thresholds": config.as_dict(),
        "threshold_selection": "EXPLORATORY_TRAIN_IN_SAMPLE_NOT_HELD_OUT",
        "online_monitor_fields": [
            "episode_key", "call_id", "main_step", "call_type",
            "native_action_prefix_count", "model_output",
            "ordinary_current_rgb_pixel_sha256"],
        "no_privileged_position_or_goal_distance_as_input": True,
        "no_motion_response_estimator_connected": True,
        "motion_stagnation_not_established": True,
        "no_policy_action_changed": True,
        "no_new_rollouts": True,
        "no_model_training": True,
        "original_data_untouched": True,
        "N_episodes": len(episodes),
        "N_S2_calls": len(chronology),
        "N_alarm_events": len(alarms),
        "N_alarm_episodes": sum(x["candidate_triggered"] for x in episodes),
        "N_native_success_episodes_OFFLINE_ONLY": sum(x["native_success_OFFLINE_ONLY"] for x in episodes),
        "alarm_episodes_by_offline_label": {
            "native_success": sum(x["candidate_triggered"] for x in episodes
                                  if x["native_success_OFFLINE_ONLY"] == 1),
            "native_failure": sum(x["candidate_triggered"] for x in episodes
                                  if x["native_success_OFFLINE_ONLY"] == 0),
        },
        "source_anomaly_audit_sha256": hash_file(path_manifest),
        "source_anomaly_summary_sha256": hash_file(path_summary),
        "sources": provenance,
        "limitations": [
            "These two failure cases informed thresholds; alarm counts are in-sample only",
            "No independent false-positive or true-positive generalization claim",
            "Repeated S2/RGB patterns do not establish motion stagnation or semantic LOST",
            "Global/episode-local action prefixes are used for audit alignment, not to choose alarms",
            "No evidence yet that any intervention at a new alarm improves navigation",
        ],
    }
    return chronology, alarms, episodes, report


def write_create_only(output, chronology, alarms, episodes, report):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Create-only output exists: " + str(output))
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("decision_prefix_timeline.jsonl", chronology),
        ("candidate_alarm_events.jsonl", alarms),
    ):
        with (output / name).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "episode_offline_comparison.csv").open(
            "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    (output / "shadow_replay_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print("VLN_ONLINE_LOOP_SHADOW_REPLAY=PASS")
    print("EPISODES:", report["N_episodes"], "S2_CALLS:", report["N_S2_calls"])
    print("CANDIDATE_EVENTS:", report["N_alarm_events"],
          "TRIGGERED_EPISODES:", report["N_alarm_episodes"])
    print("IN_SAMPLE_ALARM_EPISODES_BY_OFFLINE_LABEL:",
          report["alarm_episodes_by_offline_label"])
    for event in alarms:
        print("CANDIDATE:", event["cohort"], "/".join(event["episode_key"]),
              event["kind"], "call_id=", event["trigger_call_id"],
              "episode_action_prefix=",
              event["trigger_action_prefix_episode_OFFLINE_ALIGNMENT"])
    print("CAUTION: candidate patterns only; motion/LOST/recovery are unproven")
    print("OUTPUT_DIR:", output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--anomaly-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = args.base.resolve()
    anomaly = (args.anomaly_dir or (base / "native_anomaly_audit_v1_02")).resolve()
    result = run(base, anomaly)
    write_create_only(args.output, *result)


if __name__ == "__main__":
    main()
