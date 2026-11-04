#!/usr/bin/env python3
"""Opt-in live passive S2 shadow sink and CPU exact-replay verifier.

The sink is called AFTER an actual DualVLN generate() response is decoded.
It reads only the existing Task Evidence Recorder's online decision row;
it cannot mutate navigator actions, output, history, simulator or policy.
The verifier independently rebuilds monitor events from frozen S2 input
records, checks source-action alignment, and compares each streamed result.

This is engineering equivalence, NOT an independent efficacy evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vln_online_loop_monitor_v01 import (
    OnlineDecisionVisualLoopMonitor, LoopConfig, SCHEMA as MONITOR_SCHEMA,
    project_online_s2_call,
)
from vln_native_s2_visual_evidence_export_v1 import (
    group_action_rows, localize_call_records,
)

SCHEMA = "vln_online_loop_live_shadow.v0.1"
TIMELINE = "online_loop_shadow_timeline.jsonl"
EVENTS = "online_loop_shadow_events.jsonl"
SUMMARY = "online_loop_shadow_summary.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{i} invalid JSON") from exc


def sha256(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            dig.update(data)
    return dig.hexdigest()


def timeline_row(online, observation):
    """No raw prompt, all-images, pose or future labels in shadow log."""
    return {
        "schema": SCHEMA,
        "episode_key": list(online["episode_key"]),
        "call_id": online["call_id"],
        "call_type": online["call_type"],
        "main_step": online["main_step"],
        "action_prefix_global": online["native_action_prefix_count"],
        "model_output": online["model_output"],
        "current_rgb_pixel_sha256": online["current_rgb_pixel_sha256"],
        "status": observation["status"],
        "same_output_streak": observation["same_output_streak"],
        "candidate_types_emitted": [e["kind"] for e in observation["events"]],
    }


class LiveShadowSink:
    """Write-once files exclusively inside a new native capture directory."""

    def __init__(self, capture_root):
        self.capture_root = Path(capture_root).resolve()
        if not self.capture_root.is_dir():
            raise FileNotFoundError(self.capture_root)
        for name in (TIMELINE, EVENTS, SUMMARY):
            if (self.capture_root / name).exists():
                raise FileExistsError("Shadow output already exists: " + name)
        self.monitor = OnlineDecisionVisualLoopMonitor()
        self.timeline_file = (self.capture_root / TIMELINE).open("x", encoding="utf-8")
        self.events_file = (self.capture_root / EVENTS).open("x", encoding="utf-8")
        self.n_calls = 0
        self.n_ordinary = 0
        self.n_followups = 0
        self.n_alarms = 0
        self.alarm_episodes = set()
        self.closed = False

    def observe(self, capture_row):
        if self.closed:
            raise RuntimeError("Shadow sink already closed")
        # Monitor does not mutate capture_row or consume offline/pose metadata.
        online = project_online_s2_call(capture_row)
        result = self.monitor.observe(capture_row)
        record = timeline_row(online, result)
        self.timeline_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.timeline_file.flush()
        for event in result["events"]:
            self.events_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            self.events_file.flush()
            self.n_alarms += 1
            self.alarm_episodes.add(tuple(event["episode_key"]))
        self.n_calls += 1
        if online["call_type"] == "ordinary":
            self.n_ordinary += 1
        else:
            self.n_followups += 1
        return record

    def close(self, error=None):
        if self.closed:
            raise RuntimeError("Shadow sink closed twice")
        self.closed = True
        self.timeline_file.close()
        self.events_file.close()
        result = {
            "schema": SCHEMA,
            "monitor_schema": MONITOR_SCHEMA,
            "status": "failed" if error else "complete",
            "error": error,
            "thresholds": self.monitor.config.as_dict(),
            "N_S2_calls": self.n_calls,
            "N_ordinary_calls": self.n_ordinary,
            "N_lookdown_followup_calls": self.n_followups,
            "N_candidate_events": self.n_alarms,
            "N_triggered_episodes": len(self.alarm_episodes),
            "source": "Actual model-decoded System-2 decisions via the existing acquisition recorder",
            "no_policy_output_or_action_modified": True,
            "no_motion_estimator": True,
            "alarm_is_candidate_only": True,
            "source_action_trace_equivalence_to_external_uninstrumented_run": "NOT_TESTED",
        }
        with (self.capture_root / SUMMARY).open("x", encoding="utf-8") as output:
            output.write(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return result


def verify_run(run_dir):
    """Independently check the full live log against original S2 call records.

    The S2 records are projected on each iteration and replayed in original
    call order. Only after the full replay are native action logs used to
    validate run-global action counts. This audit never changes run files.
    """
    root = Path(run_dir).resolve()
    cap = root / "capture"
    decision_file = cap / "online_decisions.jsonl"
    actions_file = root / "control" / "actions.jsonl"
    timeline_file = cap / TIMELINE
    events_file = cap / EVENTS
    summary_file = cap / SUMMARY
    capture_summary_file = cap / "capture_summary.json"
    for path in (decision_file, actions_file, timeline_file, events_file,
                 summary_file, capture_summary_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    saved = read_json(summary_file)
    cap_summary = read_json(capture_summary_file)
    if (saved.get("schema") != SCHEMA
            or saved.get("monitor_schema") != MONITOR_SCHEMA
            or saved.get("status") != "complete"
            or cap_summary.get("status") != "complete"):
        raise ValueError("Live/capture run is not complete and audited")
    config = LoopConfig(**saved["thresholds"])
    original = list(read_jsonl(decision_file))
    live_timeline = list(read_jsonl(timeline_file))
    live_events = list(read_jsonl(events_file))
    action_rows = list(read_jsonl(actions_file))
    groups, offsets, total, order = group_action_rows(action_rows)
    localized = localize_call_records(
        original, offsets, {key:len(items) for key, items in groups.items()}, total)
    if len(original) != len(live_timeline) or len(original) != int(saved["N_S2_calls"]):
        raise ValueError("Live timeline length differs from actual S2 recorder")
    if (len(original) != int(cap_summary["N_decisions"])
            or total != int(cap_summary["N_actual_env_actions"])):
        raise ValueError("Live S2/action counts differ from capture summary")
    if set(localized) != set(order):
        raise ValueError("S2 coverage differs from action episodes")

    monitor = OnlineDecisionVisualLoopMonitor(config)
    expected_events = []
    ordinary = followups = 0
    for i, (raw, live) in enumerate(zip(original, live_timeline)):
        if raw["call_id"] != i:
            raise ValueError("S2 call IDs must equal capture row indices")
        online = project_online_s2_call(raw)
        result = monitor.observe(raw)
        expected_timeline = timeline_row(online, result)
        if live != expected_timeline:
            raise ValueError(
                "LIVE_SHADOW_NOT_COMPARABLE: mismatch at S2 call_id "
                + str(raw["call_id"]))
        expected_events.extend(result["events"])
        if online["call_type"] == "ordinary":
            ordinary += 1
        else:
            followups += 1
    if live_events != expected_events:
        raise ValueError("LIVE_SHADOW_NOT_COMPARABLE: candidate event sequence mismatch")
    event_episodes = {tuple(r["episode_key"]) for r in expected_events}
    if (len(expected_events) != int(saved["N_candidate_events"])
            or len(event_episodes) != int(saved["N_triggered_episodes"])
            or ordinary != int(saved["N_ordinary_calls"])
            or followups != int(saved["N_lookdown_followup_calls"])):
        raise ValueError("Live shadow summary differs from verified timeline")

    return {
        "schema": SCHEMA,
        "status": "LIVE_SHADOW_EXACT_PREFIX_REPLAY_MATCH",
        "run_dir": str(root),
        "N_episodes": len(order),
        "N_native_actions": total,
        "N_decisions": len(original),
        "N_ordinary_calls": ordinary,
        "N_lookdown_followup_calls": followups,
        "N_candidate_events": len(expected_events),
        "N_triggered_episodes": len(event_episodes),
        "decisions_sha256": sha256(decision_file),
        "actions_sha256": sha256(actions_file),
        "live_timeline_sha256": sha256(timeline_file),
        "live_events_sha256": sha256(events_file),
        "offline_replay_and_live_prefix_events_match": True,
        "original_run_unchanged": True,
        "external_uninstrumented_baseline_comparison": "NOT_DONE",
        "no_claim_of_held_out_detection_or_recovery_efficacy": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_run(args.run_dir)
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Create-only live-shadow verification report exists: " + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(report["status"])
    print("EPISODES:", report["N_episodes"],
          "S2_CALLS:", report["N_decisions"],
          "NATIVE_ACTIONS:", report["N_native_actions"])
    print("CANDIDATES:", report["N_candidate_events"],
          "TRIGGERED_EPISODES:", report["N_triggered_episodes"])
    print("REPORT:", dest)
    print("CAUTION: exact event replay does not prove policy non-interference or held-out efficacy")


if __name__ == "__main__":
    main()
