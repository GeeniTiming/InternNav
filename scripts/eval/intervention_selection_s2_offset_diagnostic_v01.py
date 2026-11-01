"""Read-only, CPU-only first System-2 continuation offset diagnosis.

Do NOT infer that a shifted S2 call is valid. Print exact observed offset,
expected offset under the original frozen report contract, actual preceding
action phases and all first continuation calls, without touching source files,
model weights, replay outputs, or outcome labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

ARMS = (
    "perturbed_keep", "perturbed_scan_no_injection",
    "perturbed_placebo", "perturbed_real_seek",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical(row):
    if "episode_key" in row:
        return tuple(map(str, row["episode_key"]))
    return (PurePosixPath(str(row["scene_id"]).replace("\\", "/")).stem,
            str(row["episode_id"]))


def locked_sources(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol_path = root / "focused_protocol.json"
    expected = manifest.get("generated_hashes", {}).get(str(protocol_path))
    if not isinstance(expected, str) or sha256(protocol_path) != expected:
        raise ValueError("Frozen focused protocol SHA256 missing or changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arms = protocol.get("arms")
    cases = protocol.get("cases")
    if arms != list(ARMS) or not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("Unexpected focused experiment plan")
    if len({tuple(c["key"]) for c in cases}) != len(cases):
        raise ValueError("Duplicate frozen context")
    return cases


def diagnose(root):
    root = Path(root).resolve()
    cases = locked_sources(root)
    result = []
    for arm in ARMS:
        dirname = root / arm
        event_file = dirname / "s2_call_events.jsonl"
        action_file = dirname / "turn_actions.jsonl"
        saved = json.loads((dirname / "s2_call_events_audit.json").read_text("utf-8"))
        if (saved.get("call_event_sha256") != sha256(event_file)
                or saved.get("same_arm_action_sha256") != sha256(action_file)):
            raise ValueError("Saved same-arm event/action SHA256 mismatch: " + arm)
        events, actions = rows(event_file), rows(action_file)
        if len(events) % 2:
            raise ValueError("Partial S2 enter/exit events in " + arm)
        enters = []
        for before, after in zip(events[::2], events[1::2]):
            if (before.get("event") != "enter" or after.get("event") != "exit"
                    or before.get("call_id") != after.get("call_id")
                    or canonical(before) != canonical(after)):
                raise ValueError("Unpaired S2 call event: " + arm)
            enters.append(before)
        for case in cases:
            key = tuple(case["key"])
            episode_events = [e for e in enters if canonical(e) == key]
            episode_actions = [a for a in actions if canonical(a) == key]
            if not episode_events or not episode_actions:
                raise ValueError("Missing recorded original episode: " + repr((arm, key)))
            gate = case["checkpoint"]
            expected = int(gate["prefix_action_count"]) + 3 + (
                0 if arm == "perturbed_keep" else 6)
            continuation = [e for e in episode_events if e["phase"] == "continuation"]
            if not continuation:
                raise ValueError("No recorded continuation S2 call: " + repr((arm, key)))
            first = continuation[0]
            observed = first.get("native_action_prefix_count")
            if type(observed) is not int or observed > len(episode_actions):
                raise ValueError("Unbound S2 offset: " + repr((arm, key)))
            before = episode_actions[:observed]
            phase_counts = dict(sorted(Counter(a.get("phase", "UNKNOWN") for a in before).items()))
            result.append({
                "episode_key": list(key), "arm": arm,
                "gate_main_step": gate["step"],
                "gate_prefix_actions": gate["prefix_action_count"],
                "expected_first_continuation_offset": expected,
                "observed_first_continuation_offset": observed,
                "offset_delta": observed - expected,
                "matches_original_report_contract": observed == expected,
                "first_continuation_call_id": first.get("call_id"),
                "first_continuation_episode_call_id": first.get("episode_call_id"),
                "first_continuation_main_step": first.get("main_step"),
                "action_phases_before_first_call": phase_counts,
                "preceding_3_action_phases": [x.get("phase") for x in before[-3:]],
                "preceding_3_action_codes": [x.get("action") for x in before[-3:]],
                "episode_enter_count": len(episode_events),
                "episode_continuation_enter_count": len(continuation),
                "pre_gate_last_s2_offset": max(
                    (e["native_action_prefix_count"] for e in episode_events
                     if e["phase"] == "prefix"), default=None),
                "first_three_continuation_calls": [{
                    "episode_call_id": e.get("episode_call_id"),
                    "main_step": e.get("main_step"),
                    "prefix_count": e.get("native_action_prefix_count"),
                } for e in continuation[:3]],
            })
    return {
        "schema": "focused_s2_offset_diagnostic.v0.1",
        "status": "READ_ONLY_OBSERVED_NOT_REPAIRED",
        "N_focus_case_arms": len(result),
        "N_exact_legacy_offsets": sum(x["matches_original_report_contract"] for x in result),
        "N_mismatching_legacy_offsets": sum(not x["matches_original_report_contract"] for x in result),
        "gpu_executed": False, "files_modified": False,
        "cases": result,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args(argv)
    print(json.dumps(diagnose(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
