"""Task-evidence v2: separate each instruction landmark from navigation progress.

All labels are human-reviewed hypotheses based only on native pre-decision RGB,
instruction and earlier actions. No simulator pose, goal, future or outcomes.
The v1 annotation tool and its frozen samples are not modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_task_evidence_acquire as acquire
import recovery_task_evidence_annotation as v1

SCHEMA = "task_evidence_annotation.v2"
ROLES = ("route_anchor", "intermediate_target", "goal_disambiguator",
         "stop_reference", "other")
VISIBILITY = ("current", "history_only", "both", "not_observed", "uncertain")
PROGRESS = ("before_subgoal", "at_subgoal", "after_subgoal", "unknown")
BASIS = ("visual", "actions", "visual_and_actions", "unknown")
CONSISTENCY = ("consistent", "conflicting", "insufficient", "unknown")
STATUS = ("unreviewed", "needs_review", "complete")


def make_template(native, source, actions):
    acquire.validate_online_row(native)
    if native["call_type"] != "ordinary":
        raise ValueError("Only ordinary native System-2 decisions can be annotated")
    refs = [r["path"] for r in native["images"]]
    for name in refs:
        if not (source / "capture" / name).is_file():
            raise FileNotFoundError(source / "capture" / name)
    past = v1.preceding_action_codes(native, actions)
    return {
        "schema": SCHEMA,
        "annotation_id": "evidence_" + str(native["call_id"]).zfill(7),
        "call_id": native["call_id"],
        "episode_key": native["episode_key"],
        "main_step": native["main_step"],
        "instruction": native["instruction"],
        "historical_rgb": refs[:-1],
        "current_rgb": refs[-1],
        "image_pixel_hashes_by_slot": [r["pixel_sha256"] for r in native["images"]],
        "duplicate_current_history_slots": [
            i for i, x in enumerate(refs[:-1]) if x == refs[-1]
        ],
        "native_action_prefix_count": native["native_action_prefix_count"],
        "past_action_codes": past,
        "episode_action_prefix_count": len(past),
        "source_capture_dir": str(source / "capture"),
        "annotation": {
            "status": "unreviewed", "annotator_id": "",
            "stage_span": "", "progress": "unknown", "progress_basis": "unknown",
            "action_evidence_indices": [],
            "history_consistency": "unknown", "conflict_slot_pairs": [],
            "landmarks": [], "reason": "", "confidence": "low",
        },
    }


def check_row(row, native, actions, source):
    acquire.validate_online_row(native)
    expected = make_template(native, Path(source).resolve(), actions)
    if row.get("schema") != SCHEMA:
        raise ValueError("Incorrect v2 schema")
    for key, original in expected.items():
        if key != "annotation" and row.get(key) != original:
            raise ValueError("Changed native provenance: " + key)
    a = row["annotation"]
    if (a["status"] not in STATUS or a["progress"] not in PROGRESS
            or a["progress_basis"] not in BASIS
            or a["history_consistency"] not in CONSISTENCY
            or a["confidence"] not in ("low", "medium", "high")):
        raise ValueError("Unknown evidence label")
    past = row["past_action_codes"]
    action_ids = a["action_evidence_indices"]
    if not isinstance(action_ids, list) or any(
            type(i) is not int or i < 0 or i >= len(past) for i in action_ids):
        raise ValueError("Invalid predecision episode-local action indices")
    n = len(native["images"])
    hashes = row["image_pixel_hashes_by_slot"]
    pairs = a["conflict_slot_pairs"]
    if not isinstance(pairs, list):
        raise ValueError("conflict_slot_pairs must be a list")
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(i) is not int or not 0 <= i < n for i in pair)
                or pair[0] == pair[1] or hashes[pair[0]] == hashes[pair[1]]):
            raise ValueError("Conflict requires two distinct nonidentical predecision images")
    if (a["history_consistency"] == "conflicting") != bool(pairs):
        raise ValueError("Conflict label and cited conflict pairs do not agree")
    landmarks = a["landmarks"]
    if not isinstance(landmarks, list):
        raise ValueError("landmarks must be a list")
    named = set()
    for landmark in landmarks:
        if not isinstance(landmark, dict):
            raise ValueError("Each landmark must be a dictionary")
        name, span = landmark.get("name"), landmark.get("instruction_span")
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(span, str) or not span.strip()
                or span not in native["instruction"]):
            raise ValueError("Landmark must cite a literal instruction span")
        identity = (name.strip(), span)
        if identity in named:
            raise ValueError("Duplicate landmark record")
        named.add(identity)
        vis = landmark.get("visibility")
        if landmark.get("role") not in ROLES or vis not in VISIBILITY:
            raise ValueError("Unknown landmark role or visibility")
        slots = landmark.get("support_image_slots", [])
        if (not isinstance(slots, list)
                or any(type(i) is not int or not 0 <= i < n for i in slots)
                or len(slots) != len(set(slots))):
            raise ValueError("Invalid landmark evidence slots")
        has_current = n-1 in slots
        has_history = any(i < n-1 for i in slots)
        if vis == "current" and (not has_current or has_history):
            raise ValueError("Current-only visibility requires only the current slot")
        if vis == "history_only" and (not has_history or has_current):
            raise ValueError("History-only visibility requires historical slots only")
        if vis == "both" and not (has_history and has_current):
            raise ValueError("Both visibility requires current and historical slots")
        if vis == "not_observed" and slots:
            raise ValueError("Not-observed landmark cannot cite a visible image")
        regions = landmark.get("regions", [])
        if not isinstance(regions, list):
            raise ValueError("regions must be a list")
        for reg in regions:
            if not isinstance(reg, dict) or reg.get("slot") not in slots:
                raise ValueError("Region must cite an observed slot")
            xyxy = reg.get("bbox_xyxy_norm")
            if (not isinstance(xyxy, list) or len(xyxy) != 4
                    or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in xyxy)
                    or not (xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3])):
                raise ValueError("Invalid normalized landmark bounding box")
        if a["status"] == "complete" and not str(landmark.get("reason","")).strip():
            raise ValueError("Complete landmark requires a visible/unknown justification")
    if a["status"] == "complete":
        if not str(a["annotator_id"]).strip() or not str(a["reason"]).strip():
            raise ValueError("Complete review requires annotator and reason")
        span = a["stage_span"].strip()
        if a["progress"] == "unknown":
            if a["progress_basis"] != "unknown" or span:
                raise ValueError("Unknown progress must have unknown basis and no stage span")
        elif (not span or span not in native["instruction"]
                or a["progress_basis"] == "unknown"):
            raise ValueError("Known progress needs a literal stage span and grounded basis")
        if a["progress_basis"] in ("actions", "visual_and_actions") and not action_ids:
            raise ValueError("Action-based progress needs cited earlier action indices")
    return a["status"]


def sample(args):
    source, dest = Path(args.source_run).resolve(), Path(args.out_dir).resolve()
    if dest.exists():
        raise FileExistsError("Create-only v2 annotations directory")
    acquire.report(argparse.Namespace(run_dir=str(source)))
    native = acquire.read_records(source / "capture/online_decisions.jsonl")
    selected = v1.choose_rows(native, args.count, args.seed, args.min_gap)
    actions = acquire.read_records(source / "capture/online_action_codes.jsonl")
    templates = [make_template(r, source, actions) for r in selected]
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / "annotations.jsonl").open("x", encoding="utf-8") as stream:
        for row in templates:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    acquire.pilot.write_json(dest / "sampling_manifest.json", {
        "schema": SCHEMA, "status": "AWAITING_HUMAN_REVIEW",
        "source_run": str(source), "count": len(templates),
        "source_rows": [r["call_id"] for r in selected],
        "seed": args.seed, "min_gap": args.min_gap,
        "selection": "Seeded round robin by TRAIN scene; ordinary S2 only; outcome-blind",
        "not_a_training_set_yet": True,
    })
    print("TASK_EVIDENCE_V2_SAMPLE=PASS", len(templates), dest)


def validate(args):
    root = Path(args.annotation_dir).resolve()
    manifest = acquire.pilot.read_json(root / "sampling_manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Not v2 task-evidence annotations")
    source = Path(manifest["source_run"]).resolve()
    acquire.report(argparse.Namespace(run_dir=str(source)))
    originals = {r["call_id"]: r for r in acquire.read_records(
        source / "capture/online_decisions.jsonl")}
    actions = acquire.read_records(source / "capture/online_action_codes.jsonl")
    rows = acquire.read_records(root / "annotations.jsonl")
    if (len(rows) != manifest["count"] or [r["call_id"] for r in rows] !=
            manifest["source_rows"]):
        raise ValueError("Annotation order/selection changed")
    counts = dict.fromkeys(STATUS, 0)
    for row in rows:
        native = originals[row["call_id"]]
        counts[check_row(row, native, actions, source)] += 1
    print(json.dumps({
        "schema": SCHEMA, "status": "ANNOTATION_V2_AUDIT_PASS",
        "N_rows": len(rows), "review_states": counts,
        "reliability_not_measured": True, "not_a_training_set_yet": True,
    }, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="command", required=True)
    a = cmds.add_parser("sample")
    a.add_argument("--source-run", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--count", type=int, default=50)
    a.add_argument("--seed", type=int, default=20261029)
    a.add_argument("--min-gap", type=int, default=4)
    a.set_defaults(function=sample)
    b = cmds.add_parser("validate")
    b.add_argument("--annotation-dir", required=True)
    b.set_defaults(function=validate)
    args = p.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
