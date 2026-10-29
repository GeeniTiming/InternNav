"""Outcome-blind, human-auditable task-evidence annotation sample.

Never infer a subgoal or landmark from future outcomes. Sample decisions from
completed native TRAIN run before reviewing any treatment outcomes. Templates
contain exact user-visible pre-decision RGB inputs and instruction, not model
answers, simulator poses, success, depth, GPS or future actions.

This tool does NOT train a model; labels require verified human review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_task_evidence_acquire as acquire

SCHEMA = "task_evidence_annotation.v1"
VISIBILITY = ("current", "history_only", "both", "not_observed", "uncertain")
AMBIGUITY = ("one_candidate", "multiple_candidates", "none_visible", "unknown")
CONSISTENCY = ("consistent", "conflicting", "insufficient", "unknown")
PROGRESS = ("before_subgoal", "at_subgoal", "after_subgoal", "unknown")
STATUS = ("unreviewed", "needs_review", "complete")


def choose_rows(decisions, count, seed, gap=4):
    """Round-robin over scenes; random shuffle within each scene; label-blind."""
    if int(count) <= 0 or int(gap) < 0:
        raise ValueError("count must be positive, gap nonnegative")
    groups = {}
    for row in decisions:
        if row["call_type"] != "ordinary":
            continue
        key = tuple(row["episode_key"])
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("No ordinary System-2 decision states")
    rng = random.Random(int(seed))
    buckets = {k: rng.sample(v, len(v)) for k, v in groups.items()}
    keys = sorted(buckets)
    chosen, selected_steps = [], {k: [] for k in keys}
    # Iterate scenes in seeded order (not outcome order).
    rng.shuffle(keys)
    while len(chosen) < count:
        progressed = False
        for key in keys:
            remaining = buckets[key]
            candidate = None
            for index, row in enumerate(remaining):
                if all(abs(row["main_step"] - v) >= gap for v in selected_steps[key]):
                    candidate = remaining.pop(index)
                    break
            if candidate is None:
                continue
            chosen.append(candidate)
            selected_steps[key].append(candidate["main_step"])
            progressed = True
            if len(chosen) == count:
                break
        if not progressed:
            break
    if len(chosen) < count:
        raise ValueError(
            f"Only {len(chosen)} decisions available with min-gap={gap}; "
            "use fewer labels or lower the minimum gap; never select by outcome."
        )
    return chosen


def make_template(row, source):
    # Validate provenance; existing images can be reused by many decisions.
    acquire.validate_online_row(row)
    refs = [ref["path"] for ref in row["images"]]
    for ref in refs:
        if not (source / "capture" / ref).is_file():
            raise FileNotFoundError(source / "capture" / ref)
    return {
        "schema": SCHEMA,
        "annotation_id": "evidence_" + str(row["call_id"]).zfill(7),
        "call_id": int(row["call_id"]),
        "episode_key": row["episode_key"],
        "main_step": row["main_step"],
        "instruction": row["instruction"],
        "current_rgb": refs[-1],
        "historical_rgb": refs[:-1],
        "source_capture_dir": str(source / "capture"),
        "annotation": {
            "status": "unreviewed",
            "annotator_id": "",
            "current_subinstruction_span": "",
            "progress": "unknown",
            "target_landmarks": [],
            "visibility": "uncertain",
            "ambiguity": "unknown",
            "history_consistency": "unknown",
            "support_image_refs": [],
            "reason": "",
            "confidence": "low",
        },
    }


def sample(args):
    source = Path(args.source_run).resolve()
    output = Path(args.out_dir).resolve()
    if output.exists():
        raise FileExistsError("Create-only annotation template directory: " + str(output))
    acquire.report(argparse.Namespace(run_dir=str(source)))
    records = acquire.read_records(source / "capture/online_decisions.jsonl")
    chosen = choose_rows(records, args.count, args.seed, args.min_gap)
    templates = [make_template(row, source) for row in chosen]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "annotations.jsonl").open("x", encoding="utf-8") as f:
        for row in templates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": SCHEMA, "status": "AWAITING_HUMAN_REVIEW",
        "source_run": str(source), "count": len(chosen),
        "selection": "Seeded round-robin scene-stratified sampling among ordinary System-2 calls",
        "seed": args.seed, "min_main_step_separation": args.min_gap,
        "source_rows": [x["call_id"] for x in chosen],
        "no_outcome_based_selection": True,
        "not_a_training_set_yet": True,
    }
    acquire.pilot.write_json(output / "sampling_manifest.json", report)
    print("TASK_EVIDENCE_ANNOTATION_SAMPLE=PASS", len(chosen), "rows at", output)


def check_row(row, native):
    if row["schema"] != SCHEMA:
        raise ValueError("Unknown annotation schema")
    if row["call_id"] != native["call_id"] or row["episode_key"] != native["episode_key"]:
        raise ValueError("Annotation decision/episode provenance mismatch")
    if row["instruction"] != native["instruction"] or row["main_step"] != native["main_step"]:
        raise ValueError("Instruction/decision time changed")
    originals = [ref["path"] for ref in native["images"]]
    if row["current_rgb"] != originals[-1] or row["historical_rgb"] != originals[:-1]:
        raise ValueError("Altered image inputs")
    data = row["annotation"]
    if data["status"] not in STATUS:
        raise ValueError("Invalid review status")
    if (data["progress"] not in PROGRESS or data["visibility"] not in VISIBILITY
            or data["ambiguity"] not in AMBIGUITY
            or data["history_consistency"] not in CONSISTENCY
            or data["confidence"] not in ("low", "medium", "high")):
        raise ValueError("Invalid evidence-state value")
    if any(x not in originals for x in data["support_image_refs"]):
        raise ValueError("Support images must be from the pre-decision System-2 inputs")
    if not isinstance(data["target_landmarks"], list) or any(
        not isinstance(item, str) or not item.strip() for item in data["target_landmarks"]
    ):
        raise ValueError("Landmark labels must be nonempty strings")
    if data["status"] == "complete":
        if (not data["annotator_id"].strip()
                or not data["reason"].strip()
                or not data["current_subinstruction_span"].strip()):
            raise ValueError("Completed annotation requires author, reason and subinstruction")
        if data["current_subinstruction_span"] not in row["instruction"]:
            raise ValueError("Subinstruction must be a literal span of the full instruction")
        if data["visibility"] in ("current", "history_only", "both"):
            if not data["target_landmarks"]:
                raise ValueError("Visible target requires at least one named landmark")
            expected = set(originals[-1:] if data["visibility"] == "current" else
                           originals[:-1] if data["visibility"] == "history_only" else
                           originals)
            if not set(data["support_image_refs"]) & expected:
                raise ValueError("Visible landmark must cite a corresponding image")
            if data["visibility"] == "both" and not (
                set(data["support_image_refs"]) & set(originals[:-1])
                and originals[-1] in data["support_image_refs"]
            ):
                raise ValueError("Both visibility needs current and history image references")
        if data["history_consistency"] == "conflicting" and len(set(data["support_image_refs"])) < 2:
            raise ValueError("Conflicting history requires two independently cited predecision images")
    return data["status"]


def validate(args):
    folder = Path(args.annotation_dir).resolve()
    sampling = acquire.pilot.read_json(folder / "sampling_manifest.json")
    if sampling["schema"] != SCHEMA:
        raise ValueError("Unknown annotation selection manifest")
    source = Path(sampling["source_run"]).resolve()
    acquire.report(argparse.Namespace(run_dir=str(source)))
    originals = {
        int(r["call_id"]): r
        for r in acquire.read_records(source / "capture/online_decisions.jsonl")
    }
    rows = acquire.read_records(folder / "annotations.jsonl")
    if (len(rows) != sampling["count"]
            or [int(r["call_id"]) for r in rows] != sampling["source_rows"]):
        raise ValueError("Annotation sampling cohort/order changed")
    counts = {k: 0 for k in STATUS}
    for row in rows:
        if int(row["call_id"]) not in originals:
            raise ValueError("Annotation references unknown decision")
        status = check_row(row, originals[int(row["call_id"])])
        counts[status] += 1
    result = {
        "schema": SCHEMA, "status": "ANNOTATION_AUDIT_PASS",
        "rows": len(rows), "review_states": counts,
        "ready_for_reliability_analysis": counts["complete"] == len(rows),
        "human_consistency_not_yet_assessed": True,
        "no_intervention_labels": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def compare(args):
    """Independent reviewer agreement; no adjudication by model/outcomes."""
    a, b = Path(args.review_a).resolve(), Path(args.review_b).resolve()
    ma = acquire.pilot.read_json(a / "sampling_manifest.json")
    mb = acquire.pilot.read_json(b / "sampling_manifest.json")
    if (ma["source_run"] != mb["source_run"]
            or ma["source_rows"] != mb["source_rows"]):
        raise ValueError("Reviewers are not using the same frozen decision sample")
    rows_a = acquire.read_records(a / "annotations.jsonl")
    rows_b = acquire.read_records(b / "annotations.jsonl")
    if (len(rows_a) != ma["count"] or len(rows_b) != ma["count"]):
        raise ValueError("One review has incomplete annotation rows")
    native_rows = {
        int(row["call_id"]): row
        for row in acquire.read_records(
            Path(ma["source_run"]) / "capture/online_decisions.jsonl"
        )
    }
    keys = ("progress", "visibility", "ambiguity", "history_consistency")
    matches = dict.fromkeys(keys, 0)
    disagreements = []
    for left, right in zip(rows_a, rows_b):
        if left["call_id"] != right["call_id"]:
            raise ValueError("Reviewers' decision ordering differs")
        native = native_rows[left["call_id"]]
        if check_row(left, native) != "complete" or check_row(right, native) != "complete":
            raise ValueError("Both reviewers must complete all labels before comparing")
        different = {}
        for key in keys:
            x, y = left["annotation"][key], right["annotation"][key]
            matches[key] += int(x == y)
            if x != y:
                different[key] = [x, y]
        if different:
            disagreements.append({"call_id": left["call_id"], "fields": different})
    n = len(rows_a)
    result = {
        "schema": "task_evidence_interrater.v1",
        "N_decisions": n,
        "raw_agreement": {k: matches[k] / n for k in keys},
        "disagreements": disagreements,
        "caution": "Raw agreement alone does not establish reliability on imbalanced classes.",
    }
    if args.out_json:
        out = Path(args.out_json)
        if out.exists():
            raise FileExistsError("Create-only reviewer comparison report")
        acquire.pilot.write_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    sampler = subs.add_parser("sample")
    sampler.add_argument("--source-run", required=True)
    sampler.add_argument("--out-dir", required=True)
    sampler.add_argument("--count", type=int, default=50)
    sampler.add_argument("--seed", type=int, default=20261029)
    sampler.add_argument("--min-gap", type=int, default=4)
    sampler.set_defaults(function=sample)
    checker = subs.add_parser("validate")
    checker.add_argument("--annotation-dir", required=True)
    checker.set_defaults(function=validate)
    compare_parser = subs.add_parser("compare")
    compare_parser.add_argument("--review-a", required=True)
    compare_parser.add_argument("--review-b", required=True)
    compare_parser.add_argument("--out-json")
    compare_parser.set_defaults(function=compare)
    args = p.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
