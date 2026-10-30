"""Create-only v2 per-landmark annotation templates at *actual replay checkpoints*.

Unlike the earlier six full-history samples, every record here refers to
the same pre-intervention System-2 boundary that a paired continuation will
replay. Images and past actions are from native-only checkpoint acquisition.

No privileged pose/distance/goal/future information enters annotations.
Human labels are blank; schema validation never judges semantic correctness.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_independent_checkpoint as cp
import recovery_train_disjoint_acquire as train
import recovery_task_evidence_annotation_v2 as v2

pilot, capture, physical = cp.pilot, cp.capture, cp.physical
SCHEMA = "task_evidence_checkpoint_annotations.v2"
LABEL_STATUSES = v2.STATUS


def native_contexts(checkpoint_run):
    root = Path(checkpoint_run).resolve()
    manifest, data = cp.read_run(root)
    plan, _, source, _, dataset, _, _ = data
    source_manifest = pilot.verify_manifest(source)
    train.validate_selection(source_manifest)
    report = pilot.read_json(root / "checkpoint_report.json")
    if (report.get("schema") != "independent_recovery_checkpoints.v1"
            or report.get("status") != "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION"):
        raise ValueError("All locked checkpoint cases must be eligible before v2 annotation")
    if (len(report["cases"]) != len(plan["cases"])
            or [c["key"] for c in report["cases"]] !=
               [c["key"] for c in plan["cases"]]):
        raise ValueError("Checkpoint report order differs from predeclared cohort")
    ep_by_key = {capture.episode_key(e): e for e in dataset["episodes"]}
    observed = {tuple(c["key"]): [] for c in report["cases"]}
    # Use native plain rows: capture_actions.jsonl includes an extra main_step
    # field, whereas the original locked prefix digest used untagged actions.
    for row in physical.read_lines(root / "control/actions.jsonl"):
        key = capture.episode_key(row)
        if key not in observed:
            raise ValueError("Unexpected action episode during checkpoint capture")
        observed[key].append(row)
    from PIL import Image
    results = []
    for case in report["cases"]:
        key = tuple(case["key"])
        gate = case["checkpoint"]
        if gate is None or case["status"] != "COMPARABLE_CHECKPOINT_CAPTURED":
            raise ValueError("Checkpoint not comparable: " + repr(key))
        folder = root / "control/checkpoints" / (key[0] + "_" + key[1])
        details = pilot.read_json(folder / "context_inputs.json")
        if (details.get("schema") != "independent_recovery_context_inputs.v1"
                or details["key"] != list(key) or details["main_step"] != gate["step"]
                or details["history_sha256"] != gate["history_sha256"]):
            raise ValueError("Stored predecision RGB context changed")
        if key not in ep_by_key:
            raise ValueError("Checkpoint episode absent from locked dataset")
        ins = ep_by_key[key]["instruction"]["instruction_text"]
        if ins != next(c["instruction"] for c in plan["cases"]
                       if tuple(c["key"]) == key):
            raise ValueError("Native instruction changed")
        refs = []
        hashes = []
        for item in details["sampled_history"]:
            p = folder / item["path"]
            if not p.is_file() or pilot.sha256(p) != item["png_sha256"]:
                raise ValueError("Historical RGB PNG or file changed")
            with Image.open(p) as opened:
                pixel_sha = physical.history_hash([opened.copy()])
            if pixel_sha != item["pixel_sha256"]:
                raise ValueError("Historical model RGB pixel digest mismatch")
            refs.append(str(p.relative_to(root)))
            hashes.append(item["pixel_sha256"])
        current_path = folder / details["current_model_rgb_path"]
        if (not current_path.is_file()
                or pilot.sha256(current_path) != details["current_model_rgb_png_sha256"]):
            raise ValueError("Native current-model RGB PNG changed")
        with Image.open(current_path) as opened:
            current_pixel_sha = physical.history_hash([opened.copy()])
        if current_pixel_sha != details["current_model_rgb_pixel_sha256"]:
            raise ValueError("Native current-model RGB pixel digest mismatch")
        current = str(current_path.relative_to(root))
        hashes.append(details["current_model_rgb_pixel_sha256"])
        # No future actions may enter this record. Index is local to Episode.
        prefix_n = int(gate["prefix_action_count"])
        trace = observed[key]
        if not 0 < prefix_n <= len(trace):
            raise ValueError("Invalid or missing predecision action prefix")
        if cp.cohort.digest(trace[:prefix_n]) != gate["prefix_trace_sha256"]:
            raise ValueError("Native action prefix SHA changed")
        past = [int(a["action"]) for a in trace[:prefix_n]]
        results.append({
            "schema": SCHEMA, "annotation_id": "checkpoint_" + key[0] + "_" + key[1],
            "episode_key": list(key), "main_step": int(gate["step"]),
            "instruction": ins,
            "source_context": str((folder / "context_inputs.json").relative_to(root)),
            "historical_rgb": refs, "current_rgb": current,
            "image_pixel_hashes_by_slot": hashes,
            "duplicate_current_history_slots": [
                i for i, h in enumerate(hashes[:-1]) if h == hashes[-1]
            ],
            "episode_action_prefix_count": prefix_n, "past_action_codes": past,
            "annotation": {
                "status": "unreviewed", "annotator_id": "",
                "stage_span": "", "progress": "unknown",
                "progress_basis": "unknown", "action_evidence_indices": [],
                "history_consistency": "unknown", "conflict_slot_pairs": [],
                "landmarks": [], "reason": "", "confidence": "low",
            },
        })
    if len({tuple(x["episode_key"]) for x in results}) != len(results):
        raise ValueError("Duplicate episode in checkpoint annotation candidates")
    return results


def validate_review(record):
    ann = record["annotation"]
    if (ann["status"] not in LABEL_STATUSES
            or ann["progress"] not in v2.PROGRESS
            or ann["progress_basis"] not in v2.BASIS
            or ann["history_consistency"] not in v2.CONSISTENCY
            or ann["confidence"] not in ("low", "medium", "high")):
        raise ValueError("Unknown v2 annotation label")
    n = len(record["historical_rgb"]) + 1
    hashes = record["image_pixel_hashes_by_slot"]
    prefix = record["past_action_codes"]
    aidx = ann["action_evidence_indices"]
    if (not isinstance(aidx, list) or any(
            type(x) is not int or not 0 <= x < len(prefix) for x in aidx)):
        raise ValueError("Action evidence must cite only earlier actions")
    pairs = ann["conflict_slot_pairs"]
    if not isinstance(pairs, list):
        raise ValueError("Conflict pairs must be a list")
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int or not 0 <= x < n for x in pair)
                or pair[0] == pair[1] or hashes[pair[0]] == hashes[pair[1]]):
            raise ValueError("Historical conflict cannot cite identical or invalid RGB slots")
    if (ann["history_consistency"] == "conflicting") != bool(pairs):
        raise ValueError("Conflict label and evidence must agree")
    landmarks = ann["landmarks"]
    if not isinstance(landmarks, list):
        raise ValueError("landmarks must be a list")
    seen = set()
    for lm in landmarks:
        if not isinstance(lm, dict):
            raise ValueError("Landmark evidence must be an object")
        name, span = lm.get("name"), lm.get("instruction_span")
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(span, str) or not span.strip()
                or span not in record["instruction"]):
            raise ValueError("Every landmark must cite a literal instruction span")
        ident = (name.strip(), span)
        if ident in seen:
            raise ValueError("Duplicate landmark")
        seen.add(ident)
        role, vis = lm.get("role"), lm.get("visibility")
        if role not in v2.ROLES or vis not in v2.VISIBILITY:
            raise ValueError("Unknown landmark role or visibility")
        slots = lm.get("support_image_slots", [])
        if (not isinstance(slots, list) or len(slots) != len(set(slots))
                or any(type(i) is not int or not 0 <= i < n for i in slots)):
            raise ValueError("Landmark evidence references invalid image slots")
        cur, hist = n-1 in slots, any(i < n-1 for i in slots)
        if ((vis == "current" and (not cur or hist))
                or (vis == "history_only" and (not hist or cur))
                or (vis == "both" and not (cur and hist))
                or (vis == "not_observed" and bool(slots))):
            raise ValueError("Landmark visibility disagrees with cited slots")
        regions = lm.get("regions", [])
        if not isinstance(regions, list):
            raise ValueError("Landmark regions must be a list")
        for reg in regions:
            if not isinstance(reg, dict) or reg.get("slot") not in slots:
                raise ValueError("Region must cite a supported image slot")
            box = reg.get("bbox_xyxy_norm")
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(x) not in (float, int) or not 0 <= x <= 1 for x in box)
                    or not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError("Invalid normalized bounding box")
        if ann["status"] == "complete" and not str(lm.get("reason", "")).strip():
            raise ValueError("Completed landmark needs an evidence reason")
    if ann["status"] == "complete":
        if not str(ann["annotator_id"]).strip() or not str(ann["reason"]).strip():
            raise ValueError("Complete review requires annotator and reason")
        span = ann["stage_span"].strip()
        if ann["progress"] == "unknown":
            if span or ann["progress_basis"] != "unknown":
                raise ValueError("Unknown stage must not invent evidence")
        elif (not span or span not in record["instruction"]
                or ann["progress_basis"] == "unknown"):
            raise ValueError("Known stage needs literal span and rationale")
        if ann["progress_basis"] in ("actions", "visual_and_actions") and not aidx:
            raise ValueError("Action-based progress needs predecision action citations")
    return ann["status"]


def prepare(args):
    root = Path(args.out_dir).resolve()
    if root.exists():
        raise FileExistsError("Create-only checkpoint evidence template directory")
    source = Path(args.checkpoint_run).resolve()
    # Validate every completed native checkpoint before exposing annotations.
    report = cp.report(argparse.Namespace(run_dir=str(source)))
    if report["status"] != "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION":
        raise ValueError("Not all preselected episodes reached a comparable checkpoint")
    records = native_contexts(source)
    root.mkdir(parents=True, exist_ok=False)
    f = root / "annotations.jsonl"
    with f.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": SCHEMA, "status": "AWAITING_HUMAN_ANNOTATION",
        "checkpoint_run": str(source), "count": len(records),
        "keys": [r["episode_key"] for r in records],
        "checkpoint_report_sha256": pilot.sha256(source / "checkpoint_report.json"),
        "annotation_source_script_sha256": pilot.sha256(__file__),
        "predecision_only": True, "selector_privileged_inputs": False,
        "human_labels_status": "NOT_LABELED",
        "not_held_out_evaluation": True,
    }
    pilot.write_json(root / "annotation_manifest.json", manifest)
    print("TRAIN_CHECKPOINT_EVIDENCE_PREPARE=PASS", len(records),
          "unreviewed native checkpoint contexts", root)


def validate(args):
    root = Path(args.annotation_dir).resolve()
    manifest = pilot.read_json(root / "annotation_manifest.json")
    source = Path(manifest["checkpoint_run"]).resolve()
    if (manifest["schema"] != SCHEMA
            or pilot.sha256(source / "checkpoint_report.json") !=
               manifest["checkpoint_report_sha256"]
            or pilot.sha256(__file__) != manifest["annotation_source_script_sha256"]):
        raise ValueError("Checkpoint annotation source or validator changed")
    originals = native_contexts(source)
    rows = [json.loads(x) for x in (root / "annotations.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    if (len(rows) != len(originals) or len(rows) != manifest["count"]
            or [x["episode_key"] for x in rows] != manifest["keys"]):
        raise ValueError("Unexpected or reordered checkpoint annotations")
    counts = {x: 0 for x in LABEL_STATUSES}
    for row, orig in zip(rows, originals):
        for key, value in orig.items():
            if key != "annotation" and row.get(key) != value:
                raise ValueError("Native predecision provenance changed: " + key)
        counts[validate_review(row)] += 1
    print(json.dumps({
        "schema": SCHEMA,
        "status": "TRAIN_CHECKPOINT_EVIDENCE_AUDIT_PASS",
        "N_contexts": len(rows), "review_states": counts,
        "no_privileged_selector_inputs": True,
        "no_interventions_executed_by_this_tool": True,
        "not_a_training_set_yet": True,
    }, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    a = s.add_parser("prepare")
    a.add_argument("--checkpoint-run", required=True)
    a.add_argument("--out-dir", required=True)
    a.set_defaults(function=prepare)
    a = s.add_parser("validate")
    a.add_argument("--annotation-dir", required=True)
    a.set_defaults(function=validate)
    args = p.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
