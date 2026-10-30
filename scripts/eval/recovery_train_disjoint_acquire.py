"""Outcome-blind R2R TRAIN native source acquisition excluding inspected scenes.

This wraps, without editing, the frozen task_evidence_acquire.py native logger.
No intervention, reset, hindsight success filtering, future trajectory, or oracle
geometry is used to select episodes. Entirely create-only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import recovery_task_evidence_acquire as acquire

pilot, capture = acquire.pilot, acquire.capture
SCHEMA = "train_scene_disjoint_native_acquisition.v1"
# Known visually inspected or previously intervened scenes from earlier pilots.
# Source-run exclusion is ALSO accepted and verified at preparation.
INSPECTED = frozenset({
    "oLBMNvg9in8", "X7HyMhZNoso",
    "5LpN3gDmAk7", "7y3sRwLe3Va",
    "2azQ1b91cZZ", "QUCTc6BB5sX", "TbHJrupSAjP",
    "x8F5xyUWy9e", "zsNo4HB9uLZ",
})


def scene_stems_from_locked_run(run_dir):
    root = Path(run_dir).resolve()
    manifest = pilot.verify_manifest(root)
    rows = manifest.get("episodes", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("An exclusion source contains no locked episodes")
    return {capture.episode_key(ep)[0] for ep in rows}


def select_outcome_blind_new_scenes(dataset, count, excluded):
    """Filter by pre-existing scene names; original annotation order retained."""
    excluded = frozenset(excluded)
    filtered = copy.deepcopy(dataset)
    filtered["episodes"] = [
        ep for ep in dataset["episodes"]
        if capture.episode_key(ep)[0] not in excluded
    ]
    result = acquire.select_train_scenes(filtered, count)
    keys = [capture.episode_key(ep) for ep in result["episodes"]]
    if (len(keys) != count or len({k[0] for k in keys}) != count
            or any(k[0] in excluded for k in keys)):
        raise RuntimeError("Scene-disjoint selection contract violated")
    return result


def validate_selection(manifest):
    cap = manifest.get("task_evidence_capture", {})
    selection = manifest.get("train_scene_disjoint", {})
    if (cap.get("schema") != acquire.SCHEMA or cap.get("split") != "train"
            or selection.get("schema") != SCHEMA
            or selection.get("outcome_blind") is not True
            or selection.get("prior_scenes_excluded") is not True):
        raise ValueError("Incorrect scene-disjoint train manifest")
    excluded = selection.get("excluded_scene_stems")
    if not isinstance(excluded, list) or excluded != sorted(set(excluded)):
        raise ValueError("Exclusion set is not locked and sorted")
    keys = [capture.episode_key(ep) for ep in manifest["episodes"]]
    if (len(keys) != cap["episodes"]
            or len(keys) != len({k[0] for k in keys})
            or {k[0] for k in keys} & set(excluded)):
        raise ValueError("Selected scenes overlap inspected scenes")
    return keys


def prepare(args):
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Create-only train-source directory: " + str(root))
    excluded = set(INSPECTED)
    excluded.update(str(x) for x in args.exclude_scene)
    watched = []
    for run in args.exclude_source_run:
        path = Path(run).resolve() / "manifest.json"
        excluded.update(scene_stems_from_locked_run(run))
        watched.append(path)
    if any(not x.strip() for x in excluded):
        raise ValueError("Blank scene exclusion is invalid")
    # Guarded local patch for the original source-selection callback only;
    # restore unconditionally. Do not patch any native model/evaluator code.
    selector = acquire.select_train_scenes

    def selection(dataset, count):
        return select_outcome_blind_new_scenes(dataset, count, excluded)

    try:
        acquire.select_train_scenes = selection
        acquire.prepare(SimpleNamespace(
            run_dir=str(root), episodes=args.episodes, seed=args.seed,
            max_steps=args.max_steps,
        ))
    finally:
        acquire.select_train_scenes = selector
    manifest = pilot.read_json(root / "manifest.json")
    manifest["selection"] = (
        "First instruction-bearing R2R TRAIN episode per distinct scene in "
        "original source order AFTER subtracting predeclared previously "
        "inspected scene IDs; no outcomes, future motion or goal selection."
    )
    manifest["train_scene_disjoint"] = {
        "schema": SCHEMA, "excluded_scene_stems": sorted(excluded),
        "exclude_source_runs": [str(Path(r).resolve()) for r in args.exclude_source_run],
        "outcome_blind": True, "prior_scenes_excluded": True,
        "selected_keys": [list(capture.episode_key(ep)) for ep in manifest["episodes"]],
        "not_held_out_evaluation": True,
    }
    manifest["source_hashes"][str(Path(__file__).resolve())] = pilot.sha256(__file__)
    for path in watched:
        manifest["source_hashes"][str(path)] = pilot.sha256(path)
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    validate_selection(manifest)
    print("TRAIN_DISJOINT_PREPARE=PASS", args.episodes,
          "new TRAIN scenes; excluded", len(excluded),
          "old scenes; output:", root)


def audit(args):
    root = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(root)
    keys = validate_selection(manifest)
    result = acquire.report(SimpleNamespace(run_dir=str(root)))
    if (result["status"] != "TRAIN_NATIVE_DECISION_DATASET_READY"
            or result["N_episodes"] != len(keys)):
        raise RuntimeError("Incomplete/disjoint TRAIN native decision source")
    print(json.dumps({
        "schema": SCHEMA, "status": "TRAIN_DISJOINT_NATIVE_READY",
        "N_scene_disjoint_episodes": len(keys),
        "N_s2_calls": result["N_decisions"],
        "N_unique_rgb": result["N_image_assets"],
        "selected_keys": [list(k) for k in keys],
        "no_interventions": True, "no_outcomes_used_in_selection": True,
        "ready_for_outcome_blind_checkpoint_capture": True,
        "not_held_out_evaluation": True,
    }, indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    p = cmd.add_parser("prepare")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--exclude-scene", action="append", default=[])
    p.add_argument("--exclude-source-run", action="append", default=[])
    p.set_defaults(function=prepare)
    p = cmd.add_parser("audit")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(function=audit)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
