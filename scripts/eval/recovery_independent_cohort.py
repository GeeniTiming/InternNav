"""Select never-inspected route groups from a completed native control pilot.

The selection uses source episode order, scene / route aliases, and explicit
pre-existing exclusions only. No future native success labels, intervention
outcomes or ground-truth goal distances influence *which* episodes are chosen.

This is acquisition / exploratory development, not an independent test set.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orientation_cohort as old_cohort

pilot = old_cohort.pilot
physical = old_cohort.physical
capture = old_cohort.capture
digest = old_cohort.digest
route_groups = old_cohort.route_groups

CHECKPOINT_RULE = {
    "minimum_main_step": 20,
    "minimum_prefix_translation_m": 1.0,
    "minimum_remaining_main_loop_steps": 30,
    "boundary": "first native idle System-2 decision at/after thresholds, RGB sensor level",
    "future_baseline_length_used": False,
}
SCOPE = ("New never-inspected scene-disjoint acquisition pilot. Selection uses native "
         "dataset order, pre-existing scene exclusion list and route aliases only. "
         "The first eligible idle System-2 boundary uses online counters; no "
         "future native body-action count. Frozen policy, no learned detector.")


def excluded_scenes_from_report(path):
    report = pilot.read_json(Path(path) / "checkpoint_report.json")
    if report.get("schema") != "crosscase_orientation_checkpoints.v1":
        raise ValueError("Exclusion must be the original locked checkpoint report")
    return sorted({str(c["key"][0]) for c in report["cases"]})


def select_plan(dataset, progress, traces, excluded_scenes=(), maximum_cases=6):
    if not 1 <= int(maximum_cases) <= 100:
        raise ValueError("Maximum case count out of range")
    episodes = dataset["episodes"]
    keys = [capture.episode_key(ep) for ep in episodes]
    if len(keys) != len(set(keys)) or set(keys) != set(progress) or set(keys) != set(traces):
        raise ValueError("Native episode/progress/trace key set mismatch")
    for ep, key in zip(episodes, keys):
        row = progress[key]
        if row["episode_instruction"] != ep["instruction"]["instruction_text"]:
            raise ValueError("Source instruction differs")
        if float(row["success"]) not in (0.0, 1.0):
            raise ValueError("Nonbinary native success")
        physical.validate_trace(traces[key], repr(key))
        if not traces[key] or not traces[key][-1]["done"]:
            raise ValueError("Native baseline did not finish " + repr(key))
    excluded = set(excluded_scenes)
    groups = route_groups(episodes)
    selected, seen_scenes, skipped = [], set(), []
    for group in groups:
        i = group["indices"][0]
        scene = keys[i][0]
        if scene in excluded:
            skipped.append({"scene": scene, "reason": "previously_inspected_scene"})
            continue
        if scene in seen_scenes:
            skipped.append({"scene": scene, "reason": "one_route_group_per_scene"})
            continue
        ep, key = episodes[i], keys[i]
        seen_scenes.add(scene)
        native = progress[key]
        selected.append({
            "key": list(key), "route_group": group["route_group"],
            "stratum": "unstratified_native_order",
            "source_annotation_order": i, "scene_id": ep["scene_id"],
            "episode_id": str(ep["episode_id"]),
            "trajectory_id": ep.get("trajectory_id"),
            "route_member_keys": [list(keys[j]) for j in group["indices"]],
            "instruction": ep["instruction"]["instruction_text"],
            "native_metrics": {k: native[k] for k in ("success", "spl", "os", "ne")},
            "native_actual_action_count": len(traces[key]),
            "native_nonstop_body_actions": sum(row["action"] in (1, 2, 3) for row in traces[key]),
            "checkpoint_body_threshold": 0,
            "checkpoint_status": "PENDING_NATIVE_REPLAY",
            "baseline_trace_sha256": digest(traces[key]),
            "order": len(selected),
        })
        if len(selected) == maximum_cases:
            break
    # Fail before a heavy GPU capture if the source pilot has too little new variety.
    if len(selected) != maximum_cases:
        raise ValueError(
            "Need %d unseen-scene route groups, found %d; prepare a larger native "
            "pilot / another data split (never re-include inspected scenes)"
            % (maximum_cases, len(selected)))
    return {
        "schema": "independent_recovery_cohort_plan.v1",
        "status": "LOCKED_SELECTION_PENDING_CHECKPOINT_CAPTURE",
        "split": "acquisition_development",
        "source_native_episodes": len(episodes),
        "native_successes": sum(float(progress[k]["success"]) == 1 for k in keys),
        "native_failures": sum(float(progress[k]["success"]) == 0 for k in keys),
        "excluded_scenes": sorted(excluded),
        "available_route_groups": len(groups),
        "selected_counts": {"unstratified": len(selected)},
        "maximum_cases": maximum_cases,
        "selection_rule": "First qualifying route group in source order per unseen scene; "
                          "never condition selection on native SR, intervention labels or future distances.",
        "checkpoint_rule": CHECKPOINT_RULE,
        "cases": selected,
        "navigation_executed": False,
        "scope": SCOPE,
    }


def prepare(args):
    source, root = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    excluded_path = Path(args.exclude_checkpoint_run).resolve()
    if root.exists():
        raise FileExistsError("Cohort is create-only; choose a new --run-dir")
    old = pilot.verify_manifest(source)
    summary = pilot.read_json(source / "control/pilot_summary.json")
    dataset = pilot.read_json(source / "episodes.json.gz")
    if (summary["status"] != "complete" or summary["stage_synchronized"]
            or summary["seed_base"] != old["seed_base"]
            or len(dataset["episodes"]) != int(summary["metrics"]["length"])):
        raise ValueError("Require a complete, unsynchronized native CONTROL pilot")
    cfg = pilot.load_config(source / "control.py")
    if (cfg.eval_type != "habitat_vln"
            or cfg.agent.model_settings.get("mode") != "dual_system"
            or cfg.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Only frozen native cache-disabled DualVLN is supported")
    habitat_path = (pilot.ROOT / cfg.env.env_settings["config_path"]).resolve()
    import yaml
    habitat = yaml.safe_load(habitat_path.read_text(encoding="utf-8"))
    split = habitat["habitat"]["dataset"]["split"]
    if split not in ("train", "val_unseen", "val_seen"):
        raise ValueError("Unsupported Habitat dataset split: " + repr(split))
    progress = capture.read_progress(source / "control/progress.json")
    traces = {capture.episode_key(ep): [] for ep in dataset["episodes"]}
    for row in physical.read_lines(source / "control/actions.jsonl"):
        key = capture.episode_key(row)
        if key not in traces:
            raise ValueError("Unknown source action episode")
        traces[key].append(row)
    excluded = excluded_scenes_from_report(excluded_path)
    plan = select_plan(dataset, progress, traces, excluded, args.cases)
    plan["source_run"] = str(source)
    plan["exclude_checkpoint_run"] = str(excluded_path)
    plan["source_split"] = split
    root.mkdir(parents=True, exist_ok=False)
    plan_path = root / "plan.json"
    pilot.write_json(plan_path, plan)
    watched = [
        source / "manifest.json", source / "control/pilot_summary.json",
        source / "control/progress.json", source / "control/actions.jsonl",
        source / "control.py", source / "episodes.json.gz", habitat_path,
        excluded_path / "checkpoint_report.json",
        Path(__file__).resolve(), Path(old_cohort.__file__).resolve(),
        Path(pilot.__file__).resolve(), Path(physical.__file__).resolve(),
        Path(capture.__file__).resolve(), plan_path,
    ]
    hashes = dict(old["source_hashes"])
    hashes.update(old["generated_hashes"])
    hashes.update({str(p): pilot.sha256(p) for p in watched})
    pilot.write_json(root / "cohort_manifest.json", {
        "schema": "independent_recovery_cohort_inputs.v1",
        "hashes": hashes,
        "checkpoint_shard_sizes": old["checkpoint_shard_sizes"],
        "scope": SCOPE,
    })
    print("INDEPENDENT_COHORT_PREPARE=PASS", len(plan["cases"]),
          "scenes; output:", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--exclude-checkpoint-run",
                        default="data/orientation_checkpoint_01")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cases", type=int, default=6)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
