"""Create a scene-diverse *native* DualVLN CONTROL pilot before any GPU run.

Delegates to the unchanged, locked cache_pilot.prepare() for validation and
manifest generation, but replaces its source-order prefix selection with the
first episode of each unseen scene. This is process-local only. Original
pilot.py is left untouched so hashes of already-completed experiments remain
valid. No success labels, target positions, pathfinder, GPS or future
trajectories affect selection.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_pilot as pilot
import failure_case_probe as capture


def excluded_from_checkpoint(report_file):
    report = pilot.read_json(report_file)
    if report.get("schema") != "crosscase_orientation_checkpoints.v1":
        raise ValueError("The exclusion report must be the original locked orientation checkpoint")
    return frozenset(str(item["key"][0]) for item in report["cases"])


def select_scene_diverse(source_dataset, count, excluded_scenes):
    """First episode from each available, nonexcluded scene, in source order."""
    if not 1 <= int(count) <= 100:
        raise ValueError("count must be between 1 and 100")
    selected = []
    chosen_scenes = set()
    for episode in source_dataset["episodes"]:
        scene, episode_id = capture.episode_key(episode)
        if scene in excluded_scenes or scene in chosen_scenes:
            continue
        if not episode.get("instruction", {}).get("instruction_text", "").strip():
            continue
        chosen_scenes.add(scene)
        selected.append(episode)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            "Need %d new distinct scenes, found %d in the FULL source dataset; "
            "do not reuse inspected scenes or select based on success outcomes"
            % (count, len(selected))
        )
    subset = copy.deepcopy(source_dataset)
    subset["episodes"] = copy.deepcopy(selected)
    return subset


def prepare(args):
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Pilot is create-only: " + str(root))
    source_report = (Path(args.exclude_checkpoint_run).resolve()
                     / "checkpoint_report.json")
    excluded = excluded_from_checkpoint(source_report)
    original = pilot.select_episodes

    # The patch is confined to this process and restored regardless of result.
    # The unchanged pilot.prepare() handles file existence, source/model hashes,
    # config serialization, scene assets, episode shard and manifest creation.
    def scene_diverse_selection(dataset, count):
        return select_scene_diverse(dataset, count, excluded)

    try:
        pilot.select_episodes = scene_diverse_selection
        pilot.prepare(args)
    finally:
        pilot.select_episodes = original

    manifest_path = root / "manifest.json"
    manifest = pilot.read_json(manifest_path)
    manifest["selection"] = (
        "First episode per distinct unseen scene in full source annotation order, "
        "excluding scenes recorded in the locked original six-case checkpoint "
        "report; no outcomes or future navigation data used."
    )
    manifest["scene_diverse_selection"] = {
        "schema": "recovery_scene_diverse_pilot.v1",
        "excluded_scenes": sorted(excluded),
        "selected_scene_ids": [
            capture.episode_key(e)[0] for e in manifest["episodes"]
        ],
        "source_checkpoint_exclusion_report": str(source_report),
        "source_order": True,
        "success_blind": True,
    }
    # Adding watches only to this new pilot is safe: no previously locked
    # experiment manifest or source file is altered.
    manifest["source_hashes"][str(Path(__file__).resolve())] = pilot.sha256(__file__)
    manifest["source_hashes"][str(source_report)] = pilot.sha256(source_report)
    pilot.write_json(manifest_path, manifest)
    pilot.verify_manifest(root)
    print("SCENE_DIVERSE_PREPARE=PASS", len(manifest["episodes"]),
          "distinct unseen scenes; no native navigation executed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--exclude-checkpoint-run",
                        default="data/orientation_checkpoint_01")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--config", default="scripts/eval/configs/habitat_dual_system_cfg.py")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-steps", type=int, default=None)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
