"""CPU-only pre-registration of task-evidence intervention selection dataset v0.1.

Does NOT run the model, choose samples from outcomes, capture checkpoints,
construct simulator perturbations, label task progress, or produce action
utilities. Preparing creates an immutable, auditable protocol and 96 *planned*
candidate evaluations from 12 previously uninspected TRAIN-scene episodes.

Future execution must separately prove that EACH of four treatments for BOTH
natural and physically perturbed contexts started from exactly the same
pre-intervention simulator pose, observed RGB/history, RNG, policy parameters
and checkpoint; no resumable snapshot is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_train_disjoint_acquire as train

pilot, capture = train.pilot, train.capture
SCHEMA = "intervention_selection_dataset.v0.1"
ACTION_SET = ("KEEP", "SEEK_LEFT45", "SEEK_RIGHT45", "HISTORY_RESET")
N_CASES = 12
N_STATES = 24
N_EVALUATIONS = 96
THRESHOLDS = (20, 40, 60)
MIN_PREFIX_TRANSLATION_M = 1.0
MIN_REMAINING_MAIN_STEPS = 30
SEEK_SCAN_ACTIONS = 6
PERTURB_TURN_ACTIONS = 3
DEFAULT_SEED = 20261031
FORBIDDEN_ONLINE = (
    "distance_to_goal", "goal_position", "geodesic_distance",
    "shortest_path", "reference_path", "future_actions", "success",
    "oracle_success", "sim_position", "sim_orientation",
    "gps", "depth", "pathfinder", "intervention_outcomes",
)
SCOPE = (
    "R2R TRAIN engineering/development-only v0.1. Twelve preselected "
    "scene-disjoint episodes, one checkpoint per episode, 12 native plus "
    "12 predeclared physically perturbed contexts, each with KEEP, "
    "SEEK_LEFT45, SEEK_RIGHT45 and HISTORY_RESET potential outcomes. "
    "No post-outcome filtering, no held-out evaluation, no ground-truth "
    "goal/path/GPS/depth selector features, no online task-evidence labels "
    "or candidate outcomes provided as model inputs. This stage is CPU "
    "pre-registration only: neither checkpoints nor intervention results "
    "are present."
)


def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def key_of(ep):
    if "scene_id" not in ep or "episode_id" not in ep:
        raise ValueError("Episode identity requires both scene_id and episode_id")
    return capture.episode_key(ep)


def source_episodes(native_source, previously_inspected):
    source = Path(native_source).resolve()
    manifest = pilot.verify_manifest(source)
    keys = train.validate_selection(manifest)
    if (len(keys) != N_CASES
            or manifest["task_evidence_capture"]["no_intervention"] is not True
            or manifest["train_scene_disjoint"]["outcome_blind"] is not True):
        raise ValueError("Require a fresh, outcome-blind, 12-scene native TRAIN source")
    # The exclusion argument is mandatory and is verified against the source
    # selection. Neither native progress nor future action count is inspected.
    if not previously_inspected:
        raise ValueError("At least one frozen prior source-run exclusion is mandatory")
    old_scenes = set(train.INSPECTED)
    prior_paths = []
    for ref in previously_inspected:
        old = Path(ref).resolve()
        if old == source:
            raise ValueError("Cannot exclude the new source from itself")
        prior_paths.append(old)
        old_scenes.update(train.scene_stems_from_locked_run(old))
    if len(set(prior_paths)) != len(prior_paths):
        raise ValueError("Repeated prior source run")
    selection = manifest["train_scene_disjoint"]
    declared = set(selection["excluded_scene_stems"])
    source_declared_prior = {
        str(Path(x).resolve()) for x in selection["exclude_source_runs"]
    }
    if {str(x) for x in prior_paths} - source_declared_prior:
        raise ValueError("Prior exclusion was not registered in native source manifest")
    if not old_scenes.issubset(declared):
        raise ValueError("Native source was not selected excluding the declared old scenes")
    if set(k[0] for k in keys) & old_scenes:
        raise ValueError("Previously inspected scene leaked into v0.1 cohort")
    expected = [key_of(ep) for ep in manifest["episodes"]]
    if keys != expected or len({k[0] for k in keys}) != N_CASES:
        raise ValueError("Scene-disjoint source identity/order changed")
    return manifest, expected, prior_paths


def build_plan(source_manifest, seed, native_limit):
    """Predeclare tiers and perturbations using only episode ID and seed.

    Sort a cryptographic hash of episode identity and seed to remove source
    annotation-order bias. Allocate four to each step tier and two left/two
    right perturbations per tier before any model outcome is observed.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not isinstance(native_limit, int) or native_limit < 100:
        raise ValueError("Native main-step limit insufficient for all tiers")
    eps = source_manifest["episodes"]
    if len(eps) != N_CASES:
        raise ValueError("Exactly 12 source episodes required")
    keys = [key_of(ep) for ep in eps]
    if len(set(keys)) != N_CASES or len({k[0] for k in keys}) != N_CASES:
        raise ValueError("One episode per distinct scene required")
    ranked = sorted(
        range(N_CASES),
        key=lambda i: (digest(f"{seed}|{keys[i][0]}|{keys[i][1]}"), keys[i]),
    )
    allocation = {}
    for rank, index in enumerate(ranked):
        tier = THRESHOLDS[rank // 4]
        yaw = 45 if (rank % 4) < 2 else -45
        allocation[index] = (tier, yaw, rank)
    cases, contexts, evaluations = [], [], []
    for index, ep in enumerate(eps):
        key = keys[index]
        tier, yaw, rank = allocation[index]
        short_id = digest(f"{key[0]}|{key[1]}")[:20]
        case_id = "ep_" + short_id
        case = {
            "case_id": case_id,
            "episode_key": list(key),
            "scene_group": key[0],
            "source_episode_index": index,
            "allocation_rank": rank,
            "checkpoint_rule": {
                "boundary": "first idle ordinary System-2 decision at/after threshold",
                "minimum_main_step": tier,
                "minimum_prefix_translation_m": MIN_PREFIX_TRANSLATION_M,
                "minimum_remaining_main_loop_steps": MIN_REMAINING_MAIN_STEPS,
                "camera_level_required": True,
                "no_future_baseline_length": True,
                "if_no_eligible_boundary": "INELIGIBLE_RETAINED_NO_REPLACEMENT",
            },
            "perturbation_plan": {
                "kind": "body_yaw_from_natural_checkpoint",
                "yaw_deg": yaw,
                "actual_native_yaw_actions": (
                    [2] * PERTURB_TURN_ACTIONS if yaw == 45
                    else [3] * PERTURB_TURN_ACTIONS
                ),
                "predecision_only": True,
                "observation_rule": "capture actual post-turn RGB as current view",
                "history_rule": (
                    "keep pre-gate historical RGB, replace active current "
                    "observation with post-turn RGB; do not append a new "
                    "ordinary S2 history step or generate S2 mid-perturbation"
                ),
                "preserve_policy_rng": True,
                "cost_accounting": "charge three physical yaw actions before all four arms",
            },
            "status": "CHECKPOINT_NOT_CAPTURED",
        }
        cases.append(case)
        for source_type in ("natural", "perturbed"):
            context_id = case_id + "_" + source_type
            contexts.append({
                "context_id": context_id,
                "case_id": case_id,
                "episode_key": list(key),
                "scene_group": key[0],
                "source_type": source_type,
                "candidate_actions": list(ACTION_SET),
                "state_capture": "PENDING_ACTUAL_SIMULATOR_REPLAY",
                "predecision_rgb_history": "NOT_CAPTURED",
                "annotation_status": "NOT_LABELED",
                "outcome_status": "NOT_EXECUTED",
                "perturbation_yaw_deg": yaw if source_type == "perturbed" else 0,
                "same_state_all_arms_required": True,
            })
            for action in ACTION_SET:
                evaluations.append({
                    "context_id": context_id,
                    "action": action,
                    "status": "PLANNED_NO_ROLLOUT",
                    "additional_yaw_actions": (
                        SEEK_SCAN_ACTIONS if action.startswith("SEEK_") else 0
                    ),
                    "history_policy": (
                        "post-gate_indices_only" if action == "HISTORY_RESET"
                        else "native_history"
                    ),
                    "require_identical_preintervention_input_hash": True,
                    "require_identical_policy_rng_and_parameters": True,
                    "outcome": None,
                })
    if (len(cases) != N_CASES or len(contexts) != N_STATES
            or len(evaluations) != N_EVALUATIONS
            or len({r["context_id"] for r in contexts}) != N_STATES):
        raise RuntimeError("Dataset v0.1 cardinality violated")
    return {
        "schema": SCHEMA,
        "status": "PRE_REGISTERED_NO_ROLLOUTS",
        "split": "train_development_only",
        "seed": seed,
        "sampling": (
            "First selected unique TRAIN episode per new scene; deterministic "
            "hash-ranked tier + perturb-direction assignment; no outcome "
            "or future navigation used, no replacement for ineligible gates"
        ),
        "action_set": list(ACTION_SET),
        "primary_sample_unit": "independent_scene_episode_checkpoint",
        "contexts_per_checkpoint": 2,
        "arms_per_context": len(ACTION_SET),
        "native_control_reuse": (
            "natural KEEP from verified native capture only after its "
            "first-eligible predeclared checkpoint is reached; "
            "perturbed KEEP always requires a new full replay"
        ),
        "planned_new_continuations_after_native": 84,
        "checkpoint_capture_status": "NOT_EXECUTED",
        "paired_intervention_capture_status": "NOT_EXECUTED",
        "training_features_available": False,
        "training_labels_available": False,
        "online_allowed_inputs": [
            "instruction", "current_rgb", "past_rgb", "past_actual_actions",
            "past_temporal_indices", "predecision_automatic_evidence_only",
        ],
        "online_forbidden_inputs": list(FORBIDDEN_ONLINE),
        "evidence_review_policy": (
            "annotate using pre-intervention images/actions; preserve unknown; "
            "human/LLM draft must be independently reviewed before gold use"
        ),
        "paired_gate": (
            "For each context, replay frozen prefix (+ exact yaw if "
            "perturbed) before ALL four arms; compare pose/observation RGB/"
            "history/RNG/weights; same budget accounting and locked S2 gate"
        ),
        "data_partitions": (
            "This is a TRAIN development pilot; held-out evaluation must "
            "come later from completely new scene groups"
        ),
        "scope": SCOPE,
        "cases": cases,
        "contexts": contexts,
        "evaluations": evaluations,
    }


def verify_plan(plan, source_manifest, seed, native_limit):
    if plan != build_plan(source_manifest, seed, native_limit):
        raise ValueError("Pre-registered dataset plan changed")
    allowed = {
        "case_id", "episode_key", "scene_group", "source_episode_index",
        "allocation_rank", "checkpoint_rule", "perturbation_plan", "status",
    }
    for row in plan["cases"]:
        if set(row) != allowed:
            raise ValueError("Unexpected privileged/outcome fields in cases")
    for item in plan["evaluations"]:
        if item["status"] != "PLANNED_NO_ROLLOUT" or item["outcome"] is not None:
            raise ValueError("Plan must not contain invented intervention results")
    return True


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(args):
    out = Path(args.run_dir).resolve()
    if out.exists():
        raise FileExistsError("Create-only preregistration directory")
    source = Path(args.native_source).resolve()
    native, keys, prior = source_episodes(source, args.exclude_source_run)
    cfg = pilot.load_config(source / "control.py")
    limit = int(cfg.eval_settings["max_steps_per_episode"])
    plan = build_plan(native, args.seed, limit)
    verify_plan(plan, native, args.seed, limit)
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    manifest = {
        "schema": SCHEMA, "status": plan["status"],
        "revision": revision,
        "source_native_run": str(source),
        "previously_inspected_source_runs": [str(p) for p in prior],
        "seed": args.seed,
        "native_limit": limit,
        "episodes": native["episodes"],
        "source_hashes": {**native["source_hashes"], **native["generated_hashes"]},
        "generated_hashes": {},
        "checkpoint_shard_sizes": native["checkpoint_shard_sizes"],
        "no_intervention_executed": True,
    }
    watched = [
        source / "manifest.json",
        Path(__file__).resolve(),
        Path(train.__file__).resolve(),
        Path(train.acquire.__file__).resolve(),
    ]
    watched += [p / "manifest.json" for p in prior]
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watched})
    out.mkdir(parents=True, exist_ok=False)
    planned_file = out / "preregistered_plan.json"
    write_json(planned_file, plan)
    manifest["generated_hashes"][str(planned_file)] = pilot.sha256(planned_file)
    manifest_file = out / "manifest.json"
    write_json(manifest_file, manifest)
    pilot.verify_manifest(out)
    print(
        "INTERVENTION_DATASET_V01_PREPARE=PASS",
        f"{len(plan['cases'])} episodes",
        f"{len(plan['contexts'])} contexts",
        f"{len(plan['evaluations'])} planned potential outcomes",
    )
    print("DATASET_STATUS=PRE_REGISTERED_NO_ROLLOUTS; ALL_LABELS_PENDING")
    return plan


def audit(args):
    out = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(out)
    if (manifest.get("schema") != SCHEMA
            or manifest.get("status") != "PRE_REGISTERED_NO_ROLLOUTS"
            or manifest.get("no_intervention_executed") is not True):
        raise ValueError("Not an untouched v0.1 CPU preregistration")
    source = Path(manifest["source_native_run"]).resolve()
    native, keys, prior = source_episodes(
        source, manifest["previously_inspected_source_runs"])
    if native["episodes"] != manifest["episodes"]:
        raise ValueError("Native episode keys/order changed")
    cfg = pilot.load_config(source / "control.py")
    if int(cfg.eval_settings["max_steps_per_episode"]) != manifest["native_limit"]:
        raise ValueError("Native main-loop budget changed")
    saved = pilot.read_json(out / "preregistered_plan.json")
    verify_plan(saved, native, manifest["seed"], manifest["native_limit"])
    counts = {str(t): 0 for t in THRESHOLDS}
    directions = {str(t): {"+45": 0, "-45": 0} for t in THRESHOLDS}
    for case in saved["cases"]:
        step = str(case["checkpoint_rule"]["minimum_main_step"])
        yaw = str(case["perturbation_plan"]["yaw_deg"])
        counts[step] += 1
        directions[step][("+45" if yaw == "45" else "-45")] += 1
    report = {
        "schema": SCHEMA,
        "status": "CPU_PLAN_AUDITED_NO_ROLLOUTS",
        "N_scene_disjoint_cases": len(keys),
        "N_contexts": len(saved["contexts"]),
        "N_planned_arm_evaluations": len(saved["evaluations"]),
        "threshold_counts": counts, "perturbation_yaw_by_tier": directions,
        "N_native_controls_reusable_if_completed": N_CASES,
        "N_new_continuations_after_native": 84,
        "outcomes_available": False,
        "training_examples_available": False,
        "not_held_out_evaluation": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    a = s.add_parser("prepare", help="CPU-only pre-register 12x2x4 plan")
    a.add_argument("--native-source", required=True)
    a.add_argument("--exclude-source-run", action="append", required=True)
    a.add_argument("--run-dir", required=True)
    a.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a.set_defaults(fn=prepare)
    a = s.add_parser("audit", help="Rebuild predeclared assignments and compare all hashes")
    a.add_argument("--run-dir", required=True)
    a.set_defaults(fn=audit)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
