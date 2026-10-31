"""v0.1 TRAIN dataset: replay and capture *natural* per-case frozen S2 gates.

This is stage B1 of intervention_selection_dataset.v0.1. It captures actual
native RGB/history and reproducible action/pose checkpoints for the 12
predeclared TRAIN scene-disjoint cases. It does NOT yet execute the three-turn
perturbation nor any of the four candidate arms.

Unlike the older independent checkpoint script, each case uses its OWN
predeclared minimum step from the locked v0.1 plan (20/40/60). The legacy
CheckpointProbe is reused read-only with a process-local eligibility hook.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_selection_dataset_v01 as ds
import recovery_independent_checkpoint as checkpoint

pilot, capture = ds.pilot, ds.capture
physical, common, cohort = checkpoint.physical, checkpoint.common, checkpoint.cohort
SCHEMA = "intervention_selection_native_gates.v0.1"
SCOPE = (
    "Twelve predeclared previously uninspected R2R TRAIN scenes. Each native "
    "episode is replayed unchanged to completion; its first eligible ordinary "
    "idle System-2 boundary is captured at the preallocated 20/40/60 threshold, "
    "provided 1m translation, 30 remaining steps and level RGB. Unreachable "
    "gates are recorded as INELIGIBLE without replacements. No actual physical "
    "perturbation, intervention outcomes, selector or held-out claims."
)


def per_case_eligibility(case, step, limit, body_count, translation, done,
                         action_seq, pixel_goal, local_actions, action,
                         camera_balance, camera_pitch):
    """Mirror old gate contract but read PREDECLARED case-specific tier."""
    rule = case["checkpoint_rule"]
    if rule["minimum_main_step"] not in ds.THRESHOLDS:
        raise ValueError("Unregistered temporal eligibility tier")
    remaining = int(limit) - int(step)
    checks = {
        "threshold_reached": int(step) >= rule["minimum_main_step"],
        "translated_enough": translation + 1e-8 >=
                             rule["minimum_prefix_translation_m"],
        "remaining_budget": remaining >=
                            rule["minimum_remaining_main_loop_steps"],
        "idle_s2": (not done and not action_seq and pixel_goal is None
                    and not local_actions and action in (0, 1, 2, 3)),
        "level_camera": (camera_balance == 0 and camera_pitch is not None
                         and abs(camera_pitch) <= 1e-3),
    }
    return {
        "checks": checks, "eligible": all(checks.values()),
        "remaining_native_main_steps": remaining,
    }



def native_episode_termination(key, rows, progress_row, native_limit):
    """Validate the evaluator's TWO native termination conditions.

    InternNav's native loop is:
      while (not done) and (step_id <= max_steps_per_episode)
    A final Habitat action may have done=False when the native main-loop
    budget is exhausted. Its completed progress row then has steps=limit+1.
    This does NOT waive trace/progress/runtime checks for incomplete episodes.
    """
    if not rows:
        raise ValueError("Empty native action trace for " + repr(key))
    if type(native_limit) is not int or native_limit < 1:
        raise ValueError("Invalid native main-loop limit")
    steps = progress_row.get("steps")
    if type(steps) is not int or not 1 <= steps <= native_limit + 1:
        raise ValueError(
            "Invalid native completed main-step count for "
            + repr(key) + ": " + repr(steps))
    done = rows[-1].get("done")
    if type(done) is not bool:
        raise ValueError("Native final done flag is not boolean for " + repr(key))
    if done:
        return "ENV_DONE"
    if steps == native_limit + 1:
        return "NATIVE_LOOP_BUDGET_EXHAUSTED"
    raise ValueError(
        "Incomplete native episode: final done=False without step budget "
        "exhaustion, key=" + repr(key) + ", steps=" + str(steps)
        + ", limit=" + str(native_limit))



def native_ready(source):
    source = Path(source).resolve()
    manifest = pilot.verify_manifest(source)
    ds.train.validate_selection(manifest)
    if len(manifest["episodes"]) != ds.N_CASES:
        raise ValueError("Expected original 12-scene native source")
    summary = pilot.read_json(source / "control/pilot_summary.json")
    capture_report = pilot.read_json(source / "capture/decision_report.json")
    if (summary["status"] != "complete"
            or capture_report.get("status") != "TRAIN_NATIVE_DECISION_DATASET_READY"
            or capture_report.get("N_episodes") != ds.N_CASES):
        raise ValueError("Complete native TRAIN System-2 recording required")
    progress = capture.read_progress(source / "control/progress.json")
    keys = {ds.key_of(ep) for ep in manifest["episodes"]}
    if set(progress) != keys:
        raise ValueError("Native progress episode mismatch")
    references = {key: [] for key in keys}
    for row in physical.validate_trace(
        physical.read_lines(source / "control/actions.jsonl"), "native source"
    ):
        key = capture.episode_key(row)
        if key not in references:
            raise ValueError("Unexpected native action episode")
        references[key].append(row)
    # The native evaluator can complete through environment done OR the
    # declared main-loop step limit, without a final Habitat done=True.
    # Require an actual completed progress row at limit+1 in the latter case.
    cfg = pilot.load_config(source / "control.py")
    limit = int(cfg.eval_settings["max_steps_per_episode"])
    for key in keys:
        native_episode_termination(key, references[key], progress[key], limit)
    if (int(summary["actual_action_count"]) !=
            sum(len(references[k]) for k in keys)
            or int(capture_report["N_action_records"]) !=
               int(summary["actual_action_count"])):
        raise ValueError("Native action count differs from completed source audits")
    return manifest, summary, progress, references


def locked_inputs(plan_root):
    root = Path(plan_root).resolve()
    manifest = pilot.verify_manifest(root)
    if (manifest["schema"] != ds.SCHEMA
            or manifest["status"] != "PRE_REGISTERED_NO_ROLLOUTS"
            or manifest["no_intervention_executed"] is not True):
        raise ValueError("Not the immutable v0.1 CPU-only planned cohort")
    native_root = Path(manifest["source_native_run"]).resolve()
    native_manifest, native_summary, progress, references = native_ready(native_root)
    plan = pilot.read_json(root / "preregistered_plan.json")
    ds.verify_plan(
        plan, native_manifest, manifest["seed"], manifest["native_limit"])
    if (manifest["episodes"] != native_manifest["episodes"]
            or set(references) != {tuple(c["episode_key"]) for c in plan["cases"]}):
        raise ValueError("v0.1 source/plan identity changed")
    return root, manifest, native_root, native_manifest, plan, native_summary, progress, references


def prepare(args):
    dest = Path(args.run_dir).resolve()
    if dest.exists():
        raise FileExistsError("Create-only natural checkpoint capture")
    plan_root, plan_m, source, source_m, plan, _, _, _ = locked_inputs(args.plan_dir)
    cfg = copy.deepcopy(pilot.load_config(source / "control.py"))
    if (cfg.agent.model_settings.get("mode") != "dual_system"
            or cfg.agent.model_settings.get("kv_cache_continuation", False)
            or int(cfg.eval_settings["max_steps_per_episode"]) !=
               plan_m["native_limit"]):
        raise ValueError("Require unchanged cache-disabled native model/config")
    cases = [
        {"key": c["episode_key"], "case_id": c["case_id"],
         "checkpoint_rule": c["checkpoint_rule"],
         "perturbation_plan": c["perturbation_plan"]}
        for c in plan["cases"]
    ]
    protocol = {
        "schema": SCHEMA, "status": "NATIVE_CAPTURE_PENDING",
        "plan_dir": str(plan_root), "native_source": str(source),
        "cases": cases, "no_replacements": True,
        "no_interventions": True, "scope": SCOPE,
    }
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    manifest = {
        "schema": SCHEMA, "seed_base": source_m["seed_base"],
        "revision": revision, "episodes": source_m["episodes"],
        "checkpoint_shard_sizes": source_m["checkpoint_shard_sizes"],
        "source_hashes": {
            **plan_m["source_hashes"], **plan_m["generated_hashes"],
        },
        "generated_hashes": {},
        "parent_plan": str(plan_root), "source_native_run": str(source),
        "no_perturbation_executed": True,
    }
    watched = [
        plan_root / "manifest.json", plan_root / "preregistered_plan.json",
        source / "manifest.json",
        source / "control/pilot_summary.json",
        source / "control/progress.json",
        source / "control/actions.jsonl",
        source / "capture/decision_report.json",
        Path(__file__).resolve(), Path(ds.__file__).resolve(),
        Path(checkpoint.__file__).resolve(), Path(physical.__file__).resolve(),
        Path(cohort.__file__).resolve(), Path(common.__file__).resolve(),
    ]
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watched})
    dest.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(cfg)
    config.eval_settings["output_path"] = str(dest / "control")
    config.agent.model_settings["vis_debug"] = True
    config.agent.model_settings["vis_debug_path"] = str(dest / "video" / "control")
    payload = json.dumps(config.model_dump(mode="json", exclude_none=True))
    path = dest / "control.py"
    path.write_text(
        "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
        + "eval_cfg = EvalCfg.model_validate(json.loads(" + repr(payload) + "))\n",
        encoding="utf-8",
    )
    ppath = dest / "gate_protocol.json"
    pilot.write_json(ppath, protocol)
    manifest["generated_hashes"] = {
        str(p): pilot.sha256(p) for p in (path, ppath)
    }
    pilot.write_json(dest / "manifest.json", manifest)
    pilot.verify_manifest(dest)
    print("V01_NATURAL_GATE_PREPARE=PASS", len(cases), "locked cases")
    print("GATE_CAPTURE_NOT_EXECUTED; ALL_PERTURBATIONS_PENDING")
    return protocol


def read_run(root):
    root = Path(root).resolve()
    manifest = pilot.verify_manifest(root)
    if (manifest.get("schema") != SCHEMA
            or manifest.get("no_perturbation_executed") is not True):
        raise ValueError("Not a locked untouched natural capture")
    plan_root, plan_m, source, source_m, plan, source_summary, progress, references = (
        locked_inputs(manifest["parent_plan"]))
    if source != Path(manifest["source_native_run"]).resolve():
        raise ValueError("Natural source root changed")
    cases = [
        {"key": c["episode_key"], "case_id": c["case_id"],
         "checkpoint_rule": c["checkpoint_rule"],
         "perturbation_plan": c["perturbation_plan"]}
        for c in plan["cases"]
    ]
    protocol = pilot.read_json(root / "gate_protocol.json")
    if (protocol != {
        "schema": SCHEMA, "status": "NATIVE_CAPTURE_PENDING",
        "plan_dir": str(plan_root), "native_source": str(source),
        "cases": cases, "no_replacements": True,
        "no_interventions": True, "scope": SCOPE,
    } or manifest["episodes"] != source_m["episodes"]
      or manifest["seed_base"] != source_m["seed_base"]):
        raise ValueError("Locked per-case natural threshold protocol changed")
    return manifest, source, protocol, source_summary, progress, references


def run(args):
    root = Path(args.run_dir).resolve()
    _, _, protocol, _, _, references = read_run(root)
    if (root / "control").exists():
        raise FileExistsError("Create-only natural replay; control already exists")
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    orig_init, orig_loop = cls.__init__, cls._run_eval_dual_system
    hook = physical.instrument_loop(orig_loop)
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("Unexpected model cache continuation")
        orig_init(evaluator, cfg)
        probe = checkpoint.CheckpointProbe(
            evaluator, root, protocol["cases"], references)
        probes.append(probe)
        evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record

    old_eligibility = checkpoint.eligibility
    cls.__init__, cls._run_eval_dual_system = initialize, hook
    checkpoint.eligibility = per_case_eligibility
    error = None
    try:
        pilot.run_arm(SimpleNamespace(
            run_dir=str(root), arm="control", profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Expected one native evaluator")
        probes[0].finish()
        if set(probes[0].finished) != {
            tuple(c["key"]) for c in protocol["cases"]
        }:
            raise RuntimeError("Incomplete natural checkpoint cohort")
        probes[0].status = "complete"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        checkpoint.eligibility = old_eligibility
        cls.__init__, cls._run_eval_dual_system = orig_init, orig_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.close(error)
    return report(args)


def report(args):
    root = Path(args.run_dir).resolve()
    _, source, protocol, baseline_summary, baseline_progress, references = read_run(root)
    run_summary = pilot.read_json(root / "control/pilot_summary.json")
    captured = pilot.read_json(root / "control/capture_summary.json")
    if run_summary["status"] != "complete" or captured["status"] != "complete":
        raise RuntimeError("Incomplete native checkpoint replay")
    common.same_fields(run_summary, baseline_summary, common.RUNTIME_FIELDS,
                       "natural checkpoint runtime")
    if captured["extra_env_actions"] or captured["orientation_interventions_executed"]:
        raise RuntimeError("Natural capture performed an external intervention")
    actual = capture.read_progress(root / "control/progress.json")
    observed = physical.validate_trace(
        physical.read_lines(root / "control/actions.jsonl"), "natural capture")
    tagged = physical.read_lines(root / "control/capture_actions.jsonl")
    if not pilot.compare_traces(observed, tagged)["actions_and_poses_match"]:
        raise RuntimeError("Native logger and checkpoint observer differ")
    expected_keys = {tuple(c["key"]) for c in protocol["cases"]}
    if set(actual) != expected_keys:
        raise RuntimeError("Wrong native episode set during gate capture")
    per_episode = {k: [] for k in expected_keys}
    for r in observed:
        k = capture.episode_key(r)
        if k not in per_episode:
            raise ValueError("Unexpected trace episode")
        per_episode[k].append(r)
    details = {tuple(d["key"]): d for d in captured["cases"]}
    events = physical.read_lines(root / "control/boundary_events.jsonl")
    if set(details) != expected_keys:
        raise RuntimeError("Missing natural checkpoint probe summary entries")
    rows = []
    for case in protocol["cases"]:
        key = tuple(case["key"])
        original, now = baseline_progress[key], actual[key]
        if (now["episode_instruction"] != original["episode_instruction"]
                or now["steps"] != original["steps"]
                or not pilot.metrics_match(
                    {k: now[k] for k in ("success", "spl", "os", "ne")},
                    {k: original[k] for k in ("success", "spl", "os", "ne")})):
            raise RuntimeError("Native baseline outcome differs " + repr(key))
        trace_compare = pilot.compare_traces(references[key], per_episode[key])
        if not trace_compare["actions_and_poses_match"]:
            raise RuntimeError("Native baseline action/pose diverged " + repr(key))
        detail = details[key]
        if not (detail["history_prefix_unchanged"]
                and detail["parameter_versions_unchanged"]):
            raise RuntimeError("Native checkpoint changed history/model")
        eligible = [
            x for x in events if tuple(x["key"]) == key and x["eligible"]
        ]
        gate = detail["checkpoint"]
        if gate:
            if (not eligible or any(
                    gate[k] != eligible[0][k] for k in eligible[0]
            )):
                raise RuntimeError("Checkpoint not first predeclared eligible boundary")
            folder = root / "control/checkpoints" / "_".join(key)
            if (pilot.read_json(folder / "checkpoint.json") != gate
                    or not (folder / "rgb.png").is_file()):
                raise RuntimeError("Natural gate metadata/RGB mismatch")
            ctx = pilot.read_json(folder / "context_inputs.json")
            if (ctx["schema"] != "independent_recovery_context_inputs.v1"
                    or ctx["key"] != list(key)
                    or ctx["main_step"] != gate["step"]
                    or ctx["history_sha256"] != gate["history_sha256"]
                    or pilot.sha256(folder / "current_model_rgb.png") !=
                       ctx["current_model_rgb_png_sha256"]):
                raise RuntimeError("Model-sized current RGB capture mismatch")
            for frame in ctx["sampled_history"]:
                file = folder / frame["path"]
                if (not file.is_file() or pilot.sha256(file) !=
                        frame["png_sha256"]):
                    raise RuntimeError("Saved historical model RGB changed")
            prefix = gate["prefix_action_count"]
            if (prefix < 1 or prefix >= len(references[key])
                    or gate["prefix_trace_sha256"] !=
                       cohort.digest(references[key][:prefix])):
                raise RuntimeError("Locked native action prefix digest mismatch")
            # Explicit online-only input record; never include native
            # final SR/NE, ground-truth pose or future actions here.
            pilot.write_json(folder / "predecision_input.json", {
                "schema": "intervention_selection_predecision.v0.1",
                "context_id": case["case_id"] + "_natural",
                "episode_key": list(key), "main_step": int(gate["step"]),
                "source_type": "natural",
                "instruction": now["episode_instruction"],
                "past_action_codes": [
                    int(r["action"]) for r in references[key][:prefix]
                ],
                "current_rgb": str(folder / "current_model_rgb.png"),
                "current_pixel_sha256":
                    ctx["current_model_rgb_pixel_sha256"],
                "current_png_sha256":
                    ctx["current_model_rgb_png_sha256"],
                "historical_rgb": [
                    {
                        "path": str(folder / h["path"]),
                        "native_history_index": h["native_history_index"],
                        "pixel_sha256": h["pixel_sha256"],
                        "png_sha256": h["png_sha256"],
                    } for h in ctx["sampled_history"]
                ],
                "historical_rgb_history_sha256": gate["history_sha256"],
                "perturbation_actions_before_context": [],
                "no_privileged_simulator_metrics": True,
            })
            status = "NATURAL_GATE_CAPTURED"
        else:
            if eligible or detail["status"] != "NOT_ELIGIBLE":
                raise RuntimeError("Skipped an available qualifying gate")
            status = "INELIGIBLE_RETAINED_NO_REPLACEMENT"
        rows.append({
            "case_id": case["case_id"], "episode_key": list(key),
            "checkpoint_rule": case["checkpoint_rule"],
            "perturbation_plan": case["perturbation_plan"],
            "status": status, "natural_checkpoint": gate,
            "baseline_native_success": int(now["success"]),
            "baseline_final_ne_m": float(now["ne"]),
            "native_termination": native_episode_termination(
                key, references[key], original,
                int(pilot.load_config(source / "control.py").
                    eval_settings["max_steps_per_episode"])),
            "full_native_trace_verified": True,
        })
    n_eligible = sum(c["natural_checkpoint"] is not None for c in rows)
    outcome = {
        "schema": SCHEMA, "status": (
            "NATURAL_GATES_AUDITED" if n_eligible == len(rows)
            else "NATURAL_GATES_AUDITED_WITH_INELIGIBLE"
        ),
        "N_cases_predeclared": len(rows), "N_captured_gates": n_eligible,
        "N_ineligible_retained": len(rows) - n_eligible,
        "N_interventions_executed": 0,
        "N_perturbed_contexts_created": 0,
        "N_paired_arm_outcomes": 0,
        "cases": rows, "scope": SCOPE,
    }
    pilot.write_json(root / "natural_checkpoint_report.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="CPU-only prepare natural replay")
    a.add_argument("--plan-dir", required=True)
    a.add_argument("--run-dir", required=True)
    a.set_defaults(fn=prepare)
    for name, fn in (("run", run), ("report", report)):
        a = sub.add_parser(name)
        a.add_argument("--run-dir", required=True)
        a.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
