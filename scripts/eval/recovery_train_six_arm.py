"""Six-scene TRAIN paired intervention pilot (five NEW arms, native KEEP reused).

This is an engineering/development intervention experiment, NOT an online
selector, robust recovery result, task-aware SEEK, validated evidence label
dataset, or independently held-out efficacy estimate.

KEEP is the exact frozen native control.
RESET: no extra motion; post-checkpoint System-2 history indices only.
PLACEBO_KEEP and SEEK_KEEP: real 3x15-degree right then 3x15-degree left
environment steps, identical physical scan and supplementary image token.
PLACEBO_RESET and SEEK_RESET: identical scan+input, with post-gate history
selection reset. In PLACEBO the supplementary view is a duplicate current
image; in SEEK it is actual right45 outward RGB.

Before GPU: validate and lock pre-intervention evidence annotations and every
underlying source/checkpoint/config/script hash. The evidence labels are
PRELIMINARY (needs_review), never considered gold.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_intervention_choice_pilot as choice
import recovery_train_checkpoint_evidence as evidence

iso, prior = choice.iso, choice.iso.prior
pilot, physical, capture, common = iso.pilot, iso.physical, iso.capture, iso.common
pose = iso.pose
SCHEMA = "train_six_arm_intervention.v1"
ARMS = (
    "history_reset",
    "lookback_right45_placebo_keep",
    "lookback_right45_evidence_keep",
    "lookback_right45_placebo_reset",
    "lookback_right45_evidence_reset",
)
SCANS = frozenset(ARMS[1:])
RESETS = frozenset(("history_reset", ARMS[3], ARMS[4]))
VIEW = {
    ARMS[1]: "lookback_right45_placebo",
    ARMS[2]: "lookback_right45_evidence",
    ARMS[3]: "lookback_right45_placebo",
    ARMS[4]: "lookback_right45_evidence",
}
COMPARISONS = {
    "keep": "checkpoint/control",
    "reset": ARMS[0],
    "placebo_keep": ARMS[1],
    "seek_keep": ARMS[2],
    "placebo_reset": ARMS[3],
    "seek_reset": ARMS[4],
}
GATE_FIELDS = (
    "step", "prefix_action_count", "state", "observation_hashes",
    "history_length", "history_sha256", "rng_sha256",
    "parameters_sha256", "prefix_trace_sha256",
)
SCOPE = (
    "Six predeclared R2R TRAIN source-order scenes with native success=1. "
    "One first-eligible System-2 checkpoint per scene; four fixed right45 "
    "outbound-and-return real scans charge six actions, reset is sampled "
    "history index filtering (NOT persistent model/KV memory erasure). "
    "Task evidence labels are pre-intervention preliminary only. Cannot "
    "estimate failure rescue, held-out generalization or learned selector."
)


def check_source(checkpoint_run, annotation_dir):
    """Fail closed on checkpoint / locked annotation provenance, without edits."""
    source = Path(checkpoint_run).resolve()
    ann = Path(annotation_dir).resolve()
    # The older independent checkpoint audit is read-only here.
    old, captured = iso.turn.source_check(source)
    if (captured.get("schema") != "independent_recovery_checkpoints.v1"
            or captured.get("status") != "CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION"
            or len(captured["cases"]) != 6
            or any(c["checkpoint"] is None for c in captured["cases"])):
        raise ValueError("Require six fully comparable predeclared checkpoints")
    source_run = Path(old["orientation_checkpoint"]["source_run"]).resolve()
    native_manifest = pilot.verify_manifest(source_run)
    evidence.train.validate_selection(native_manifest)
    # The independent cohort must have locked TRAIN before interventions.
    cohort_root = Path(old["orientation_checkpoint"]["cohort_dir"]).resolve()
    plan = pilot.read_json(cohort_root / "plan.json")
    if (plan.get("source_split") != "train"
            or plan.get("maximum_cases") != 6
            or plan.get("cases") is None
            or [c["key"] for c in plan["cases"]] !=
                [c["key"] for c in captured["cases"]]):
        raise ValueError("Six-case TRAIN cohort identity has changed")
    native = capture.read_progress(source / "control/progress.json")
    keys = {tuple(c["key"]) for c in captured["cases"]}
    if set(native) != keys or any(native[k]["success"] != 1 for k in keys):
        raise ValueError("This pilot is predeclared for six native-SUCCESS scenes only")
    # A validation PASS proves structure, not human review correctness.
    with contextlib.redirect_stdout(io.StringIO()):
        evidence.validate(SimpleNamespace(annotation_dir=str(ann)))
    source_annotation_manifest = pilot.read_json(ann / "annotation_manifest.json")
    if (Path(source_annotation_manifest["checkpoint_run"]).resolve() != source
            or source_annotation_manifest["count"] != 6):
        raise ValueError("Evidence review does not reference this exact checkpoint run")
    ann_rows = [
        json.loads(line) for line in (ann / "annotations.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    if ([r["episode_key"] for r in ann_rows] !=
            [r["key"] for r in captured["cases"]]):
        raise ValueError("Evidence labels not aligned to the six checkpoint keys")
    if any(r["annotation"]["status"] == "unreviewed" for r in ann_rows):
        raise ValueError("Pre-intervention annotation review has not even begun")
    if any(r["annotation"]["status"] not in ("needs_review", "complete")
           for r in ann_rows):
        raise ValueError("Unexpected preliminary review state")
    return old, captured, native, source_run, ann_rows


def build_protocol(cases, native_limit):
    if len(cases) != 6 or any(not c["checkpoint"] for c in cases):
        raise ValueError("Exactly six captured checkpoints required")
    if native_limit < iso.SCAN_ACTIONS + 1:
        raise ValueError("No scan action budget")
    protocol = {
        "schema": SCHEMA, "cases": cases,
        "arms": list(ARMS), "label_mapping": COMPARISONS,
        "scan_angles": dict.fromkeys(SCANS, -45),
        "scan_actions": iso.SCAN_ACTIONS,
        "native_main_step_limit": native_limit,
        "original_main_loop_limit": native_limit,
        "charged_main_loop_limit": native_limit - iso.SCAN_ACTIONS,
        "charged_main_step_limit": native_limit - iso.SCAN_ACTIONS,
        "charged_native_main_step_limit": native_limit - iso.SCAN_ACTIONS,
        "forced_native_actions": iso.SCAN_ACTIONS,
        "scan_view_token": iso.EXTRA_PROMPT,
        "predeclared_control": "reused exactly from checkpoint/control",
        "split": "train_development_only",
        "frozen_scan": "right45 fixed, NOT task-driven",
        "no_online_selector": True, "no_new_training": True,
        "scope": SCOPE,
    }
    if any(c["checkpoint"]["step"] + iso.SCAN_ACTIONS >= native_limit for c in cases):
        raise ValueError("Insufficient charged main-loop budget at checkpoint")
    iso.verify_inherited_runtime_contract(protocol)
    return protocol


def prepare(args):
    root = Path(args.run_dir).resolve()
    source = Path(args.checkpoint_run).resolve()
    annotations = Path(args.annotation_dir).resolve()
    if root.exists():
        raise FileExistsError("Run directory already exists, create-only")
    old, captured, native, _, labels = check_source(source, annotations)
    cfg = pilot.load_config(source / "control.py")
    if (cfg.agent.model_settings.get("kv_cache_continuation", False)
            or cfg.agent.model_settings.get("mode") != "dual_system"):
        raise ValueError("Frozen cache-disabled DualVLN required")
    protocol = build_protocol(captured["cases"],
                              int(cfg.eval_settings["max_steps_per_episode"]))
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    manifest = {
        "schema": SCHEMA, "seed_base": old["seed_base"],
        "episodes": old["episodes"],
        "checkpoint_shard_sizes": dict(old["checkpoint_shard_sizes"]),
        "source_hashes": {**old["source_hashes"], **old["generated_hashes"]},
        "generated_hashes": {},
        "revision": revision,
        "train_six_arm": {
            "source_checkpoint_run": str(source),
            "annotation_dir": str(annotations),
            "preintervention_statuses": [r["annotation"]["status"] for r in labels],
            "scope": SCOPE,
        },
    }
    watched = [
        source / "manifest.json", source / "checkpoint_report.json",
        annotations / "annotation_manifest.json",
        annotations / "annotations.jsonl",
        Path(__file__).resolve(), Path(choice.__file__).resolve(),
        Path(evidence.__file__).resolve(), Path(iso.__file__).resolve(),
        Path(prior.__file__).resolve(), Path(iso.turn.__file__).resolve(),
        Path(iso.checkpoint.__file__).resolve(), Path(physical.__file__).resolve(),
        Path(pose.__file__).resolve(), Path(common.__file__).resolve(),
    ]
    watched += [p for p in (source / "control").rglob("*")
                if p.is_file() and p.suffix in (".json", ".jsonl", ".png")]
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watched})
    root.mkdir(parents=True, exist_ok=False)
    path = root / "six_arm_protocol.json"
    pilot.write_json(path, protocol)
    manifest["generated_hashes"][str(path)] = pilot.sha256(path)
    for arm in ARMS:
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.eval_settings["output_path"] = str(root / arm)
        arm_cfg.agent.model_settings["vis_debug"] = True
        arm_cfg.agent.model_settings["vis_debug_path"] = str(root / "video" / arm)
        path = root / (arm + ".py")
        serialized = json.dumps(arm_cfg.model_dump(mode="json", exclude_none=True))
        path.write_text(
            "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
            + "eval_cfg = EvalCfg.model_validate(json.loads("
            + repr(serialized) + "))\n", encoding="utf-8")
        manifest["generated_hashes"][str(path)] = pilot.sha256(path)
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    print("TRAIN_SIX_ARM_PREPARE=PASS",
          len(captured["cases"]), "contexts;",
          len(ARMS), "new arms = 30 rollouts; KEEP reused")
    print("PREINTERVENTION_EVIDENCE=LOCKED_PRELIMINARY_NOT_GOLD")
    return protocol


def read_run(root):
    manifest = pilot.verify_manifest(root)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Wrong TRAIN paired intervention manifest")
    linked = manifest["train_six_arm"]
    source = Path(linked["source_checkpoint_run"]).resolve()
    annotation = Path(linked["annotation_dir"]).resolve()
    _, captured, _, _, labels = check_source(source, annotation)
    protocol = pilot.read_json(root / "six_arm_protocol.json")
    limit = int(pilot.load_config(source / "control.py").eval_settings[
        "max_steps_per_episode"])
    if (protocol != build_protocol(captured["cases"], limit)
            or manifest["episodes"] !=
            pilot.verify_manifest(source)["episodes"]
            or linked["preintervention_statuses"] !=
                [r["annotation"]["status"] for r in labels]):
        raise ValueError("Locked six-arm protocol drift")
    return source, annotation, protocol


# Canonical scan names are only an internal behavior selector. All saved
# action streams and diagnostic files must remain under the NEW arm directory.
# Importantly, TurnProbe.__init__ opens the per-arm action file immediately:
# pass the actual new arm name to its constructor, not the canonical alias.
class ScanKeepProbe(iso.MechanismProbe):
    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.output_arm = arm
        self.arm = VIEW[arm]


class ScanResetProbe(choice.JointProbe):
    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.output_arm = arm
        self.arm = VIEW[arm]


def run(args):
    root = Path(args.run_dir).resolve()
    source, annotation, protocol = read_run(root)
    if args.arm not in ARMS:
        raise ValueError("Unrecognized predeclared arm")
    if (root / args.arm).exists():
        raise FileExistsError("Arm already exists, start new run directory on error")
    refs = {tuple(c["key"]): [] for c in protocol["cases"]}
    for row in physical.read_lines(source / "control/actions.jsonl"):
        key = capture.episode_key(row)
        if key not in refs:
            raise RuntimeError("New episode in original frozen native trace")
        refs[key].append(row)
    for c in protocol["cases"]:
        key = tuple(c["key"])
        if (not refs[key]
                or c["checkpoint"]["prefix_action_count"] >= len(refs[key])):
            raise RuntimeError("No validated prefix/continuation for " + repr(key))
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    native_init, native_loop = cls.__init__, cls._run_eval_dual_system
    probes = []
    is_scan = args.arm in SCANS
    selected_probe = (ScanResetProbe if args.arm in RESETS else ScanKeepProbe) \
        if is_scan else prior.CandidateProbe

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("Cache continuation forbidden")
        native_init(evaluator, cfg)
        probe = selected_probe(evaluator, root, args.arm, protocol, refs)
        probes.append(probe)
        evaluator._recovery_candidate = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate

        def counted_generate(*a, **kw):
            probe.s2_calls += 1
            return probe.original_generate(*a, **kw)
        evaluator.model.generate = counted_generate

    cls.__init__ = initialize
    cls._run_eval_dual_system = (
        iso.instrument_loop(native_loop) if is_scan else
        prior.instrument_loop(native_loop))
    error = None
    try:
        pilot.run_arm(SimpleNamespace(
            run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Not exactly one native evaluator")
        probes[0].finish()
        if set(probes[0].finished) != set(refs):
            raise RuntimeError("NOT_COMPARABLE: incomplete six-scene cohort")
        probes[0].status = "complete"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = native_init, native_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.evaluator.model.generate = probe.original_generate
            probe.close(error)


def validate_histories(root, arm, key, detail, checkpoint, reset):
    folder = root / arm / "checkpoints" / (key[0] + "_" + key[1])
    records = pilot.read_json(folder / "history_inputs.json")
    if len(records) != detail["history_prompt_events"]:
        raise ValueError("History audit count differs")
    if reset:
        if (not records or not detail["handoff"]["history_reset"]
                or detail["handoff"]["history_cutoff"] != checkpoint["history_length"]):
            raise ValueError("Post-gate history RESET not executed")
        for event in records:
            if (event["cutoff"] != checkpoint["history_length"]
                    or any(i < event["cutoff"] or i >= event["rgb_count"] - 1
                           for i in event["selected_history_ids"])):
                raise ValueError("Pre-gate history leaked into reset System-2 query")
        if (records[0]["main_step"] != checkpoint["step"]
                or records[0]["selected_history_ids"]):
            raise ValueError("First RESET System-2 readout must contain no pre-gate history")
    elif records or detail["handoff"].get("history_reset", False):
        raise ValueError("KEEP history arm unexpectedly changed sampled history")
    return records


def audit_scan_image(root, arm, key, detail, checkpoint):
    folder = root / arm / "checkpoints" / (key[0] + "_" + key[1])
    handoff = detail["handoff"]
    if (handoff["external_actions"] != iso.SCAN_ACTIONS
            or handoff["charged_native_limit"] !=
            detail["_expected_charged_limit"]):
        raise ValueError("Physical scan action budget changed")
    if (math.dist(handoff["state"]["position"], checkpoint["state"]["position"]) >
            iso.POSE_TOLERANCE_M
            or pose.heading_error(handoff["state"]["orientation_wxyz"],
                                  checkpoint["state"]["orientation_wxyz"]) >
               iso.HEADING_TOLERANCE_DEG):
        raise ValueError("Scan did not return to original checkpoint pose")
    evidence_input = pilot.read_json(folder / "evidence_input.json")
    events = evidence_input["events"]
    if (len(events) != 1 or events[0]["main_step"] != checkpoint["step"]
            or events[0].get("same_image_count") is not True
            or events[0].get("prompt") != iso.EXTRA_PROMPT
            or events[0].get("insertion_order") != "before_original_current"):
        raise ValueError("Supplementary token / image slot not controlled")
    if events[0]["source"] != (
            "outward_view" if arm.endswith("evidence_keep") or arm.endswith("evidence_reset")
            else "duplicate_current"):
        raise ValueError("Wrong supplementary image source")
    from PIL import Image
    with Image.open(folder / "outward_view_model_rgb.png") as im:
        image_hash = physical.history_hash([im.copy()])
    if image_hash != evidence_input["scan_view_sha256"]:
        raise ValueError("Saved outward RGB differs from injected provenance")
    expected = (
        evidence_input["scan_view_sha256"] if "evidence" in arm
        else evidence_input["post_return_current_sha256"]
    )
    if events[0]["supplementary_pixel_sha256"] != expected:
        raise ValueError("Actual injected pixels don't match declared view")
    if detail["injection_events"] != events:
        raise ValueError("Injected event audit differs from runtime summary")
    return evidence_input


def report(args):
    root = Path(args.run_dir).resolve()
    source, annotation, protocol = read_run(root)
    native_summary = pilot.read_json(source / "control/pilot_summary.json")
    native = capture.read_progress(source / "control/progress.json")
    keys = [tuple(c["key"]) for c in protocol["cases"]]
    context = {tuple(c["key"]): c for c in protocol["cases"]}
    if set(native) != set(keys):
        raise ValueError("Native control cohort has drifted")
    reference = {k: [] for k in keys}
    for row in physical.read_lines(source / "control/actions.jsonl"):
        reference[capture.episode_key(row)].append(row)
    tagged = {}
    details = {}
    progress = {}
    for arm in ARMS:
        summary = pilot.read_json(root / arm / "pilot_summary.json")
        diag = pilot.read_json(root / arm / "turn_summary.json")
        if (summary["status"] != "complete" or diag["status"] != "complete"):
            raise RuntimeError("Incomplete intervention arm: " + arm)
        common.same_fields(summary, native_summary, common.RUNTIME_FIELDS, arm + " runtime")
        tagged[arm] = physical.validate_trace(
            physical.read_lines(root / arm / "turn_actions.jsonl"), arm)
        unt = physical.validate_trace(
            physical.read_lines(root / arm / "actions.jsonl"), arm + " native")
        if not pilot.compare_traces(tagged[arm], unt)["actions_and_poses_match"]:
            raise ValueError("Native/observer per-action trace mismatch")
        details[arm] = {tuple(r["key"]): r for r in diag["cases"]}
        progress[arm] = capture.read_progress(root / arm / "progress.json")
        if set(details[arm]) != set(keys) or set(progress[arm]) != set(keys):
            raise ValueError("Changed six-episode cohort in " + arm)
    rows = []
    for key in keys:
        gate = context[key]["checkpoint"]
        data = {}
        data["keep"] = {
            "success": native[key]["success"], "ne": native[key]["ne"],
            "spl": native[key]["spl"], "os": native[key]["os"],
            "extra_actions": 0,
        }
        meta = {"key": list(key), "step": gate["step"],
                "prefix_action_count": gate["prefix_action_count"],
                "boundary_verified": True,
                "scan_identical_across_four_arms": True,
                "same_current_rgb_after_return": True,
                "annotation_frozen": True}
        scan_frames = []
        external_sets = []
        for label, arm in COMPARISONS.items():
            if label == "keep":
                continue
            detail = details[arm][key]
            common.same_fields(detail["gate"], gate, GATE_FIELDS, arm + " gate")
            pcount = gate["prefix_action_count"]
            trace = [r for r in tagged[arm] if capture.episode_key(r) == key]
            if not pilot.compare_traces(reference[key][:pcount],
                                        trace[:pcount])["actions_and_poses_match"]:
                raise ValueError("Frozen native prefix diverged in " + arm)
            if progress[arm][key]["episode_instruction"] != native[key]["episode_instruction"]:
                raise ValueError("Instruction changed during paired intervention")
            external = [r for r in trace if r["phase"] == "external_recovery"]
            expected_n = iso.SCAN_ACTIONS if arm in SCANS else 0
            if len(external) != expected_n:
                raise ValueError("Incorrect external action charge for " + arm)
            if arm in SCANS:
                if [r["action"] for r in external] != [3, 3, 3, 2, 2, 2]:
                    raise ValueError("Not the fixed right45 scan action sequence")
                if any(math.dist(r["position"], gate["state"]["position"]) >
                       iso.POSE_TOLERANCE_M for r in external):
                    raise ValueError("Scan translated instead of rotating")
                detail["_expected_charged_limit"] = protocol["charged_native_main_step_limit"]
                img = audit_scan_image(root, arm, key, detail, gate)
                scan_frames.append(img)
                external_sets.append(external)
            else:
                if (detail["handoff"]["charged_native_limit"] !=
                        protocol["native_main_step_limit"]):
                    raise ValueError("RESET-only motion budget unexpectedly changed")
                if (math.dist(detail["handoff"]["state"]["position"],
                              gate["state"]["position"]) > iso.POSE_TOLERANCE_M
                    or pose.heading_error(detail["handoff"]["state"]["orientation_wxyz"],
                                          gate["state"]["orientation_wxyz"]) >
                    iso.HEADING_TOLERANCE_DEG):
                    raise ValueError("RESET-only changed physical state")
            validate_histories(root, arm, key, detail, gate, arm in RESETS)
            val = progress[arm][key]
            if val["success"] not in (0, 1):
                raise ValueError("Nonbinary downstream success")
            if not math.isfinite(float(val["ne"])):
                raise ValueError("Nonfinite final navigation error")
            data[label] = {
                "success": int(val["success"]), "ne": float(val["ne"]),
                "spl": float(val["spl"]), "os": float(val["os"]),
                "extra_actions": expected_n,
            }
        if (len(scan_frames) != 4 or
                len({x["post_return_current_sha256"] for x in scan_frames}) != 1):
            raise ValueError("Returned current model RGB differs among four scans")
        # KEEP/RESET comparisons must have identical real yaw scan and final RGB
        # within their history strata, and even across strata.
        if any(not pilot.compare_traces(external_sets[0], x)[
                "actions_and_poses_match"] for x in external_sets[1:]):
            raise ValueError("External scan action/pose trajectories differ")
        if len({x["scan_view_sha256"] for x in scan_frames}) != 1:
            raise ValueError("Four physical scans saw different outward RGB")
        if scan_frames[0]["events"][0]["supplementary_pixel_sha256"] != \
           scan_frames[2]["events"][0]["supplementary_pixel_sha256"]:
            raise ValueError("Placebo image differs between KEEP and RESET")
        if scan_frames[1]["events"][0]["supplementary_pixel_sha256"] != \
           scan_frames[3]["events"][0]["supplementary_pixel_sha256"]:
            raise ValueError("SEEK image differs between KEEP and RESET")
        contrasts = choice.compare_6_conditions({
            key: {k: {"success": v["success"], "ne": v["ne"]}
                  for k, v in data.items()}
        })[0]
        rows.append({
            "key": list(key), "checkpoint_step": gate["step"],
            "outcomes": data,
            "binary_success_contrasts": contrasts["binary_success_contrasts"],
            "navigation_error_contrasts_m": contrasts["navigation_error_contrasts_m"],
            "checks": meta,
        })
    result = {
        "schema": "train_six_arm_intervention_report.v1",
        "status": "SIX_TRAIN_SCENE_COMPARABLE",
        "N_contexts": len(rows), "N_new_rollouts": len(ARMS) * len(rows),
        "N_reused_controls": len(rows),
        "all_native_baselines_successful": True,
        "scan_direction": "right45", "scan_actions": iso.SCAN_ACTIONS,
        "preintervention_evidence_sha256": pilot.sha256(
            annotation / "annotations.jsonl"),
        "review_status": "PRELIMINARY_NOT_HUMAN_CONFIRMED",
        "arm_mapping": COMPARISONS, "per_context": rows,
        "scope": SCOPE,
        "not_estimated": [
            "failure_rescue", "out_of_scene_generalization",
            "evidence_to_intervention_effect_predictability",
            "learned_policy", "task_conditional_scan_direction"
        ],
    }
    pilot.write_json(root / "six_arm_report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_all(args):
    root = Path(args.run_dir).resolve()
    read_run(root)
    if any((root / a).exists() for a in ARMS):
        raise FileExistsError("Run-all is create-only; run missing arms individually")
    for arm in ARMS:
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), "run",
            "--run-dir", str(root), "--arm", arm,
        ], check=True)
    return report(SimpleNamespace(run_dir=str(root)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    ss = p.add_subparsers(dest="command", required=True)
    a = ss.add_parser("prepare")
    a.add_argument("--checkpoint-run", required=True)
    a.add_argument("--annotation-dir", required=True)
    a.add_argument("--run-dir", required=True)
    a.set_defaults(function=prepare)
    for name, fn in (("run", run), ("run-all", run_all), ("report", report)):
        a = ss.add_parser(name)
        a.add_argument("--run-dir", required=True)
        if name == "run":
            a.add_argument("--arm", choices=ARMS, required=True)
        a.set_defaults(function=fn)
    args = p.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
