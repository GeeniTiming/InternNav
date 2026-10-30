"""Six-arm *comparability* pilot, reusing four frozen controls and running two joint arms.

CPU prepare/preflight. GPU run only for right45 supplementary RGB/placebo plus
selective post-checkpoint history RESET. Report performs strict paired checks.

Controls (not rerun):
  KEEP                       checkpoint native control
  RESET                      independent factorial / history_reset
  PLACEBO_KEEP               evidence isolation / lookback_right45_placebo
  SEEK_KEEP                  evidence isolation / lookback_right45_evidence
New, exactly two:
  PLACEBO_RESET              same 6-action right45 scan + duplicate current RGB,
                             history indices restricted to post-gate observations
  SEEK_RESET                 same scan + actual rotated RGB, same reset condition

The study is TWO DEVELOPMENT SCENES ONLY. NOT efficacy, task-grounded SEEK, a
learned intervention selector, generalization, or proof of causal mediation.
Source checkpoints and every earlier experiment stay byte-for-byte untouched.
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
import recovery_evidence_isolation as iso

pilot, physical, common = iso.pilot, iso.physical, iso.common
capture, pose = iso.capture, iso.pose
JOINT_ARMS = ("lookback_right45_placebo", "lookback_right45_evidence")
LABELS = {
    "keep": "source/control",
    "reset": "factorial/history_reset",
    "placebo_keep": "mechanism/lookback_right45_placebo",
    "seek_keep": "mechanism/lookback_right45_evidence",
    "placebo_reset": "joint/lookback_right45_placebo",
    "seek_reset": "joint/lookback_right45_evidence",
}
SCHEMA = "intervention_choice_comparability.v1"
SCOPE = (
    "Two previously inspected development scenes, fixed right45 non-task-specific "
    "yaw scan, exactly 6 charged actual actions, original frozen policy. Matched "
    "evidence/placebo image injection at same returned pose; post-boundary history "
    "selection reset. Not a new held-out experiment, online evidence detector, "
    "task-adaptive scan, statistical efficacy study or trained selector."
)


def original_runs(mechanism):
    mechanism = Path(mechanism).resolve()
    source, factorial, old_protocol = iso.read_run(mechanism)
    saved = pilot.read_json(mechanism / "mechanism_report.json")
    if (saved.get("schema") != "evidence_isolation_report.v1"
            or saved.get("status") != "MECHANISM_PILOT_VALIDATED"
            or saved.get("N_contexts") != len(old_protocol["cases"])):
        raise ValueError("Complete original evidence-isolation mechanism report required")
    old = pilot.read_json(factorial / "candidate_report.json")
    if (old.get("schema") != "recovery_independent_factorial_report.v1"
            or old.get("status") != "READY_FOR_INTERPRETATION"):
        raise ValueError("Original history_reset comparator is incomplete")
    for arm in JOINT_ARMS:
        if (pilot.read_json(mechanism / arm / "pilot_summary.json")["status"] != "complete"
                or pilot.read_json(mechanism / arm / "turn_summary.json")["status"] != "complete"):
            raise ValueError("Original right45 evidence/placebo missing: " + arm)
    for arm in ("history_reset",):
        if (pilot.read_json(factorial / arm / "pilot_summary.json")["status"] != "complete"
                or pilot.read_json(factorial / arm / "turn_summary.json")["status"] != "complete"):
            raise ValueError("Original RESET is incomplete")
    # Old mechanism_report previously audited the original four arms, and
    # scan pair identity, using original source action/handoff hashes.
    keys = {tuple(c["key"]) for c in old_protocol["cases"]}
    if (len(keys) != len(old_protocol["cases"])
            or keys != {tuple(r["key"]) for r in saved["per_context"]}):
        raise ValueError("Old mechanism report has inconsistent episode keys")
    return source, factorial, old_protocol, saved


def compare_6_conditions(outcomes):
    """All contrasts are paired within an *identical* pre-intervention state."""
    required = frozenset(LABELS)
    rows = []
    for key, arms in outcomes.items():
        if set(arms) != required:
            raise ValueError("Missing six comparison arms: " + repr(key))
        for arm, value in arms.items():
            if (type(value["success"]) not in (bool, int, float)
                    or float(value["success"]) not in (0.0, 1.0)):
                raise ValueError("Nonbinary success: " + arm)
            if not math.isfinite(float(value["ne"])):
                raise ValueError("Nonfinite navigation error: " + arm)
        success = {a: int(v["success"]) for a, v in arms.items()}
        ne = {a: float(v["ne"]) for a, v in arms.items()}
        contrasts = {
            "reset_minus_keep": success["reset"] - success["keep"],
            "extra_rgb_minus_placebo_keep": success["seek_keep"] - success["placebo_keep"],
            "extra_rgb_minus_placebo_reset": success["seek_reset"] - success["placebo_reset"],
            "reset_minus_keep_given_placebo_scan": success["placebo_reset"] - success["placebo_keep"],
            "reset_minus_keep_given_real_scan": success["seek_reset"] - success["seek_keep"],
            "rgb_reset_interaction": (success["seek_reset"] - success["placebo_reset"])
                                     - (success["seek_keep"] - success["placebo_keep"]),
        }
        rows.append({
            "key": list(key), "success": success, "navigation_error_m": ne,
            "binary_success_contrasts": contrasts,
            "navigation_error_contrasts_m": {
                "reset_minus_keep": ne["reset"] - ne["keep"],
                "extra_rgb_minus_placebo_keep": ne["seek_keep"] - ne["placebo_keep"],
                "extra_rgb_minus_placebo_reset": ne["seek_reset"] - ne["placebo_reset"],
                "reset_minus_keep_given_placebo_scan": ne["placebo_reset"] - ne["placebo_keep"],
                "reset_minus_keep_given_real_scan": ne["seek_reset"] - ne["seek_keep"],
            },
        })
    return rows


def verify_scan_pair(tagged, control_tagged, key, expected_count=iso.SCAN_ACTIONS):
    """Require the SAME externally executed yaw actions and poses."""
    rows = [r for r in tagged if capture.episode_key(r) == key
            and r["phase"] == "external_recovery"]
    other = [r for r in control_tagged if capture.episode_key(r) == key
             and r["phase"] == "external_recovery"]
    required = [3, 3, 3, 2, 2, 2]
    if (len(rows) != expected_count or len(other) != expected_count
            or [int(x["action"]) for x in rows] != required
            or [int(x["action"]) for x in other] != required):
        raise ValueError("NOT_COMPARABLE: right45 scan actions differ")
    if not pilot.compare_traces(rows, other)["actions_and_poses_match"]:
        raise ValueError("NOT_COMPARABLE: right45 scan trajectories differ")
    return rows


def validate_joint_case(detail, folder, arm, checkpoint):
    if (detail["gate"]["step"] != checkpoint["step"]
            or detail["gate"]["history_sha256"] != checkpoint["history_sha256"]
            or not detail["prefix_comparison"]["actions_and_poses_match"]):
        raise ValueError("NOT_COMPARABLE: original prefix/gate changed")
    handoff = detail["handoff"]
    cutoff = detail["gate"]["history_length"]
    if (handoff.get("history_reset") is not True
            or handoff.get("history_cutoff") != cutoff
            or handoff["external_actions"] != iso.SCAN_ACTIONS
            or detail.get("history_prompt_events", 0) < 1):
        raise ValueError("NOT_COMPARABLE: post-gate history reset or cost missing")
    history = pilot.read_json(folder / "history_inputs.json")
    if len(history) != detail["history_prompt_events"]:
        raise ValueError("NOT_COMPARABLE: reset history event count changed")
    for e in history:
        if e["cutoff"] != cutoff:
            raise ValueError("History cutoff changed after handoff")
        if any(i < cutoff or i >= e["rgb_count"] - 1 for i in e["selected_history_ids"]):
            raise ValueError("NOT_COMPARABLE: pre-gate history leaked into reset S2 input")
    first = next((e for e in history if e["main_step"] == checkpoint["step"]), None)
    if first is None or first["selected_history_ids"]:
        raise ValueError("NOT_COMPARABLE: the first reset S2 prompt must contain NO old history")
    evidence = pilot.read_json(folder / "evidence_input.json")
    events = evidence.get("events", [])
    if (len(events) != 1 or events[0]["main_step"] != checkpoint["step"]
            or events[0]["prompt"] != iso.EXTRA_PROMPT
            or events[0].get("same_image_count") is not True):
        raise ValueError("NOT_COMPARABLE: one supplementary image required")
    expected = "outward_view" if arm.endswith("_evidence") else "duplicate_current"
    if events[0]["source"] != expected:
        raise ValueError("NOT_COMPARABLE: incorrect supplementary image source")
    expected_hash = (evidence["scan_view_sha256"] if arm.endswith("_evidence")
                     else evidence["post_return_current_sha256"])
    if events[0]["supplementary_pixel_sha256"] != expected_hash:
        raise ValueError("NOT_COMPARABLE: supplementary image hashes changed")
    return evidence


class JointProbe(iso.MechanismProbe):
    """Read-only subclass: scans as before, then filters history at S2 readout."""

    def on_boundary(self, step, observations, done, history, action_seq,
                    pixel_goal, local_actions, action):
        # First native loop iteration precedes TurnProbe.begin(), which
        # initializes self.gate. Do not access a missing attribute here.
        before = getattr(self, "gate", None) is not None
        result = super().on_boundary(
            step, observations, done, history, action_seq,
            pixel_goal, local_actions, action,
        )
        if not before and self.gate is not None:
            # No history tensors or native rgb_list are edited. Only indices
            # selected for post-gate ordinary S2 prompts change (as in the
            # previously validated independent factorial RESET condition).
            self.reset_enabled = True
            self.history_cutoff = len(history)
            self.handoff["history_reset"] = True
            self.handoff["history_cutoff"] = self.history_cutoff
            key = self.current
            folder = self.directory / "checkpoints" / (key[0] + "_" + key[1])
            pilot.write_json(folder / "handoff.json", self.handoff)
        return result

    def filter_history_ids(self, step, original_ids, rgb_count):
        if self.gate is not None and self.history_cutoff is None:
            raise RuntimeError("NOT_COMPARABLE: joint reset not configured before S2")
        return super().filter_history_ids(step, original_ids, rgb_count)


def prepare(args):
    mechanism, root = Path(args.mechanism_run).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Create-only: choose a NEW run-dir")
    source, factorial, old_protocol, saved = original_runs(mechanism)
    if len(old_protocol["cases"]) != 2:
        raise ValueError("This engineering pilot is restricted to the two original development contexts")
    if any(iso.ARM_ANGLES[a] != -45 for a in JOINT_ARMS):
        raise ValueError("Only fixed right45 scan is predeclared")
    cfg = pilot.load_config(source / "control.py")
    if (cfg.agent.model_settings.get("kv_cache_continuation", False)
            or cfg.agent.model_settings.get("mode") != "dual_system"):
        raise ValueError("Require frozen native cache-disabled DualVLN")
    protocol = {
        "schema": SCHEMA, "source_run": str(source),
        "factorial_run": str(factorial), "mechanism_run": str(mechanism),
        "cases": old_protocol["cases"], "arms": list(JOINT_ARMS),
        "six_arm_mapping": LABELS, "fixed_scan_direction": "right45",
        "original_main_loop_limit": old_protocol["original_main_loop_limit"],
        "native_main_step_limit": old_protocol["native_main_step_limit"],
        "charged_main_loop_limit": old_protocol["charged_main_loop_limit"],
        "charged_main_step_limit": old_protocol["charged_main_step_limit"],
        "charged_native_main_step_limit": old_protocol["charged_native_main_step_limit"],
        "forced_native_actions": iso.SCAN_ACTIONS,
        "source_split": "existing_development_checkpoints_only",
        "source_selection_frozen": True,
        "scope": SCOPE,
        "separation": (
            "KEEP/RESET cost 0 forced scan actions; four scan-based arms cost "
            "6 true yaw actions at the same locked checkpoint. SEEK is a fixed "
            "right45 observation (NOT task-conditioned). Extra RGB vs duplicate "
            "image is paired within each history condition."
        ),
    }
    iso.verify_inherited_runtime_contract(protocol)
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        check=False, text=True, capture_output=True,
    ).stdout.strip()
    old_manifest = pilot.verify_manifest(mechanism)
    manifest = {
        "schema": SCHEMA, "seed_base": old_manifest["seed_base"],
        "episodes": old_manifest["episodes"],
        "checkpoint_shard_sizes": old_manifest["checkpoint_shard_sizes"],
        "revision": revision, "source_hashes": dict(old_manifest["source_hashes"]),
        "generated_hashes": {},
        "intervention_choice": {"mechanism_run": str(mechanism), "scope": SCOPE},
    }
    manifest["source_hashes"].update(old_manifest["generated_hashes"])
    watched = [
        Path(__file__).resolve(), Path(iso.__file__).resolve(),
        Path(iso.prior.__file__).resolve(), Path(iso.turn.__file__).resolve(),
        Path(iso.checkpoint.__file__).resolve(), Path(iso.physical.__file__).resolve(),
        Path(iso.pose.__file__).resolve(), Path(iso.common.__file__).resolve(),
        mechanism / "manifest.json", mechanism / "mechanism_report.json",
        factorial / "manifest.json", factorial / "candidate_report.json",
        source / "manifest.json", source / "checkpoint_report.json",
    ]
    for base, arms in ((mechanism, JOINT_ARMS), (factorial, ("history_reset",))):
        for arm in arms:
            folder = base / arm
            for name in ("pilot_summary.json", "turn_summary.json", "progress.json",
                         "actions.jsonl", "turn_actions.jsonl"):
                watched.append(folder / name)
            watched.extend(p for p in (folder / "checkpoints").rglob("*")
                           if p.is_file() and p.suffix in (".json", ".png"))
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watched})
    root.mkdir(parents=True, exist_ok=False)
    protocol_path = root / "choice_protocol.json"
    pilot.write_json(protocol_path, protocol)
    manifest["generated_hashes"][str(protocol_path)] = pilot.sha256(protocol_path)
    for arm in JOINT_ARMS:
        new_cfg = copy.deepcopy(cfg)
        new_cfg.eval_settings["output_path"] = str(root / arm)
        new_cfg.agent.model_settings["vis_debug"] = True
        new_cfg.agent.model_settings["vis_debug_path"] = str(root / "video" / arm)
        text_cfg = json.dumps(new_cfg.model_dump(mode="json", exclude_none=True))
        p = root / (arm + ".py")
        p.write_text(
            "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
            + "eval_cfg = EvalCfg.model_validate(json.loads(" + repr(text_cfg) + "))\n",
            encoding="utf-8",
        )
        manifest["generated_hashes"][str(p)] = pilot.sha256(p)
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    print("INTERVENTION_CHOICE_PREPARE=PASS", len(protocol["cases"]),
          "contexts; new GPU arms=2; existing controls=4")


def read_run(root):
    manifest = pilot.verify_manifest(root)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Wrong choice-comparability manifest")
    mechanism = Path(manifest["intervention_choice"]["mechanism_run"])
    source, factorial, old_protocol, saved = original_runs(mechanism)
    protocol = pilot.read_json(root / "choice_protocol.json")
    if (protocol.get("schema") != SCHEMA
            or protocol["source_run"] != str(source)
            or protocol["factorial_run"] != str(factorial)
            or protocol["mechanism_run"] != str(mechanism)
            or protocol["cases"] != old_protocol["cases"]
            or protocol["arms"] != list(JOINT_ARMS)
            or protocol["six_arm_mapping"] != LABELS):
        raise ValueError("Choice protocol/source mismatch")
    iso.verify_inherited_runtime_contract(protocol)
    return source, factorial, mechanism, protocol


def run(args):
    root = Path(args.run_dir).resolve()
    source, factorial, mechanism, protocol = read_run(root)
    if args.arm not in JOINT_ARMS:
        raise ValueError("Unknown joint arm")
    if (root / args.arm).exists():
        raise FileExistsError("Create-only joint arm; do not resume failed runs")
    refs = {tuple(case["key"]): [] for case in protocol["cases"]}
    for row in physical.read_lines(source / "control/actions.jsonl"):
        refs[capture.episode_key(row)].append(row)
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    old_init, old_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise ValueError("Cache continuation must remain disabled")
        old_init(evaluator, cfg)
        probe = JointProbe(evaluator, root, args.arm, protocol, refs)
        probes.append(probe)
        evaluator._recovery_candidate = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate

        def counted_generate(*a, **kw):
            probe.s2_calls += 1
            return probe.original_generate(*a, **kw)
        evaluator.model.generate = counted_generate

    cls.__init__, cls._run_eval_dual_system = initialize, iso.instrument_loop(old_loop)
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Expected exactly one native evaluator")
        probes[0].finish()
        if set(probes[0].finished) != set(refs):
            raise RuntimeError("NOT_COMPARABLE: incomplete paired cohort")
        probes[0].status = "complete"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = old_init, old_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.evaluator.model.generate = probe.original_generate
            probe.close(error)


def report(args):
    root = Path(args.run_dir).resolve()
    source, factorial, mechanism, protocol = read_run(root)
    old_report = pilot.read_json(mechanism / "mechanism_report.json")
    prior = pilot.read_json(factorial / "candidate_report.json")
    controls = capture.read_progress(source / "control/progress.json")
    old_factorial = {(tuple(x["key"]), x["arm"]): x for x in prior["outcomes"]}
    old_mechanism = {tuple(x["key"]): x for x in old_report["per_context"]}
    native_runtime = pilot.read_json(source / "control/pilot_summary.json")
    cases = {tuple(x["key"]): x for x in protocol["cases"]}
    if (set(cases) != set(controls) or set(cases) != set(old_mechanism)
            or any((k, "history_reset") not in old_factorial for k in cases)):
        raise ValueError("Control cohort has drifted")
    trace_source = {key: [] for key in cases}
    for t in physical.read_lines(source / "control/actions.jsonl"):
        trace_source[capture.episode_key(t)].append(t)
    old_traces = {}
    for a in JOINT_ARMS:
        old_traces[a] = physical.validate_trace(
            physical.read_lines(mechanism / a / "turn_actions.jsonl"), "old " + a)
    new_traces = {}
    new_details = {}
    new_progress = {}
    for arm in JOINT_ARMS:
        summary = pilot.read_json(root / arm / "pilot_summary.json")
        detail_file = pilot.read_json(root / arm / "turn_summary.json")
        if summary["status"] != "complete" or detail_file["status"] != "complete":
            raise RuntimeError("Incomplete joint arm " + arm)
        common.same_fields(summary, native_runtime, common.RUNTIME_FIELDS, arm + " runtime")
        native = physical.validate_trace(
            physical.read_lines(root / arm / "actions.jsonl"), arm + " native")
        tagged = physical.validate_trace(
            physical.read_lines(root / arm / "turn_actions.jsonl"), arm + " tagged")
        if not pilot.compare_traces(native, tagged)["actions_and_poses_match"]:
            raise RuntimeError("Native/observer trace mismatch")
        new_traces[arm] = tagged
        new_details[arm] = {tuple(x["key"]): x for x in detail_file["cases"]}
        new_progress[arm] = capture.read_progress(root / arm / "progress.json")
        if set(new_details[arm]) != set(cases) or set(new_progress[arm]) != set(cases):
            raise ValueError("New joint arm changed the cohort")

    outcomes = {}
    per_case_checks = []
    for key, case in cases.items():
        checkpoint = case["checkpoint"]
        n = checkpoint["prefix_action_count"]
        if trace_source[key][:n] == []:
            raise RuntimeError("Empty source prefix")
        arms = {}
        arms["keep"] = {"success": controls[key]["success"], "ne": controls[key]["ne"]}
        arm_reset = old_factorial[(key, "history_reset")]
        arms["reset"] = {"success": arm_reset["success"], "ne": arm_reset["ne"]}
        old = old_mechanism[key]
        for label, name in (("placebo_keep", JOINT_ARMS[0]), ("seek_keep", JOINT_ARMS[1])):
            m = old[name]
            arms[label] = {"success": m["success"], "ne": m["ne"]}
        audit = {"key": list(key), "checkpoint_step": checkpoint["step"],
                 "same_native_prefix": True, "same_physical_scan_across_4_arms": True,
                 "same_current_rgb_after_return": True,
                 "post_reset_history_isolated": True,
                 "supplementary_image_provenance_verified": True}
        end_current = []
        joint_scan = []
        for label, name in (("placebo_reset", JOINT_ARMS[0]), ("seek_reset", JOINT_ARMS[1])):
            d = new_details[name][key]
            folder = root / name / "checkpoints" / (key[0] + "_" + key[1])
            evidence = validate_joint_case(d, folder, name, checkpoint)
            trace = [t for t in new_traces[name] if capture.episode_key(t) == key]
            if not pilot.compare_traces(trace_source[key][:n], trace[:n])["actions_and_poses_match"]:
                raise RuntimeError("NOT_COMPARABLE: old/native prefix diverged")
            verify_scan_pair(new_traces[name], old_traces[name], key)
            joint_scan.append([t for t in trace if t["phase"] == "external_recovery"])
            end_current.append(evidence["post_return_current_sha256"])
            expected = (evidence["scan_view_sha256"] if name.endswith("_evidence")
                        else evidence["post_return_current_sha256"])
            if evidence["events"][0]["supplementary_pixel_sha256"] != expected:
                raise RuntimeError("Supplementary view origin mismatch")
            val = new_progress[name][key]
            if val["episode_instruction"] != controls[key]["episode_instruction"]:
                raise RuntimeError("Original instruction changed")
            arms[label] = {"success": val["success"], "ne": val["ne"]}
        if not pilot.compare_traces(joint_scan[0], joint_scan[1])["actions_and_poses_match"]:
            raise RuntimeError("Physical scan differs between reset pair")
        old_current = old["lookback_right45_placebo"]["post_return_current_sha256"]
        if old_current != old["lookback_right45_evidence"]["post_return_current_sha256"]:
            raise RuntimeError("Old scan current RGB mismatch")
        if len(set(end_current + [old_current])) != 1:
            raise RuntimeError("Returned current RGB differs across keep/reset")
        for name in JOINT_ARMS:
            m = old[name]
            d = new_details[name][key]
            folder = root / name / "checkpoints" / (key[0] + "_" + key[1])
            extra = pilot.read_json(folder / "evidence_input.json")
            if extra["scan_view_sha256"] != m["supplementary_sha256"] and name.endswith("_evidence"):
                raise RuntimeError("Outward scan RGB changed in new conditions")
            if extra["events"][0]["supplementary_pixel_sha256"] != m["supplementary_sha256"]:
                raise RuntimeError("Injected supplementary RGB changed under reset")
        outcomes[key] = arms
        per_case_checks.append(audit)
    rows = compare_6_conditions(outcomes)
    result = {
        "schema": "intervention_choice_comparability_report.v1",
        "status": "TWO_SCENE_MECHANISM_COMPARABLE",
        "N_contexts": len(rows), "N_new_rollouts": 2 * len(rows),
        "N_reused_arm_contexts": 4 * len(rows),
        "scan_direction": "right45", "scan_actions": iso.SCAN_ACTIONS,
        "arm_mapping": LABELS, "per_context": rows, "comparability": per_case_checks,
        "scope": SCOPE,
        "limitations": (
            "Reused development episodes; no unseen-scene efficacy. Scan direction fixed "
            "and not task-conditioned. Binary success with N=2 cannot establish predictive "
            "relationship; no evidence labels, learned selector, or formal effect CI."
        ),
    }
    pilot.write_json(root / "six_arm_comparability_report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_all(args):
    root = Path(args.run_dir).resolve()
    read_run(root)
    if any((root / a).exists() for a in JOINT_ARMS):
        raise FileExistsError("run-all is create-only; run missing arm explicitly")
    for arm in JOINT_ARMS:
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), "run",
            "--run-dir", str(root), "--arm", arm,
        ], check=True)
    report(SimpleNamespace(run_dir=str(root)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="CPU-only locked source audit and config creation")
    prep.add_argument("--mechanism-run", required=True)
    prep.add_argument("--run-dir", required=True)
    prep.set_defaults(function=prepare)
    for name, handler in (("run", run), ("report", report), ("run-all", run_all)):
        p = subs.add_parser(name)
        p.add_argument("--run-dir", required=True)
        if name == "run":
            p.add_argument("--arm", required=True, choices=JOINT_ARMS)
        p.set_defaults(function=handler)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
