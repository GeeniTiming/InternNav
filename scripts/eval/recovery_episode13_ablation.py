"""Episode 13: isolate scan motion vs extra input under image-history RESET.

Exactly one predeclared R2R TRAIN context (cV4RVeZvu5T/13); no scene selection,
new outcomes, endpoint goals, failure labels or future actions used in selecting
the intervention. KEEP + six-arm results are frozen references only.

Four single-episode native replays:
  virtual_placebo_reset: RESET + identical supplementary duplicate current RGB
                         and exact old prompt, but NO physical scan.
  scan_no_injection_reset: RESET + exactly six real right45 out-and-back turns,
                           but NO extra RGB token or injection.
  replay_placebo_reset: exact preexisting (scan + placebo + RESET), with new
                        all-System-2-output trace capture for comparison.
  replay_seek_keep: exact preexisting (scan + actual outward RGB + KEEP), with
                    new all-System-2-output trace capture for comparison.

This is a diagnostic, NOT a method training, independent validation, or proof of
memory/progress mechanism. Old six-arm files are never overwritten or edited.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_train_six_arm as six

iso, prior = six.iso, six.prior
pilot, physical, capture, common, pose = (
    six.pilot, six.physical, six.capture, six.common, six.pose
)
TARGET = ("cV4RVeZvu5T", "13")
SCHEMA = "episode13_mechanism_ablation.v1"
ARMS = (
    "virtual_placebo_reset",
    "scan_no_injection_reset",
    "replay_placebo_reset",
    "replay_seek_keep",
)
SCANS = frozenset(ARMS[1:])
RESETS = frozenset(ARMS[:3])
OLD_MAP = {
    "replay_placebo_reset": six.ARMS[3],
    "replay_seek_keep": six.ARMS[2],
}
CANONICAL = {
    "scan_no_injection_reset": "lookback_right45_placebo",
    "replay_placebo_reset": "lookback_right45_placebo",
    "replay_seek_keep": "lookback_right45_evidence",
}
SCOPE = (
    "One inspected R2R TRAIN Episode13 engineering diagnosis, no reranking or "
    "held-out claim. Only two new interventions, two traced reference replays. "
    "Virtual placebo has no physical scan and full native budget; the two "
    "scan conditions each expend exactly six real yaw actions and charge six "
    "main-loop steps. RESET filters S2 sampled RGB history, not persistent "
    "model/KV memory. Text trace captures actual ordinary/followup S2 "
    "decoded output, not unobservable internal task belief."
)


def digest_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def source_check(six_arm_run):
    src = Path(six_arm_run).resolve()
    checkpoint, annotation, six_protocol = six.read_run(src)
    report = pilot.read_json(src / "six_arm_report.json")
    if (report.get("schema") != "train_six_arm_intervention_report.v1"
            or report.get("status") != "SIX_TRAIN_SCENE_COMPARABLE"
            or len(six_protocol["cases"]) != 6
            or len(report["per_context"]) != 6):
        raise ValueError("Complete six-scene comparable report required")
    matches = [c for c in six_protocol["cases"] if tuple(c["key"]) == TARGET]
    rows = [c for c in report["per_context"] if tuple(c["key"]) == TARGET]
    if len(matches) != 1 or len(rows) != 1:
        raise ValueError("Episode13 is not the exact locked six-scene case")
    if (rows[0]["outcomes"]["keep"]["success"] != 1
            or rows[0]["outcomes"]["reset"]["success"] != 1
            or rows[0]["outcomes"]["placebo_keep"]["success"] != 1
            or rows[0]["outcomes"]["seek_keep"]["success"] != 1
            or rows[0]["outcomes"]["placebo_reset"]["success"] != 0
            or rows[0]["outcomes"]["seek_reset"]["success"] != 0):
        raise ValueError("Historical target outcome pattern differs from frozen pilot")
    return checkpoint, annotation, six_protocol, matches[0], rows[0]


def make_protocol(case, limit):
    data = {
        "schema": SCHEMA,
        "cases": [case],
        "arms": list(ARMS),
        "native_main_step_limit": limit,
        "original_main_loop_limit": limit,
        "charged_main_loop_limit": limit - iso.SCAN_ACTIONS,
        "charged_main_step_limit": limit - iso.SCAN_ACTIONS,
        "charged_native_main_step_limit": limit - iso.SCAN_ACTIONS,
        "forced_native_actions": iso.SCAN_ACTIONS,
        "per_arm_extra_actions": {a: (iso.SCAN_ACTIONS if a in SCANS else 0)
                                  for a in ARMS},
        "per_arm_supplied_rgb": {
            ARMS[0]: "duplicate_current",
            ARMS[1]: "none",
            ARMS[2]: "duplicate_current",
            ARMS[3]: "actual_right45_outward",
        },
        "per_arm_reset": {a: a in RESETS for a in ARMS},
        "extra_prompt": iso.EXTRA_PROMPT,
        "original_checkpoint_target": list(TARGET),
        "predeclared_first_divergence_search": "full auto-action stream, first mismatch",
        "system2_trace": "each native decoded output and selected history indices",
        "scope": SCOPE,
    }
    iso.verify_inherited_runtime_contract(data)
    return data


def prepare(args):
    root = Path(args.run_dir).resolve()
    six_arm = Path(args.six_arm_run).resolve()
    if root.exists():
        raise FileExistsError("Create-only diagnostic directory")
    checkpoint, annotation, original, case, prior_row = source_check(six_arm)
    native_cfg = pilot.load_config(checkpoint / "control.py")
    if (native_cfg.agent.model_settings.get("mode") != "dual_system"
            or native_cfg.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Frozen native DualVLN, no KV continuation, required")
    limit = int(native_cfg.eval_settings["max_steps_per_episode"])
    protocol = make_protocol(case, limit)
    # Only one episode. No additional scene sampling or cherry-picking.
    old_data = pilot.read_json(checkpoint / "episodes.json.gz")
    matches = [ep for ep in old_data["episodes"] if capture.episode_key(ep) == TARGET]
    if len(matches) != 1:
        raise ValueError("Locked checkpoint dataset does not include exact target once")
    single = copy.deepcopy(old_data)
    single["episodes"] = matches
    import yaml
    habitat = yaml.safe_load((checkpoint / "habitat.yaml").read_text("utf-8"))
    if habitat["habitat"]["dataset"]["split"] != "train":
        raise ValueError("R2R TRAIN split must remain frozen")
    rev = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    old_manifest = pilot.verify_manifest(six_arm)
    manifest = {
        "schema": SCHEMA,
        "revision": rev,
        "seed_base": old_manifest["seed_base"],
        "episodes": [{"scene_id": matches[0]["scene_id"],
                      "episode_id": str(matches[0]["episode_id"])}],
        "checkpoint_shard_sizes": old_manifest["checkpoint_shard_sizes"],
        "source_hashes": {
            **old_manifest["source_hashes"],
            **old_manifest["generated_hashes"],
        },
        "generated_hashes": {},
        "diagnosis": {
            "source_six_arm_run": str(six_arm),
            "checkpoint_run": str(checkpoint),
            "annotation_dir": str(annotation),
            "target": list(TARGET),
            "scope": SCOPE,
        },
    }
    watch = [
        six_arm / "manifest.json", six_arm / "six_arm_report.json",
        checkpoint / "manifest.json", checkpoint / "checkpoint_report.json",
        annotation / "annotations.jsonl",
        Path(__file__).resolve(), Path(six.__file__).resolve(),
        Path(iso.__file__).resolve(), Path(prior.__file__).resolve(),
        Path(six.choice.__file__).resolve(),
        Path(physical.__file__).resolve(), Path(common.__file__).resolve(),
    ]
    for old_name in OLD_MAP.values():
        watch.extend([
            six_arm / old_name / "turn_actions.jsonl",
            six_arm / old_name / "progress.json",
            six_arm / old_name / "turn_summary.json",
        ])
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watch})
    root.mkdir(parents=True, exist_ok=False)
    ep_path = root / "episodes.json.gz"
    with ep_path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as out:
            out.write(json.dumps(single, ensure_ascii=False).encode("utf-8"))
    habitat["habitat"]["dataset"]["data_path"] = str(ep_path)
    yaml_path = root / "habitat.yaml"
    yaml_path.write_text("# @package _global_\n" + yaml.safe_dump(
        habitat, sort_keys=False), encoding="utf-8")
    protocol_path = root / "diagnostic_protocol.json"
    pilot.write_json(protocol_path, protocol)
    for path in (ep_path, yaml_path, protocol_path):
        manifest["generated_hashes"][str(path)] = pilot.sha256(path)
    for arm in ARMS:
        cfg = copy.deepcopy(native_cfg)
        cfg.env.env_settings["config_path"] = str(yaml_path)
        cfg.eval_settings["output_path"] = str(root / arm)
        cfg.agent.model_settings["vis_debug"] = True
        cfg.agent.model_settings["vis_debug_path"] = str(root / "video" / arm)
        path = root / (arm + ".py")
        payload = json.dumps(cfg.model_dump(mode="json", exclude_none=True))
        path.write_text(
            "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
            + "eval_cfg = EvalCfg.model_validate(json.loads("
            + repr(payload) + "))\n", encoding="utf-8")
        manifest["generated_hashes"][str(path)] = pilot.sha256(path)
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    print("EP13_ABLATION_PREPARE=PASS one frozen case; four single-episode runs")
    print("EXPERIMENTAL_SCOPE=DEVELOPMENT_MECHANISM_ONLY; NO_LEARNED_SELECTOR")


def read_run(root):
    manifest = pilot.verify_manifest(root)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unexpected diagnostic manifest")
    six_arm = Path(manifest["diagnosis"]["source_six_arm_run"]).resolve()
    checkpoint, annotation, old_protocol, case, row = source_check(six_arm)
    if (manifest["episodes"] != [{"scene_id": case["scene_id"],
                                   "episode_id": str(case["episode_id"])}]
            or Path(manifest["diagnosis"]["checkpoint_run"]).resolve() != checkpoint
            or Path(manifest["diagnosis"]["annotation_dir"]).resolve() != annotation):
        raise ValueError("Diagnosis source identity changed")
    protocol = pilot.read_json(root / "diagnostic_protocol.json")
    limit = int(pilot.load_config(checkpoint / "control.py").eval_settings[
        "max_steps_per_episode"])
    if protocol != make_protocol(case, limit):
        raise ValueError("Frozen Episode13 protocol drifted")
    return six_arm, checkpoint, case, row, protocol


class DecisionLogger:
    """Runtime only: native System-2 decoded responses, pre-input history slots."""

    def begin(self, key):
        super().begin(key)
        self.decisions = []
        self.history_selections = []
        self.injection_diagnostic = []

    def filter_history_ids(self, step, original_ids, rgb_count):
        actual = super().filter_history_ids(step, original_ids, rgb_count)
        if self.gate is not None:
            self.history_selections.append({
                "step": int(step),
                "original_history_indices": [int(x) for x in original_ids],
                "selected_history_indices": [int(x) for x in actual],
                "rgb_count": int(rgb_count),
            })
        return actual

    def record_system2_output(self, step, message):
        # Decoded native output, including down-look followup if the native
        # code generates one. Do not reinterpret text as task-state labels.
        self.decisions.append({
            "episode_key": list(self.current),
            "step": int(step),
            "generation_index": len(self.decisions),
            "text": str(message),
            "phase": self.phase,
            "action_count_at_generation": len(self.records),
        })
        if isinstance(self, iso.MechanismProbe):
            iso.MechanismProbe.record_system2_output(self, step, message)

    def _save_decisions(self):
        if self.current is None:
            return
        if self.gate is None:
            raise RuntimeError("No verified checkpoint before decision capture")
        folder = self.directory / "checkpoints" / "_".join(self.current)
        folder.mkdir(parents=True, exist_ok=True)
        pilot.write_json(folder / "system2_outputs.json", self.decisions)
        pilot.write_json(folder / "system2_history_selections.json",
                         self.history_selections)
        pilot.write_json(folder / "diagnostic_inputs.json",
                         self.injection_diagnostic)

    def finish(self):
        if self.current is not None:
            self._save_decisions()
        return super().finish()


class VirtualPlacebo(DecisionLogger, prior.CandidateProbe):
    """Only supplementary image+prompt added to history reset, no physical scan."""

    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.arm = "history_reset"

    def begin(self, key):
        super().begin(key)
        self.pending_virtual = False
        self.injected_virtual = 0

    def on_boundary(self, *args):
        before = getattr(self, "gate", None) is not None
        result = super().on_boundary(*args)
        if not before and self.gate is not None:
            self.pending_virtual = True
        return result

    def inject_extra_view(self, step, sources, images):
        if not self.pending_virtual:
            return
        if (int(step) != self.gate["step"] or self.phase != "continuation"
                or not images or self.injected_virtual):
            raise RuntimeError("Virtual supplementary input not at locked S2 readout")
        image = images[-1].copy()
        sources[0]["value"] += iso.EXTRA_PROMPT
        images.insert(-1, image)
        if sources[0]["value"].count("<image>") + 1 != len(images):
            raise RuntimeError("Virtual RGB prompt/image count mismatch")
        self.pending_virtual = False
        self.injected_virtual += 1
        self.injection_diagnostic.append({
            "step": int(step), "injected": True, "source": "duplicate_current",
            "image_pixel_sha256": physical.history_hash([image]),
            "prompt": iso.EXTRA_PROMPT, "external_actions": 0,
        })

    def finish(self):
        if self.current is not None and (self.injected_virtual != 1
                                        or self.pending_virtual):
            raise RuntimeError("Exactly one virtual placebo image required")
        return super().finish()


class ScanNoInjection(DecisionLogger, six.choice.JointProbe):
    """Original physical 6-step scan and history RESET, but NO extra image."""

    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.arm = CANONICAL[arm]

    def inject_extra_view(self, step, sources, images):
        if not self.inject_pending:
            return
        if int(step) != self.gate["step"] or self.phase != "continuation":
            raise RuntimeError("Scan no-input suppression moved away from locked readout")
        self.inject_pending = False
        self.injection_diagnostic.append({
            "step": int(step), "injected": False,
            "source": "none", "image_count_before": len(images),
            "prompt_suffix_added": False,
            "external_actions": self.external_actions,
        })

    def finish(self):
        # MechanismProbe.finish requires one injection and is unsuitable for
        # this negative intervention. Use its inherited pure RESET checker.
        if self.current is None:
            return
        if self.inject_pending or self.inject_count or len(self.injection_diagnostic) != 1:
            raise RuntimeError("No-injection diagnostic was not exactly once suppressed")
        self._save_decisions()
        return prior.CandidateProbe.finish(self)


class ReplayPlaceboReset(DecisionLogger, six.choice.JointProbe):
    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.arm = CANONICAL[arm]


class ReplaySeekKeep(DecisionLogger, iso.MechanismProbe):
    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.arm = CANONICAL[arm]


CLASSES = dict(zip(ARMS, (
    VirtualPlacebo, ScanNoInjection, ReplayPlaceboReset, ReplaySeekKeep
)))


def run(args):
    root = Path(args.run_dir).resolve()
    six_arm, checkpoint, case, old, protocol = read_run(root)
    if args.arm not in ARMS:
        raise ValueError("Not a predeclared Episode13 diagnostic arm")
    if (root / args.arm).exists():
        raise FileExistsError("Create-only arm; failed attempts must remain archived")
    refs = {TARGET: []}
    for row in physical.read_lines(checkpoint / "control/actions.jsonl"):
        if capture.episode_key(row) == TARGET:
            refs[TARGET].append(row)
    if not refs[TARGET]:
        raise RuntimeError("No source frozen Episode13 actions")
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    old_init, old_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, config):
        if config.agent.model_settings.get("kv_cache_continuation", False):
            raise ValueError("KV continuation forbidden")
        old_init(evaluator, config)
        probe = CLASSES[args.arm](evaluator, root, args.arm, protocol, refs)
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
        pilot.run_arm(SimpleNamespace(
            run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("One native evaluator expected")
        probes[0].finish()
        if set(probes[0].finished) != {TARGET}:
            raise RuntimeError("Episode13 targeted replay did not finish")
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
    six_arm, checkpoint, case, baseline, protocol = read_run(root)
    original = capture.read_progress(checkpoint / "control/progress.json")[TARGET]
    frozen = {a: baseline["outcomes"][label]
              for a, label in OLD_MAP.items()}
    source_actions = [
        r for r in physical.read_lines(checkpoint / "control/actions.jsonl")
        if capture.episode_key(r) == TARGET
    ]
    gate = case["checkpoint"]
    results, scans = {}, {}
    for arm in ARMS:
        if (pilot.read_json(root / arm / "pilot_summary.json")["status"] != "complete"
                or pilot.read_json(root / arm / "turn_summary.json")["status"] != "complete"):
            raise ValueError("Incomplete Episode13 diagnostic: " + arm)
        details = pilot.read_json(root / arm / "turn_summary.json")["cases"]
        if len(details) != 1 or tuple(details[0]["key"]) != TARGET:
            raise ValueError("Changed targeted Episode13 cohort")
        detail = details[0]
        common.same_fields(detail["gate"], gate, six.GATE_FIELDS, arm + " gate")
        rows = [
            r for r in physical.read_lines(root / arm / "turn_actions.jsonl")
            if capture.episode_key(r) == TARGET
        ]
        plain = physical.read_lines(root / arm / "actions.jsonl")
        if not pilot.compare_traces(rows, plain)["actions_and_poses_match"]:
            raise ValueError("Native/observer trace diverged")
        prefix = int(gate["prefix_action_count"])
        if not pilot.compare_traces(source_actions[:prefix],
                                    rows[:prefix])["actions_and_poses_match"]:
            raise ValueError("Original native prefix diverged")
        external = [r for r in rows if r["phase"] == "external_recovery"]
        expected_external = protocol["per_arm_extra_actions"][arm]
        if len(external) != expected_external:
            raise ValueError("Wrong number of real external actions")
        if arm in SCANS:
            if [r["action"] for r in external] != [3, 3, 3, 2, 2, 2]:
                raise ValueError("Not the original six real right45 yaw actions")
            if any(math.dist(r["position"], gate["state"]["position"]) > 1e-5
                   for r in external):
                raise ValueError("Actual scan translated the robot")
            handoff = detail["handoff"]
            if (math.dist(handoff["state"]["position"], gate["state"]["position"]) > 1e-5
                    or pose.heading_error(handoff["state"]["orientation_wxyz"],
                                          gate["state"]["orientation_wxyz"]) > 0.05
                    or handoff["charged_native_limit"] !=
                       protocol["charged_native_main_step_limit"]):
                raise ValueError("Physical scan did not restore state or charge budget")
            scans[arm] = external
        elif detail["handoff"]["charged_native_limit"] != protocol["native_main_step_limit"]:
            raise ValueError("Virtual placebo incorrectly charged a yaw scan")
        six.validate_histories(root, arm, TARGET, detail, gate, arm in RESETS)
        folder = root / arm / "checkpoints" / "_".join(TARGET)
        inputs = pilot.read_json(folder / "diagnostic_inputs.json")
        decisions = pilot.read_json(folder / "system2_outputs.json")
        chosen = pilot.read_json(folder / "system2_history_selections.json")
        if not decisions or not chosen or len(inputs) != 1:
            raise ValueError("Incomplete S2 text / history / injection audit")
        if not any(r["step"] == gate["step"] for r in decisions):
            raise ValueError("Locked first System2 response not captured")
        if (inputs[0]["injected"] is not (arm != "scan_no_injection_reset")
                or inputs[0]["step"] != gate["step"]):
            raise ValueError("Diagnostic input condition mislabeled")
        if arm == "virtual_placebo_reset" and (
                inputs[0]["external_actions"] != 0
                or inputs[0]["prompt"] != iso.EXTRA_PROMPT):
            raise ValueError("Virtual placebo input is not matched")
        result = capture.read_progress(root / arm / "progress.json")
        if set(result) != {TARGET}:
            raise ValueError("New episode in single-context diagnostic")
        val = result[TARGET]
        if not math.isfinite(float(val["ne"])):
            raise ValueError("Nonfinite final navigation error")
        if arm in OLD_MAP:
            reference = [
                r for r in physical.read_lines(
                    six_arm / OLD_MAP[arm] / "turn_actions.jsonl")
                if capture.episode_key(r) == TARGET
            ]
            compared = pilot.compare_traces(reference, rows)
            if not compared["actions_and_poses_match"]:
                raise ValueError("Logged reference replay differs from frozen old arm")
            if (val["success"] != frozen[arm]["success"]
                    or abs(val["ne"] - frozen[arm]["ne"]) > 1e-5):
                raise ValueError("Reference outcome is not reproducible")
        results[arm] = {
            "success": int(val["success"]), "ne": float(val["ne"]),
            "spl": float(val["spl"]), "os": float(val["os"]),
            "extra_actions": expected_external,
            "total_s2_outputs": len(decisions),
            "s2_outputs_near_step33": [
                d for d in decisions if 28 <= d["step"] <= 36
            ],
            "injection": inputs,
            "num_history_selections": len(chosen),
        }
    # Validate that real scans are identical to the previously audited
    # source arm, including all six actual Habitat action poses.
    old_scan = [
        r for r in physical.read_lines(
            six_arm / OLD_MAP["replay_placebo_reset"] / "turn_actions.jsonl")
        if capture.episode_key(r) == TARGET and
        r["phase"] == "external_recovery"
    ]
    if any(not pilot.compare_traces(old_scan, trace)[
            "actions_and_poses_match"] for trace in scans.values()):
        raise ValueError("Diagnostic yaw scan differs from prior frozen scan")
    output = {
        "schema": "episode13_mechanism_ablation_report.v1",
        "status": "EP13_MECHANISM_COMPARABLE",
        "target": list(TARGET), "checkpoint_step": gate["step"],
        "reference_replays_equal_frozen_traces": True,
        "real_scan_actions_identical_to_frozen_reference": True,
        "prior_keep_success": int(original["success"]),
        "prior_placebo_reset_success": int(
            baseline["outcomes"]["placebo_reset"]["success"]),
        "results": results,
        "scope": SCOPE,
        "not_estimated": [
            "held_out_generalization", "task_progress_drift",
            "causal_memory_semantics", "recovering_native_failure",
        ],
    }
    pilot.write_json(root / "diagnostic_report.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    a = cmd.add_parser("prepare")
    a.add_argument("--six-arm-run", required=True)
    a.add_argument("--run-dir", required=True)
    a.set_defaults(fn=prepare)
    a = cmd.add_parser("run")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--arm", choices=ARMS, required=True)
    a.set_defaults(fn=run)
    a = cmd.add_parser("report")
    a.add_argument("--run-dir", required=True)
    a.set_defaults(fn=report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
