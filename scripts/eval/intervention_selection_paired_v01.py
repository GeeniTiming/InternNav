"""v0.1 paired intervention rollouts: two contexts x three new actions.

Frozen source: 12 registered TRAIN scenes; only 10 eligible natural gates and
their audited physical-perturbation KEEP continuations are available. Each
run replays the original native prefix. The perturbed arm also repeats the
three actual, predeclared yaw actions and checks the resulting RGB/pose/RNG
against *perturbed KEEP* before adding the selected operation.

PREPARE: CPU only. RUN: 6 create-only multi-episode GPU arms.
REPORT: fail-closed paired audit. No model training / held-out claims.

This file does not edit the original InternNav evaluator or previous data.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_selection_perturbed_keep_v01 as pert
import recovery_evidence_isolation as iso

natural, ds = pert.natural, pert.ds
pilot, capture, physical, common, pose = (
    pert.pilot, pert.capture, pert.physical, pert.common, pert.pose
)
prior = iso.prior
SCHEMA = "intervention_selection_paired.v0.1"
MODES = ("natural", "perturbed")
NEW_ACTIONS = ("SEEK_LEFT45", "SEEK_RIGHT45", "HISTORY_RESET")
ARMS = tuple(m + "_" + a.lower() for m in MODES for a in NEW_ACTIONS)
SCAN_NAMES = {
    "SEEK_LEFT45": "lookback_left45_evidence",
    "SEEK_RIGHT45": "lookback_right45_evidence",
}
SCOPE = (
    "Prospectively locked TRAIN development contexts only. Actual paired "
    "rollouts from identical pre-arm natural or physically perturbed gate; "
    "real yaw for lookback scan and explicit post-gate history indexing for "
    "HISTORY_RESET. No inferred Q labels, held-out efficacy, oracle state "
    "features, or outcome-dependent replacement."
)


def split_arm(arm):
    if arm not in ARMS:
        raise ValueError("Invalid registered v0.1 arm: " + str(arm))
    m, a = arm.split("_", 1)
    operation = a.upper()
    if m not in MODES or operation not in NEW_ACTIONS:
        raise ValueError("Malformed mode/action")
    return m, operation


def eligible_sources(pert_root, max_cases):
    pert_root = Path(pert_root).resolve()
    natural_root, native_root, perturb_protocol = pert.read_run(pert_root)
    pert_report = pilot.read_json(pert_root / "perturbed_keep_report.json")
    if (pert_report.get("schema") != pert.SCHEMA
            or pert_report.get("status") != "PERTURBED_KEEP_AUDITED"
            or pert_report.get("N_original_cases") != ds.N_CASES
            or pert_report.get("N_eligible_cases") != len(perturb_protocol["cases"])
            or pert_report.get("N_perturbed_keep_outcomes") !=
               len(perturb_protocol["cases"])):
        raise ValueError("Require audited perturbed KEEP on all eligible contexts")
    natural_report = pilot.read_json(
        natural_root / "natural_checkpoint_report.json")
    if (natural_report.get("schema") != natural.SCHEMA
            or natural_report.get("N_cases_predeclared") != ds.N_CASES
            or natural_report.get("N_captured_gates") !=
               len(perturb_protocol["cases"])
            or natural_report.get("N_ineligible_retained") +
               len(perturb_protocol["cases"]) != ds.N_CASES):
        raise ValueError("The original 12-case denominator was changed")
    keys = [tuple(c["key"]) for c in perturb_protocol["cases"]]
    outcome_keys = [tuple(c["episode_key"]) for c in pert_report["cases"]]
    if keys != outcome_keys or len(keys) != len(set(keys)):
        raise ValueError("Perturbed KEEP results differ from frozen cohort/order")
    if max_cases not in (2, len(keys)):
        raise ValueError(
            "Only preregistered first-two-case pilot or all eligible cases allowed")
    if max_cases > len(keys):
        raise ValueError("Not enough eligible cases for pilot")
    selected = copy.deepcopy(perturb_protocol["cases"][:max_cases])
    if not selected or any(c["checkpoint"] is None for c in selected):
        raise ValueError("A previously ineligible checkpoint cannot be selected")
    return natural_root, native_root, pert_root, perturb_protocol, selected


def prepare(args):
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Paired directory is create-only")
    natural_root, source, pert_root, inherited, selected = eligible_sources(
        args.perturbed_keep_run, args.max_cases)
    source_m = pilot.verify_manifest(pert_root)
    config = pilot.load_config(source / "control.py")
    native_limit = int(config.eval_settings["max_steps_per_episode"])
    if (config.agent.model_settings.get("mode") != "dual_system"
            or config.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Requires frozen cache-disabled DualVLN")
    if any(c["checkpoint"]["step"] + 9 >= native_limit for c in selected):
        raise ValueError("No continuation budget after perturbation and scan")
    native_dataset = pilot.read_json(source / "episodes.json.gz")
    eps = {ds.key_of(e): e for e in native_dataset["episodes"]}
    selected_dataset = copy.deepcopy(native_dataset)
    selected_dataset["episodes"] = [
        copy.deepcopy(eps[tuple(c["key"])]) for c in selected
    ]
    if len(selected_dataset["episodes"]) != len(selected):
        raise ValueError("Subset episode selection mismatch")
    import yaml
    habitat = yaml.safe_load((source / "habitat.yaml").read_text("utf-8"))
    if habitat["habitat"]["dataset"]["split"] != "train":
        raise ValueError("Only TRAIN data allowed")
    protocol = {
        "schema": SCHEMA, "status": "SIX_ARMS_PENDING",
        "source_natural_gate_run": str(natural_root),
        "source_native_run": str(source),
        "source_perturbed_keep_run": str(pert_root),
        "original_cases": ds.N_CASES,
        "original_eligible_cases": len(inherited["cases"]),
        "selected_cases": selected,
        "max_cases": args.max_cases,
        "mode": "PILOT_FIRST_TWO" if args.max_cases == 2 else "FULL_ELIGIBLE",
        "arms": list(ARMS),
        "native_main_step_limit": native_limit,
        "costs": {"perturbed_prefix_yaw": 3, "seek_extra_yaw": 6,
                  "reset_extra_yaw": 0},
        "no_outcome_dependent_selection": True, "scope": SCOPE,
    }
    root.mkdir(parents=True, exist_ok=False)
    data_file = root / "episodes.json.gz"
    with data_file.open("wb") as fh:
        with gzip.GzipFile(fileobj=fh, filename="", mode="wb", mtime=0) as gz:
            gz.write(json.dumps(
                selected_dataset, ensure_ascii=False).encode("utf-8"))
    habitat["habitat"]["dataset"]["data_path"] = str(data_file)
    hpath = root / "habitat.yaml"
    hpath.write_text("# @package _global_\n" + yaml.safe_dump(
        habitat, sort_keys=False), encoding="utf-8")
    files = [data_file, hpath]
    for arm in ARMS:
        cfg = copy.deepcopy(config)
        cfg.env.env_settings["config_path"] = str(hpath)
        cfg.eval_settings["output_path"] = str(root / arm)
        cfg.agent.model_settings["vis_debug"] = True
        cfg.agent.model_settings["vis_debug_path"] = str(root / "video" / arm)
        serialized = json.dumps(cfg.model_dump(mode="json", exclude_none=True))
        p = root / (arm + ".py")
        p.write_text(
            "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
            + "eval_cfg = EvalCfg.model_validate(json.loads("
            + repr(serialized) + "))\n", encoding="utf-8")
        files.append(p)
    protocol_file = root / "paired_protocol.json"
    pilot.write_json(protocol_file, protocol)
    files.append(protocol_file)
    watched = [
        pert_root / "manifest.json", pert_root / "perturbed_keep_report.json",
        natural_root / "manifest.json",
        natural_root / "natural_checkpoint_report.json",
        natural_root / "control/actions.jsonl",
        pert_root / pert.ARM / "turn_actions.jsonl",
        pert_root / pert.ARM / "progress.json",
        source / "manifest.json",
        Path(__file__).resolve(), Path(pert.__file__).resolve(),
        Path(natural.__file__).resolve(), Path(iso.__file__).resolve(),
        Path(prior.__file__).resolve(), Path(pose.__file__).resolve(),
        Path(physical.__file__).resolve(),
    ]
    for c in selected:
        suffix = "_".join(c["key"])
        natural_folder = natural_root / "control/checkpoints" / suffix
        pert_folder = pert_root / pert.ARM / "checkpoints" / suffix
        watched.extend([
            natural_folder / "checkpoint.json",
            natural_folder / "predecision_input.json",
            natural_folder / "current_model_rgb.png",
            pert_folder / "handoff.json",
            pert_folder / "predecision_input.json",
            pert_folder / "post_perturb_current_model_rgb.png",
        ])
    prior_hashes = {
        **source_m["source_hashes"], **source_m["generated_hashes"]
    }
    prior_hashes.update({str(p): pilot.sha256(p) for p in watched})
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "schema": SCHEMA, "revision": revision,
        "seed_base": source_m["seed_base"],
        "episodes": [{"scene_id": e["scene_id"],
                      "episode_id": str(e["episode_id"])}
                     for e in selected_dataset["episodes"]],
        "source_hashes": prior_hashes,
        "checkpoint_shard_sizes": source_m["checkpoint_shard_sizes"],
        "generated_hashes": {str(f): pilot.sha256(f) for f in files},
        "no_interventions_executed_at_prepare": True,
        "linked_perturbed_keep": str(pert_root),
    }
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    print("V01_PAIRED_PREPARE=PASS", len(selected),
          "episodes x 2 modes x 3 missing arms =", len(selected)*6)
    print("GPU_ARMS_NOT_EXECUTED; LABELS_UNAVAILABLE")
    return protocol


def read_run(root):
    root = Path(root).resolve()
    manifest = pilot.verify_manifest(root)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Wrong paired v0.1 manifest")
    natural_root, source, pert_root, inherited, selected = eligible_sources(
        manifest["linked_perturbed_keep"],
        len(manifest["episodes"]),
    )
    saved = pilot.read_json(root / "paired_protocol.json")
    rebuilt = {
        "schema": SCHEMA, "status": "SIX_ARMS_PENDING",
        "source_natural_gate_run": str(natural_root),
        "source_native_run": str(source),
        "source_perturbed_keep_run": str(pert_root),
        "original_cases": ds.N_CASES,
        "original_eligible_cases": len(inherited["cases"]),
        "selected_cases": selected,
        "max_cases": len(selected),
        "mode": "PILOT_FIRST_TWO" if len(selected)==2 else "FULL_ELIGIBLE",
        "arms": list(ARMS),
        "native_main_step_limit": int(
            pilot.load_config(source / "control.py").
            eval_settings["max_steps_per_episode"]),
        "costs": {"perturbed_prefix_yaw": 3, "seek_extra_yaw": 6,
                  "reset_extra_yaw": 0},
        "no_outcome_dependent_selection": True, "scope": SCOPE,
    }
    if saved != rebuilt or manifest["episodes"] != [
        {"scene_id": e["scene_id"], "episode_id": str(e["episode_id"])}
        for e in pilot.read_json(root / "episodes.json.gz")["episodes"]
    ]:
        raise ValueError("Paired protocol changed after frozen prepare")
    return natural_root, source, pert_root, saved


class PairProbe(iso.MechanismProbe):
    def __init__(self, evaluator, root, arm, protocol, references):
        super().__init__(evaluator, root, arm, protocol, references)
        self.output_arm = arm
        self.mode, self.operation = split_arm(arm)
        self.arm = (
            SCAN_NAMES[self.operation] if self.operation in SCAN_NAMES
            else "history_reset"
        )
        self.prearm = None
        self.first_reset_output = None
        self.prearm_rgb_digest = None

    def on_boundary(self, step, observations, done, history, action_seq,
                    pixel_goal, local_actions, action):
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({
            "scene_id": ep.scene_id, "episode_id": ep.episode_id
        })
        if key != self.current:
            self.finish()
            self.begin(key)
        self.current_step = int(step)
        expected = self.cases[key]["checkpoint"]
        if self.gate is not None or step < expected["step"]:
            return observations, done
        if (step != expected["step"]
                or len(self.records) != expected["prefix_action_count"]
                or done or action_seq or pixel_goal is not None
                or local_actions or action not in (0, 1, 2, 3)):
            raise RuntimeError("Paired native S2 gate changed")
        pitch = natural.checkpoint.camera_pitch_degrees(
            natural.checkpoint.quaternion_list(
                self.evaluator.env._env.sim.get_agent_state().
                sensor_states["rgb"].rotation))
        if abs(pitch) > 1e-3:
            raise RuntimeError("S2 camera not level")
        gate = {
            "step": int(step), "prefix_action_count": len(self.records),
            "state": self.state(),
            "observation_hashes": physical.observation_hashes(observations),
            "history_length": len(history),
            "history_sha256": physical.history_hash(history),
            "rng_sha256": common.rng_digest(),
            "parameters_sha256": self.parameters,
            "prefix_trace_sha256":
                natural.checkpoint.cohort.digest(self.records),
        }
        common.same_fields(gate, expected, tuple(gate), "native paired prefix")
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError("Frozen native parameters changed")
        self.gate, self.history_prefix = gate, tuple(history)
        latest = observations
        self.phase = "alignment" if self.mode == "perturbed" else "continuation"
        perturb_actions = (
            pert.expected_yaw_actions(self.cases[key]["perturbation_plan"])
            if self.mode == "perturbed" else []
        )
        if perturb_actions:
            with physical.preserve_policy_rng():
                for code in perturb_actions:
                    latest, _, ended, _ = self.evaluator.env.step(code)
                    if ended:
                        raise RuntimeError("Perturbation ended the episode")
            if self.forced_count != 3:
                raise RuntimeError("Not three real perturbation turns")
        pre_state = self.state()
        pre_hash = physical.observation_hashes(latest)
        if (common.rng_digest() != gate["rng_sha256"]
                or physical.history_hash(history) != gate["history_sha256"]
                or common.parameter_versions(self.evaluator.model) != self.parameters):
            raise RuntimeError("Paired pre-arm state changed RNG/history/model")
        from PIL import Image
        current = Image.fromarray(latest["rgb"]).convert("RGB").resize(
            (self.evaluator.model_args.resize_w,
             self.evaluator.model_args.resize_h))
        model_rgb_hash = physical.history_hash([current])
        natural_root = Path(self.protocol["source_natural_gate_run"])
        folder_id = "_".join(key)
        if self.mode == "natural":
            reference = pilot.read_json(
                natural_root / "control/checkpoints" / folder_id /
                "predecision_input.json")
            expected_obs = gate["observation_hashes"]
            expected_state = gate["state"]
            expected_actions = []
        else:
            pert_root = Path(self.protocol["source_perturbed_keep_run"])
            frozen = pert_root / pert.ARM / "checkpoints" / folder_id
            reference = pilot.read_json(frozen / "predecision_input.json")
            handoff = pilot.read_json(frozen / "handoff.json")
            expected_obs = handoff["final_observation_hashes"]
            expected_state = handoff["state"]
            expected_actions = perturb_actions
            if (handoff["forced_actions"] != 3
                    or handoff["charged_native_limit"] !=
                       self.original_limit - 3):
                raise RuntimeError("Original perturbed KEEP budget changed")
        if (pre_hash != expected_obs or
                math.dist(pre_state["position"],
                          expected_state["position"]) > 1e-5 or
                pose.heading_error(pre_state["orientation_wxyz"],
                                   expected_state["orientation_wxyz"]) > .05 or
                reference["current_pixel_sha256"] != model_rgb_hash or
                reference["perturbation_actions_before_context"] !=
                   expected_actions or
                reference["historical_rgb_history_sha256"] !=
                   gate["history_sha256"]):
            raise RuntimeError("NOT_COMPARABLE: pre-arm context differs from KEEP")
        self.prearm = {
            "context_id": reference["context_id"],
            "mode": self.mode, "key": list(key),
            "state": pre_state, "observation_hashes": pre_hash,
            "current_rgb_pixel_sha256": model_rgb_hash,
            "history_sha256": gate["history_sha256"],
            "rng_sha256": gate["rng_sha256"],
            "parameters_sha256": self.parameters,
            "perturbation_actions": perturb_actions,
            "predecision_reference": reference["context_id"],
        }
        self.prearm_rgb_digest = model_rgb_hash
        folder = self.directory / "checkpoints" / folder_id
        folder.mkdir(parents=True, exist_ok=False)
        pilot.write_json(folder / "boundary.json", gate)
        pilot.write_json(folder / "prearm.json", self.prearm)

        if self.operation in SCAN_NAMES:
            self.phase = "external_recovery"
            outbound_angle = iso.ARM_ANGLES[self.arm]
            origin = pre_state["orientation_wxyz"]
            target = pose.yaw_quaternion(
                pose.yaw_degrees(origin) + outbound_angle)
            collected = [latest]
            def advance(code):
                result = self.evaluator.env.step(code)
                collected[0] = result[0]
                if result[2]:
                    raise RuntimeError("Physical scan terminated episode")
                return result
            with physical.preserve_policy_rng():
                n1 = pose.turn_to_heading(
                    target, self.state, advance, budget=3)
                if n1 != 3:
                    raise RuntimeError("Outbound scan did not take 3 turns")
                self.view_image = Image.fromarray(
                    collected[0]["rgb"]).convert("RGB").resize((
                        self.evaluator.model_args.resize_w,
                        self.evaluator.model_args.resize_h,
                    ))
                self.view_sha256 = physical.history_hash([self.view_image])
                n2 = pose.turn_to_heading(
                    origin, self.state, advance, budget=3)
                if n2 != 3:
                    raise RuntimeError("Inbound scan did not take 3 turns")
            latest = collected[0]
            end = self.state()
            if (math.dist(end["position"], pre_state["position"]) > 1e-5
                    or pose.heading_error(
                        end["orientation_wxyz"], origin) > .05):
                raise RuntimeError("Scan did not return to same pose")
            self.external_actions = 6
            self.inject_pending = True
        else:
            self.external_actions = 0
            self.history_cutoff = len(history)
            self.reset_enabled = True

        if (common.rng_digest() != gate["rng_sha256"]
                or physical.history_hash(history) != gate["history_sha256"]
                or common.parameter_versions(self.evaluator.model) !=
                   self.parameters):
            raise RuntimeError("Scan/reset changed policy RNG/history/parameters")
        cost = len(perturb_actions) + self.external_actions
        self.evaluator.max_steps_per_episode = self.original_limit - cost
        if self.evaluator.max_steps_per_episode <= int(step):
            raise RuntimeError("No budget left after charged actions")
        self.phase = "continuation"
        handoff = {
            "state": self.state(), "prearm": self.prearm,
            "external_actions": self.external_actions,
            "history_reset": bool(self.reset_enabled),
            "history_cutoff": self.history_cutoff,
            "original_history_unchanged": True,
            "policy_rng_restored": True, "model_unchanged": True,
            "charged_native_limit": self.evaluator.max_steps_per_episode,
            "final_observation_hashes": physical.observation_hashes(latest),
        }
        self.handoff = handoff
        pilot.write_json(folder / "handoff.json", handoff)
        if self.operation in SCAN_NAMES:
            self.view_image.save(folder / "outward_view_model_rgb.png")
            Image.fromarray(latest["rgb"]).save(folder / "after_return.png")
        print("V01_PAIR_PREARM_VERIFIED", self.output_arm, key, step,
              "extra", cost, flush=True)
        return latest, False

    def inject_extra_view(self, step, sources, input_images):
        if self.operation in SCAN_NAMES:
            iso.MechanismProbe.inject_extra_view(
                self, step, sources, input_images)
        elif self.inject_pending:
            raise RuntimeError("Reset unexpectedly requested supplementary image")

    def record_system2_output(self, step, message):
        if self.operation in SCAN_NAMES:
            iso.MechanismProbe.record_system2_output(self, step, message)
        elif self.first_reset_output is None and self.gate is not None:
            self.first_reset_output = {
                "key": list(self.current), "step": int(step),
                "text": str(message),
            }

    def finish(self):
        key = self.current
        if key is None:
            return
        if self.prearm is None:
            raise RuntimeError("No verified context for paired outcome")
        if self.operation in SCAN_NAMES:
            iso.MechanismProbe.finish(self)
        else:
            prior.CandidateProbe.finish(self)
        self.finished[key]["prearm"] = self.prearm
        self.finished[key]["operation"] = self.operation
        self.finished[key]["mode"] = self.mode
        if self.operation == "HISTORY_RESET":
            if self.first_reset_output is None:
                raise RuntimeError("RESET had no first S2 response")
            self.finished[key]["first_system2_response"] = (
                self.first_reset_output)


def run(args):
    root = Path(args.run_dir).resolve()
    natural_root, native_root, pert_root, protocol = read_run(root)
    mode, operation = split_arm(args.arm)
    if (root / args.arm).exists():
        raise FileExistsError("Create-only arm exists; retain attempts and retry fresh")
    keys = {tuple(c["key"]) for c in protocol["selected_cases"]}
    references = {k: [] for k in keys}
    for row in physical.read_lines(natural_root / "control/actions.jsonl"):
        k = capture.episode_key(row)
        if k in references:
            references[k].append(row)
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    orig_init, orig_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("Unexpected policy KV cache enabled")
        orig_init(evaluator, cfg)
        probe = PairProbe(
            evaluator, root, args.arm, protocol, references)
        probes.append(probe)
        evaluator._recovery_candidate = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate

        def counted(*aa, **kw):
            probe.s2_calls += 1
            return probe.original_generate(*aa, **kw)

        evaluator.model.generate = counted

    cls.__init__, cls._run_eval_dual_system = (
        initialize, iso.instrument_loop(orig_loop))
    error = None
    try:
        pilot.run_arm(SimpleNamespace(
            run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Expected one native evaluator")
        probes[0].finish()
        if set(probes[0].finished) != keys:
            raise RuntimeError("Incomplete predeclared paired cohort")
        probes[0].status = "complete"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = orig_init, orig_loop
        for p in probes:
            p.evaluator.env.step = p.original_step
            p.evaluator.model.generate = p.original_generate
            p.close(error)
    print("V01_PAIR_ARM_RUN=COMPLETE", args.arm,
          "AUDIT_NOT_YET_COMPLETE", flush=True)


def audit_arm(root, arm, protocol):
    mode, operation = split_arm(arm)
    folder = root / arm
    summary = pilot.read_json(folder / "pilot_summary.json")
    report = pilot.read_json(folder / "turn_summary.json")
    if summary["status"] != "complete" or report["status"] != "complete":
        raise ValueError("Incomplete arm: " + arm)
    native_root = Path(protocol["source_native_run"])
    common.same_fields(
        summary, pilot.read_json(native_root / "control/pilot_summary.json"),
        common.RUNTIME_FIELDS, "paired " + arm)
    detail = {tuple(x["key"]): x for x in report["cases"]}
    keys = {tuple(c["key"]) for c in protocol["selected_cases"]}
    progress = capture.read_progress(folder / "progress.json")
    if set(detail) != keys or set(progress) != keys:
        raise ValueError("Unexpected paired episode identities " + arm)
    tagged = physical.validate_trace(
        physical.read_lines(folder / "turn_actions.jsonl"), arm)
    plain = physical.validate_trace(
        physical.read_lines(folder / "actions.jsonl"), arm + " native")
    if not pilot.compare_traces(tagged, plain)["actions_and_poses_match"]:
        raise RuntimeError("Observer/native trace changed " + arm)
    ref = {k: [] for k in keys}
    natural_root = Path(protocol["source_natural_gate_run"])
    for r in physical.read_lines(natural_root / "control/actions.jsonl"):
        k = capture.episode_key(r)
        if k in ref:
            ref[k].append(r)
    rows = []
    for c in protocol["selected_cases"]:
        k = tuple(c["key"])
        d, gate = detail[k], c["checkpoint"]
        common.same_fields(d["gate"], gate, tuple(gate), "paired native gate")
        trace = [r for r in tagged if capture.episode_key(r) == k]
        prefix = gate["prefix_action_count"]
        if not pilot.compare_traces(
            ref[k][:prefix], trace[:prefix])[
                "actions_and_poses_match"]:
            raise RuntimeError("Paired native action prefix diverged")
        if (progress[k]["episode_instruction"] !=
                capture.read_progress(native_root / "control/progress.json")[
                    k]["episode_instruction"]):
            raise RuntimeError("Instruction differed in paired rollout")
        alignment = [r for r in trace if r["phase"] == "alignment"]
        expected_alignment = (
            pert.expected_yaw_actions(c["perturbation_plan"])
            if mode == "perturbed" else []
        )
        if [r["action"] for r in alignment] != expected_alignment:
            raise RuntimeError("Pre-arm perturbation actions changed")
        if mode == "perturbed":
            frozen = [
                r for r in physical.read_lines(
                    Path(protocol["source_perturbed_keep_run"]) /
                    pert.ARM / "turn_actions.jsonl")
                if capture.episode_key(r) == k and r["phase"] == "alignment"
            ]
            if not pilot.compare_traces(
                    frozen, alignment)["actions_and_poses_match"]:
                raise RuntimeError("Three-turn perturbation no longer matches KEEP")
        prearm = pilot.read_json(
            folder / "checkpoints" / "_".join(k) / "prearm.json")
        if prearm != d["prearm"]:
            raise RuntimeError("Pre-arm fingerprints do not match runtime")
        expected_input_path = (
            natural_root / "control/checkpoints" / "_".join(k) /
            "predecision_input.json"
            if mode == "natural" else
            Path(protocol["source_perturbed_keep_run"]) /
            pert.ARM / "checkpoints" / "_".join(k) /
            "predecision_input.json"
        )
        expected_input = pilot.read_json(expected_input_path)
        if (prearm["current_rgb_pixel_sha256"] !=
                expected_input["current_pixel_sha256"]
                or prearm["history_sha256"] != gate["history_sha256"]
                or prearm["rng_sha256"] != gate["rng_sha256"]
                or prearm["context_id"] != expected_input["context_id"]):
            raise RuntimeError("Pre-arm RGB/history/RNG mismatch with KEEP")
        external = [r for r in trace if r["phase"] == "external_recovery"]
        if operation in SCAN_NAMES:
            direction = 2 if operation == "SEEK_LEFT45" else 3
            seq = [direction] * 3 + [5-direction] * 3
            if [r["action"] for r in external] != seq:
                raise RuntimeError("Not the frozen 6-action SEEK scan")
            if (d["handoff"]["external_actions"] != 6
                    or d["handoff"]["history_reset"]
                    or d["handoff"]["charged_native_limit"] !=
                       protocol["native_main_step_limit"] -
                       len(expected_alignment) - 6):
                raise RuntimeError("Scan history/cost accounting changed")
            input_file = pilot.read_json(
                folder / "checkpoints" / "_".join(k) / "evidence_input.json")
            if (len(input_file["events"]) != 1
                    or input_file["events"][0]["source"] != "outward_view"
                    or input_file["events"][0]["supplementary_pixel_sha256"]
                       != input_file["scan_view_sha256"]
                    or len(d["injection_events"]) != 1):
                raise RuntimeError("SEEK did not inject exactly one true scan RGB")
            from PIL import Image
            with Image.open(
                folder / "checkpoints" / "_".join(k) /
                "outward_view_model_rgb.png"
            ) as im:
                if physical.history_hash([im.copy()]) != (
                        input_file["scan_view_sha256"]):
                    raise RuntimeError("Injected RGB file hash differs")
            if not d.get("first_system2_response"):
                raise RuntimeError("No first S2 response for scan")
        else:
            if external or not d["handoff"]["history_reset"]:
                raise RuntimeError("RESET moved robot or was not activated")
            if d["handoff"]["charged_native_limit"] != (
                protocol["native_main_step_limit"]-len(expected_alignment)
            ):
                raise RuntimeError("RESET budget mismatch")
            events = pilot.read_json(
                folder / "checkpoints" / "_".join(k) / "history_inputs.json")
            if not events or len(events) != d["history_prompt_events"]:
                raise RuntimeError("No actual post-gate history reset events")
            if (events[0]["main_step"] != gate["step"]
                    or events[0]["selected_history_ids"]
                    or any(i < event["cutoff"]
                           for event in events
                           for i in event["selected_history_ids"])):
                raise RuntimeError("Pre-gate history leaked into RESET")
        rows.append({
            "context_id": prearm["context_id"], "episode_key": list(k),
            "source_type": mode, "action": operation,
            "success": int(progress[k]["success"]),
            "ne": float(progress[k]["ne"]),
            "spl": float(progress[k]["spl"]),
            "os": float(progress[k]["os"]),
            "extra_yaw_actions": len(external),
            "prefix_perturbation_actions": len(expected_alignment),
            "audit": "PAIRED_PREARM_VERIFIED",
        })
    return rows


def report(args):
    root = Path(args.run_dir).resolve()
    natural_root, native_root, pert_root, protocol = read_run(root)
    from collections import defaultdict
    all_rows = []
    for arm in ARMS:
        all_rows.extend(audit_arm(root, arm, protocol))
    control_natural = capture.read_progress(
        natural_root / "control/progress.json")
    control_perturbed = capture.read_progress(
        pert_root / pert.ARM / "progress.json")
    by_context = defaultdict(dict)
    for row in all_rows:
        key = (row["context_id"], row["action"])
        if row["action"] in by_context[row["context_id"]]:
            raise ValueError("Duplicate action in same context")
        by_context[row["context_id"]][row["action"]] = row
    for c in protocol["selected_cases"]:
        key = tuple(c["key"])
        cid = c["case_id"]
        for mode, control in (
            ("natural", control_natural),
            ("perturbed", control_perturbed),
        ):
            name = cid + "_" + mode
            if name not in by_context:
                raise ValueError("Missing paired context " + name)
            if set(by_context[name]) != set(NEW_ACTIONS):
                raise ValueError("Incomplete predeclared context")
            baseline = control[key]
            by_context[name]["KEEP"] = {
                "context_id": name, "episode_key": list(key),
                "source_type": mode, "action": "KEEP",
                "success": int(baseline["success"]),
                "ne": float(baseline["ne"]),
                "spl": float(baseline["spl"]),
                "os": float(baseline["os"]),
                "extra_yaw_actions": 0,
                "prefix_perturbation_actions": 3 if mode=="perturbed" else 0,
                "audit": "REUSED_AUDITED_KEEP",
            }
    context_rows = []
    outcome_rows = []
    for c in protocol["selected_cases"]:
        key = tuple(c["key"])
        for mode in MODES:
            cid = c["case_id"] + "_" + mode
            original_input = (
                natural_root / "control/checkpoints" / "_".join(key) /
                "predecision_input.json" if mode == "natural" else
                pert_root / pert.ARM / "checkpoints" / "_".join(key) /
                "predecision_input.json"
            )
            online = pilot.read_json(original_input)
            if (online["context_id"] != cid
                    or online["episode_key"] != list(key)
                    or online.get("no_privileged_simulator_metrics") is not True):
                raise ValueError("Offline or mismatched online feature record")
            context_rows.append(online)
            outcome_rows.extend(by_context[cid][a] for a in
                                ("KEEP",) + NEW_ACTIONS)
    if (len(context_rows) != 2*len(protocol["selected_cases"])
            or len(outcome_rows) != 8*len(protocol["selected_cases"])):
        raise ValueError("Incomplete audited four-arm outcome table")
    outfile = root / "contexts_predecision.jsonl"
    outcomes = root / "intervention_outcomes.jsonl"
    if outfile.exists() or outcomes.exists():
        raise FileExistsError("Already exported; outputs create-only")
    for file, items in ((outfile, context_rows), (outcomes, outcome_rows)):
        with file.open("x", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "schema": SCHEMA,
        "status": "PILOT_PAIRED_AUDITED" if protocol["max_cases"] == 2
                  else "TEN_CASE_PAIRED_AUDITED",
        "N_original_cases": ds.N_CASES,
        "N_eligible_cases": protocol["original_eligible_cases"],
        "N_selected_cases": len(protocol["selected_cases"]),
        "N_contexts": len(context_rows),
        "N_new_intervention_results": len(all_rows),
        "N_four_arm_outcomes": len(outcome_rows),
        "N_hindsight_preference_labels": 0,
        "outcome_labels_used_as_online_features": False,
        "scope": SCOPE,
    }
    pilot.write_json(root / "paired_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--perturbed-keep-run", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--max-cases", type=int, default=2)
    p.set_defaults(fn=prepare)
    p = commands.add_parser("run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.set_defaults(fn=run)
    p = commands.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(fn=report)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
