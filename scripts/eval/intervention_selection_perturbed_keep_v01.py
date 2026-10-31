"""v0.1 stage B2: real ±45 degree perturbation, then frozen native KEEP.

Only natural gates that passed the separate native checkpoint audit are
eligible. Every perturbed KEEP replays the entire original action prefix,
applies exactly 3 actual Habitat yaw actions at the gate (not a teleport),
records the real returned RGB and sampled pre-gate history, then continues.
No RESET, SEEK or online Q model is trained/executed here.

Natural and perturbed contexts are different starting states. The original
12-case plan and all INELIGIBLE entries remain untouched and count as missing,
not silently replaced. One full replay per eligible perturbed KEEP.
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
import intervention_selection_native_gates_v01 as natural
import recovery_independent_turn as turn

ds = natural.ds
pilot, capture, physical, common = (
    natural.pilot, natural.capture, natural.physical, natural.common
)
pose, checkpoint = turn.pose, natural.checkpoint
SCHEMA = "intervention_selection_perturbed_keep.v0.1"
ARM = "perturbed_keep"
SCOPE = (
    "First predeclared v0.1 physical perturbation (3 real native ±15° yaw "
    "actions) at each previously audited natural TRAIN gate, with 3-step "
    "charged continuation. Unreachable gates retained as ineligible. "
    "The original history is retained, actual new RGB is taken after yaw, "
    "policy RNG restored. This captures only perturbed KEEP, NOT four-arm "
    "selection labels, recovery efficacy, or held-out generalization."
)


def input_cases(natural_root):
    natural_root = Path(natural_root).resolve()
    manifest, source, protocol, summary, progress, refs = natural.read_run(
        natural_root)
    saved = pilot.read_json(natural_root / "natural_checkpoint_report.json")
    if (saved.get("schema") != natural.SCHEMA
            or saved.get("status") not in (
                "NATURAL_GATES_AUDITED",
                "NATURAL_GATES_AUDITED_WITH_INELIGIBLE",
            )
            or len(saved["cases"]) != ds.N_CASES):
        raise ValueError("Require fully audited natural gate report")
    if [tuple(c["episode_key"]) for c in saved["cases"]] != [
            tuple(c["key"]) for c in protocol["cases"]]:
        raise ValueError("Natural report differs from predeclared ordering")
    eligible = []
    for planned, observed in zip(protocol["cases"], saved["cases"]):
        key = tuple(planned["key"])
        if observed["case_id"] != planned["case_id"]:
            raise ValueError("Natural checkpoint case identity changed")
        gate = observed["natural_checkpoint"]
        if gate is None:
            if observed["status"] != "INELIGIBLE_RETAINED_NO_REPLACEMENT":
                raise ValueError("Unrecorded reason for unavailable natural gate")
            continue
        if observed["status"] != "NATURAL_GATE_CAPTURED":
            raise ValueError("Unverified natural checkpoint not allowed")
        original = pilot.read_json(
            natural_root / "control/checkpoints" / "_".join(key) /
            "checkpoint.json")
        if original != gate:
            raise ValueError("Natural RGB checkpoint audit hash changed")
        eligible.append({
            "key": planned["key"],
            "case_id": planned["case_id"],
            "checkpoint": gate,
            "perturbation_plan": planned["perturbation_plan"],
        })
    if not eligible:
        raise ValueError("Zero eligible gates; preserve original plan and stop")
    return natural_root, manifest, source, protocol, saved, eligible


def expected_yaw_actions(perturbation):
    yaw = perturbation["yaw_deg"]
    if yaw not in (-45, 45):
        raise ValueError("Only predeclared ±45 degree perturbations allowed")
    codes = [2, 2, 2] if yaw == 45 else [3, 3, 3]
    if (perturbation["actual_native_yaw_actions"] != codes
            or perturbation["predecision_only"] is not True
            or perturbation["preserve_policy_rng"] is not True):
        raise ValueError("Modified locked perturbation action protocol")
    return codes


def prepare(args):
    dest = Path(args.run_dir).resolve()
    if dest.exists():
        raise FileExistsError("Create-only perturbed KEEP run")
    natural_root, prior, source, natural_protocol, saved, eligible = input_cases(
        args.natural_gate_run)
    for case in eligible:
        expected_yaw_actions(case["perturbation_plan"])
    native_config = copy.deepcopy(pilot.load_config(source / "control.py"))
    if (native_config.agent.model_settings.get("mode") != "dual_system"
            or native_config.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Require frozen cache-disabled native model")
    maximum = int(native_config.eval_settings["max_steps_per_episode"])
    if any(c["checkpoint"]["step"] + 3 >= maximum for c in eligible):
        raise ValueError("Perturbation requires three charged main steps")
    dataset = pilot.read_json(source / "episodes.json.gz")
    ep_by_key = {ds.key_of(e): e for e in dataset["episodes"]}
    if set(ep_by_key) != {tuple(x["key"]) for x in natural_protocol["cases"]}:
        raise ValueError("Original TRAIN native dataset identity changed")
    subset = copy.deepcopy(dataset)
    subset["episodes"] = [copy.deepcopy(ep_by_key[tuple(c["key"])] )
                          for c in eligible]
    import yaml
    habitat = yaml.safe_load((source / "habitat.yaml").read_text("utf-8"))
    if habitat["habitat"]["dataset"]["split"] != "train":
        raise ValueError("Not the frozen TRAIN simulator split")
    revision = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    manifest = {
        "schema": SCHEMA, "seed_base": prior["seed_base"],
        "revision": revision,
        "episodes": [{"scene_id": e["scene_id"],
                      "episode_id": str(e["episode_id"])}
                     for e in subset["episodes"]],
        "source_hashes": {**prior["source_hashes"], **prior["generated_hashes"]},
        "generated_hashes": {},
        "checkpoint_shard_sizes": prior["checkpoint_shard_sizes"],
        "source_natural_gate_run": str(natural_root),
        "no_paired_comparison_finished": True,
        "original_cases_including_ineligible": ds.N_CASES,
    }
    watched = [
        natural_root / "manifest.json",
        natural_root / "natural_checkpoint_report.json",
        natural_root / "control/capture_summary.json",
        natural_root / "control/actions.jsonl",
        source / "manifest.json",
        Path(__file__).resolve(), Path(natural.__file__).resolve(),
        Path(turn.__file__).resolve(), Path(checkpoint.__file__).resolve(),
        Path(pose.__file__).resolve(), Path(physical.__file__).resolve(),
    ]
    for c in eligible:
        folder = natural_root / "control/checkpoints" / "_".join(c["key"])
        watched.extend([
            folder / "checkpoint.json", folder / "context_inputs.json",
            folder / "current_model_rgb.png", folder / "rgb.png",
        ])
        ctx = pilot.read_json(folder / "context_inputs.json")
        watched.extend(folder / frame["path"] for frame in ctx["sampled_history"])
    manifest["source_hashes"].update({str(p): pilot.sha256(p) for p in watched})
    dest.mkdir(parents=True, exist_ok=False)
    data_path = dest / "episodes.json.gz"
    with data_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            out.write(json.dumps(subset, ensure_ascii=False).encode("utf-8"))
    habitat["habitat"]["dataset"]["data_path"] = str(data_path)
    yaml_path = dest / "habitat.yaml"
    yaml_path.write_text(
        "# @package _global_\n" + yaml.safe_dump(habitat, sort_keys=False),
        encoding="utf-8",
    )
    native_config.env.env_settings["config_path"] = str(yaml_path)
    native_config.eval_settings["output_path"] = str(dest / ARM)
    native_config.agent.model_settings["vis_debug"] = True
    native_config.agent.model_settings["vis_debug_path"] = str(dest / "video" / ARM)
    serialized = json.dumps(native_config.model_dump(
        mode="json", exclude_none=True))
    config_path = dest / (ARM + ".py")
    config_path.write_text(
        "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
        + "eval_cfg = EvalCfg.model_validate(json.loads("
        + repr(serialized) + "))\n", encoding="utf-8")
    protocol = {
        "schema": SCHEMA, "status": "PERTURBED_KEEP_PENDING",
        "arm": ARM, "source_natural_gate_run": str(natural_root),
        "cases": eligible, "original_case_count": ds.N_CASES,
        "ineligible_case_keys": [
            c["episode_key"] for c in saved["cases"]
            if c["natural_checkpoint"] is None
        ],
        "native_main_step_limit": maximum,
        "charged_main_step_limit": maximum - 3,
        "perturbation_actions": 3,
        "no_outcome_dependent_replacement": True,
        "scope": SCOPE,
    }
    ppath = dest / "perturb_protocol.json"
    pilot.write_json(ppath, protocol)
    manifest["generated_hashes"] = {
        str(p): pilot.sha256(p)
        for p in (data_path, yaml_path, config_path, ppath)
    }
    pilot.write_json(dest / "manifest.json", manifest)
    pilot.verify_manifest(dest)
    print("V01_PERTURBED_KEEP_PREPARE=PASS", len(eligible),
          "eligible; ineligible retained=", ds.N_CASES-len(eligible))
    return protocol


def read_run(root):
    root = Path(root).resolve()
    manifest = pilot.verify_manifest(root)
    if (manifest.get("schema") != SCHEMA
            or manifest["no_paired_comparison_finished"] is not True):
        raise ValueError("Not an unchanged perturbed KEEP source")
    natural_root, prior, source, previous, saved, eligible = input_cases(
        manifest["source_natural_gate_run"])
    protocol = pilot.read_json(root / "perturb_protocol.json")
    if (protocol["schema"] != SCHEMA
            or protocol["cases"] != eligible
            or protocol["ineligible_case_keys"] != [
                c["episode_key"] for c in saved["cases"]
                if c["natural_checkpoint"] is None
            ]
            or protocol["perturbation_actions"] != 3
            or protocol["charged_main_step_limit"] !=
               protocol["native_main_step_limit"]-3
            or protocol["arm"] != ARM
            or len(manifest["episodes"]) != len(eligible)
            or {ds.key_of(e) for e in manifest["episodes"]} !=
               {tuple(c["key"]) for c in eligible}):
        raise ValueError("Changed original v0.1 perturbation or eligible cohort")
    for c in eligible:
        expected_yaw_actions(c["perturbation_plan"])
    return natural_root, source, protocol


class PerturbedKeepProbe(turn.TurnProbe):
    """Native physical turn plus audit; no injected extra image/no history reset."""

    def on_boundary(self, step, observations, done, history, action_seq,
                    pixel_goal, local_actions, action):
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({
            "scene_id": ep.scene_id, "episode_id": ep.episode_id,
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
            raise RuntimeError("NOT_COMPARABLE: perturb at wrong native S2 gate")
        pitch = checkpoint.camera_pitch_degrees(
            checkpoint.quaternion_list(
                self.evaluator.env._env.sim.get_agent_state().
                sensor_states["rgb"].rotation))
        if abs(pitch) > 1e-3:
            raise RuntimeError("Unlevel native camera at perturbation gate")
        gate = {
            "step": int(step), "prefix_action_count": len(self.records),
            "state": self.state(),
            "observation_hashes": physical.observation_hashes(observations),
            "history_length": len(history),
            "history_sha256": physical.history_hash(history),
            "rng_sha256": common.rng_digest(),
            "parameters_sha256": self.parameters,
            "prefix_trace_sha256": checkpoint.cohort.digest(self.records),
        }
        common.same_fields(gate, expected, tuple(gate), "perturbation native gate")
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError("Native model weights changed")
        self.gate, self.history_prefix = gate, tuple(history)
        planned = self.cases[key]["perturbation_plan"]
        actions = expected_yaw_actions(planned)
        target = pose.yaw_quaternion(
            pose.yaw_degrees(gate["state"]["orientation_wxyz"])
            + planned["yaw_deg"])
        self.phase = "alignment"
        returned = observations
        with physical.preserve_policy_rng():
            for code in actions:
                returned, _, ended, _ = self.evaluator.env.step(code)
                if ended:
                    raise RuntimeError("Perturbation terminated episode")
        if self.forced_count != 3:
            raise RuntimeError("Not exactly 3 real Habitat yaw actions")
        after = self.state()
        if (math.dist(after["position"], gate["state"]["position"]) > 1e-5
                or pose.heading_error(after["orientation_wxyz"], target) > .05):
            raise RuntimeError("Perturbed pose diverged from real ±45-degree target")
        camera_pitch = checkpoint.camera_pitch_degrees(
            checkpoint.quaternion_list(
                self.evaluator.env._env.sim.get_agent_state().
                sensor_states["rgb"].rotation))
        if abs(camera_pitch) > 1e-3:
            raise RuntimeError("Camera no longer level after body yaw")
        if (common.rng_digest() != gate["rng_sha256"]
                or physical.history_hash(history) != gate["history_sha256"]
                or common.parameter_versions(self.evaluator.model) != self.parameters):
            raise RuntimeError("History/RNG/weights changed during perturbation")
        self.evaluator.max_steps_per_episode = self.protocol[
            "charged_main_step_limit"]
        self.phase = "continuation"
        self.handoff = {
            "state": after, "target_orientation_wxyz": target,
            "forced_actions": 3, "history_retained": True,
            "policy_rng_restored": True,
            "final_observation_hashes": physical.observation_hashes(returned),
            "charged_native_limit": self.evaluator.max_steps_per_episode,
        }
        folder = self.directory / "checkpoints" / "_".join(key)
        folder.mkdir(parents=True, exist_ok=False)
        pilot.write_json(folder / "boundary.json", gate)
        pilot.write_json(folder / "handoff.json", self.handoff)
        from PIL import Image
        img = Image.fromarray(returned["rgb"]).convert("RGB")
        img.save(folder / "post_perturb_rgb.png")
        model_img = img.resize((
            self.evaluator.model_args.resize_w,
            self.evaluator.model_args.resize_h,
        ))
        model_img.save(folder / "post_perturb_current_model_rgb.png")
        original_dir = (Path(self.protocol["source_natural_gate_run"]) /
                        "control/checkpoints" / "_".join(key))
        history_desc = pilot.read_json(original_dir / "context_inputs.json")
        pilot.write_json(folder / "predecision_input.json", {
            "schema": "intervention_selection_predecision.v0.1",
            "context_id": self.cases[key]["case_id"] + "_perturbed",
            "episode_key": list(key), "main_step": int(step),
            "source_type": "perturbed",
            "current_rgb": "post_perturb_current_model_rgb.png",
            "current_pixel_sha256": physical.history_hash([model_img]),
            "current_png_sha256": pilot.sha256(
                folder / "post_perturb_current_model_rgb.png"),
            "historical_rgb": [
                {"path": str(original_dir / h["path"]),
                 "native_history_index": h["native_history_index"],
                 "pixel_sha256": h["pixel_sha256"],
                 "png_sha256": h["png_sha256"]}
                for h in history_desc["sampled_history"]
            ],
            "historical_rgb_history_sha256": gate["history_sha256"],
            "perturbation_actions_before_context": actions,
            "no_privileged_simulator_metrics": True,
        })
        print("V01_PERTURBED_GATE=CAPTURED", key, "step=", step,
              "yaw=", planned["yaw_deg"], "charged_actions=3")
        return returned, False


def run(args):
    root = Path(args.run_dir).resolve()
    source_natural, source_native, protocol = read_run(root)
    if (root / ARM).exists():
        raise FileExistsError("Create-only perturb replay already exists")
    references = {tuple(c["key"]): [] for c in protocol["cases"]}
    for row in physical.read_lines(
            source_natural / "control/actions.jsonl"):
        key = capture.episode_key(row)
        if key in references:
            references[key].append(row)
    for c in protocol["cases"]:
        k = tuple(c["key"])
        if len(references[k]) <= c["checkpoint"]["prefix_action_count"]:
            raise ValueError("Not enough original actions to verify gate prefix")
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("KV continuation not allowed")
        original_init(evaluator, cfg)
        probe = PerturbedKeepProbe(evaluator, root, ARM, protocol, references)
        probes.append(probe)
        evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate

        def counted_generate(*aa, **kw):
            probe.s2_calls += 1
            return probe.original_generate(*aa, **kw)

        evaluator.model.generate = counted_generate

    cls.__init__, cls._run_eval_dual_system = (
        initialize, physical.instrument_loop(original_loop))
    error = None
    try:
        pilot.run_arm(SimpleNamespace(
            run_dir=str(root), arm=ARM, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Exactly one model evaluator required")
        probes[0].finish()
        if set(probes[0].finished) != set(references):
            raise RuntimeError("Missing predeclared perturbed cases")
        probes[0].status = "complete"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for p in probes:
            p.evaluator.env.step = p.original_step
            p.evaluator.model.generate = p.original_generate
            p.close(error)
    return report(args)


def report(args):
    root = Path(args.run_dir).resolve()
    natural_root, source, protocol = read_run(root)
    summary = pilot.read_json(root / ARM / "pilot_summary.json")
    diagnostic = pilot.read_json(root / ARM / "turn_summary.json")
    if summary["status"] != "complete" or diagnostic["status"] != "complete":
        raise RuntimeError("Incomplete actual perturbed KEEP capture")
    progress = capture.read_progress(root / ARM / "progress.json")
    rows = physical.validate_trace(
        physical.read_lines(root / ARM / "turn_actions.jsonl"),
        "perturbed KEEP")
    plain = physical.validate_trace(
        physical.read_lines(root / ARM / "actions.jsonl"),
        "perturbed native logger")
    if not pilot.compare_traces(rows, plain)["actions_and_poses_match"]:
        raise RuntimeError("Physical yaw observer and native trace mismatch")
    details = {tuple(x["key"]): x for x in diagnostic["cases"]}
    expected = {tuple(c["key"]) for c in protocol["cases"]}
    if set(progress) != expected or set(details) != expected:
        raise RuntimeError("Perturbed KEEP cohort differs from frozen eligible gates")
    prior = {k: [] for k in expected}
    for r in physical.read_lines(natural_root / "control/actions.jsonl"):
        k = capture.episode_key(r)
        if k in prior:
            prior[k].append(r)
    results = []
    for case in protocol["cases"]:
        key = tuple(case["key"])
        expected_gate = case["checkpoint"]
        detail = details[key]
        gate = detail["gate"]
        common.same_fields(gate, expected_gate, tuple(gate),
                           "real perturb native prefix")
        if not detail["prefix_comparison"]["actions_and_poses_match"]:
            raise RuntimeError("Native prefix replay diverged")
        current = [r for r in rows if capture.episode_key(r) == key]
        prefix = expected_gate["prefix_action_count"]
        if not pilot.compare_traces(
                prior[key][:prefix], current[:prefix])[
                "actions_and_poses_match"]:
            raise RuntimeError("Frozen native prefix does not match")
        yaw = [r for r in current if r["phase"] == "alignment"]
        expected_actions = expected_yaw_actions(case["perturbation_plan"])
        if [r["action"] for r in yaw] != expected_actions:
            raise RuntimeError("Not exactly 3 planned real yaw actions")
        if any(math.dist(
                r["position"], gate["state"]["position"]) > 1e-5
               for r in yaw):
            raise RuntimeError("Perturbation yaw introduced translation")
        handoff = detail["handoff"]
        target = pose.yaw_quaternion(
            pose.yaw_degrees(gate["state"]["orientation_wxyz"]) +
            case["perturbation_plan"]["yaw_deg"])
        if (math.dist(handoff["state"]["position"],
                      gate["state"]["position"]) > 1e-5
                or pose.heading_error(
                    handoff["state"]["orientation_wxyz"], target) > .05
                or handoff["forced_actions"] != 3
                or handoff["charged_native_limit"] !=
                   protocol["charged_main_step_limit"]):
            raise RuntimeError("Post-perturbation pose/step budget mismatch")
        folder = root / ARM / "checkpoints" / "_".join(key)
        if (pilot.read_json(folder / "boundary.json") != gate
                or pilot.read_json(folder / "handoff.json") != handoff):
            raise RuntimeError("Perturbed gate or handoff serialization drift")
        input_record = pilot.read_json(folder / "predecision_input.json")
        image = folder / "post_perturb_current_model_rgb.png"
        if (input_record["schema"] != "intervention_selection_predecision.v0.1"
                or input_record["episode_key"] != list(key)
                or input_record["context_id"] != case["case_id"] + "_perturbed"
                or pilot.sha256(image) != input_record["current_png_sha256"]
                or input_record["perturbation_actions_before_context"] !=
                   expected_actions):
            raise RuntimeError("Perturbed current RGB not saved as true input")
        from PIL import Image
        with Image.open(image) as img:
            if physical.history_hash([img.copy()]) != (
                    input_record["current_pixel_sha256"]):
                raise RuntimeError("Perturbed resized RGB pixel digest mismatch")
        for ref in input_record["historical_rgb"]:
            f = Path(ref["path"])
            if not f.is_file() or pilot.sha256(f) != ref["png_sha256"]:
                raise RuntimeError("Pre-gate historical RGB changed")
        result = progress[key]
        results.append({
            "case_id": case["case_id"], "episode_key": list(key),
            "context_id": case["case_id"] + "_perturbed",
            "status": "PERTURBED_KEEP_AUDITED",
            "perturbation_yaw_deg": case["perturbation_plan"]["yaw_deg"],
            "actual_perturbation_actions": 3,
            "outcome_action": "KEEP",
            "success": int(result["success"]),
            "final_ne_m": float(result["ne"]),
            "spl": float(result["spl"]), "os": float(result["os"]),
            "prefix_verified": True, "actual_post_yaw_rgb_verified": True,
            "history_prefix_unchanged": True,
        })
    out = {
        "schema": SCHEMA, "status": "PERTURBED_KEEP_AUDITED",
        "N_original_cases": ds.N_CASES,
        "N_eligible_cases": len(expected),
        "N_ineligible_retained": ds.N_CASES - len(expected),
        "N_perturbed_contexts_captured": len(results),
        "N_paired_four_arm_contexts_complete": 0,
        "N_perturbed_keep_outcomes": len(results),
        "cases": results, "scope": SCOPE,
    }
    pilot.write_json(root / "perturbed_keep_report.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--natural-gate-run", required=True)
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
