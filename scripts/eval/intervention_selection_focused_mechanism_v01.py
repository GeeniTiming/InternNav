"""Posthoc three-scene, four-arm, same-checkpoint S2 mechanism pilot.

Separate development-only branch. Original v0.1 frozen drivers are imported
unchanged. Explicit outcome-selected scenes are for mechanism investigation,
never a held-out efficacy estimate or training data.
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
import intervention_selection_paired_v01 as paired
import intervention_selection_s2_call_events_v01 as s2

pert, pilot, physical, capture = paired.pert, paired.pilot, paired.physical, paired.capture
iso, common, pose, ds = paired.iso, paired.common, paired.pose, paired.ds

SCHEMA = "intervention_selection_focused_mechanism.v0.1"
FOCUS = (
    ("i5noydFURQK", "70", "right"),
    ("r1Q1Z4BcV1o", "106", "right"),
    ("PX4nDJXEHrG", "79", "left"),
)
ARMS = (
    "perturbed_keep",
    "perturbed_scan_no_injection",
    "perturbed_placebo",
    "perturbed_real_seek",
)
SCOPE = (
    "Posthoc selected three-scene mechanism cases. Same predecision state and "
    "predeclared within-case scan direction. NOT training or held-out efficacy."
)


def select_cases(all_cases):
    indexed = {tuple(map(str, c["key"])): c for c in all_cases}
    if len(indexed) != len(all_cases):
        raise ValueError("Duplicate original case identity")
    selected = []
    for scene, episode, side in FOCUS:
        key = (scene, episode)
        if key not in indexed:
            raise ValueError("Focus checkpoint missing from eligible source: " + repr(key))
        case = copy.deepcopy(indexed[key])
        if not isinstance(case.get("checkpoint"), dict):
            raise ValueError("Uncaptured checkpoint for " + repr(key))
        case["focus_scan_side"] = side
        selected.append(case)
    if len({tuple(c["key"])[0] for c in selected}) != len(FOCUS):
        raise ValueError("Focus selection is not scene-disjoint")
    return selected


def original_sources(pert_root):
    pert_root = Path(pert_root).resolve()
    natural_root, source_root, inherited = pert.read_run(pert_root)
    all_n = len(inherited["cases"])
    # This original validation is read-only and checks the complete 10-case
    # denominator rather than modifying the original selection protocol.
    paired.eligible_sources(pert_root, all_n)
    return natural_root, source_root, pert_root, select_cases(inherited["cases"])


def prepare(args):
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError("Focused experiment directory must be new")
    natural_root, source, pert_root, cases = original_sources(args.perturbed_keep_run)
    src_m = pilot.verify_manifest(pert_root)
    model_cfg = pilot.load_config(source / "control.py")
    native_limit = int(model_cfg.eval_settings["max_steps_per_episode"])
    if (model_cfg.agent.model_settings.get("mode") != "dual_system"
            or model_cfg.agent.model_settings.get("kv_cache_continuation", False)):
        raise ValueError("Require frozen cache-disabled DualVLN")
    if any(int(c["checkpoint"]["step"]) + 9 >= native_limit for c in cases):
        raise ValueError("Insufficient step budget after perturbation and scan")
    import yaml
    habitat = yaml.safe_load((source / "habitat.yaml").read_text(encoding="utf-8"))
    if habitat["habitat"]["dataset"]["split"] != "train":
        raise ValueError("Only TRAIN development scenes allowed")
    full_dataset = pilot.read_json(source / "episodes.json.gz")
    episode_by_key = {ds.key_of(e): e for e in full_dataset["episodes"]}
    if any(tuple(c["key"]) not in episode_by_key for c in cases):
        raise ValueError("Selected episode absent from frozen source data")
    subset = copy.deepcopy(full_dataset)
    subset["episodes"] = [copy.deepcopy(episode_by_key[tuple(c["key"])]) for c in cases]

    root.mkdir(parents=True, exist_ok=False)
    ep_path = root / "episodes.json.gz"
    with ep_path.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as gz:
            gz.write(json.dumps(subset, ensure_ascii=False).encode("utf-8"))
    habitat["habitat"]["dataset"]["data_path"] = str(ep_path)
    hab_path = root / "habitat.yaml"
    hab_path.write_text("# @package _global_\n" + yaml.safe_dump(
        habitat, sort_keys=False), encoding="utf-8")
    generated = [ep_path, hab_path]
    for arm in ARMS:
        cfg = copy.deepcopy(model_cfg)
        cfg.env.env_settings["config_path"] = str(hab_path)
        cfg.eval_settings["output_path"] = str(root / arm)
        cfg.agent.model_settings["vis_debug"] = False
        encoded = json.dumps(cfg.model_dump(mode="json", exclude_none=True))
        path = root / (arm + ".py")
        path.write_text(
            "import json\nfrom internnav.configs.evaluator import EvalCfg\n"
            + "eval_cfg = EvalCfg.model_validate(json.loads(" + repr(encoded) + "))\n",
            encoding="utf-8")
        generated.append(path)
    protocol = {
        "schema": SCHEMA, "status": "POSTHOC_PROTOCOL_LOCKED_NO_ROLLOUT",
        "source_natural_gate_run": str(natural_root),
        "source_native_run": str(source),
        "source_perturbed_keep_run": str(pert_root),
        "selected_cases": cases, "cases": cases,
        "native_main_step_limit": native_limit,
        "charged_main_step_limit": native_limit - 3,
        "arms": list(ARMS), "scan_action_count": 6,
        "focus_keys": [list(c["key"]) for c in cases],
        "scope": SCOPE, "uses_outcomes_for_case_selection": True,
        "selector_features_from_outcomes": False,
        "gpu_rollouts_executed": False,
    }
    pp = root / "focused_protocol.json"
    pilot.write_json(pp, protocol)
    generated.append(pp)
    watch = [
        pert_root / "manifest.json", pert_root / "perturbed_keep_report.json",
        natural_root / "manifest.json", natural_root / "natural_checkpoint_report.json",
        natural_root / "control/actions.jsonl", source / "manifest.json",
        pert_root / pert.ARM / "turn_actions.jsonl",
        Path(__file__).resolve(), Path(paired.__file__).resolve(),
        Path(pert.__file__).resolve(), Path(iso.__file__).resolve(),
        Path(paired.prior.__file__).resolve(), Path(s2.__file__).resolve(),
        Path(physical.__file__).resolve(),
    ]
    for c in cases:
        suffix = "_".join(c["key"])
        for path in (
            natural_root / "control/checkpoints" / suffix / "checkpoint.json",
            natural_root / "control/checkpoints" / suffix / "predecision_input.json",
            pert_root / pert.ARM / "checkpoints" / suffix / "handoff.json",
            pert_root / pert.ARM / "checkpoints" / suffix / "predecision_input.json",
            pert_root / pert.ARM / "checkpoints" / suffix / "post_perturb_current_model_rgb.png",
        ):
            watch.append(path)
    hashes = {**src_m["source_hashes"], **src_m["generated_hashes"]}
    hashes.update({str(p): pilot.sha256(p) for p in watch})
    sha = subprocess.run(["git", "-C", str(pilot.ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "schema": SCHEMA, "revision": sha, "seed_base": src_m["seed_base"],
        "episodes": [{"scene_id": ep["scene_id"], "episode_id": str(ep["episode_id"])}
                     for ep in subset["episodes"]],
        "source_hashes": hashes,
        "checkpoint_shard_sizes": src_m["checkpoint_shard_sizes"],
        "generated_hashes": {str(f): pilot.sha256(f) for f in generated},
        "focus_is_posthoc": True,
    }
    pilot.write_json(root / "manifest.json", manifest)
    pilot.verify_manifest(root)
    print("FOCUSED_MECHANISM_PREPARE=PASS cases=3 arms=4 planned_rollouts=12 GPU=0")
    return protocol


def read_run(root):
    root = Path(root).resolve()
    manifest = pilot.verify_manifest(root)
    if manifest.get("schema") != SCHEMA or manifest.get("focus_is_posthoc") is not True:
        raise ValueError("Wrong posthoc mechanism manifest")
    protocol = pilot.read_json(root / "focused_protocol.json")
    natural_root, source, pert_root, cases = original_sources(
        protocol["source_perturbed_keep_run"])
    if (protocol.get("selected_cases") != cases or protocol.get("cases") != cases
            or protocol.get("focus_keys") != [list(c["key"]) for c in cases]
            or protocol.get("arms") != list(ARMS)
            or protocol.get("native_main_step_limit") != int(
                pilot.load_config(source / "control.py").eval_settings["max_steps_per_episode"])
            or protocol.get("charged_main_step_limit") !=
                protocol["native_main_step_limit"] - 3
            or protocol.get("source_natural_gate_run") != str(natural_root)
            or protocol.get("source_native_run") != str(source)
            or protocol.get("schema") != SCHEMA):
        raise ValueError("Focused protocol drift")
    selected_eps = pilot.read_json(root / "episodes.json.gz")["episodes"]
    if ([ds.key_of(x) for x in selected_eps] != [tuple(c["key"]) for c in cases]
            or manifest["episodes"] != [
                {"scene_id": x["scene_id"], "episode_id": str(x["episode_id"])}
                for x in selected_eps]):
        raise ValueError("Focused episode order/identity changed")
    return natural_root, source, pert_root, protocol


class FocusScanProbe(paired.PairProbe):
    """Reuse the old paired physical scan and prearm KEEP equality checks."""

    def __init__(self, evaluator, root, arm, protocol, references):
        if arm not in ARMS[1:]:
            raise ValueError("Not a focused scan arm")
        # Bypass only PairProbe's original arm-name parser: its original
        # prearm check, six physical turns, image hook and reset gate are reused.
        iso.MechanismProbe.__init__(self, evaluator, root, arm, protocol, references)
        self.output_arm = arm
        self.mode = "perturbed"
        self.operation = "SEEK_RIGHT45"  # scan membership; actual side set per case
        self.arm = "lookback_right45_evidence"
        self.prearm = None
        self.first_reset_output = None
        self.prearm_rgb_digest = None
        self.focus_role = arm

    def begin(self, key):
        super().begin(key)
        side = self.cases[key]["focus_scan_side"]
        variant = "placebo" if self.focus_role == ARMS[2] else "evidence"
        if side not in ("left", "right"):
            raise ValueError("Unknown locked scan direction")
        self.arm = "lookback_" + side + "45_" + variant
        self.first_noinjection_output = None

    def on_boundary(self, *args):
        was_at_gate = self.gate is None
        result = super().on_boundary(*args)
        if (self.focus_role == ARMS[1] and was_at_gate
                and self.gate is not None):
            # Scan completes identically; the first subsequent query receives
            # NO additional image or prompt. Do not call the injection hook.
            self.inject_pending = False
        return result

    def inject_extra_view(self, step, sources, input_images):
        if self.focus_role == ARMS[1]:
            if self.inject_pending or self.inject_count:
                raise RuntimeError("SCAN_NO_INJECTION changed S2 prompt")
            return
        return super().inject_extra_view(step, sources, input_images)

    def record_system2_output(self, step, message):
        if self.focus_role == ARMS[1]:
            if self.gate is not None and self.first_noinjection_output is None:
                self.first_noinjection_output = str(message)
            return
        super().record_system2_output(step, message)

    def finish(self):
        if self.current is None:
            return
        if self.focus_role == ARMS[1]:
            key = self.current
            if (self.prearm is None or self.inject_pending or self.inject_count
                    or self.external_actions != 6
                    or self.first_noinjection_output is None):
                raise RuntimeError("SCAN_NO_INJECTION contract violated")
            paired.prior.CandidateProbe.finish(self)
            self.finished[key].update(
                prearm=self.prearm, operation="SCAN_NO_INJECTION",
                mode="perturbed", supplementary_views=0)
        else:
            super().finish()


def run(args):
    root = Path(args.run_dir).resolve()
    natural_root, source, pert_root, protocol = read_run(root)
    arm = args.arm
    if arm not in ARMS or (root / arm).exists():
        raise FileExistsError("Invalid or already-created arm: " + str(arm))
    keys = {tuple(c["key"]) for c in protocol["cases"]}
    references = {k: [] for k in keys}
    for row in physical.read_lines(natural_root / "control/actions.jsonl"):
        key = capture.episode_key(row)
        if key in references:
            references[key].append(row)
    if any(not records for records in references.values()):
        raise RuntimeError("Missing full frozen native reference trace")
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators["habitat_vln"]
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("KV cache continuation prohibited")
        original_init(evaluator, cfg)
        if arm == ARMS[0]:
            probe = pert.PerturbedKeepProbe(evaluator, root, arm, protocol, references)
            evaluator._physical_recovery_probe = probe
        else:
            probe = FocusScanProbe(evaluator, root, arm, protocol, references)
            evaluator._recovery_candidate = probe
        probes.append(probe)
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate
        rec = s2.ArmS2Events(probe, root / arm / "s2_call_events.jsonl")
        probe.s2_events = rec
        traced = rec.wrap(probe.original_generate)

        def counted(*aa, **kw):
            probe.s2_calls += 1
            return traced(*aa, **kw)
        evaluator.model.generate = counted

    cls.__init__, cls._run_eval_dual_system = (
        initialize, physical.instrument_loop(original_loop) if arm == ARMS[0]
        else iso.instrument_loop(original_loop))
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Expected one evaluator in single-process replay")
        probes[0].finish()
        if set(probes[0].finished) != keys:
            raise RuntimeError("Incomplete 3-episode focus cohort")
        probes[0].status = "complete"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.evaluator.model.generate = probe.original_generate
            probe.s2_events.close()
            probe.close(error)
    events = s2.audit_events(root / arm / "s2_call_events.jsonl",
                             root / arm / "turn_actions.jsonl", expected_arm=arm)
    events["focused_protocol_sha256"] = pilot.sha256(root / "focused_protocol.json")
    pilot.write_json(root / arm / "s2_call_events_audit.json", events)
    print("FOCUSED_MECHANISM_ARM=PASS", arm, events["N_calls"])
    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--perturbed-keep-run", required=True)
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("preflight")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=lambda a: print("FOCUSED_MECHANISM_PREFLIGHT=PASS",
                                       len(read_run(a.run_dir)[3]["cases"])))
    p = sub.add_parser("run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.set_defaults(func=run)
    a = parser.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
