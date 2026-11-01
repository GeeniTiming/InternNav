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

# CheckpointProbe stores additional preselection/eligibility audit metadata in
# frozen checkpoints. PairProbe recreates precisely these nine runtime fields;
# comparing dict keys from the frozen checkpoint is therefore invalid.
REPLAY_GATE_FIELDS = (
    "step", "prefix_action_count", "state", "observation_hashes",
    "history_length", "history_sha256", "rng_sha256",
    "parameters_sha256", "prefix_trace_sha256",
)


def verify_frozen_replay_gate(replayed, frozen, label):
    """Compare actual replay state against the frozen checkpoint, fail closed.

    Frozen selection-only fields (checks, eligible, prefix_translation_m,
    remaining_native_main_steps, etc.) must be validated in their own schema,
    not required to reappear in a replay-generated state record.
    """
    if not isinstance(replayed, dict) or set(replayed) != set(REPLAY_GATE_FIELDS):
        raise ValueError("Unexpected runtime gate schema: " + label)
    if not isinstance(frozen, dict) or any(
            field not in frozen for field in REPLAY_GATE_FIELDS):
        raise ValueError("Frozen gate missing required replay state: " + label)
    checks = frozen.get("checks")
    if (frozen.get("eligible") is not True or not isinstance(checks, dict)
            or not checks or any(v is not True for v in checks.values())):
        raise ValueError("Frozen gate was not eligible: " + label)
    common.same_fields(replayed, frozen, REPLAY_GATE_FIELDS, label)
    return True


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
        self.operation = "SEEK_LEFT45" if side == "left" else "SEEK_RIGHT45"
        self.first_noinjection_output = None

    def on_boundary(self, *args):
        result = super().on_boundary(*args)
        if self.focus_role == ARMS[1] and self.gate is not None:
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
    if arm not in ARMS:
        raise ValueError("Unknown focus arm: " + str(arm))
    if (root / arm).exists():
        raise FileExistsError("Arm is create-only: " + str(arm))
    if arm != ARMS[0]:
        evidence = root / "keep_replay_gate.json"
        if not evidence.is_file():
            raise RuntimeError("KEEP replay gate not audited; refuse scanned GPU arms")
        barrier = pilot.read_json(evidence)
        if (barrier.get("status") != "KEEP_FULL_REPLAY_VERIFIED_NEW_S2_EVENTS"
                or barrier.get("N_verified_cases") != len(protocol["cases"])
                or barrier.get("replayed_action_sha256") != pilot.sha256(
                    root / ARMS[0] / "turn_actions.jsonl")
                or barrier.get("S2_call_events_sha256") != pilot.sha256(
                    root / ARMS[0] / "s2_call_events.jsonl")):
            raise RuntimeError("KEEP replay gate missing, changed or invalid")
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




def audit_keep(args):
    """CPU barrier: no further GPU arms until exact historical KEEP replay."""
    root = Path(args.run_dir).resolve()
    _, _, pert_root, protocol = read_run(root)
    outfile = root / "keep_replay_gate.json"
    if outfile.exists():
        raise FileExistsError("KEEP replay gate is create-only")
    arm = ARMS[0]
    current_dir = root / arm
    summary = pilot.read_json(current_dir / "pilot_summary.json")
    diag = pilot.read_json(current_dir / "turn_summary.json")
    if summary["status"] != "complete" or diag["status"] != "complete":
        raise ValueError("New instrumented KEEP not complete")
    source_summary = pilot.read_json(pert_root / pert.ARM / "pilot_summary.json")
    if source_summary["status"] != "complete":
        raise ValueError("Frozen original KEEP incomplete")
    new_trace = physical.validate_trace(
        physical.read_lines(current_dir / "turn_actions.jsonl"), "new KEEP")
    old_trace = physical.validate_trace(
        physical.read_lines(pert_root / pert.ARM / "turn_actions.jsonl"),
        "original KEEP")
    new_prog = capture.read_progress(current_dir / "progress.json")
    old_prog = capture.read_progress(pert_root / pert.ARM / "progress.json")
    detail = {tuple(row["key"]): row for row in diag["cases"]}
    checked = []
    for case in protocol["cases"]:
        key = tuple(case["key"])
        subset = lambda trace: [
            row for row in trace if capture.episode_key(row) == key]
        if not pilot.compare_traces(
                subset(new_trace), subset(old_trace))["actions_and_poses_match"]:
            raise ValueError("KEEP full action/pose replay diverged: " + repr(key))
        a, b = new_prog[key], old_prog[key]
        if (int(a["success"]) != int(b["success"])
                or a["episode_instruction"] != b["episode_instruction"]
                or not math.isclose(float(a["ne"]), float(b["ne"]),
                                    abs_tol=1e-5, rel_tol=1e-6)):
            raise ValueError("KEEP original outcome changed: " + repr(key))
        original = pilot.read_json(
            pert_root / pert.ARM / "checkpoints" / "_".join(key)
            / "handoff.json")
        common.same_fields(
            detail[key]["handoff"], original, tuple(original), "KEEP gate")
        checked.append(list(key))
    result = s2.audit_events(
        current_dir / "s2_call_events.jsonl",
        current_dir / "turn_actions.jsonl", expected_arm=arm)
    saved = pilot.read_json(current_dir / "s2_call_events_audit.json")
    if any(saved.get(k) != v for k, v in result.items()):
        raise ValueError("KEEP S2 event audit drift")
    if result["N_episodes"] != len(checked):
        raise ValueError("KEEP per-call S2 events missing from a focus case")
    out = {"schema": SCHEMA + ".keep_gate",
           "status": "KEEP_FULL_REPLAY_VERIFIED_NEW_S2_EVENTS",
           "N_verified_cases": len(checked), "episode_keys": checked,
           "original_action_sha256": pilot.sha256(
               pert_root / pert.ARM / "turn_actions.jsonl"),
           "replayed_action_sha256": pilot.sha256(
               current_dir / "turn_actions.jsonl"),
           "S2_call_events_sha256": pilot.sha256(
               current_dir / "s2_call_events.jsonl")}
    pilot.write_json(outfile, out)
    print("FOCUSED_KEEP_GATE=PASS", len(checked), "historical case replays")
    return out

def report(args):
    """Require four complete, source-matched replays; never infer causal effects."""
    root = Path(args.run_dir).resolve()
    if (root / "focused_mechanism_report.json").exists():
        raise FileExistsError("Final audit report is create-only")
    natural, native, pert_root, protocol = read_run(root)
    keys = [tuple(c["key"]) for c in protocol["cases"]]
    original_keep = capture.read_progress(pert_root / pert.ARM / "progress.json")
    original_tagged = physical.read_lines(
        pert_root / pert.ARM / "turn_actions.jsonl")
    old_actions = {
        key: [x for x in original_tagged if capture.episode_key(x) == key]
        for key in keys
    }
    summaries, diagnostics, tagged, outcomes, decisions = {}, {}, {}, {}, {}
    for arm in ARMS:
        directory = root / arm
        summary = pilot.read_json(directory / "pilot_summary.json")
        diag = pilot.read_json(directory / "turn_summary.json")
        if (summary["status"] != "complete" or diag["status"] != "complete"):
            raise ValueError("Incomplete GPU rollout: " + arm)
        rows = physical.validate_trace(
            physical.read_lines(directory / "turn_actions.jsonl"), arm)
        plain = physical.validate_trace(
            physical.read_lines(directory / "actions.jsonl"), arm + " native")
        if not pilot.compare_traces(rows, plain)["actions_and_poses_match"]:
            raise ValueError("Native and tagged action logs diverge: " + arm)
        computed = s2.audit_events(
            directory / "s2_call_events.jsonl",
            directory / "turn_actions.jsonl", expected_arm=arm)
        saved = pilot.read_json(directory / "s2_call_events_audit.json")
        if any(saved.get(k) != value for k, value in computed.items()):
            raise ValueError("Per-arm S2 audit changed since capture")
        per_ep = {tuple(row["key"]): row for row in diag["cases"]}
        progress = capture.read_progress(directory / "progress.json")
        if set(per_ep) != set(keys) or set(progress) != set(keys):
            raise ValueError("Missing focus episode from " + arm)
        summaries[arm], diagnostics[arm] = summary, per_ep
        tagged[arm] = {
            key: [r for r in rows if capture.episode_key(r) == key]
            for key in keys
        }
        outcomes[arm] = progress
        events = physical.read_lines(directory / "s2_call_events.jsonl")
        decisions[arm] = {
            key: [e for e in events if e["event"] == "enter"
                  and tuple(e["episode_key"]) == key]
            for key in keys
        }
    cases = []
    for case in protocol["cases"]:
        key = tuple(case["key"])
        gate = case["checkpoint"]
        orig = original_keep[key]
        keep = outcomes[ARMS[0]][key]
        if (int(orig["success"]) != int(keep["success"])
                or not math.isclose(float(orig["ne"]), float(keep["ne"]),
                                    abs_tol=1e-5, rel_tol=1e-6)
                or not pilot.compare_traces(
                    old_actions[key], tagged[ARMS[0]][key])["actions_and_poses_match"]):
            raise ValueError("Instrumented KEEP failed original complete replay: " + repr(key))
        expected_pre = pilot.read_json(
            pert_root / pert.ARM / "checkpoints" / "_".join(key) / "handoff.json")
        observed_keep = diagnostics[ARMS[0]][key]["handoff"]
        common.same_fields(
            observed_keep, expected_pre, tuple(expected_pre),
            "instrumented KEEP post-yaw context")
        expected_actions = (
            [2, 2, 2, 3, 3, 3] if case["focus_scan_side"] == "left"
            else [3, 3, 3, 2, 2, 2]
        )
        external_scans, scan_evidence = [], {}
        prearm_ref = None
        for arm in ARMS[1:]:
            detail = diagnostics[arm][key]
            verify_frozen_replay_gate(
                detail["gate"], gate, arm + " frozen S2 checkpoint")
            pre = detail["prearm"]
            if prearm_ref is None:
                prearm_ref = pre
            elif pre != prearm_ref:
                raise ValueError("Three scan arms began at different pre-arm contexts")
            if (pre["state"] != observed_keep["state"]
                    or pre["observation_hashes"] !=
                       observed_keep["final_observation_hashes"]
                    or pre["history_sha256"] != gate["history_sha256"]
                    or pre["rng_sha256"] != gate["rng_sha256"]):
                raise ValueError("Scan arm differs from KEEP at pre-arm gate")
            scan = [row for row in tagged[arm][key]
                    if row["phase"] == "external_recovery"]
            if [row["action"] for row in scan] != expected_actions:
                raise ValueError("Unexpected 45-degree scan and return: " + arm)
            if (int(detail["handoff"]["external_actions"]) != 6
                    or detail["handoff"]["history_reset"] is not False
                    or math.dist(detail["handoff"]["state"]["position"],
                                 pre["state"]["position"]) > 1e-5
                    or pose.heading_error(
                        detail["handoff"]["state"]["orientation_wxyz"],
                        pre["state"]["orientation_wxyz"]) > .05):
                raise ValueError("Scan cost, return pose or history differs")
            external_scans.append(scan)
            img_file = root / arm / "checkpoints" / "_".join(key) / "evidence_input.json"
            if arm == ARMS[1]:
                if img_file.exists() or detail.get("injection_events"):
                    raise ValueError("SCAN_NO_INJECTION unexpectedly injected an image")
            else:
                entry = pilot.read_json(img_file)
                if (len(entry.get("events", [])) != 1
                        or entry["events"][0]["source"] != (
                            "duplicate_current" if arm == ARMS[2] else "outward_view")
                        or entry["events"][0]["same_image_count"] is not True):
                    raise ValueError("Wrong placebo/evidence view provenance")
                scan_evidence[arm] = entry
        if any(not pilot.compare_traces(
                external_scans[0], other)["actions_and_poses_match"]
               for other in external_scans[1:]):
            raise ValueError("Physical scans differ among scan arms")
        if (scan_evidence[ARMS[2]]["scan_view_sha256"] !=
                scan_evidence[ARMS[3]]["scan_view_sha256"]
                or scan_evidence[ARMS[2]]["post_return_current_sha256"] !=
                   scan_evidence[ARMS[3]]["post_return_current_sha256"]):
            raise ValueError("Real/placebo scan camera frames differ")
        # Exact per-episode model input/output sequence before intervention.
        def prefix_events(arm):
            return [
                (e["input_ids_sha256"], e["episode_call_id"],
                 e["native_action_prefix_count"])
                for e in decisions[arm][key] if e["phase"] == "prefix"
            ]
        baseline_calls = prefix_events(ARMS[0])
        if not baseline_calls or any(prefix_events(arm) != baseline_calls
                                     for arm in ARMS[1:]):
            raise ValueError("S2 call anchors before checkpoint are not paired")
        for arm in ARMS:
            cont = [e for e in decisions[arm][key]
                    if e["phase"] == "continuation"]
            if not cont or cont[0]["native_action_prefix_count"] != (
                    gate["prefix_action_count"] + 3
                    + (6 if arm != ARMS[0] else 0)):
                raise ValueError("No first post-gate S2 call at exact action offset")
        scores = {
            arm: {"success": int(outcomes[arm][key]["success"]),
                  "ne_m": float(outcomes[arm][key]["ne"])}
            for arm in ARMS
        }
        cases.append({
            "episode_key": list(key), "focus_scan_side": case["focus_scan_side"],
            "checkpoint_main_step": gate["step"],
            "prearm_matches_original_KEEP": True,
            "three_scan_arms_matched_in_pose_and_yaw": True,
            "prefix_S2_inputs_and_action_offsets_matched": True,
            "first_post_gate_S2_offsets_observed": True,
            "outcomes_OFFLINE_ONLY": scores,
        })
    result = {
        "schema": SCHEMA + ".report",
        "status": "POSTHOC_THREE_SCENE_MATCHED_S2_CALLS_DESCRIPTIVE_NOT_CAUSAL",
        "N_scene_groups": len(cases), "N_contexts": len(cases),
        "N_actual_rollouts": len(cases) * len(ARMS),
        "N_S2_call_aligned_prefixes": len(cases) * (len(ARMS) - 1),
        "arms": list(ARMS), "posthoc_selection": True,
        "not_estimated": [
            "held_out_efficacy", "generalization", "learned_action_selector",
            "causal_effect_of_task_relevance"
        ],
        "per_context": cases, "scope": SCOPE,
    }
    pilot.write_json(root / "focused_mechanism_report.json", result)
    print("FOCUSED_MECHANISM_REPORT=PASS", len(cases), "scenes, 12 actual rollouts")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

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
    p = sub.add_parser("audit-keep")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=audit_keep)
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=report)
    a = parser.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
