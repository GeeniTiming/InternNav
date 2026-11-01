"""Separate v0.1 replay entrypoint with actual per-arm S2 call telemetry.

The original frozen intervention_selection_{paired,perturbed_keep}_v01.py
files remain byte-for-byte unchanged so historical manifest source hashes still
validate. This entrypoint is ONLY for entirely new prepare/run directories.

The existing dataset design supports first-two or all-eligible cohorts; it
DOES NOT yet support the posthoc-selected three-case mechanism ablation.
Do not treat these logs as a completed KEEP/PLACEBO/SCAN_NO_INJECTION experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_selection_paired_v01 as paired
import intervention_selection_perturbed_keep_v01 as pert
import intervention_selection_s2_call_events_v01 as s2

pilot, physical, capture, iso = paired.pilot, paired.physical, paired.capture, paired.iso


def _replay(args, *, paired_arm):
    root = Path(args.run_dir).resolve()
    if paired_arm:
        _, _, _, protocol = paired.read_run(root)
        mode, operation = paired.split_arm(args.arm)
        arm = args.arm
        keys = {tuple(c["key"]) for c in protocol["selected_cases"]}
        source = Path(protocol["source_natural_gate_run"])
        Probe = paired.PairProbe
        native_instrument = iso.instrument_loop
    else:
        _, _, protocol = pert.read_run(root)
        arm = pert.ARM
        keys = {tuple(c["key"]) for c in protocol["cases"]}
        source = Path(protocol["source_natural_gate_run"])
        Probe = pert.PerturbedKeepProbe
        native_instrument = physical.instrument_loop
    if (root / arm).exists():
        raise FileExistsError("Create-only arm already exists: " + str(root / arm))
    references = {k: [] for k in keys}
    for row in physical.read_lines(source / "control/actions.jsonl"):
        k = capture.episode_key(row)
        if k in references:
            references[k].append(row)
    if not keys or any(not rows for rows in references.values()):
        raise ValueError("Missing frozen native prefix references")

    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa:F401
    cls = Evaluator.evaluators["habitat_vln"]
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get("kv_cache_continuation", False):
            raise RuntimeError("Expected cache-disabled frozen DualVLN")
        original_init(evaluator, cfg)
        probe = Probe(evaluator, root, arm, protocol, references)
        probe.s2_events = s2.ArmS2Events(probe, root / arm / "s2_call_events.jsonl")
        probes.append(probe)
        if paired_arm:
            evaluator._recovery_candidate = probe
        else:
            evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate
        traced = probe.s2_events.wrap(probe.original_generate)

        def counted(*a, **kw):
            probe.s2_calls += 1
            return traced(*a, **kw)
        evaluator.model.generate = counted

    cls.__init__, cls._run_eval_dual_system = initialize, native_instrument(original_loop)
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError("Expected exactly one evaluator")
        probes[0].finish()
        if set(probes[0].finished) != keys:
            raise RuntimeError("Incomplete predeclared cohort")
        probes[0].status = "complete"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for p in probes:
            p.evaluator.env.step = p.original_step
            p.evaluator.model.generate = p.original_generate
            p.s2_events.close()
            p.close(error)

    result = s2.audit_events(root / arm / "s2_call_events.jsonl",
                             root / arm / "turn_actions.jsonl",expected_arm=arm)
    audit_file = root / arm / "s2_call_events_audit.json"
    if audit_file.exists():
        raise FileExistsError("New replay event audit is create-only")
    pilot.write_json(audit_file, result)
    print("S2_INSTRUMENTED_REPLAY_V01=PASS", arm, result["N_calls"])
    # Do not declare prior causality or edit any old paired datasets.
    if not paired_arm:
        pert.report(SimpleNamespace(run_dir=str(root)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run-paired", help="New prepared paired v0.1 run only")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", required=True, choices=paired.ARMS)
    k = sub.add_parser("run-perturbed-keep", help="New prepared perturbed KEEP only")
    k.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    _replay(args, paired_arm=args.command == "run-paired")


if __name__ == "__main__":
    main()
