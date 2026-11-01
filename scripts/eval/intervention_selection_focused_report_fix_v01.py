"""Read-only reporting fix for frozen v0.1 three-case mechanism runs.

The original focused experiment script is source-hash pinned in completed
manifests and MUST NOT be changed. The original report mistakenly compares the
entire frozen checkpoint dict (including 'checks' and 'eligible') against the
smaller runtime gate dict. This separate report-only CLI fixes only that
schema mismatch, without weakening any original data/trace checks.

No model/rollout, no changes to old output dirs, no rewriting manifests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intervention_selection_focused_mechanism_v01 as focus

SCHEMA = "focused_mechanism_report_gate_hotfix.v0.1"
RUNTIME_GATE_FIELDS = (
    "step", "prefix_action_count", "state", "observation_hashes",
    "history_length", "history_sha256", "rng_sha256",
    "parameters_sha256", "prefix_trace_sha256",
)
EXPECTED_CHECKS = frozenset((
    "threshold_reached", "translated_enough", "remaining_budget",
    "idle_s2", "level_camera",
))
GATE_LABELS = frozenset(
    arm + " frozen S2 checkpoint" for arm in focus.ARMS[1:]
)


def verify_replay_gate(replayed, frozen, label, original_same_fields):
    """Validate complete runtime gate against frozen core, retaining audit flags."""
    if label not in GATE_LABELS:
        raise ValueError("Not a recognized focus gate label")
    if not isinstance(replayed, dict) or set(replayed) != set(RUNTIME_GATE_FIELDS):
        raise ValueError("Unexpected replay gate schema for " + label)
    if not isinstance(frozen, dict) or any(
            k not in frozen for k in RUNTIME_GATE_FIELDS):
        raise ValueError("Frozen checkpoint missing runtime gate fields for " + label)
    checks = frozen.get("checks")
    if (frozen.get("eligible") is not True or
            not isinstance(checks, dict) or set(checks) != EXPECTED_CHECKS or
            any(v is not True for v in checks.values())):
        raise ValueError("Frozen source checkpoint was not an eligible System-2 gate")
    # Compare ALL replay-observed fields, including original RNG, model hash,
    # history, observation, pose, step and action-prefix commitments.
    original_same_fields(replayed, frozen, RUNTIME_GATE_FIELDS, label)
    return True


def audited_comparator(original_same_fields):
    """Intercept exactly one documented report comparison; delegate all others."""
    def guarded(actual, expected, fields, label):
        if label not in GATE_LABELS:
            return original_same_fields(actual, expected, fields, label)
        # The original failing call always passes tuple(frozen_checkpoint).
        # Reject another call site rather than silently redefining its contract.
        if not isinstance(expected, dict) or tuple(fields) != tuple(expected):
            raise ValueError("Unexpected frozen-gate comparison invocation")
        return verify_replay_gate(actual, expected, label, original_same_fields)
    return guarded


def report(run_dir):
    run_dir = Path(run_dir).resolve()
    output = run_dir / "focused_mechanism_report.json"
    if output.exists():
        raise FileExistsError(
            "Focused report already exists; never overwrite audited artifacts")
    # The old source-hash checks are done by the ORIGINAL report/read_run;
    # do not modify its script or patch its manifest.
    original = focus.common.same_fields
    focus.common.same_fields = audited_comparator(original)
    try:
        result = focus.report(SimpleNamespace(run_dir=str(run_dir)))
    finally:
        focus.common.same_fields = original
    if result.get("N_actual_rollouts") != 12:
        raise ValueError("Focused report incomplete despite old report result")
    print("FOCUSED_REPORT_GATE_HOTFIX_V01=PASS")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report", help="Audit existing 12-rollout run without altering pinned drivers")
    p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report(args.run_dir)


if __name__ == "__main__":
    main()
