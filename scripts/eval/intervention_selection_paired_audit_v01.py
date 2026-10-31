"""Additive, audit-only v0.1 gate-field compatibility layer.

WHY A SEPARATE FILE? Frozen existing paired RUN manifests hash
intervention_selection_paired_v01.py. Editing that script would invalidate
finished GPU trials merely to fix an audit-only KeyError. This companion
does not modify any previously hashed source or data file.

Existing native checkpoint snapshots contain *eligibility metadata* in
addition to the core replay gate. The paired runtime gate intentionally
contains only the exact core fields. Compare those same nine core fields,
fail closed if fields are missing/extra, and retain every other existing
pose/RGB/history/RNG/scan/budget/metrics audit unchanged.

No GPU, no source edits, no bypassing pilot.verify_manifest.
"""
from __future__ import annotations

import argparse
import contextlib
from unittest.mock import patch

import intervention_selection_paired_v01 as paired

GATE_CORE_FIELDS = (
    "step",
    "prefix_action_count",
    "state",
    "observation_hashes",
    "history_length",
    "history_sha256",
    "rng_sha256",
    "parameters_sha256",
    "prefix_trace_sha256",
)


def compare_native_gate(actual, expected, fields, label, original):
    """Perform the *complete* runtime-gate contract against enriched checkpoint.

    Existing audit_arm() supplies fields=tuple(expected), incorrectly
    requesting eligibility-only keys absent in actual. Enforce exact core
    shape and all nine values. Other same_fields calls remain unchanged.
    """
    if label != "paired native gate":
        return original(actual, expected, fields, label)
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise RuntimeError("NOT_COMPARABLE: native gate must be mapping")
    actual_names = set(actual)
    expected_names = set(expected)
    required = set(GATE_CORE_FIELDS)
    if actual_names != required:
        raise RuntimeError(
            "NOT_COMPARABLE: paired runtime gate core schema differs: "
            + repr({"missing": sorted(required - actual_names),
                    "unexpected": sorted(actual_names - required)})
        )
    if not required.issubset(expected_names):
        raise RuntimeError(
            "NOT_COMPARABLE: source checkpoint missing core gate fields: "
            + repr(sorted(required - expected_names))
        )
    if set(fields) != expected_names:
        raise RuntimeError("NOT_COMPARABLE: caller unexpectedly changed gate comparison fields")
    return original(actual, expected, GATE_CORE_FIELDS, label)


@contextlib.contextmanager
def audited_compatibility():
    """Scoped in-process audit correction; no native model or run patches."""
    original = paired.common.same_fields

    def checked(actual, expected, fields, label):
        return compare_native_gate(actual, expected, fields, label, original)

    with patch.object(paired.common, "same_fields", new=checked):
        yield


def audit_arm(args):
    with audited_compatibility():
        return paired.audit_single(args)


def report(args):
    with audited_compatibility():
        return paired.report(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("audit-arm", help="Audit one already-completed, hashed GPU arm")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=paired.ARMS, required=True)
    p.set_defaults(fn=audit_arm)
    p = subs.add_parser("report", help="Run all unchanged arm audits and export tables")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(fn=report)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
