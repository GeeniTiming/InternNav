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
import ast
import functools
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
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


def pinned_focus_module(run_dir):
    """Load the EXACT source version pinned by the original experiment.

    The report-hotfix checkout can have a different historical revision:
    using its local focused module would audit the wrong runner. Require the
    on-disk original source SHA256 to match the experiment manifest first.
    """
    manifest_path = Path(run_dir).resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("source_hashes", {})
    matches = [
        (Path(path), digest) for path, digest in records.items()
        if Path(path).name == "intervention_selection_focused_mechanism_v01.py"
    ]
    if len(matches) != 1:
        raise ValueError("Require exactly one pinned focused source in manifest")
    source, expected = matches[0]
    if not source.is_absolute() or not source.is_file():
        raise ValueError("Pinned focused source path missing")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(
            "FROZEN_SOURCE_SHA_MISMATCH: " + str(source) +
            " expected=" + expected + " actual=" + observed +
            ". Restore an exact Git version with the separate recovery helper; "
            "do NOT edit the manifest or skip checksum verification."
        )
    source_dir = str(source.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(
        "_v01_focused_source_exact_manifest", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pinned original report driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.ARMS) != tuple(focus.ARMS):
        raise RuntimeError("Pinned original experiment arm vocabulary changed")
    return module



# The frozen native evaluator performs four actual camera env.step() calls
# BEFORE a new ordinary S2 generate: LOOKDOWN x2 (5), LOOKUP x2 (4).
# The original focused report omitted those four from its offset assertion.
NATIVE_PRE_S2_CAMERA_ACTIONS = (5, 5, 4, 4)


def verify_first_post_gate_s2(key, gate, tagged, decisions, arms):
    """Verify four real native camera actions at exact first S2 call offsets.

    This is NOT a +4 tolerance. It independently checks each action row's
    phase, code and main_step against a measured model.generate entry. It
    rejects surplus actions or any missing/changed camera actions.
    """
    if tuple(arms) != tuple(focus.ARMS):
        raise ValueError("Unexpected arm vocabulary at S2 boundary")
    prefix = gate.get("prefix_action_count")
    gate_step = gate.get("step")
    if type(prefix) is not int or prefix < 1 or type(gate_step) is not int:
        raise ValueError("Malformed frozen checkpoint S2 anchor")
    first_call_ids = []
    for arm in arms:
        actions = tagged[arm][key]
        continuation = [
            row for row in decisions[arm][key]
            if row["phase"] == "continuation"
        ]
        if not continuation:
            raise ValueError("No observed first continuation S2 call: " + repr((key, arm)))
        entry = continuation[0]
        offset = entry.get("native_action_prefix_count")
        if type(offset) is not int or offset > len(actions):
            raise ValueError("First S2 call missing valid action offset: " + repr((key, arm)))
        scan_count = 0 if arm == arms[0] else 6
        pre_camera = prefix + 3 + scan_count
        expected_offset = pre_camera + len(NATIVE_PRE_S2_CAMERA_ACTIONS)
        if (offset != expected_offset
                or entry.get("main_step") != gate_step):
            raise ValueError("First S2 call not at exact native camera completion: "
                             + repr((key, arm, offset, expected_offset, entry.get("main_step"))))
        before = actions[:offset]
        segments = (
            (before[:prefix], "prefix", prefix),
            (before[prefix:prefix+3], "alignment", 3),
            (before[prefix+3:pre_camera], "external_recovery", scan_count),
            (before[pre_camera:offset], "continuation", 4),
        )
        for segment, expected_phase, expected_count in segments:
            if len(segment) != expected_count or any(
                    row.get("phase") != expected_phase for row in segment):
                raise ValueError("Wrong actual action phase before first S2: "
                                 + repr((key, arm, expected_phase)))
        camera = before[pre_camera:offset]
        if (tuple(row.get("action") for row in camera)
                != NATIVE_PRE_S2_CAMERA_ACTIONS
                or any(row.get("main_step") != gate_step for row in camera)):
            raise ValueError("Actual native camera LOOKDOWN/LOOKUP sequence changed: "
                             + repr((key, arm)))
        cid = entry.get("episode_call_id")
        if type(cid) is not int or cid < 0:
            raise ValueError("Missing actual first continuation episode call ID")
        first_call_ids.append(cid)
    if len(set(first_call_ids)) != 1:
        raise ValueError("First post-gate S2 calls differ in episode call ordinal")
    return True


def focused_report_with_camera_audit(original_report):
    """Guarded AST correction of exactly one frozen report assertion.

    Reject any unexpected report-source change; the original source bytes
    and all GPU trajectory recordings remain unchanged.
    """
    source = textwrap.dedent(inspect.getsource(original_report))
    tree = ast.parse(source)
    expected = ast.parse(
        'for arm in ARMS:\n'
        '    cont = [e for e in decisions[arm][key]\n'
        '            if e["phase"] == "continuation"]\n'
        '    if not cont or cont[0]["native_action_prefix_count"] != (\n'
        '            gate["prefix_action_prefix_count"] + 3\n'
        '            + (6 if arm != ARMS[0] else 0)):\n'
        '        raise ValueError("No first post-gate S2 call at exact action offset")\n'
    ).body[0]
    # Correctly spell the frozen source key. Keeping this explicit makes
    # drift detectable when comparing syntax trees below.
    expected.body[1].test.comparators[0].left.value.value = "prefix_action_count"
    expected_dump = ast.dump(expected, include_attributes=False)
    matches = 0

    class ReplaceOriginalOffsetCheck(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal matches
            if ast.dump(node, include_attributes=False) == expected_dump:
                matches += 1
                return ast.copy_location(ast.parse(
                    '_s2_camera_gate_verify(key, gate, tagged, decisions, ARMS)'
                ).body[0], node)
            return self.generic_visit(node)

    tree = ReplaceOriginalOffsetCheck().visit(tree)
    if matches != 1:
        raise RuntimeError("Pinned focused report changed S2 offset assertion shape; "
                           "refuse unreviewed instrumentation")
    ast.fix_missing_locations(tree)
    namespace = dict(original_report.__globals__)
    namespace["_s2_camera_gate_verify"] = verify_first_post_gate_s2
    exec(compile(tree, "<focused-s2-report-camera-audit>", "exec"), namespace)
    clone = namespace.get(original_report.__name__)
    if clone is None:
        raise RuntimeError("Unable to compile strict read-only report")
    return functools.update_wrapper(clone, original_report)



def report(run_dir):
    run_dir = Path(run_dir).resolve()
    output = run_dir / "focused_mechanism_report.json"
    if output.exists():
        raise FileExistsError(
            "Focused report already exists; never overwrite audited artifacts")
    # Ensure that we execute the original experiment's exact pinned report
    # implementation, not the possibly different local hotfix checkout version.
    pinned = pinned_focus_module(run_dir)
    original = pinned.common.same_fields
    pinned.common.same_fields = audited_comparator(original)
    try:
        checked_report = focused_report_with_camera_audit(pinned.report)
        result = checked_report(SimpleNamespace(run_dir=str(run_dir)))
    finally:
        pinned.common.same_fields = original
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
