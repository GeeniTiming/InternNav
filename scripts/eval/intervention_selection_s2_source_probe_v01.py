"""Read-only, CPU-only source-log readiness probe for intervention selection v0.1.

The posthoc trace ZIP contains action-indexed histories, not true S2 call
boundaries. This script checks the original server action-log commitments and
inventories possible per-call logs from those SAME run directories. It never
reconstructs call IDs, matches distinct replays by main_step, or starts Habitat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

SCHEMA = "intervention_selection_s2_source_probe.v0.1"
TRACE_REPORT = "trace_divergence_report.json"
EXPECTED_BUNDLE_FILES = (
    "continuation_action_traces.jsonl",
    "trace_divergence_vs_keep.jsonl",
    "decision_comparisons.jsonl",
)
CANDIDATE_NAME = re.compile(r"(?:system.?2|s2|decision|call|event)", re.I)
MAX_CANDIDATE_BYTES = 64 * 1024 * 1024
MAX_CALL_ROWS = 200_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(bundle: Path) -> tuple[dict, dict[str, str]]:
    """Read only source report and verify commitments for other ZIP members."""
    bundle = Path(bundle)
    hashes = {}
    if bundle.is_dir():
        report = json.loads((bundle / TRACE_REPORT).read_text(encoding="utf-8"))
        for name in EXPECTED_BUNDLE_FILES:
            hashes[name] = sha256_file(bundle / name)
    elif bundle.is_file() and zipfile.is_zipfile(bundle):
        with zipfile.ZipFile(bundle) as archive:
            if any(name not in archive.namelist() for name in (TRACE_REPORT,) + EXPECTED_BUNDLE_FILES):
                raise ValueError("Missing one or more required trace bundle members")
            if archive.getinfo(TRACE_REPORT).file_size > 2 * 1024 * 1024:
                raise ValueError("Oversized report")
            report = json.loads(archive.read(TRACE_REPORT).decode("utf-8"))
            for name in EXPECTED_BUNDLE_FILES:
                if archive.getinfo(name).file_size > 128 * 1024 * 1024:
                    raise ValueError("Oversized trace member: " + name)
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                hashes[name] = digest.hexdigest()
    else:
        raise ValueError("Expected directory with trace JSON/JSONL or ZIP file")
    if report.get("schema") != "intervention_selection_trace_divergence.v0.1":
        raise ValueError("Unexpected source schema")
    expected = report.get("output_hashes")
    if not isinstance(expected, dict) or set(expected) != set(EXPECTED_BUNDLE_FILES):
        raise ValueError("Incomplete source output hashes")
    for name in EXPECTED_BUNDLE_FILES:
        if hashes[name] != expected[name]:
            raise ValueError("Trace ZIP member SHA256 mismatch: " + name)
    if not isinstance(report.get("original_trace_sha256"), dict) or not report["original_trace_sha256"]:
        raise ValueError("No source raw action-log hash commitments")
    return report, hashes


def source_paths(report: dict, source_root: Path | None):
    """An optional root remaps an intact SOURCE tree, not individual arms."""
    entries = report["original_trace_sha256"]
    old = [Path(key) for key in entries]
    if any(not path.is_absolute() for path in old):
        raise ValueError("Original server raw paths must be absolute")
    old_root = Path(os.path.commonpath([str(path) for path in old]))
    if len(old) == 1:
        old_root = old[0].parent
    result = []
    for old_path, expected in sorted(entries.items()):
        p = Path(old_path)
        mapped = (Path(source_root) / p.relative_to(old_root)) if source_root else p
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid SHA256 commitment for " + old_path)
        result.append((p, mapped, expected))
    return old_root, result


def candidate_dirs(action_file: Path):
    """Limit searches to files associated with this same original run/arm."""
    arm = action_file.parent
    run = arm.parent
    return (arm, arm / "capture", run / "capture", arm / "events", arm / "diagnostics")


def discover_candidate_files(action_file: Path):
    found = set()
    for directory in candidate_dirs(action_file):
        if not directory.is_dir():
            continue
        for file in directory.glob("*.jsonl"):
            if file.is_file() and CANDIDATE_NAME.search(file.stem) and file != action_file:
                found.add(file.resolve())
    return sorted(found)


def inspect_candidate(file: Path) -> dict:
    """Never emit prompts, model outputs, RGBs or simulator poses."""
    entry = {"file": str(file), "sha256": sha256_file(file), "size_bytes": file.stat().st_size}
    if file.stat().st_size > MAX_CANDIDATE_BYTES:
        return dict(entry, status="TOO_LARGE_NOT_INSPECTED")
    count = 0
    seen = set()
    per_ep_previous = {}
    errors = set()
    episodes = set()
    required = {"call_id", "episode_key", "main_step", "native_action_prefix_count",
                "call_type", "model_output"}
    try:
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                count += 1
                if count > MAX_CALL_ROWS:
                    errors.add("TOO_MANY_ROWS")
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    errors.add("ROW_NOT_OBJECT")
                    break
                missing = required - set(row)
                if missing:
                    errors.add("MISSING_REQUIRED_FIELDS:" + ",".join(sorted(missing)))
                    break
                call_id, prefix = row["call_id"], row["native_action_prefix_count"]
                key = row["episode_key"]
                if (not isinstance(call_id, int) or isinstance(call_id, bool) or call_id < 0
                    or not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0
                    or not isinstance(key, list) or len(key) != 2
                    or not all(isinstance(v, str) for v in key)
                    or not isinstance(row["main_step"], int) or isinstance(row["main_step"], bool)
                    or row["main_step"] < 0
                    or row["call_type"] not in ("ordinary", "lookdown_followup")
                    or not isinstance(row["model_output"], str)
                    or not row["model_output"]):
                    errors.add("MALFORMED_CALL_FIELDS")
                    break
                key_tuple = tuple(key)
                if call_id in seen:
                    errors.add("DUPLICATE_CALL_ID")
                    break
                seen.add(call_id)
                prev = per_ep_previous.get(key_tuple)
                if prev is not None and prefix < prev:
                    errors.add("DECREASING_ACTION_PREFIX")
                    break
                per_ep_previous[key_tuple] = prefix
                episodes.add(key_tuple)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.add("UNREADABLE_OR_INVALID_JSONL:" + type(exc).__name__)
    if not count:
        errors.add("EMPTY")
    # These are candidate anchors ONLY. A proof requires matching exact
    # per-arm action stream and recorded replay checkpoint/policy provenance.
    status = "CALL_ANCHOR_CANDIDATE_UNVERIFIED" if not errors else "NOT_USABLE_AS_CALL_ANCHOR"
    return dict(entry, status=status, rows_inspected=count,
                distinct_episode_keys=len(episodes), failure_reasons=sorted(errors))


def probe(bundle: Path, source_root: Path | None = None):
    report, hashes = load_bundle(bundle)
    old_root, records = source_paths(report, source_root)
    raw = []
    candidate_cache = {}
    for original, path, expected in records:
        if not path.is_file():
            status, actual = "NOT_FOUND", None
        else:
            actual = sha256_file(path)
            status = "MATCHED" if actual == expected else "HASH_MISMATCH"
        candidates = []
        if status == "MATCHED":
            for file in discover_candidate_files(path):
                if file not in candidate_cache:
                    candidate_cache[file] = inspect_candidate(file)
                candidates.append(str(file))
        raw.append({
            "recorded_source_path": str(original), "resolved_source_path": str(path),
            "recorded_sha256": expected, "observed_sha256": actual,
            "status": status, "call_log_candidates": candidates,
        })
    inventory = [candidate_cache[k] for k in sorted(candidate_cache)]
    n_match = sum(x["status"] == "MATCHED" for x in raw)
    n_missing = sum(x["status"] == "NOT_FOUND" for x in raw)
    n_bad = sum(x["status"] == "HASH_MISMATCH" for x in raw)
    n_candidates = sum(x["status"] == "CALL_ANCHOR_CANDIDATE_UNVERIFIED" for x in inventory)
    if n_bad:
        status = "BLOCKED_SOURCE_HASH_MISMATCH"
    elif n_missing:
        status = "BLOCKED_ORIGINAL_ACTION_LOGS_MISSING"
    elif not n_candidates:
        status = "RAW_ACTION_LOGS_VERIFIED_NO_LINKABLE_S2_CALL_ANCHORS"
    else:
        status = "RAW_ACTION_LOGS_VERIFIED_S2_CANDIDATES_REQUIRE_EXACT_REPLAY_BINDING"
    summary = {
        "schema": SCHEMA, "status": status,
        "trace_bundle": str(Path(bundle).resolve()),
        "trace_member_sha256_verified": hashes,
        "recorded_source_root": str(old_root),
        "remapped_source_root": str(Path(source_root).resolve()) if source_root else None,
        "N_original_action_logs": len(raw), "N_matched_action_logs": n_match,
        "N_missing_action_logs": n_missing, "N_changed_action_logs": n_bad,
        "N_candidate_files": len(inventory), "N_candidate_anchor_files": n_candidates,
        "N_SYSTEM2_call_aligned_pairs": 0,
        "can_execute_paired_ablation": False,
        "reason": (
            "No action-indexed trace contains proof of per-arm S2 entry/exit. "
            "Even a candidate call log must be bound to the exact same replay, "
            "checkpoint, per-episode action offsets, history/RNG and S2 outputs. "
            "Do not merge native-only call logs into treated trajectories."
        ),
        "next_gate": (
            "If no same-arm call logs exist, instrument new matched checkpoint "
            "replays with explicit per-call action-offset events; do not "
            "reconstruct call IDs from main_step or decoded text."
        ),
        "gpu_rollouts_executed": False, "original_data_modified": False,
    }
    return summary, raw, inventory


def write_create_only(output: Path, summary: dict, raw: list, candidates: list):
    if output.exists():
        raise FileExistsError("Create-only output path already exists")
    output.mkdir(parents=True, exist_ok=False)
    items = {
        "s2_readiness_report.json": summary,
        "source_action_hash_audit.json": raw,
        "s2_call_candidate_inventory.json": candidates,
    }
    for name, value in items.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe",))
    parser.add_argument("--input", required=True, type=Path, help="Original trace bundle ZIP or directory")
    parser.add_argument("--run-dir", required=True, type=Path, help="NEW create-only output directory")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Optional relocated root of the raw server action-log tree")
    args = parser.parse_args(argv)
    if args.run_dir.exists():
        parser.error("run-dir already exists (create-only)")
    summary, raw, inventory = probe(args.input, args.source_root)
    write_create_only(args.run_dir, summary, raw, inventory)
    print("S2_SOURCE_PROBE_V01=PASS")
    print(json.dumps({k: summary[k] for k in (
        "status", "N_original_action_logs", "N_matched_action_logs",
        "N_missing_action_logs", "N_changed_action_logs", "N_candidate_files",
        "N_candidate_anchor_files", "N_SYSTEM2_call_aligned_pairs")}, indent=2))
    # Hash mismatches are stop conditions even though they are recorded in report.
    if summary["N_changed_action_logs"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())