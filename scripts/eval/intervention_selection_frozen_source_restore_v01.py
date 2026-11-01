"""Check or restore the exact source code frozen by a v0.1 run manifest.

Safe by default: inspect only; restore requires explicit command and creates
a byte-for-byte backup. Never patches a manifest, skips digest validation,
runs Habitat, or overwrites an existing backup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE_NAME = "intervention_selection_focused_mechanism_v01.py"
# Git revisions that contained distinct versions of the posthoc focused driver.
# Each candidate's bytes are verified against the RUN-SPECIFIC manifest SHA256.
CANDIDATE_REVISIONS = (
    "a2f5696089ea059210c040c64e9b76fc0c26c4c8",
    "d7e65e5b450da3f280b159cd9236f01b2274fc6f",
    "9edf8eb194046c917560ea03aa9c51affd78c00c",
)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def frozen_source(run_dir):
    manifest = json.loads(
        (Path(run_dir).resolve() / "manifest.json").read_text(encoding="utf-8"))
    matches = [(Path(p), h) for p, h in manifest.get("source_hashes", {}).items()
               if Path(p).name == SOURCE_NAME]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original focused script in source_hashes")
    path, expected = matches[0]
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError("Original focused source path missing: " + str(path))
    if not isinstance(expected, str) or len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected):
        raise ValueError("Malformed original source SHA256 commitment")
    return path, expected


def git_versions(original):
    # Locate the real repo containing the prepared experiment's original path.
    repo = subprocess.run(
        ["git", "-C", str(original.parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True).stdout.strip()
    root = Path(repo).resolve()
    original = original.resolve()
    if not original.is_relative_to(root):
        raise ValueError("Original file escaped its containing Git repository")
    rel = original.relative_to(root).as_posix()
    versions = []
    for revision in CANDIDATE_REVISIONS:
        result = subprocess.run(["git", "-C", str(root), "show",
                                 revision + ":" + rel], capture_output=True)
        if result.returncode == 0:
            versions.append((revision, result.stdout, sha256(result.stdout)))
    return versions


def resolve(run_dir, restore=False):
    source, expected = frozen_source(run_dir)
    original = source.read_bytes()
    actual = sha256(original)
    versions = git_versions(source)
    matches = [(rev, data) for rev, data, digest in versions
               if digest == expected]
    report = {
        "source": str(source), "manifest_expected_sha256": expected,
        "current_sha256": actual, "source_matches_manifest": actual == expected,
        "available_versions": [
            {"commit": rev, "sha256": digest, "matches_manifest": digest == expected}
            for rev, _, digest in versions],
        "matching_git_revisions": [rev for rev, _ in matches],
        "restored": False,
    }
    if actual == expected:
        return report
    if not restore:
        return report
    if not matches:
        raise RuntimeError(
            "NO_EXACT_GIT_MATCH: restore refused. Original remains unchanged. "
            "Collect source hashes and investigate which uncommitted source "
            "was used at prepare; do not bypass the manifest."
        )
    content = matches[0][1]
    if sha256(content) != expected:
        raise RuntimeError("Internal candidate hash changed")
    # Backup the CURRENT file before changing anything, even if its current
    # bytes came from a modified Git working tree.
    backup = source.with_name(source.name + ".before-manifest-recovery-"
                              + actual[:16] + ".bak")
    if backup.exists():
        raise FileExistsError(
            "Backup path already exists; refuse overwrite: " + str(backup))
    shutil.copy2(source, backup)
    if sha256(backup.read_bytes()) != actual:
        raise RuntimeError("Backup verification failed; original unchanged")
    fd, temporary = tempfile.mkstemp(
        prefix=source.name + ".restore-", dir=str(source.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, source.stat().st_mode)
        if sha256(Path(temporary).read_bytes()) != expected:
            raise RuntimeError("Prepared source failed manifest SHA256")
        os.replace(temporary, source)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if sha256(source.read_bytes()) != expected:
        raise RuntimeError("Post-restore source hash did not match manifest")
    report.update(
        restored=True, restored_from_commit=matches[0][0],
        source_matches_manifest=True, current_sha256=expected,
        backup_path=str(backup))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "restore"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = resolve(args.run_dir, restore=args.command == "restore")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["source_matches_manifest"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
