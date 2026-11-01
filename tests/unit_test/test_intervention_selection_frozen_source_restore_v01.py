"""CPU-only tests for exact-manifest, backup-first source recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
import intervention_selection_frozen_source_restore_v01 as recover


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / recover.SOURCE_NAME
        self.original = b"# changed post-run\n"
        self.locked = b"# frozen code at prepare\n"
        self.source.write_bytes(self.original)
        self.expected = recover.sha256(self.locked)
        (self.root / "manifest.json").write_text(json.dumps({
            "source_hashes": {str(self.source): self.expected}
        }))
        self.version = ("a2f5696089ea059210c040c64e9b76fc0c26c4c8",
                        self.locked, self.expected)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inspect_only_never_changes_original(self):
        with patch.object(recover, "git_versions", return_value=[self.version]):
            result = recover.resolve(self.root, restore=False)
        self.assertFalse(result["source_matches_manifest"])
        self.assertFalse(result["restored"])
        self.assertEqual(result["matching_git_revisions"], [self.version[0]])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse(list(self.root.glob("*.bak")))

    def test_explicit_restore_requires_matching_sha_and_keeps_backup(self):
        with patch.object(recover, "git_versions", return_value=[self.version]):
            result = recover.resolve(self.root, restore=True)
        self.assertTrue(result["source_matches_manifest"])
        self.assertTrue(result["restored"])
        self.assertEqual(self.source.read_bytes(), self.locked)
        self.assertEqual(Path(result["backup_path"]).read_bytes(), self.original)
        self.assertEqual(recover.sha256(self.source.read_bytes()), self.expected)

    def test_no_matching_git_revision_never_overwrites(self):
        bad = ("wrong-commit", b"wrong content", recover.sha256(b"wrong content"))
        with patch.object(recover, "git_versions", return_value=[bad]):
            with self.assertRaisesRegex(RuntimeError, "NO_EXACT_GIT_MATCH"):
                recover.resolve(self.root, restore=True)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse(list(self.root.glob("*.bak")))

    def test_existing_backup_is_not_overwritten(self):
        backup = self.source.with_name(
            self.source.name + ".before-manifest-recovery-" +
            recover.sha256(self.original)[:16] + ".bak")
        backup.write_bytes(b"previous backup")
        with patch.object(recover, "git_versions", return_value=[self.version]):
            with self.assertRaises(FileExistsError):
                recover.resolve(self.root, restore=True)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(backup.read_bytes(), b"previous backup")

    def test_already_matching_source_is_not_replaced(self):
        self.source.write_bytes(self.locked)
        with patch.object(recover, "git_versions", return_value=[self.version]):
            result = recover.resolve(self.root, restore=True)
        self.assertTrue(result["source_matches_manifest"])
        self.assertFalse(result["restored"])
        self.assertEqual(self.source.read_bytes(), self.locked)
        self.assertFalse(list(self.root.glob("*.bak")))

    def test_frozen_source_identity_is_required(self):
        (self.root / "manifest.json").write_text(json.dumps({
            "source_hashes": {"other.py": self.expected}
        }))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            recover.frozen_source(self.root)


if __name__ == "__main__":
    unittest.main()
