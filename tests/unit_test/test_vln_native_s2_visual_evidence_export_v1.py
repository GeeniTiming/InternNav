"""Unit tests for the read-only native S2 visual evidence exporter."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import vln_native_s2_visual_evidence_export_v1 as exporter


def action(scene, episode):
    return {
        "scene_id": "data/scene_data/mp3d/{0}/{0}.glb".format(scene),
        "episode_id": str(episode),
        "action": 1,
    }


def call(call_id, key, offset, call_type="ordinary", output="→", main_step=None):
    image_path = "images/" + ("a" * 64) + ".png"
    rendered = "test prompt"
    return {
        "schema": "task_evidence_decisions.v1",
        "call_id": call_id,
        "episode_key": list(key),
        "main_step": call_id if main_step is None else main_step,
        "call_type": call_type,
        "native_action_prefix_count": offset,
        "history_native_indices": [],
        "images": [{
            "path": image_path, "pixel_sha256": "a" * 64,
            "width": 2, "height": 2, "mode": "RGB",
        }],
        "current_image_ref": image_path if call_type == "ordinary" else None,
        "prompt_text": "test",
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "model_output": output,
        "native_parsed_actions": [],
    }


class ActionOffsetTests(unittest.TestCase):
    def test_episode_global_action_offsets_follow_native_file_order(self):
        rows = (
            [action("scene_a", 1) for _ in range(3)]
            + [action("scene_b", 2) for _ in range(4)]
            + [action("scene_c", 3) for _ in range(2)]
        )
        grouped, offsets, total, order = exporter.group_action_rows(rows)
        self.assertEqual(total, 9)
        self.assertEqual(offsets, {
            ("scene_a", "1"): 0,
            ("scene_b", "2"): 3,
            ("scene_c", "3"): 7,
        })
        self.assertEqual([len(grouped[key]) for key in order], [3, 4, 2])

    def test_noncontiguous_episode_action_rows_fail_closed(self):
        rows = [action("scene_a", 1), action("scene_b", 2), action("scene_a", 1)]
        with self.assertRaisesRegex(ValueError, "not episode-contiguous"):
            exporter.group_action_rows(rows)

    def test_global_s2_prefix_is_localized_without_losing_global_value(self):
        rows = [action("scene_a", 1) for _ in range(3)] + [
            action("scene_b", 2) for _ in range(4)
        ]
        grouped, offsets, total, _ = exporter.group_action_rows(rows)
        source = [
            call(0, ("scene_a", "1"), 0),
            call(1, ("scene_a", "1"), 3),
            call(2, ("scene_b", "2"), 3),
            call(3, ("scene_b", "2"), 4),
            call(4, ("scene_b", "2"), 7),
        ]
        localized = exporter.localize_call_records(
            source, offsets, {key: len(value) for key, value in grouped.items()}, total)
        self.assertEqual(
            [x["action_prefix_count_episode"] for x in localized[("scene_b", "2")]],
            [0, 1, 4],
        )
        self.assertEqual(
            [x["action_prefix_count_global"] for x in localized[("scene_b", "2")]],
            [3, 4, 7],
        )
        self.assertEqual(source[2]["native_action_prefix_count"], 3)

    def test_global_prefix_outside_episode_block_is_rejected(self):
        rows = [action("scene_a", 1) for _ in range(3)] + [
            action("scene_b", 2) for _ in range(4)
        ]
        grouped, offsets, total, _ = exporter.group_action_rows(rows)
        source = [call(0, ("scene_b", "2"), 2)]
        with self.assertRaisesRegex(ValueError, "does not fall inside its episode block"):
            exporter.localize_call_records(
                source, offsets, {key: len(value) for key, value in grouped.items()}, total)

    def test_call_id_and_global_prefix_must_be_monotonic(self):
        rows = [action("scene_a", 1) for _ in range(3)]
        grouped, offsets, total, _ = exporter.group_action_rows(rows)
        calls = [
            call(1, ("scene_a", "1"), 2),
            call(2, ("scene_a", "1"), 1),
        ]
        with self.assertRaisesRegex(ValueError, "global action-prefix counter decreases"):
            exporter.localize_call_records(
                calls, offsets, {key: len(value) for key, value in grouped.items()}, total)


class SelectionTests(unittest.TestCase):
    def test_longest_run_uses_call_type_and_exact_output(self):
        key = ("scene_a", "1")
        rows = [
            call(0, key, 0, "ordinary", "→"),
            call(1, key, 1, "ordinary", "←←←←"),
            call(2, key, 2, "lookdown_followup", "100 100", main_step=1),
            call(3, key, 3, "ordinary", "←←←←"),
            call(4, key, 4, "ordinary", "←←←←"),
            call(5, key, 5, "ordinary", "←←←←"),
            call(6, key, 6, "ordinary", "→"),
        ]
        run = exporter.longest_output_run(rows, "ordinary", "←←←←", 0, 10)
        self.assertEqual([x["call_id"] for x in run], [1, 3, 4, 5])
        sampled = exporter.sample_run(run, (0, 2, 3))
        self.assertEqual([x["call_id"] for x in sampled], [1, 4, 5])

    def test_selection_adds_pre_post_stop_and_same_step_followup(self):
        key = ("scene_a", "1")
        rows = [
            call(0, key, 0, "ordinary", "→", main_step=0),
            call(1, key, 1, "ordinary", "↓", main_step=1),
            call(2, key, 2, "lookdown_followup", "120 80", main_step=1),
            call(3, key, 3, "ordinary", "↓", main_step=2),
            call(4, key, 4, "ordinary", "↓", main_step=3),
            call(5, key, 5, "ordinary", "→", main_step=4),
            call(6, key, 6, "ordinary", "STOP", main_step=5),
        ]
        spec = {
            "key": key, "window": (0, 5),
            "primary": {"call_type": "ordinary", "model_output": "↓",
                        "minimum_run": 3, "sample_indices": (0, 1, 2)},
            "secondary": None, "special_outputs": (),
        }
        selected, summary = exporter.select_target_calls({key: rows}, spec)
        ids = [x["call"]["call_id"] for x in selected]
        self.assertEqual(set(ids), {0, 1, 2, 3, 4, 5, 6})
        reasons = {x["call"]["call_id"]: x["reasons"] for x in selected}
        self.assertIn("paired_same_main_step", reasons[2])
        self.assertEqual(summary["primary_run_length"], 3)

    def test_selection_rejects_shorter_than_locked_run(self):
        key = ("scene_a", "1")
        rows = [call(i, key, i, "ordinary", "↓") for i in range(2)]
        spec = {
            "key": key, "window": (0, 10),
            "primary": {"call_type": "ordinary", "model_output": "↓",
                        "minimum_run": 3, "sample_indices": (0, 1, 2)},
            "secondary": None, "special_outputs": (),
        }
        with self.assertRaisesRegex(ValueError, "Primary repeated-output run too short"):
            exporter.select_target_calls({key: rows}, spec)


class ImageIntegrityTests(unittest.TestCase):
    def test_source_image_pixel_digest_and_metadata_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_root = Path(tmp) / "capture"
            image_dir = capture_root / "images"
            image_dir.mkdir(parents=True)
            image = Image.new("RGB", (3, 2), (10, 20, 30))
            digest = exporter.pixel_digest(image)
            source = image_dir / (digest + ".png")
            image.save(str(source), format="PNG")
            ref = {
                "path": "images/" + digest + ".png",
                "pixel_sha256": digest, "width": 3, "height": 2, "mode": "RGB",
            }
            actual, file_sha = exporter.verify_source_image(capture_root, ref)
            self.assertEqual(actual, source)
            self.assertEqual(file_sha, exporter.sha256_file(source))

    def test_modified_image_fails_pixel_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_root = Path(tmp) / "capture"
            image_dir = capture_root / "images"
            image_dir.mkdir(parents=True)
            image = Image.new("RGB", (3, 2), (10, 20, 30))
            digest = exporter.pixel_digest(image)
            source = image_dir / (digest + ".png")
            Image.new("RGB", (3, 2), (90, 80, 70)).save(str(source), format="PNG")
            ref = {
                "path": "images/" + digest + ".png",
                "pixel_sha256": digest, "width": 3, "height": 2, "mode": "RGB",
            }
            with self.assertRaisesRegex(ValueError, "Pixel SHA256 mismatch"):
                exporter.verify_source_image(capture_root, ref)

    def test_image_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = {
                "path": "images/../escape.png", "pixel_sha256": "a" * 64,
                "width": 1, "height": 1, "mode": "RGB",
            }
            with self.assertRaisesRegex(ValueError, "Unsafe or unexpected"):
                exporter.verify_source_image(tmp, ref)


if __name__ == "__main__":
    unittest.main()
