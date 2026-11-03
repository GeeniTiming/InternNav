"""CPU-only tests for global S2 action-prefix localization."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import vln_native_anomaly_audit_v1 as audit


def action(scene, episode, code=1):
    return {"scene_id": f"mp3d/{scene}/{scene}.glb",
            "episode_id": str(episode), "action": code}


class GlobalActionPrefixTests(unittest.TestCase):
    def test_action_trace_builds_exact_episode_global_intervals(self):
        rows = (
            [action("scene_a", 1) for _ in range(3)]
            + [action("scene_b", 2) for _ in range(4)]
            + [action("scene_c", 3) for _ in range(2)]
        )
        grouped, offsets, total, order = audit.group_ordered_action_rows(rows)
        self.assertEqual(total, 9)
        self.assertEqual(offsets, {
            ("scene_a", "1"): 0,
            ("scene_b", "2"): 3,
            ("scene_c", "3"): 7,
        })
        self.assertEqual([len(grouped[k]) for k in order], [3, 4, 2])

    def test_global_s2_prefix_is_localized_without_overwriting_source(self):
        rows = (
            [action("scene_a", 1) for _ in range(3)]
            + [action("scene_b", 2) for _ in range(4)]
            + [action("scene_c", 3) for _ in range(2)]
        )
        grouped, offsets, total, _ = audit.group_ordered_action_rows(rows)
        source = {
            ("scene_a", "1"): [
                {"native_action_prefix_count": 0, "call_id": 0},
                {"native_action_prefix_count": 3, "call_id": 1},
            ],
            ("scene_b", "2"): [
                {"native_action_prefix_count": 3, "call_id": 2},
                {"native_action_prefix_count": 4, "call_id": 3},
                {"native_action_prefix_count": 7, "call_id": 4},
            ],
            ("scene_c", "3"): [
                {"native_action_prefix_count": 7, "call_id": 5},
                {"native_action_prefix_count": 9, "call_id": 6},
            ],
        }
        result = audit.localize_global_s2_prefixes(
            source, offsets, {k: len(v) for k, v in grouped.items()}, total)
        self.assertEqual(
            [r["native_action_prefix_count_local"] for r in result[("scene_b", "2")]],
            [0, 1, 4],
        )
        self.assertEqual(
            [r["native_action_prefix_count_global"] for r in result[("scene_b", "2")]],
            [3, 4, 7],
        )
        self.assertEqual(
            source[("scene_b", "2")][0]["native_action_prefix_count"], 3)
        self.assertEqual(
            result[("scene_c", "3")][-1]["native_action_prefix_count_local"], 2)

    def test_global_prefix_outside_own_episode_block_fails_closed(self):
        rows = (
            [action("scene_a", 1) for _ in range(3)]
            + [action("scene_b", 2) for _ in range(4)]
        )
        grouped, offsets, total, _ = audit.group_ordered_action_rows(rows)
        source = {("scene_b", "2"): [
            {"native_action_prefix_count": 2, "call_id": 0}
        ]}
        with self.assertRaisesRegex(ValueError, "does not fall within its episode action block"):
            audit.localize_global_s2_prefixes(
                source, offsets, {k: len(v) for k, v in grouped.items()}, total)

    def test_global_prefix_beyond_entire_trace_fails_closed(self):
        rows = [action("scene_a", 1) for _ in range(3)]
        grouped, offsets, total, _ = audit.group_ordered_action_rows(rows)
        source = {("scene_a", "1"): [
            {"native_action_prefix_count": 4, "call_id": 0}
        ]}
        with self.assertRaisesRegex(ValueError, "outside full native trace"):
            audit.localize_global_s2_prefixes(
                source, offsets, {k: len(v) for k, v in grouped.items()}, total)

    def test_noncontiguous_episode_blocks_are_rejected(self):
        rows = [
            action("scene_a", 1), action("scene_b", 2),
            action("scene_a", 1),
        ]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            audit.group_ordered_action_rows(rows)

    def test_episode_local_boundary_prefix_remains_per_episode(self):
        # The boundary recorder resets self.records at each episode begin;
        # unlike DecisionRecorder's global S2 counter, this prefix is local.
        grouped, offsets, total, _ = audit.group_ordered_action_rows(
            [action("scene_a", 1) for _ in range(3)]
            + [action("scene_b", 2) for _ in range(4)])
        boundary_prefix = 2
        self.assertLessEqual(boundary_prefix, len(grouped[("scene_b", "2")]))
        self.assertEqual(offsets[("scene_b", "2")], 3)
        self.assertEqual(total, 7)


if __name__ == "__main__":
    unittest.main()
