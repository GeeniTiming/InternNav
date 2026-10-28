"""Lightweight scene-diverse pilot selection tests; no GPU/model/Habitat."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from recovery_scene_diverse_pilot import select_scene_diverse


def item(scene, n, text="go"):
    return {
        "scene_id": "data/scene_datasets/mp3d/" + scene + "/" + scene + ".glb",
        "episode_id": n, "instruction": {"instruction_text": text},
    }


class DiversePilotTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            item("old1", 1), item("old1", 2), item("newA", 3),
            item("newA", 4), item("old2", 5), item("newB", 6),
            item("newC", 7),
        ]
        self.source = {"episodes": copy.deepcopy(self.episodes), "meta": "original"}
        self.excluded = {"old1", "old2"}

    def test_selects_new_distinct_scenes_in_first_occurrence_order(self):
        subset = select_scene_diverse(self.source, 3, self.excluded)
        self.assertEqual([ep["episode_id"] for ep in subset["episodes"]], [3, 6, 7])
        self.assertEqual(self.source["episodes"], self.episodes)
        self.assertEqual(subset["meta"], "original")

    def test_filters_only_metadata_not_results(self):
        subset = select_scene_diverse(self.source, 2, self.excluded)
        self.assertEqual([ep["episode_id"] for ep in subset["episodes"]], [3, 6])
        # A duplicate scene appearing sooner does not change the scene identity.
        self.assertEqual(len({ep["scene_id"] for ep in subset["episodes"]}), 2)

    def test_refuses_insufficient_scene_diversity(self):
        with self.assertRaisesRegex(ValueError, "Need 4 new distinct scenes, found 3"):
            select_scene_diverse(self.source, 4, self.excluded)

    def test_refuses_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            select_scene_diverse(self.source, 0, self.excluded)

    def test_skips_missing_instruction(self):
        original = copy.deepcopy(self.source)
        original["episodes"][2]["instruction"]["instruction_text"] = "  "
        subset = select_scene_diverse(original, 2, self.excluded)
        self.assertEqual([ep["episode_id"] for ep in subset["episodes"]], [4, 6])


if __name__ == "__main__":
    unittest.main()
