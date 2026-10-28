"""CPU-only checks for new-scene selection, online checkpoint gate and 5x2 grid."""
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import recovery_independent_cohort as cohort
import recovery_independent_checkpoint as checkpoint
import recovery_independent_factorial as factorial


def episode(scene, number):
    return {
        "scene_id": f"data/scene_datasets/mp3d/{scene}/{scene}.glb",
        "episode_id": str(number),
        "instruction": {"instruction_text": f"Go to location {number}"},
        "reference_path": [[0., 0., 0.], [1., 0., 1.]],
        "trajectory_id": f"route{number}",
    }


def native_trace(ep):
    return [
        {"scene_id": ep["scene_id"], "episode_id": str(ep["episode_id"]),
         "action": 1, "done": False, "position": [0., 0., 0.],
         "orientation_wxyz": [1., 0., 0., 0.]},
        {"scene_id": ep["scene_id"], "episode_id": str(ep["episode_id"]),
         "action": 0, "done": True, "position": [1., 0., 0.],
         "orientation_wxyz": [1., 0., 0., 0.]},
    ]


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            episode("oldScene", 1), episode("newA", 2), episode("newB", 3),
            episode("newA", 4), episode("newC", 5),
        ]
        self.dataset = {"episodes": self.episodes}
        self.progress, self.traces = {}, {}
        for ep in self.episodes:
            key = cohort.capture.episode_key(ep)
            self.progress[key] = {
                "episode_instruction": ep["instruction"]["instruction_text"],
                "success": 0., "spl": 0., "os": 0., "ne": 8.,
            }
            self.traces[key] = native_trace(ep)

    def test_scene_disjoint_no_success_based_selection(self):
        plan = cohort.select_plan(self.dataset, self.progress, self.traces,
                                  ["oldScene"], maximum_cases=2)
        self.assertEqual([x["key"] for x in plan["cases"]],
                         [["newA", "2"], ["newB", "3"]])
        # Do not change membership after reversing outcome labels
        for p in self.progress.values():
            p["success"] = 1. - p["success"]
        other = cohort.select_plan(self.dataset, self.progress, self.traces,
                                   ["oldScene"], maximum_cases=2)
        self.assertEqual([x["key"] for x in plan["cases"]],
                         [x["key"] for x in other["cases"]])

    def test_not_enough_new_scenes_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Need 8 unseen-scene"):
            cohort.select_plan(self.dataset, self.progress, self.traces,
                               ["oldScene"], maximum_cases=8)

    def test_first_online_idle_gate(self):
        args = dict(case={}, step=20, limit=500, body_count=2,
                    translation=1.1, done=False, action_seq=[],
                    pixel_goal=None, local_actions=[], action=1,
                    camera_balance=0, camera_pitch=0.)
        self.assertTrue(checkpoint.eligibility(**args)["eligible"])
        self.assertFalse(checkpoint.eligibility(**dict(args, step=19))["eligible"])
        self.assertFalse(checkpoint.eligibility(**dict(args, translation=.5))["eligible"])
        self.assertFalse(checkpoint.eligibility(**dict(args, pixel_goal=10))["eligible"])
        self.assertFalse(checkpoint.eligibility(**dict(args, limit=45))["eligible"])
        self.assertNotIn("native_nonstop_body_actions",
                         inspect.getsource(checkpoint.eligibility))
        self.assertNotIn("checkpoint_body_threshold",
                         inspect.getsource(checkpoint.eligibility))

    def test_complete_angle_x_history_grid(self):
        self.assertEqual(len(factorial.ARMS), 9)
        self.assertEqual(set(factorial.RESET_ARMS),
                         {"history_reset", "left45_reset", "right45_reset",
                          "left90_reset", "right90_reset"})
        self.assertEqual(set(factorial.ROTATIONS.values()), {-90, -45, 45, 90})
        self.assertEqual(len(factorial.ROTATIONS), 8)
        self.assertTrue(all(abs(a) % 15 == 0 for a in factorial.ROTATIONS.values()))
        self.assertNotIn("backtrack_replan", factorial.ARMS)


if __name__ == "__main__":
    unittest.main()
