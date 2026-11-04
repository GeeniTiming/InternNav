"""Behavioral and causal-contract tests for the passive VLN S2 loop monitor."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import vln_online_loop_monitor_v01 as monitor
import vln_online_loop_replay_v01 as replay


def s2_row(call_id, key=("scene", "1"), output="↓", rgb=None,
           kind="ordinary", main_step=None, prefix=None):
    rgb = rgb if rgb is not None else ("a" * 64)
    ref = "images/" + rgb + ".png"
    if kind == "lookdown_followup":
        # The original followup's last image is a separate source image and
        # must NOT be passed as ordinary current RGB.
        ref = "images/" + "f" * 64 + ".png"
    return {
        "episode_key": list(key),
        "call_id": call_id,
        "main_step": call_id if main_step is None else main_step,
        "call_type": kind,
        "native_action_prefix_count": call_id if prefix is None else prefix,
        "model_output": output,
        "images": [{"path": ref, "pixel_sha256": rgb}],
        "current_image_ref": ref if kind == "ordinary" else None,
        "native_parsed_actions": [5] if output == "↓" else [2] * 4,
    }


def hashes(symbols):
    return [hashlib.sha256(ch.encode("ascii")).hexdigest() for ch in symbols]


class MonitorPrefixTests(unittest.TestCase):
    def test_static_current_rgb_and_output_repeat_fires_at_sixth_decision(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        rows = [s2_row(i, rgb="a" * 64) for i in range(9)]
        observed = [engine.observe(row)["events"] for row in rows]
        self.assertEqual([len(x) for x in observed], [0, 0, 0, 0, 0, 1, 0, 0, 0])
        event = observed[5][0]
        self.assertEqual(event["kind"], monitor.STATIC_KIND)
        self.assertEqual(event["trigger_call_id"], 5)
        self.assertEqual(event["evidence_online_prefix_only"]["consecutive_same_s2_decisions"], 6)
        self.assertFalse(event["uses_ground_truth_motion"])
        self.assertEqual(event["motion_progress_status"], "NOT_OBSERVED")
        self.assertIsNone(event["recovery_action_selected"])

    def test_followup_does_not_reset_or_count_as_ordinary_decision(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        events = []
        for i in range(6):
            rows = [
                s2_row(i * 2, rgb="a" * 64, prefix=i * 2,
                       main_step=i),
                s2_row(i * 2 + 1, kind="lookdown_followup",
                       output="371 170", prefix=i * 2 + 1, main_step=i),
            ]
            events.extend(engine.observe(rows[0])["events"])
            followup = engine.observe(rows[1])
            self.assertEqual(followup["status"], "IGNORED_FOLLOWUP")
            self.assertFalse(followup["events"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["trigger_call_id"], 10)
        self.assertEqual(engine.ordinary_count, 6)
        self.assertEqual(engine.followup_count, 6)

    def test_six_distinct_rgb_prevents_static_candidate(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        out = [engine.observe(s2_row(i, rgb=dig)) for i, dig in
               enumerate(hashes("abcdefghij"))]
        self.assertFalse(any(x["events"] for x in out))

    def test_periodic_visual_cycle_needs_three_full_cycles(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        sequence = hashes("abcdef" * 3)
        emitted = []
        for i, digest in enumerate(sequence):
            emitted.extend(engine.observe(
                s2_row(i, output="←←←←", rgb=digest))["events"])
        self.assertEqual([x["kind"] for x in emitted], [monitor.PERIODIC_KIND])
        self.assertEqual(emitted[0]["trigger_call_id"], 17)
        evidence = emitted[0]["evidence_online_prefix_only"]
        self.assertEqual(evidence["period_ordinary_decisions"], 6)
        self.assertEqual(evidence["complete_matching_visual_cycles"], 3)
        self.assertEqual(evidence["distinct_current_rgb_hashes_per_cycle"], 6)

    def test_five_left_turn_decisions_with_changed_images_do_not_alarm(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        for i, digest in enumerate(hashes("abcde")):
            self.assertFalse(engine.observe(s2_row(
                i, output="←←←←", rgb=digest))["events"])

    def test_identical_rgb_but_changing_outputs_do_not_alarm(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        for i in range(24):
            output = "↓" if i % 2 == 0 else "←←←←"
            self.assertFalse(engine.observe(s2_row(
                i, output=output, rgb="a" * 64))["events"])

    def test_prefix_invariance_of_alarms_and_causal_no_future_dependence(self):
        digest = hashes("abcdef" * 3 + "uvwxyz")
        rows = [s2_row(i, output="←←←←", rgb=img)
                for i, img in enumerate(digest)]
        online = monitor.OnlineDecisionVisualLoopMonitor()
        online_outputs = [online.observe(row)["events"] for row in rows]
        for end in range(1, len(rows) + 1):
            fresh = monitor.OnlineDecisionVisualLoopMonitor()
            prefix_outputs = [fresh.observe(row)["events"] for row in rows[:end]]
            self.assertEqual(prefix_outputs, online_outputs[:end])

    def test_episode_boundary_resets_streak_and_prior_episode_cannot_reappear(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        for i in range(5):
            self.assertFalse(engine.observe(s2_row(i, key=("s1", "1")))["events"])
        for i in range(5, 10):
            self.assertFalse(engine.observe(s2_row(i, key=("s2", "2")))["events"])
        with self.assertRaisesRegex(ValueError, "Previously closed"):
            engine.observe(s2_row(10, key=("s1", "1")))

    def test_privileged_and_invalid_current_image_are_rejected(self):
        row = s2_row(0)
        for field in ("sim_position", "distance_to_goal", "success", "future_actions"):
            with self.subTest(field=field):
                bad = dict(row)
                bad[field] = 0
                with self.assertRaisesRegex(ValueError, "Privileged"):
                    monitor.OnlineDecisionVisualLoopMonitor().observe(bad)
        bad = dict(row)
        bad["current_image_ref"] = "images/other.png"
        with self.assertRaisesRegex(ValueError, "not final ordinary"):
            monitor.OnlineDecisionVisualLoopMonitor().observe(bad)
        bad = dict(row)
        bad["images"] = [{"path": row["current_image_ref"], "pixel_sha256": "NOT_HASH"}]
        with self.assertRaisesRegex(ValueError, "digest is malformed"):
            monitor.OnlineDecisionVisualLoopMonitor().observe(bad)

    def test_nonmonotonic_call_id_or_action_prefix_fails_closed(self):
        engine = monitor.OnlineDecisionVisualLoopMonitor()
        engine.observe(s2_row(10, prefix=20))
        with self.assertRaisesRegex(ValueError, "call_id must increase"):
            engine.observe(s2_row(10, prefix=21))
        with self.assertRaisesRegex(ValueError, "action prefix must never decrease"):
            engine.observe(s2_row(11, prefix=19))

    def test_invalid_config_does_not_silently_disable_tests(self):
        with self.assertRaises(ValueError):
            monitor.LoopConfig(static_max_unique_rgb=6)
        with self.assertRaises(ValueError):
            monitor.LoopConfig(periodic_min_period=2, periodic_min_unique_rgb_per_cycle=3)


class ReplayContractTests(unittest.TestCase):
    def fixture(self):
        ordered = []
        order = []
        lengths = {}
        offline = {}
        call_id = 0
        for idx in range(12):
            key = ("scene" + str(idx), str(idx))
            order.append(key)
            offline[("development", *key)] = {
                "native_success": "0" if idx == 0 else "1",
            }
            row_count = 6 if idx == 0 else 1
            for i in range(row_count):
                dig = "a" * 64 if idx == 0 else hashes(chr(97 + idx))[0]
                row = s2_row(
                    call_id, key=key, output="↓", rgb=dig,
                    main_step=i, prefix=call_id)
                row["action_prefix_count_episode"] = i
                ordered.append(row)
                call_id += 1
            lengths[key] = 40
        source = {"ordered": ordered, "episode_order": order, "lengths": lengths}
        return source, offline

    def test_12_episode_replay_labels_are_offline_and_dont_enter_monitor(self):
        source, offline = self.fixture()
        decisions, alarms, summary = replay.replay_stream(
            "development", source, offline)
        self.assertEqual(len(summary), 12)
        self.assertEqual(len(decisions), 17)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["trigger_action_prefix_episode_OFFLINE_ALIGNMENT"], 5)
        self.assertTrue(all("native_success" not in x for x in decisions))
        self.assertEqual(summary[0]["native_success_OFFLINE_ONLY"], 0)
        self.assertEqual(sum(x["candidate_triggered"] for x in summary), 1)
        self.assertEqual(sum(x["native_success_OFFLINE_ONLY"] for x in summary), 11)

    def test_create_only_report_and_separate_offline_results(self):
        source, offline = self.fixture()
        decisions, alarms, summary = replay.replay_stream(
            "development", source, offline)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report"
            report = {
                "N_episodes": 12, "N_S2_calls": len(decisions),
                "N_alarm_events": len(alarms), "N_alarm_episodes": 1,
                "alarm_episodes_by_offline_label": {"native_success": 0, "native_failure": 1},
            }
            replay.write_create_only(path, decisions, alarms, summary, report)
            self.assertEqual(len(list(replay.load_jsonl(
                path / "candidate_alarm_events.jsonl"))), 1)
            with (path / "episode_offline_comparison.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 12)
            with self.assertRaises(FileExistsError):
                replay.write_create_only(path, decisions, alarms, summary, report)

    def test_replay_fails_if_local_action_prefix_is_invalid(self):
        source, offline = self.fixture()
        source["ordered"][0]["action_prefix_count_episode"] = 41
        with self.assertRaisesRegex(ValueError, "outside Native action block"):
            replay.replay_stream("development", source, offline)


if __name__ == "__main__":
    unittest.main()
