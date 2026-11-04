#!/usr/bin/env python3
"""Passive, prefix-causal VLN decision/visual loop monitor (v0.1).

Input: current System-2 output and current ordinary RGB pixel digest, plus
call identity/time. No future actions, Habitat pose, goal distance, or outcome.
Ignored LOOKDOWN follow-ups do not count as ordinary decisions.

An alarm means a *repeated visual-decision pattern candidate*, NOT collision,
zero physical motion, LOST, erroneous navigation, or justified intervention.
Thresholds are exploratory and were informed by two TRAIN failure episodes;
do not present an in-sample replay as independent detection validation.

The model/navigation policy and its actions are never modified here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "vln_online_loop_monitor.v0.1"
STATIC_KIND = "STATIC_VISUAL_DECISION_REPEAT_CANDIDATE"
PERIODIC_KIND = "PERIODIC_VISUAL_DECISION_REPEAT_CANDIDATE"
DISALLOWED_ONLINE_FIELDS = frozenset((
    "sim_position", "sim_orientation", "distance_to_goal", "success",
    "reference_path", "future_actions", "oracle_success", "depth",
    "gps", "final_ne", "native_success", "final_goal_euclidean_m",
    "evidence_OFFLINE_ONLY", "OFFLINE_ONLY",
))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LoopConfig:
    # Candidate thresholds only; retrospectively selected on observed TRAIN cases.
    static_min_same_decisions: int = 6
    static_recent_images: int = 6
    static_max_unique_rgb: int = 2
    periodic_min_same_decisions: int = 10
    periodic_min_period: int = 3
    periodic_max_period: int = 8
    periodic_complete_cycles: int = 3
    periodic_min_unique_rgb_per_cycle: int = 3
    max_streak_kept: int = 512

    def __post_init__(self):
        if (self.static_recent_images < 2
                or self.static_min_same_decisions < self.static_recent_images
                or not 1 <= self.static_max_unique_rgb < self.static_recent_images):
            raise ValueError("Invalid static-repeat thresholds")
        if (self.periodic_min_period < 2
                or self.periodic_max_period < self.periodic_min_period
                or self.periodic_complete_cycles < 2
                or self.periodic_min_same_decisions < 2
                or self.periodic_min_unique_rgb_per_cycle < 2
                or self.periodic_min_unique_rgb_per_cycle > self.periodic_min_period
                or self.max_streak_kept < max(
                    self.static_min_same_decisions,
                    self.periodic_min_same_decisions,
                    self.periodic_complete_cycles * self.periodic_max_period)):
            raise ValueError("Invalid periodic-repeat thresholds")

    def as_dict(self):
        return asdict(self)


def project_online_s2_call(raw):
    """Explicit allowlist: only fields physically observable at the S2 call.

    This projection intentionally discards prompt/history text and NEVER reads
    future or privileged simulator fields even when present in an offline log.
    """
    if not isinstance(raw, dict):
        raise TypeError("S2 call must be a dict")
    if DISALLOWED_ONLINE_FIELDS.intersection(raw):
        raise ValueError("Privileged/future fields must not enter online loop monitor")
    for required in ("episode_key", "call_id", "main_step", "call_type",
                     "native_action_prefix_count", "model_output"):
        if required not in raw:
            raise ValueError("Missing online S2 field: " + required)
    call_type = raw["call_type"]
    if call_type not in ("ordinary", "lookdown_followup"):
        raise ValueError("Unknown S2 call type: " + str(call_type))
    key = raw["episode_key"]
    if (not isinstance(key, (list, tuple)) or len(key) != 2
            or not all(str(value).strip() for value in key)):
        raise ValueError("Malformed S2 episode identity")
    digest = None
    if call_type == "ordinary":
        refs = raw.get("images")
        if not isinstance(refs, list) or not refs:
            raise ValueError("Ordinary S2 call has no model RGB input")
        ref = refs[-1]
        if not isinstance(ref, dict) or raw.get("current_image_ref") != ref.get("path"):
            raise ValueError("Current RGB is not final ordinary S2 model input")
        digest = ref.get("pixel_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("Ordinary current RGB digest is malformed")
    elif raw.get("current_image_ref") is not None:
        raise ValueError("LOOKDOWN follow-up must not masquerade as current RGB")
    if not isinstance(raw["model_output"], str):
        raise ValueError("S2 model output must be text")
    for field in ("call_id", "main_step", "native_action_prefix_count"):
        if type(raw[field]) is not int or raw[field] < 0:
            raise ValueError("Online S2 counter must be nonnegative integer: " + field)
    return {
        "episode_key": tuple(str(value) for value in key),
        "call_id": raw["call_id"],
        "main_step": raw["main_step"],
        "call_type": call_type,
        "native_action_prefix_count": raw["native_action_prefix_count"],
        "model_output": raw["model_output"].strip(),
        "current_rgb_pixel_sha256": digest,
    }


class OnlineDecisionVisualLoopMonitor:
    """One monotonically ordered run, reset at every episode boundary.

    observe() consumes exactly the present call and past state and returns the
    newly emitted candidate event(s), at most once per kind per episode.
    It does not recommend or execute recovery actions.
    """
    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config if config is not None else LoopConfig()
        self.episode_key = None
        self.closed_episodes = set()
        self.last_call_id = -1
        self.last_global_action_prefix = -1
        self.last_main_step = -1
        self.last_output = None
        self.same_output_streak: List[Dict[str, Any]] = []
        self.emitted_kinds = set()
        self.ordinary_count = 0
        self.followup_count = 0

    def _new_episode(self, key):
        if self.episode_key is not None:
            self.closed_episodes.add(self.episode_key)
        if key in self.closed_episodes:
            raise ValueError("Previously closed S2 episode reappeared: " + repr(key))
        self.episode_key = key
        self.last_main_step = -1
        self.last_output = None
        self.same_output_streak = []
        self.emitted_kinds = set()
        self.ordinary_count = 0
        self.followup_count = 0

    def _emit(self, kind, row, evidence):
        self.emitted_kinds.add(kind)
        return {
            "schema": SCHEMA,
            "kind": kind,
            "episode_key": list(self.episode_key),
            "trigger_call_id": row["call_id"],
            "trigger_main_step": row["main_step"],
            "trigger_global_action_prefix": row["native_action_prefix_count"],
            "repeated_model_output": row["model_output"],
            "evidence_online_prefix_only": evidence,
            "uses_ground_truth_motion": False,
            "motion_progress_status": "NOT_OBSERVED",
            "causal_root_cause_established": False,
            "recovery_action_selected": None,
        }

    def observe(self, raw):
        row = project_online_s2_call(raw)
        key = row["episode_key"]
        if row["call_id"] <= self.last_call_id:
            raise ValueError("S2 call_id must increase strictly across run")
        if row["native_action_prefix_count"] < self.last_global_action_prefix:
            raise ValueError("Global action prefix must never decrease")
        if key != self.episode_key:
            self._new_episode(key)
        if row["main_step"] < self.last_main_step:
            raise ValueError("Within-episode main_step must never decrease")
        self.last_call_id = row["call_id"]
        self.last_global_action_prefix = row["native_action_prefix_count"]
        self.last_main_step = row["main_step"]

        if row["call_type"] == "lookdown_followup":
            self.followup_count += 1
            return {
                "call_id": row["call_id"], "call_type": row["call_type"],
                "episode_key": list(key), "status": "IGNORED_FOLLOWUP",
                "same_output_streak": len(self.same_output_streak),
                "events": [],
            }

        self.ordinary_count += 1
        if row["model_output"] and row["model_output"] == self.last_output:
            self.same_output_streak.append(row)
            if len(self.same_output_streak) > self.config.max_streak_kept:
                self.same_output_streak.pop(0)
        else:
            self.same_output_streak = [row]
        self.last_output = row["model_output"]
        streak = self.same_output_streak
        k = len(streak)
        hashes = [x["current_rgb_pixel_sha256"] for x in streak]
        events = []
        cfg = self.config

        if (STATIC_KIND not in self.emitted_kinds
                and row["model_output"]
                and k >= cfg.static_min_same_decisions
                and len(set(hashes[-cfg.static_recent_images:]))
                    <= cfg.static_max_unique_rgb):
            events.append(self._emit(STATIC_KIND, row, {
                "consecutive_same_s2_decisions": k,
                "recent_ordinary_decision_count": cfg.static_recent_images,
                "distinct_current_rgb_hashes_in_recent_decisions": len(
                    set(hashes[-cfg.static_recent_images:])),
                "recent_current_rgb_hashes": hashes[-cfg.static_recent_images:],
                "criterion": "output_recurrence_AND_low_visual_diversity",
            }))

        if (PERIODIC_KIND not in self.emitted_kinds
                and row["model_output"]
                and k >= cfg.periodic_min_same_decisions):
            for period in range(cfg.periodic_min_period, cfg.periodic_max_period + 1):
                window_len = period * cfg.periodic_complete_cycles
                if k < window_len:
                    continue
                tail = hashes[-window_len:]
                cycles = [tail[i * period:(i + 1) * period]
                          for i in range(cfg.periodic_complete_cycles)]
                if (len(set(cycles[-1])) >= cfg.periodic_min_unique_rgb_per_cycle
                        and all(cycle == cycles[-1] for cycle in cycles[:-1])):
                    events.append(self._emit(PERIODIC_KIND, row, {
                        "consecutive_same_s2_decisions": k,
                        "period_ordinary_decisions": period,
                        "complete_matching_visual_cycles": cfg.periodic_complete_cycles,
                        "distinct_current_rgb_hashes_per_cycle": len(set(cycles[-1])),
                        "cycle_current_rgb_hashes": cycles[-1],
                        "criterion": "output_recurrence_AND_periodic_visual_recurrence",
                    }))
                    break

        return {
            "call_id": row["call_id"],
            "call_type": row["call_type"],
            "episode_key": list(key),
            "status": "CANDIDATE" if events else "NO_NEW_CANDIDATE",
            "same_output_streak": k,
            "events": events,
        }
