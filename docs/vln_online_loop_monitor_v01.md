# Online decision–visual loop monitor v0.1 (shadow only)

## Scope

A passive, prefix-causal monitor of decision repetition and RGB recurrence
in DualVLN's actual System-2 input/output stream. The monitor does not modify
the policy, history, navigation actions or simulator. An emitted event means
"candidate recurring decision–visual pattern" only, not established motion
stagnation, collision, LOST, semantic failure or recovery benefit.

The two R2R TRAIN failures VLzqgDo317F/58 and i5noydFURQK/70 motivated
the provisional thresholds. In-sample replay is NOT an independent detection
validation, and thresholds have not been calibrated on scene-disjoint data.

## Observable inputs and explicit exclusions

OnlineDecisionVisualLoopMonitor.observe(raw_s2_record) uses only:
- Episode identity, S2 call ID, main step and call type.
- Actual ordinary S2 output text and the final current RGB pixel SHA256.
- Current run-global action prefix for timestamping (not measuring motion).

The current RGB digest must match the final ordinary S2 image reference.
LOOKDOWN follow-ups are ignored for ordinary decision counts; they cannot
reset or inflate a consecutive output streak. Episode changes reset streaks;
reentry of a previously closed episode is rejected. Calls must be in strict
global call ID order and nondecreasing global action prefix order.

No ground-truth simulator positions, goal distances, success labels, future
images/actions, depth or GPS are accepted by the online monitor. There is
NO validated online motion/odometry response in this version. Action-prefix
changes are NOT evidence of actual movement. Motion-stagnation and semantic
root-cause claims remain unproven.

## Exploratory prefix-only candidate patterns

Static: six or more consecutive identical ordinary S2 outputs, with no more
than two distinct current RGB SHA256 values in the latest six ordinary calls.

Periodic: ten or more consecutive identical ordinary S2 outputs, together
with three identical complete visual cycles of length 3–8 ordinary decisions.
Each cycle must contain at least three different RGB SHA256 values. Three
complete cycles is intentionally stricter than the previous exploratory
four-position visual matching test.

A candidate of a given type is emitted once at the first qualifying S2 call
per episode. Events explicitly say motion_progress_status=NOT_OBSERVED and
recovery_action_selected=null. No physical recovery action is executed.

## CPU-only reproduction

From the original ~/InternNav checkout:

    git pull --recurse-submodules
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
      python -m unittest discover -s tests/unit_test \
      -p 'test_vln_online_loop_monitor_v01.py' -v

    python -u scripts/eval/vln_online_loop_replay_v01.py \
      --output /mnt/data8t/geenitiming/internnav_data/online_loop_shadow_replay_v01_01

Use a fresh suffix if the chosen output directory exists. Replaying reads
the frozen anomaly audit and its referenced original Native action/S2 files;
global S2 prefixes are localized against the complete native action log,
but episode-local indices are reported for OFFLINE_ALIGNMENT only.

The replay writes:
- decision_prefix_timeline.jsonl: all S2 call observations and emitted events.
- candidate_alarm_events.jsonl: prefix-causal candidate alarms.
- episode_offline_comparison.csv: 24 episodes; Native success is OFFLINE_ONLY.
- shadow_replay_manifest.json: source hashes, thresholds, evidence limits.

The monitor never receives Native success labels: those are joined only after
the replay to describe development-case coverage and success-case alarms.

## What is not done

This is not integrated into the live DualVLN evaluator and is not a
trigger-based paired recovery experiment. After CPU replay, the next steps
are passive online integration and trajectory-equivalence checks, a validated
online motion-response channel, then preregistered trigger-timed paired arms
with KEEP and explicit separation of SEEK's physical scan and extra RGB
effects. Do not claim held-out efficacy or train a Q-selector from this data.
