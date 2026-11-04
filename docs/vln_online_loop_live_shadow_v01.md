# DualVLN live S2 shadow monitor v0.1

The 24-episode replay in vln_online_loop_replay_v01.py has two in-sample
candidate alarms. This stage connects the **same state machine** to the
actual native S2 generation callback, without proposing or executing recovery
actions. It is an *engineering integration*, not a held-out effectiveness
study.

## Integration

An existing AST-guarded, native-only capture hook logs actual S2 generation
in recovery_task_evidence_acquire.py. Its --shadow-monitor option is OFF by
default. When explicitly enabled, it creates a LiveShadowSink in the fresh
capture/ directory. The sink receives the unchanged, complete recorder row
only after the original model response was decoded and parsed. It writes:

- capture/online_loop_shadow_timeline.jsonl
- capture/online_loop_shadow_events.jsonl
- capture/online_loop_shadow_summary.json

The original online_decisions.jsonl, image catalog, action trace and
OFFLINE_ONLY position logging have unchanged schemas. The shadow sink never
writes back to the evaluator, action parser, model or Habitat. All outputs
are create-only; existing Native runs and previous evidence are untouched.
A monitor exception fails the new run instead of silently ignoring corrupted
state.

The monitor never receives true simulator position, goal distance, success
labels, future trajectory or an odometry estimate. Candidate alarms mean
repeated RGB/decision patterns, not confirmed zero movement or LOST.
Only online-decision fields are written to the shadow timeline, no prompt
or full image history. Thresholds were motivated by two TRAIN failures.

## A small native live engineering smoke test

Use the current ~/InternNav checkout (no new working directory). The exact
root below must not exist before prepare.

    cd ~/InternNav
    git pull --recurse-submodules

    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
      python -m unittest discover -s tests/unit_test \
      -p 'test_vln_online_loop_live_shadow_v01.py' -v

    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
      python -m unittest discover -s tests/unit_test \
      -p 'test_vln_online_loop_monitor_v01.py' -v

    ROOT=/mnt/data8t/geenitiming/internnav_data/online_loop_live_shadow_smoke_v01_01

    python scripts/eval/recovery_task_evidence_acquire.py prepare \
      --run-dir "$ROOT" --episodes 2 --seed 20261029 --max-steps 80

    python -u scripts/eval/recovery_task_evidence_acquire.py run \
      --run-dir "$ROOT" --shadow-monitor

Run needs the usual model checkpoint, Habitat data and CUDA. The capture is
a NEW run in the existing experiment data tree, not a rerun/overwrite of
earlier Native studies. This is a short engineering smoke test, so zero
alerts are possible and not evidence of lack of anomalies.

Next, CPU-check exact shadow/replay equivalence:

    python -u scripts/eval/vln_online_loop_live_shadow_v01.py \
      --run-dir "$ROOT" \
      --output "$ROOT/live_shadow_exact_replay_report_v01.json"

Expected terminal status:
LIVE_SHADOW_EXACT_PREFIX_REPLAY_MATCH

The verification independently recalculates every call's event and timeline
from original captured S2 rows; it also checks global action alignment
against the captured action log. It is read-only except the create-only
verification report path. If the run was interrupted, start a new run
directory; never overwrite/fix the old capture in place.

## What this proves and what it does not

Passing verifier means **the live observed outputs match an independent CPU
replay for the same original S2 decisions**, including call timing, episode
identity, decision output and RGB hash. CPU unit tests assert that adding the
observer does not change the original recorder's S2 rows or native fake
actions in a deterministic fixture.

It does NOT by itself prove that the full GPU live navigator's policy,
actions, SR or SPL are unchanged relative to a separate noninstrumented run.
That requires a matched-seed, identical-configuration, identical-episode
control trace comparison. The verifier explicitly reports this status as
NOT_DONE. It also does not establish detection generalization or recovery
efficacy. After the engineering pilot, undertake trajectory-equivalence
control and then scene-disjoint shadow evaluation before trigger-time recovery.
