# Case15: two positions by three headings

This predeclared diagnostic follows the completed in-place prompt controls.
It tests the same selected development case at main step 85, without training,
new prompt tuning, or expansion of the closed 24-case train investigation.

## Why the design changed

The recorded historical-anchor heading and intervention-entry heading differ
by only about 0.00000153 degrees. Returning while preserving one versus restoring
the other would be effectively the same treatment. We therefore declare the
full two-position by three-heading grid before seeing any new outcomes:

| Arm | Position at handoff | Heading relative to the original step-85 heading |
| --- | --- | --- |
| control | Original intervention location | 0 degrees; unchanged native control |
| turn_left90 | Same intervention location | +90 degrees (LEFT) |
| turn_right90 | Same intervention location | -90 degrees (RIGHT) |
| anchor_same_heading | Recorded historical-anchor neighborhood | 0 degrees |
| anchor_left90 | Same returned location | +90 degrees |
| anchor_right90 | Same returned location | -90 degrees |

The anchor is the position of native actual-action record 393 (zero-based),
already locked during the earlier audit. The trigger is after record 445.
Candidate construction uses only these recorded prefix poses. It does not add
the earlier physical reference's second waypoint, which followed the correct
reference-route direction. The original anchor was selected in this inspected
case, so this is not an online anchor-selection algorithm.

## Physical execution and comparability

- All changes use actual Habitat FORWARD/LEFT/RIGHT actions. No pose setter,
  teleport, or fixed-pose sensor query is used.
- Rotation-only arms perform six native 15-degree turns without translation.
- Return arms use the existing simulator shortest-path follower to reach the
  same historical target, then use native turns to align the final heading.
  The follower is privileged execution, not a deployable RGB-only solution.
- The return target tolerance is 0.35 m, consistent with the earlier diagnostic.
  The three return arms must nevertheless have identical actual return action
  and pose traces (within 1e-5), and matching return endpoints before alignment.
  Their final positions must also agree within 1e-5. Requested and actual
  handoff poses are both reported.
- Native turn angle is checked against the recorded prefix. A turn must change
  heading by the expected 15 degrees and leave position unchanged. An off-grid
  target, budget exhaustion or unexpected displacement aborts the diagnostic.
- Each correction has a fixed maximum of 80 actual body actions. Follower STOP
  signals are consumed internally, never executed as episode STOP.
- Correction cannot enter the goal region; final handoff remains more than
  one metre outside the goal radius. Goal distance is an offline exclusion
  check, not an input to candidate selection.
- Original visual history is retained. Intermediate correction views are saved
  separately and are not ingested. The true final observation enters the next
  native iteration once. Policy RNG is restored and parameters remain frozen.
- Native episode budgets remain unchanged; actual correction actions count in
  the environment. Action costs differ across arms and are reported.
- A fresh control must reproduce the complete 547-action native failure. Every
  treatment first matches the source and fresh-control prefix, pose, observed
  image/depth hashes, history and RNG. A failed gate is NOT_COMPARABLE.

This varies physical state and corresponding observations together; it does not
isolate an internal memory cause. Simulator navigation and recorded poses are
privileged diagnostic resources. The full-reference correction remains a
privileged comparison rather than an autonomous baseline.

## Run in the existing checkout, foreground in tmux

```bash
cd /home/geenitiming/InternNav
conda activate internnav_habitat
git switch perf/dualvln-cache-pilot
git pull --ff-only

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
  -p test_case15_pose_factorial.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/case15_pose_factorial.py prepare \
  --source-run data/case15_reinstruction_01 \
  --run-dir data/case15_pose_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/case15_pose_factorial.py run-all \
  --run-dir data/case15_pose_01
```

Preparation is CPU-only and create-only. After it passes, run-all starts six
separate single-episode processes sequentially, then prints the report. It
stops immediately when a child fails. It does not launch a multi-GPU or
parallel workload, and does not signal other processes. Existing directories
are not overwritten or silently resumed.

Individual arms remain available through `run --run-dir ... --arm ARM`.
For manual execution use the declared order above; anchor_same_heading must
finish before the other two return arms. A report can be regenerated without
GPU work:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/case15_pose_factorial.py report \
  --run-dir data/case15_pose_01
```

## Outputs and interpretation

The result is `data/case15_pose_01/pose_factorial_report.json`.
Each arm records native actions, `probe_actions.jsonl`, the boundary gate,
`pose_summary.json`, and (for treatments) `after_pose_recovery.json`.
Autonomous debug videos are under `video/ARM/epoch_0/`; these jump over the
external correction. Actual correction observations are saved in
`ARM/recovery_frames/`.

Rotation-only success supports a viewpoint/heading intervention in this case.
Return success at the original heading supports a return intervention.
Different outcomes at the same returned location support heading-dependent
continuation under this protocol. If all five treatments fail while the full
reference-informed correction succeeds, these historical-position/heading
choices were insufficient; that does not establish that every other local
intervention must fail.

Report every declared arm, correction cost, handoff pose and terminal action.
Review videos separately from metric success. Do not tune new angles/anchors
on this selected case to manufacture a winning method. No held-out efficacy,
causal memory diagnosis, generalization or timing speedup follows.
