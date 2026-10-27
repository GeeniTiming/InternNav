# Cross-case orientation diagnostic: selection only

Case15 suggests that a heading intervention can change frozen-policy continuation.
The next question is whether this occurs on other routes and whether the same
intervention harms native successes. This script locks a small diagnostic cohort;
it does not implement an autonomous recovery method.

## Prepare on CPU

From the existing project and environment:

```bash
cd /home/geenitiming/InternNav
conda activate internnav_habitat
git pull --ff-only

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
    -p test_orientation_cohort.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/orientation_cohort.py prepare \
    --source-run data/cache_pilot_05 \
    --run-dir data/orientation_crosscase_01
```

Only `prepare` is available. It reads the completed, unprofiled 20-episode native
control run, validates its input manifest, annotations, progress and action traces,
and writes two new files: `plan.json` and `cohort_manifest.json`. The latter locks
input/code/plan hashes and the existing checkpoint-shard sizes. No Habitat/model
rollout, checkpoint capture, arm directory or training is started. Existing output
directories are rejected; source results remain unchanged.

## Selection fixed before interventions

- Merge same-scene route siblings transitively by trajectory ID or reference-path
  geometry rounded to four decimal places.
- Exclude the entire group containing scene `zsNo4HB9uLZ`, episode `15`, including
  its other instruction variants.
- A group with any native failure belongs to the failure stratum. Its successful
  siblings cannot become success controls. Choose its first failed instruction in
  source annotation order. For an all-success group, choose its first instruction.
- Choose at most three groups per stratum, first preferring distinct scenes, then
  filling in source annotation order. Interleave failure and success cases.
- Use only native-control outcomes. Cache or recovery outcomes never affect this
  selection. Keep shortages and ineligible cases; do not replace or expand them.

The completed 24-route train investigation remains closed. This uses a separate,
already inspected val-unseen development pool. It is not a new training dataset,
an estimate of failure prevalence, or held-out evaluation.

## Checkpoint rule locked now, capture deferred

For each selected native trace, count non-STOP body actions (FORWARD/LEFT/RIGHT;
exclude camera actions). Set the threshold to `ceil(0.60 * count)`. In a future
native replay, capture the first idle, level-camera System-2 boundary at or after
that threshold, after at least 1 m of accumulated prefix translation, with at
least six baseline non-STOP body actions remaining.

This timing uses completed baseline length and is privileged offline diagnosis,
not an online error detector. If no boundary qualifies, retain the case as
`NOT_ELIGIBLE`; do not move the threshold or choose another case. Preparation can
flag an insufficient remaining baseline budget, but cannot yet establish boundary
eligibility. `PENDING_NATIVE_REPLAY` does not mean a checkpoint has been captured.

The future runner must use all three declared headings at that same position:
native control, +90 degrees (left), and -90 degrees (right). Use actual native
turn actions, verify the turn and unchanged position, preserve history and frozen
parameters, restore policy RNG, log intermediate views without ingesting them,
and normally ingest the final observation once. Intervention actions consume the
original budget. Require full native-control reproduction before interpreting
continuations. No external translation or GT waypoint is part of this grid.

## Read the output

`LOCKED_SELECTION_PENDING_CHECKPOINT_CAPTURE` means three groups per stratum.
`LOCKED_FEWER_GROUPS_PENDING_CHECKPOINT_CAPTURE` means at least one group in each
stratum, with the available count retained. `INSUFFICIENT_STRATA_STOP` means one
stratum is absent and this protocol stops. Inspect the plan before implementing
checkpoint capture; there is no GPU command at this stage.

Even a later positive result would support only a bounded orientation-intervention
effect on this development cohort. It would not by itself establish memory
corruption, language completion, a general recovery algorithm, or an online rule
for choosing a useful heading. Native-success controls are needed to measure harm.