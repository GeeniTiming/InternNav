# Next phase: new TRAIN scene-disjoint native checkpoint/evidence acquisition

## What is and is not done

The old two-case six-arm pilot is frozen on
\`exp/intervention-choice-pilot-v1\`. Its outcome differences are development
examples and are NOT used to choose new training trajectories or checkpoints.

This new branch adds:

- \`recovery_train_disjoint_acquire.py\`: prepare an entirely new create-only
  R2R TRAIN native acquisition, removing previously inspected scene IDs
  *before* selecting the first instruction-bearing episode per new scene
- \`recovery_train_checkpoint_evidence.py\`: create source-hashed, task-grounded
  v2 **unreviewed** annotation templates for the exact original System-2
  checkpoint boundaries, using original model-sized RGB history and actual
  predecision action prefixes only
- CPU regression tests (selection, prefix/index guards, provenance, and
  evidence-label semantics)

Old native evaluator/checkpoint/factorial scripts are reused unchanged.
This is a source-and-checkpoint **qualification stage**, not a 72-rollout
study, an online predictor, or a model training job.

## 1 — Clone/switch and CPU test

\`\`\`bash
cd ~/InternNav
git fetch origin
git switch exp/train-scene-checkpoint-acquisition-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_train_checkpoint_acquisition.py' -v
\`\`\`

Do **not** continue with a failing CPU test. This branch does not modify the
already locked sources or old \`task_evidence_train_smoke_02_03\`.

## 2 — Predeclare a 12-new-scene R2R TRAIN source

Previously examined scene IDs are explicitly excluded in the new script:
\`oLBMNvg9in8\`, \`X7HyMhZNoso\`, \`5LpN3gDmAk7\`,
\`7y3sRwLe3Va\` and the five previously examined R2R/VLN scenes.
The frozen previous TRAIN source is additionally checked from its manifest.
Extra exclusion flags are optional.

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_disjoint_acquire.py prepare \
  --episodes 12 \
  --exclude-source-run data/task_evidence_train_smoke_02_03 \
  --run-dir data/train_disjoint_native_12_01
\`\`\`

Expected CPU status: \`TRAIN_DISJOINT_PREPARE=PASS\` with 12 distinct new
TRAIN scene keys. Selection uses ONLY the original dataset source order,
literal instruction presence and predeclared scene exclusions, never native
success, NE, future actions, visual oracle features or target coordinates.
Frozen source hashes and the exclusion set are written to \`manifest.json\`.

## 3 — Run original native controls / collect all System-2 decisions

**GPU phase 1**; not an intervention.

\`\`\`bash
set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_task_evidence_acquire.py run \
  --run-dir data/train_disjoint_native_12_01 \
  2>&1 | tee train_disjoint_native_12_01.log

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_disjoint_acquire.py audit \
  --run-dir data/train_disjoint_native_12_01
\`\`\`

Expected audit: \`TRAIN_DISJOINT_NATIVE_READY\`, \`N_scene_disjoint_episodes=12\`.
This says only that native data is complete, not that the model succeeded.

## 4 — Lock an outcome-blind 6-scene checkpoint cohort

The existing \`recovery_independent_cohort.py\` accepts TRAIN inputs and
chooses the first source-order route-group representative from distinct scenes.
Because every source Episode is in a predeclared new scene, this becomes the
first six new scenes. The original, already-existing
\`data/orientation_checkpoint_01\` path is an illustrative earlier
cross-case exclusion report required by the old cohort module. Check it
before running. It must contain a **completed**
\`crosscase_orientation_checkpoints.v1\` \`checkpoint_report.json\`.
Use the actual previously frozen original cross-case path if yours differs.

\`\`\`bash
test -f data/orientation_checkpoint_01/checkpoint_report.json

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_cohort.py \
  --source-run data/train_disjoint_native_12_01 \
  --exclude-checkpoint-run data/orientation_checkpoint_01 \
  --cases 6 \
  --run-dir data/train_disjoint_cohort_06_01
\`\`\`

Expected: \`INDEPENDENT_COHORT_PREPARE=PASS 6 scenes\`.
This cohort plan explicitly records native successes/NE for later auditing
but DOES NOT use them to decide which scenes or checkpoints to include.

## 5 — Native checkpoint replay at the first eligible idle System-2 boundary

The unchanged checkpoint logic uses ONLY online conditions:

- Main loop step >= 20
- Accumulated predecision translation >= 1.0 m
- At least 30 main-loop steps remain
- Idle ordinary System-2 boundary; camera level

Checkpoint selection never uses future baseline length or downstream
intervention outcomes. Native controls are fully replayed and checked against
the original per-action poses/hashes, RNG, history, weights, and final metrics.

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_checkpoint.py prepare \
  --cohort-dir data/train_disjoint_cohort_06_01 \
  --run-dir data/train_disjoint_checkpoints_06_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_independent_checkpoint.py run \
  --run-dir data/train_disjoint_checkpoints_06_01 \
  2>&1 | tee train_disjoint_checkpoints_06_01.log
\`\`\`

**STOP GATE:** inspect
\`data/train_disjoint_checkpoints_06_01/checkpoint_report.json\`.

- \`CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION\`: all 6 source-order
  scenes have a valid replayed checkpoint; proceed to stage 6
- \`INSUFFICIENT_ELIGIBLE_STRATA_STOP\`: at least one scene lacked a
  predeclared eligible boundary; **do not** silently drop it or choose an
  easier intervention winner. Keep the run as failed qualification and
  revise the selection protocol *before* running treatments

There is no automatic replacement/alternative scene selection. This
stage performs no forced movement, no history editing and no interventions.

## 6 — Create annotations for THOSE SAME checkpoint boundaries

Only after the checkpoint report says READY:

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_checkpoint_evidence.py prepare \
  --checkpoint-run data/train_disjoint_checkpoints_06_01 \
  --out-dir data/train_checkpoint_evidence_v2_06_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_checkpoint_evidence.py validate \
  --annotation-dir data/train_checkpoint_evidence_v2_06_01
\`\`\`

Expected:
\`TRAIN_CHECKPOINT_EVIDENCE_PREPARE=PASS\` then
\`TRAIN_CHECKPOINT_EVIDENCE_AUDIT_PASS\`.

The source template \`annotations.jsonl\` contains each checkpoint's
instruction, original model-sized sampled historical RGB, current RGB,
verified image hashes, original episode-local predecision action codes,
and separate entries for landmark roles and task-progress labels.

It contains **no** simulator poses, GPS/depth, target coordinates, final SR/NE,
oracle distances, future actions or treatment outcomes. The accompanying
source checkpoint report DOES have privileged offline metadata for
reproducibility; never copy those fields into the online annotation
templates or the future selector.

The six templates initially say \`unreviewed\`; structural audit PASS
**does not imply human annotation is complete or reliable**.
This stage is pre-intervention only.

## Next gate: treatment comparison

Freeze and human-review pre-intervention evidence annotations **before**
seeing any new treatment outcomes. Have two independent reviewers label
the same subset and compare disagreements; preserve \`unknown\` for
uncertain task-stage or ambiguous landmarks.

The old two-context treatment runner is intentionally NOT pointed at these
new scenes: it is constrained to the two original development checkpoints,
and the original factorial runs a larger 5x2 angle grid. The subsequent
engineering task is a new fresh-scene six-arm runner matched to the
new \`train_disjoint_checkpoints_06_01\` source, using only necessary
controls and stricter cost-matched contrasts. **Do not run a 54+ continuation
grid automatically** while this checkpoint qualification is incomplete.

Six independent scenes and one checkpoint each are a small development
sample; this does not test the previously proposed 12-checkpoint/72-rollout
design. Reserve untouched TRAIN scenes for model development checks and a
strict scene-disjoint later evaluation cohort.
