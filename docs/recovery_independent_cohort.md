# Next experiment: NEW route groups + actual System-2 history RGB + 5x2 continuations

Branch: \`exp/recovery-independent-cohort-v1\`.

**Not a continuation of the original six cases.** New acquisition uses one selected
route group per unseen scene, excluding **all scenes** that appeared in
\`orientation_checkpoint_01\`. It does not use the native outcome or any
treatment outcomes to select cases. The checkpoint is the **first** native
idle System-2 boundary with:

- \`step_id >= 20\`
- observed translation >= 1 m
- >= 30 main loop iterations left under the original budget
- level RGB camera and no queued local actions or pixel goal

No completed-baseline length, goal distance, depth, GPS, pathfinder or
reference path is used to *decide the boundary*. Some newly selected routes
may have no eligible boundary. Report that as a limitation; **do not switch
episodes after seeing their outcomes or silently change the thresholds**.

## 1. Update and test without GPU

\`\`\`bash
cd /home/geenitiming/InternNav
git fetch origin
git switch exp/recovery-independent-cohort-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_independent_cohort.py' -v
\`\`\`

## 2. Fix for "Need 2 unseen-scene groups, found 1"

The legacy `cache_pilot.py prepare --episodes 40` uses the first **40 source
episodes**, which are often clustered by scene. If the old 40-control pilot
has already completed, **do not delete or rerun it**.

**Option A — immediate one-scene pipeline smoke test:** reuse that pilot,
choose a NEW create-only cohort directory and `--cases 1`:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_cohort.py \
  --source-run data/recovery_native_pilot_40_01 \
  --exclude-checkpoint-run data/orientation_checkpoint_01 \
  --cases 1 \
  --run-dir data/recovery_new_cohort_01_01
```

This is useful only as a pipeline test, not as evidence of generalization.
Use matching `01_01` paths at downstream stages rather than the 2-scene
paths below.

**Option B — preferable next 2-scene pilot:** run the new *scene-diverse*
CPU preparation first. It selects the **first episode of each distinct
uninspected scene from the full source annotations** without conditioning on
success or future trajectories. It leaves the original `cache_pilot.py`
untouched to preserve previously locked hashes:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_scene_diverse_pilot.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_scene_diverse_pilot.py \
  --episodes 2 \
  --exclude-checkpoint-run data/orientation_checkpoint_01 \
  --run-dir data/recovery_native_diverse_02_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/cache_pilot.py run \
  --run-dir data/recovery_native_diverse_02_01 \
  --arm control \
  2>&1 | tee recovery_native_diverse_02_01.log

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_cohort.py \
  --source-run data/recovery_native_diverse_02_01 \
  --exclude-checkpoint-run data/orientation_checkpoint_01 \
  --cases 2 \
  --run-dir data/recovery_new_cohort_02_02
```

Then use this `02_02` cohort for checkpoint capture and a new matching
directory for factorial arms. All outputs are create-only.

## 2b. Alternative: create an expanded unmodified native CONTROL pilot

If you **already have** a completed, non-profiled native pilot with
enough additional scenes, use it at step 3; it must have \`manifest.json\`,
\`episodes.json.gz\`, \`control.py\`, \`control/pilot_summary.json\`,
\`control/progress.json\` and \`control/actions.jsonl\`.

Otherwise use a **fresh** directory (these commands do not overwrite
the old pilot or the 6 historical cases):

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/cache_pilot.py prepare \
  --episodes 40 \
  --run-dir data/recovery_native_pilot_40_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/cache_pilot.py run \
  --run-dir data/recovery_native_pilot_40_01 \
  --arm control \
  2>&1 | tee recovery_native_pilot_40_01.log
\`\`\`

A \`cache\` arm is **not** required; do not call
\`cache_pilot.py report\`, since it expects both \`control\` and \`cache\`.
Only native \`control\` is a valid source for the new cohort.

## 3. Lock a *new*, non-overlapping scene cohort (CPU)

Start with **two independent scenes** to check the entire pipeline before
collecting 30–50 route groups. More scenes require a sufficiently diverse
source pilot.

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_cohort.py \
  --source-run data/recovery_native_pilot_40_01 \
  --exclude-checkpoint-run data/orientation_checkpoint_01 \
  --cases 2 \
  --run-dir data/recovery_new_cohort_02_01
\`\`\`

The script refuses to select already-inspected scenes and fails explicitly
if there are fewer than two new scenes. It never fills that gap with the
old six scenes. If the source 40-episode pilot is too narrow, collect a
new, larger native pilot in a *new* directory and prepare a fresh cohort.

## 4. Capture online-only checkpoint + exact sampled RGB history

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_checkpoint.py prepare \
  --cohort-dir data/recovery_new_cohort_02_01 \
  --run-dir data/recovery_new_checkpoint_02_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_independent_checkpoint.py run \
  --run-dir data/recovery_new_checkpoint_02_01 \
  2>&1 | tee recovery_new_checkpoint_02_01.log
\`\`\`

The capture replays the native control **without intervention**, verifying
the full action and pose trajectory and RNG/history state. At each
eligible boundary it saves:

- \`rgb.png\`: actual native checkpoint observation RGB
- \`current_model_rgb.png\`: exact resized current RGB for System 2
- \`historical_rgb/history_XXXX.png\`: sampled native pre-boundary
  historical images (same model-size PIL data, correct history indices)
- \`context_inputs.json\`: indexed frame inventory, pixel / PNG SHA-256
- \`checkpoint.json\`: strict native state, history and RNG match fingerprint

Sensor depth and GPS remain available to the unchanged *original* DualVLN
executor, but **not** as selector inputs or exported training features.

Only continue if report status is
\`CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION\`. A route with no
eligible idle boundary is explicitly reported as \`NOT_ELIGIBLE\`;
keep the original run unchanged.

## 5. Run the 5x2 paired intervention grid (nine NEW arms + reused RELEASE)

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_factorial.py prepare \
  --source-run data/recovery_new_checkpoint_02_01 \
  --run-dir data/recovery_new_factorial_02_01

set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_independent_factorial.py run-all \
  --run-dir data/recovery_new_factorial_02_01 \
  2>&1 | tee recovery_new_factorial_02_01.log
\`\`\`

The nine new arms are:
\`history_reset\`, \`left45_keep\`, \`right45_keep\`,
\`left45_reset\`, \`right45_reset\`, \`left90_keep\`,
\`right90_keep\`, \`left90_reset\`, \`right90_reset\`.
RELEASE is the validated native control, not a tenth GPU rerun.

Every turn uses 15-degree native body actions, with 0/3/6 external actions
charged to its native continuation. The simulator is **not** teleported,
there is no retrospective best-angle selection, and the model remains frozen.
Each run arm is create-only. **Never restart run-all after partial failure**;
save the log and use a fresh experiment folder once a root cause is fixed.

## 6. Export complete features and paired labels (CPU)

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_independent_export.py \
  --checkpoint-run data/recovery_new_checkpoint_02_01 \
  --factorial-run data/recovery_new_factorial_02_01 \
  --out-dir data/recovery_new_dataset_02_01
\`\`\`

Outputs: \`contexts.jsonl\`, \`paired_labels.jsonl\`,
\`candidate_rows.csv\`, \`dataset_report.json\`, and all actual sampled
RGB history frames plus action-only prefix histories.

The labels describe **ten correlated potential continuations** for the same
state; they are not ten independent examples. The small smoke-test cohort
remains \`acquisition_development_only\`, not held-out evaluation data.

### First stop point

Run CPU unit tests and steps **2–3**. Share the output from
\`recovery_independent_cohort.py\`. Then capture the two native checkpoints.
If eligible, continue to the paired GPU arms.

After the two-scene smoke test, fix the acquisition protocol and collect
more **scene-disjoint route groups**. Reserve disjoint groups for a future
selector evaluation, and do not call that test independent after tuning it.
