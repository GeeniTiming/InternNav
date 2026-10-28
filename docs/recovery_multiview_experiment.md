# Multiview physical–history experiment v1 (controlled expansion)

This work **does not claim RIGHT90 is an optimal physical recovery**. We
predeclare symmetric +/-45-degree turns to test whether the original
six-case 90-degree interaction persists under different physical view changes.
LEFT90 and RIGHT90 are **both** retained; no post-hoc right-turn-only filter.

The combined completed grid is:

| Physical action | Keep original System-2 history | Reset System-2 history |
| --- | --- | --- |
| No turn | already complete: checkpoint/control | already complete: candidates/history_reset |
| LEFT45 (3 native turns) | **new GPU** | **new GPU** |
| RIGHT45 (3 native turns) | **new GPU** | **new GPU** |
| LEFT90 (6 native turns) | already complete: orientation_turn/left90 | already complete: candidates/left90_reset |
| RIGHT90 (6 native turns) | already complete: orientation_turn/right90 | already complete: candidates/right90_reset |

Each arm replays the exact existing native checkpoint; actual simulator
actions, prefix poses, original policy history, policy RNG and checkpoint
metadata must match. No teleport, new training, offline oracle decision or
outcome-dependent action adjustment. Every extra real turn is charged to
the remaining native main-loop limit. Its reported cost is in **Habitat body
actions**, not abstract high-level decisions.

## Run CPU tests (no model/GPU)

Use the existing environment/working directory. Make no changes to prior runs.

\`\`\`bash
cd /home/geenitiming/InternNav
git fetch origin
git switch exp/physical-memory-multiview-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_multiview.py' -v
\`\`\`

## First: analyze completed LEFT90/RIGHT90 and history factorial (3x2)

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_multiview_compare.py \
  --checkpoint-run data/orientation_checkpoint_01 \
  --turn-run data/orientation_turn_01 \
  --candidate-run data/recovery_candidates_01 \
  --out-dir data/recovery_multiview_existing_3x2_01
\`\`\`

This reuses six existing arms and should export six per-checkpoint
factorial matrices (6 checkpoints x 6 arms, **not 36 independent routes**).
It tests *both* signs, and distinguishes positive gamma from actual
joint-only rescue.

## Then: four new +/-45-degree GPU arms

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_multiview_45.py prepare \
  --source-run data/orientation_checkpoint_01 \
  --run-dir data/recovery_multiview_45_01

set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_multiview_45.py run-all \
  --run-dir data/recovery_multiview_45_01 \
  2>&1 | tee recovery_multiview_45_run.log
\`\`\`

If an arm fails, the sequential command stops and **preserves all output**.
Do not overwrite an incomplete arm or silently replace a case. Report
the failure and make a new run directory after fixing its cause.

When all four arms finish:

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_multiview_compare.py \
  --checkpoint-run data/orientation_checkpoint_01 \
  --turn-run data/orientation_turn_01 \
  --candidate-run data/recovery_candidates_01 \
  --angle45-run data/recovery_multiview_45_01 \
  --out-dir data/recovery_multiview_full_5x2_01
\`\`\`

Result: \`angle_factorial_report.json\` and
\`per_checkpoint_angle_matrix.jsonl\`. Key quantities by angle:

\`\`\`text
Gamma(angle) =
 Y(angle,reset) - Y(angle,keep)
 -Y(no_turn,reset) + Y(no_turn,keep)
\`\`\`

**A positive gamma is not necessarily a rescue.** Also compare
\`joint_only_rescue_angles\`, SR, the count of baseline failures rescued,
baseline successes harmed, and body-action overhead per angle.

## Limits and next study

- These six cases were already used to formulate the hypothesis, so they
  remain **exploratory development data**, irrespective of how many new
  interventions run on them.
- Testing the +/-45 angle on these six cannot establish direction/angle
  generalization. A new independent route-group cohort, predeclared
  triggers and full RGB-history capture are still necessary.
- Historic RGB is not present in the old checkpoint export. This grid
  characterizes outcomes, **not** a ready multimodal selector dataset.
- A 5x2 angular factorial is **not** all forms of physical recovery:
  changing location, looking around with a pan-tilt camera, and returning
  to a previously visited site are distinct interventions and require
  cost-matched and privilege-controlled follow-up experiments.
- Do not select a direction/angle by its *observed* success when
  evaluating a learned choice. Oracle best-arm is an analysis upper bound.

### Send back

For the CPU stage, send \`data/recovery_multiview_existing_3x2_01/angle_factorial_report.json\`.
For the GPU stage, send the full 5x2 \`angle_factorial_report.json\` and
\`recovery_multiview_45_run.log\`. A run failure needs only its traceback
and output log; do not restart previously completed runs.
