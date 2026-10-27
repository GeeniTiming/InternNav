# Native checkpoint capture for the locked orientation cohort

This is the second stage after `orientation_cohort.py prepare`. It executes only
native controls, captures the first boundary satisfying the already declared
rule, and verifies complete native reproduction before exposing a checkpoint for
future intervention preparation. No left/right intervention is executed here.

## Locked cases

The server-prepared cohort contains these six route groups (episode IDs refer to
the InternNav annotation, not the old ETP-R1 experiments):

| Stratum | Scene | Episode | Body-action threshold |
| --- | --- | --- | --- |
| Failure | TbHJrupSAjP | 4 | 34 |
| Success | zsNo4HB9uLZ | 1 | 31 |
| Failure | x8F5xyUWy9e | 9 | 39 |
| Success | 2azQ1b91cZZ | 16 | 45 |
| Failure | 2azQ1b91cZZ | 10 | 69 |
| Success | QUCTc6BB5sX | 19 | 54 |

The saved cohort plan remains authoritative and read-only. Cases 13/14/15 from the
case15 route are excluded. Instruction siblings 4/5/6 and 7/8/9 are route groups,
not additional independent samples. A failure group with successful instruction
siblings cannot provide a success control.

## Commands

Run in the existing checkout/environment and foreground tmux session:

```bash
cd /home/geenitiming/InternNav
conda activate internnav_habitat
git pull --ff-only

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
    -p test_orientation_checkpoint.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/orientation_checkpoint.py prepare \
    --cohort-dir data/orientation_crosscase_01 \
    --run-dir data/orientation_checkpoint_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/orientation_checkpoint.py run \
    --run-dir data/orientation_checkpoint_01
```

`prepare` is CPU-only and create-only. It verifies the cohort hashes, reconstructs
the locked selection from the original native source, and creates a six-case
subset and native config. Original seeds, model settings and action budgets are
preserved; only the dataset subset and output/video destinations change.

`run` loads the model once for the selected cohort. It adds process-local logging
hooks around the native loop and step recorder, and restores them even if the run
fails. It does not force recorded actions: each native policy action is computed
normally and immediately checked against its source action/pose. Existing capture
outputs are never resumed or overwritten.

## Fixed capture rule

Count native non-STOP body actions (FORWARD/LEFT/RIGHT, excluding camera actions).
Capture the first idle System-2 boundary at or after each saved 60% threshold, with
at least 1 m accumulated prefix translation and six baseline body actions still
remaining. Require no pending action sequence, pixel goal or local plan, and a
level RGB sensor. Level is checked using the native camera-action balance and the
live sensor rotation; querying agent state does not render a new view or move it.

The timing uses completed baseline length, so it remains an offline diagnostic.
If no boundary qualifies, record `NOT_ELIGIBLE_KEEP_LOCKED`. Do not choose a later
fraction, replace the case, or add cases. Proximity to the goal is logged only as
an offline fact; it is not an additional checkpoint selection condition.

At capture, save RGB plus fingerprints of current observations, history, policy
RNG, parameters and the executed prefix. Check these again after logging. Existing
history images are retained and checked for later mutation. The observation is
returned unchanged and ingested once by the original native loop.

These files are replay metadata and image evidence. They are not serialized,
resumable simulator/model states. Future contrasts must perform a fresh native
replay to the same verified boundary.

## Output and interpretation

The final report is `data/orientation_checkpoint_01/checkpoint_report.json`. It
contains all six cases, full action/pose comparisons, native metrics and capture
metadata. Per-case RGB/JSON evidence is under `control/checkpoints/`; complete
native debug videos are under `video/control/`. `control/boundary_events.jsonl`
records all tested boundaries so the first-eligible rule can be checked.

A changed native action/pose, runtime, instruction, main-step count, or outcome is
`NOT_COMPARABLE`, not an intervention effect. No checkpoint is eligible for later
interpretation until the full native trace and per-case outcome match. A report
with `CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION` has at least one comparable
checkpoint in each stratum. If a stratum has none, the protocol stops without
replacement. Inspect the report before preparing any orientation arms.

This stage establishes comparable starting states only. It does not measure
recovery benefit or harm, isolate memory corruption, demonstrate language
completion, train a selector, or provide held-out efficacy. The original 24-case
train investigation and the case15 development results remain unchanged.