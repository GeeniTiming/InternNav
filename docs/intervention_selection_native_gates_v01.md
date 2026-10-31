# v0.1 Stage B: real natural and physically perturbed state capture

## What this branch does / does not do

Implements two actual InternNav/Habitat collection stages **after** the frozen CPU selection plan exists. No changes to the original native model, old Episode13 or six-scene reports, or the original CPU v0.1 preregistration.

1. `intervention_selection_native_gates_v01.py` replays 12 native TRAIN episodes *without changing behavior* and captures exactly one natural ordinary System-2 gate per case using its preallocated minimum step (20, 40 or 60). It verifies every original native action/pose, full-episode results, predecision RGB/history SHA, RNG and model parameter versions.
2. `intervention_selection_perturbed_keep_v01.py` uses only audited eligible natural gates. It replays the exact native action prefix, executes exactly three **real Habitat ±15° body yaw actions** to reach the plan's ±45° perturbation, captures the genuine post-yaw RGB at native model size and unchanged pre-gate sampled RGB history, charges three main-loop steps, then completes **KEEP**. This is *one action outcome per perturbed context*, not the full four-arm dataset. Natural-gate verification is an additional **12-episode native replay** beyond the source-native run; do not omit its compute cost when budgeting later full four-arm acquisition.

CHECKPOINT FILES ARE METADATA + REAL PNG ONLY, **NOT** resumable simulator/model snapshots: every evaluation starts from the beginning with locked native prefix replay. Ineligible cases are retained in reports and are **never** replaced based on SR, NE, or future length.

## Stage 0: required already-frozen CPU plan

You must have completed `docs/intervention_selection_dataset_v01.md` CPU preregistration. Use the actual source and plan paths on your server; names below are examples. If the source is already run and audited, do not rerun it.

```bash
cd ~/InternNav
git fetch origin
git switch exp/intervention-dataset-native-gates-v01
git pull --ff-only
conda activate internnav_habitat

NEW_NATIVE=data/intervention_v01_native_12_01
PLAN=data/intervention_selection_v01_plan_01
NATURAL=data/intervention_v01_natural_gates_01
PERTURBED=data/intervention_v01_perturbed_keep_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_native_gates_v01.py' -v

# If the NEW_NATIVE source is only CPU-prepared, run its original frozen
# unmodified dual-system policy once to generate real source trajectories.
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_task_evidence_acquire.py run \
  --run-dir "$NEW_NATIVE" \
  2>&1 | tee v01_native_source_12.log

# This audit is CPU-only and does not restart Habitat.
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_disjoint_acquire.py audit \
  --run-dir "$NEW_NATIVE"
```

Source `capture/decision_report.json` must have `TRAIN_NATIVE_DECISION_DATASET_READY` and 12 completed episodes. If it already exists and passes audit, **skip source GPU run**, not resume or overwrite it.

## Native completion semantics (source readiness)

The original DualVLN evaluator uses `while (not done) and (step_id <= max_steps_per_episode)`. Therefore a **completed**, audited source episode can end with `done=False` on its final actual Habitat action **if and only if** its saved progress has `steps == max_steps_per_episode+1`. The v0.1 `native_ready` check now verifies that exact budget-exhaustion case or ordinary `done=True`, and still requires full 12-episode completion, consistent actual action counts, and future full replay action/pose equivalence. Partial runs with `done=False` before the main-loop limit remain rejected.

The GitHub fix for this readiness check is in the **existing branch**; no change to the old native source, locked preregistration, or original outcomes is needed. If a natural-gate `prepare` failed at this readiness check, it failed **before** creating the natural-gate output directory and can be retried after pulling. If a directory already exists, do not overwrite it.

## Stage B1: immutable natural gate capture

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_native_gates_v01.py prepare \
  --plan-dir "$PLAN" --run-dir "$NATURAL"

set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/intervention_selection_native_gates_v01.py run \
  --run-dir "$NATURAL" \
  2>&1 | tee v01_natural_gates.log
```

`$NATURAL/natural_checkpoint_report.json` shows `N_captured_gates` and `N_ineligible_retained`. Status is `NATURAL_GATES_AUDITED` if all eligible or `NATURAL_GATES_AUDITED_WITH_INELIGIBLE` if some did not reach their predeclared gate. Both are truthful and retain all original cases; don't silently relax Step 60.

The natural RGB artifacts are under `$NATURAL/control/checkpoints/<scene>_<episode>/`: `current_model_rgb.png`, `historical_rgb/*.png`, `context_inputs.json`, **`predecision_input.json`** (instruction, legal past action prefix and model RGB references only), `checkpoint.json`, raw `rgb.png`.

## Stage B2: perturb each audited gate and execute KEEP

Only after B1 succeeds:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_perturbed_keep_v01.py prepare \
  --natural-gate-run "$NATURAL" --run-dir "$PERTURBED"

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/intervention_selection_perturbed_keep_v01.py run \
  --run-dir "$PERTURBED" \
  2>&1 | tee v01_perturbed_keep.log
```

`$PERTURBED/perturbed_keep_report.json` must report `PERTURBED_KEEP_AUDITED` only after pose and hash checks pass. Per-episode artifacts reside at `$PERTURBED/perturbed_keep/checkpoints/<scene>_<episode>/`: `predecision_input.json` (original instruction, actual past prefix **plus three perturbation yaw actions**, sampled pre-gate history and new actual model RGB), `post_perturb_rgb.png`, `post_perturb_current_model_rgb.png`, `boundary.json`, `handoff.json`. Pose, yaw and offline goal metric stay in the offline boundary/handoff files; never pass them to a selector.

## Critical paired-comparison and data leakage invariants

- Native and perturbed are **different contexts**: do not compare their KEEP as if same pre-arm state.
- The original three-turn perturbation cost is common to all four arms at *perturbed* context; SEEK will later add its own six-turn observation cost.
- At each perturbed context KEEP starts immediately after three actual yaw turns, using actual post-yaw RGB and the original history; no additional fake RGB history steps or intermediate S2 decision.
- All new SEEK/RESET runs must replay exact recorded prefix and perturbation and verify SAME postperturb RGB/history/RNG/weights before choosing actions. These are not yet implemented in this branch.
- Any unobserved outcome remains NULL. `N_paired_four_arm_contexts_complete` remains zero after B2.
- No GPS, depth, pathfinder, goal distance, future navigation, human outcome labels or other simulator-only features may enter online selector inputs.
- Read-only original gate scenes and existing source reports stay frozen. A partial GPU run is create-only: save log and directory, fix on a NEW run root rather than overwriting.

## Acceptance gate before four-arm collection

The minimum acceptable result of this branch is a native source report, a natural-gate report with exact per-case stage eligibility and complete matched native controls, and a perturbed-KEEP report with exact three-turn action/pose and actual postturn RGB hashes. This **does not** yet yield a full intervention selection dataset, oracle preference labels, or an online policy.

If some Episode cannot reach its predeclared gate, keep `INELIGIBLE` in the registered cohort and report the denominator. Plan a larger independent cohort *after* this pilot, not retroactively replace current episodes.

## Next coding step after actual B1/B2 logs

Implement paired KEEP/LEFT45/RIGHT45/HISTORY_RESET from each verified natural and perturbed gate with exact per-context fingerprints, and export **separate** online-only contexts, offline outcomes, and evidence annotation tables. Do not train Q before labels have passed paired audit.