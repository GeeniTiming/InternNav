# Six-scene TRAIN paired intervention: reproducible procedure

Purpose: study unwanted interventions at six previously locked, scene-disjoint R2R TRAIN checkpoints. All six original KEEP continuations succeeded. Hence this is a **harm study, not a failure rescue study**. It uses preliminary, unconfirmed `needs_review` annotations, not gold supervision.

Exactly six arms per checkpoint: KEEP (native, reused), RESET (new, zero extra yaw actions), PLACEBO_KEEP (new, 6 yaw actions + duplicate RGB), SEEK_KEEP (new, 6 yaw actions + actual outward RGB), PLACEBO_RESET (new, same yaw/placebo + reset), SEEK_RESET (new, same yaw/evidence + reset). This yields 6 native controls + 30 new continuation outcomes. The fixed right45 scan is not task-dependent; RESET only changes S2 history image index selection, not model memory or KV cache.

## 1. Branch and CPU tests

```bash
cd ~/InternNav
git fetch origin
git switch exp/train-six-arm-intervention-v1
git pull --ff-only
conda activate internnav_habitat
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test -p 'test_recovery_train_six_arm.py' -v
```

Tests are CPU-only synthetic checks and do not imply GPU success.

## 2. Freeze prior annotations and prepare (CPU only)

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_checkpoint_evidence.py validate \
  --annotation-dir data/train_checkpoint_evidence_v2_review_06_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_six_arm.py prepare \
  --checkpoint-run data/train_disjoint_checkpoints_06_01 \
  --annotation-dir data/train_checkpoint_evidence_v2_review_06_01 \
  --run-dir data/train_six_arm_06_01
```

Expected: `TRAIN_SIX_ARM_PREPARE=PASS` with 6 contexts and 5 new arms; `PREINTERVENTION_EVIDENCE=LOCKED_PRELIMINARY_NOT_GOLD`. The script pins the annotation file SHA before any new intervention. It fails if source hashes change, a checkpoint differs, or a review is still `unreviewed`. Structural validation never establishes that a preliminary human-style label is factually correct.

## 3. Run GPU arms in sequence

Start with the zero-scan RESET arm:

```bash
set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_train_six_arm.py run \
  --run-dir data/train_six_arm_06_01 --arm history_reset \
  2>&1 | tee train_six_arm_history_reset.log
```

Once successful, run the four real yaw scans individually and sequentially:

```bash
for arm in \
  lookback_right45_placebo_keep \
  lookback_right45_evidence_keep \
  lookback_right45_placebo_reset \
  lookback_right45_evidence_reset
do
  PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/recovery_train_six_arm.py run \
    --run-dir data/train_six_arm_06_01 --arm "$arm" \
    2>&1 | tee "train_six_arm_$arm.log" || exit 1
done
```

All arms are create-only; do not rerun a failed arm over partial artifacts. Preserve failures and prepare a new directory for corrected experiments.

## 4. Strict comparability report (CPU only)

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_six_arm.py report \
  --run-dir data/train_six_arm_06_01 \
  | tee train_six_arm_06_01_report.log
```

Expected on complete audit: `SIX_TRAIN_SCENE_COMPARABLE`; output `data/train_six_arm_06_01/six_arm_report.json`. It checks source manifests, full native prefix actions/poses, per-arm runtime, charged six actions and matched scans, current RGB hash, actual outward RGB vs duplicated current RGB, exact image-slot/token insertion, no old historical image indices in RESET, and unchanged frozen preliminary annotations.

## Interpretation constraints

- Keep success was 6/6; test harmful interventions, not success rescue.
- SEEK is a fixed 45-degree scan, not an online evidence-guided viewing policy.
- RESET changes sampled history, not all prior temporal information.
- Four scan arms have 6 additional Habitat actions, unlike KEEP and RESET. Always compare extra RGB and placebo within the same history and action budget.
- These are 6 development contexts. Report descriptive, paired per-scene effects only, and do not train a selection policy on unconfirmed evidence labels.
- Duplicate current RGB among historical slots is a known property of the original checkpoint inputs; do not count it as independent corroboration.

Next: analyze first System-2 output divergence and match with the frozen evidence categories; only after that, design a larger scene-held-out cohort including native failures.