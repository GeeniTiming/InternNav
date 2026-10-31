# v0.1 paired intervention selection — 2-case pilot and 10-case cohort

The original pre-registered 12 TRAIN-scene episodes have 10 audited natural gates and 2 ineligible gates. Ten perturbed KEEP outcomes have been audited (7 successful, 3 failed). This stage preserves those source runs and runs three NEW actions per context: SEEK_LEFT45, SEEK_RIGHT45 and HISTORY_RESET. Natural and physically perturbed are distinct states; comparisons are made only *within* the same context.

Each SEEK adds six real yaw actions and a genuine outward RGB image at the next System-2 decision; each HISTORY_RESET only filters sampled history indices, not the neural weights/cache. A perturbed arm always replays the same three predeclared real yaw actions and checks the exact original perturb KEEP's pose/RGB/history/RNG before taking its intervention. One run is one create-only arm across predeclared episodes.

No new Git branch, no changed sampling, no hindsight episode selection, no fabricated labels. The two ineligible cases are retained permanently.

## CPU test and 2-case GPU preflight

Use the existing branch and the actual original folder names on the server.

~~~bash
cd ~/InternNav
git fetch origin
git switch exp/intervention-dataset-native-gates-v01
git pull --ff-only
conda activate internnav_habitat

PERTURBED=data/intervention_v01_perturbed_keep_01
PILOT=data/intervention_v01_paired_first2_01
FULL=data/intervention_v01_paired_all10_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_paired_v01.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_paired_v01.py prepare \
  --perturbed-keep-run "$PERTURBED" --max-cases 2 \
  --run-dir "$PILOT"

set -o pipefail
for arm in \
  natural_seek_left45 natural_seek_right45 natural_history_reset \
  perturbed_seek_left45 perturbed_seek_right45 perturbed_history_reset
do
  PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/intervention_selection_paired_v01.py run \
    --run-dir "$PILOT" --arm "$arm" \
    2>&1 | tee "v01_first2_$arm.log" || break
done

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_paired_v01.py report \
  --run-dir "$PILOT"
~~~

Expected only after real GPU runs and all checks pass: PILOT_PAIRED_AUDITED, 2 cases, 4 contexts, **12 new** intervention continuations and **16 total** four-arm outcomes including KEEP.

**If any arm fails:** stop, preserve its full log and run directory, do not rerun over the same root or skip it. Fix the code on the existing Git branch and re-prepare a new create-only root. Do not run all 60 until pilot passes.

## Full 10 eligible cases

~~~bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_paired_v01.py prepare \
  --perturbed-keep-run "$PERTURBED" --max-cases 10 \
  --run-dir "$FULL"

set -o pipefail
for arm in \
  natural_seek_left45 natural_seek_right45 natural_history_reset \
  perturbed_seek_left45 perturbed_seek_right45 perturbed_history_reset
do
  PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/intervention_selection_paired_v01.py run \
    --run-dir "$FULL" --arm "$arm" \
    2>&1 | tee "v01_all10_$arm.log" || break
done

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_paired_v01.py report \
  --run-dir "$FULL"
~~~

Expected only after complete and audited GPU runs: TEN_CASE_PAIRED_AUDITED, 10 original eligible cases, 20 contexts, **60 newly executed** intervention continuations, **80** total four-arm results with KEEP. Two ineligible cases remain recorded in the frozen source.

## Key audit requirements and output

Before every operation, the runner checks full original native prefix pose/actions, identical natural gate; in perturbed mode it also verifies original three real yaw turn trace and *actual post-turn* RGB/heading against frozen perturbed KEEP. It checks exact historical RGB digest, RNG, model parameter digest, genuine scan view injection vs reset-index filtering, charged steps and per-arm outcome completion. Do not use simulator-only pose, goal distance, depth, GPS, pathfinder or future results as online selector inputs.

After the complete report succeeds, export:
- **contexts_predecision.jsonl:** legal pre-arm instruction, RGB images, historical RGB and earlier actual action prefix.
- **intervention_outcomes.jsonl:** offline successful/failed continuations, NE, SPL, extra yaw costs and audit.
- **paired_report.json:** result counts and current status; not a trained Q score or future test estimate.

This is engineering/development data from 10 scene groups; paired contexts are not independent scenes. No held-out evaluation and no claim that fixed SEEK is an adaptive task-aware gaze method. HISTORY_RESET is not full neural memory erasure.
