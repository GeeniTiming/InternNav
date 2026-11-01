# Intervention Selection Dataset v0.1 — offline paired outcome labels

## Objective

Transform the already audited 10-scene/20-context/80-outcome R2R TRAIN development pilot into immutable offline benefit/risk label tables. No GPU runs or retraining, no new branches, and no modifications of old source manifests or RGB.

## Source contract

Required source folder: data/intervention_v01_paired_all10_01 with paired_report.json marked TEN_CASE_PAIRED_AUDITED, original 12 registered scenes, 10 eligible scenes, 20 contexts, 80 results; complete predecision contexts and action outcomes files; unchanged source hashes. The two ineligible scenes stay permanently ineligible. Natural and physically perturbed contexts in one scene are NOT independent scenes.

## Commands

CPU test and build using existing git branch:

~~~bash
cd ~/InternNav
git fetch origin
git switch exp/intervention-dataset-native-gates-v01
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_outcome_labels_v01.py' -v

SOURCE=data/intervention_v01_paired_all10_01
LABELS=data/intervention_v01_offline_labels_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_outcome_labels_v01.py build \
  --run-dir "$SOURCE" \
  --out-dir "$LABELS" \
  --spl-tolerance 0.02

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_outcome_labels_v01.py audit \
  --label-dir "$LABELS"
~~~

LABELS must not exist. This is create-only and does not modify the source.

Expected output ONLY after execution and audit:
- OFFLINE_LABELS_PREPARED_NOT_TRAINED
- OFFLINE_LABELS_AUDITED_NO_Q_TRAINING 20 contexts 60 effects 120 pairs
- Source data should reproduce 2 RESCUE, 6 HARM, 39 BOTH_SUCCESS and 13 BOTH_FAILURE. Do not change data to force those counts.

## Files

| Output | Expected records | Meaning |
|---|---:|---|
| context_label_summary.jsonl | 20 | Within-state successful arms, rescue/harm flags, no positive arm on all-failure; best_action deliberately null |
| action_effects_vs_keep.jsonl | 60 | Each new action vs KEEP: delta_success, delta_spl, delta_ne_improvement_m, physical yaw cost, RESCUE/HARM/BOTH_SUCCESS/BOTH_FAILURE |
| pairwise_preferences.jsonl | 120 | Six unordered action pairs per context, observed SR/SPL differences and conservative, potentially null preference |
| label_report.json | 1 | Original source and generated file SHA256, threshold, counts, offline-only caveats |

Outputs are OFFLINE targets only, never new online features. The generator rejects source target fields such as success, NE, SPL and GPS/depth/simulator position if present in the predecision feature row.

## Explicit conservative labels

1. One successful arm and one failed arm: choose the observed successful arm (SINGLE real rollout, NOT expected Q).
2. Both failed: UNRESOLVED_BOTH_FAILED and no preferred action. Lower final NE remains diagnostic, NOT rescue evidence.
3. Both successful, SPL difference <= 0.02: prefer lower extra physical yaw cost; equal cost is TIE.
4. Both successful, SPL difference > 0.02: higher SPL and no higher yaw cost wins; higher SPL *with higher yaw cost* is SPL_COST_TRADEOFF, no winner.

The original three physical perturbation turns are shared prefix cost for all four candidates in a perturbed context and are never incorrectly charged only to one action. Each SEEK adds 6 turns; RESET adds 0. This is a physical action count proxy, not model latency or token-cost reward. The SPL tolerance was defined after v0.1 rollouts, so it is an EXPLICIT provisional engineering heuristic, not a pre-registered ground-truth preference.

No continuous Q targets or forced best actions are emitted. Sensitivity to SPL tolerance can be tested by building into separate fresh output folders using --spl-tolerance 0.00 and --spl-tolerance 0.05, but never adapt it to a final held-out test.

## Interpretation and next milestone

The current 80 source outcomes contain 2 rescue pairs, 6 harmful pairs, 39 both-success pairs and 13 both-failure pairs. At the context level: 10 all-success, 5 KEEP success with some harmful intervention, 2 KEEP failure rescued, and 3 KEEP failure with no successful candidate. The 20 contexts represent only 10 distinct scene groups, so do not randomly split correlated natural and perturbed contexts across train/test.

Next: inspect the predecision task evidence, first System-2 decisions and STOP logic in recovered episodes 70 and 106 and harmed episodes 73/79; use placebo and scan-no-injection controls to separate informational benefit from physical yaw; grow a new blind TRAIN-scene cohort before attempting generalizable Q(s,a) training.
