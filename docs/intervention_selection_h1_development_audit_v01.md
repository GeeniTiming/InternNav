# H1: Existing paired-rollout development audit (v0.1)

## Fixed research question

Given the same frozen InternNav / InternVLA-N1-DualVLN predecision
observation, instruction, native history and action prefix, do different
recovery operations have **state-dependent** relative success and harm
outcomes? If such heterogeneity exists, can it eventually be predicted from
**pre-intervention online-available** evidence?

This step evaluates **H1 existence/heterogeneity only**. It does NOT train,
estimate online predictability (H2), or validate a risk-constrained selector
(H3). Do not redefine the research question based on individual Episode
Rescue/Harm examples.

## Reuse what exists; do not rerun GPU

Existing frozen dataset: the original 12 previously uninspected R2R TRAIN
scenes, 10 checkpoint-eligible independently selected scenes, and TWO
correlated predecision contexts (natural and predeclared perturbed) per scene.

20 x 4 = 80 actually executed outcomes, with operations:
- KEEP: unchanged original continuation;
- HISTORY_RESET: filters pre-gate history selection (not model-memory reset);
- SEEK_LEFT45 / SEEK_RIGHT45: six actual physical yaw actions, returned pose,
  supplementary outward RGB AND prompt change.

20 x 3 = 60 paired action-vs-KEEP effects. The script reads the ORIGINAL
`intervention_outcomes.jsonl` through the original v0.1 audited loader, which
verifies source manifest and outcome identities. Optionally pass
`--label-dir` to rerun the original immutable offline label audit.

The recently outcome-selected three scenes Episode 70 / 106 / 79 are **NOT**
part of this original ten-scene H1 analysis and must NEVER be reported as an
independent held-out success estimate. If any scene appears in multiple
analysis datasets, scene-level independence must be rechecked before
combining them.

## H1 descriptive outputs and their meaning

Primary quantity is actual binary SR per operation; secondary quantities
include SPL, final NE (diagnostic), harms/rescues and physical-yaw cost.

- Each operation's Rescue / Harm / Both-Success / Both-Failure counts against
  the exact corresponding KEEP in the SAME context;
- Natural and perturbed contexts reported separately and jointly;
- Scene-mean success effects, and number of scenes where the effect is
  positive or negative. Two contexts of the same scene count as ONE scene;
- Best fixed operation **selected in-sample**, with deterministic KEEP-first
  tie priority. It is *not* a held-out fixed baseline;
- Per-context Oracle SR using true future outcomes, descriptively upper
  bounding any online selector on this measured sample;
- Difference Oracle minus best in-sample fixed, which is an optimistic
  sample-level headroom proxy, not an estimated deployable gain;
- Optional scene-cluster bootstrap percentiles with fixed random seed and
  in-resample best fixed selection. These are EXPLORATORY intervals, not
  confirmatory confidence intervals or a p-value.

Do not arbitrarily scalarize SR, SPL and yaw cost into an unvalidated Q
value, or assign a uniquely best action in all-failure contexts.

## Single-directory Git workflow

Work ONLY in `/home/geenitiming/InternNav`; do NOT make worktrees or
change frozen scripts. Fetch only these NEW analysis files on the current
branch (after checking there are no uncommitted versions of those paths):

```bash
cd /home/geenitiming/InternNav
git fetch origin
git status --short
git restore --source=origin/exp/intervention-h1-development-audit-v01 -- \
  scripts/eval/intervention_selection_h1_development_audit_v01.py \
  tests/unit_test/test_intervention_selection_h1_development_audit_v01.py \
  docs/intervention_selection_h1_development_audit_v01.md
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_h1_development_audit_v01.py' -v
python -m py_compile scripts/eval/intervention_selection_h1_development_audit_v01.py
```

Locate the exact already audited original paired-run folder:

```bash
find /mnt/data8t/geenitiming/internnav_data \
  -name paired_report.json -type f -print
```

Do not use the three-scene `focused_mechanism_report.json` folder. Set
`PAIRED` to the folder containing `paired_report.json`,
`paired_protocol.json`, and `intervention_outcomes.jsonl`.

```bash
PAIRED=/replace/with/original/audited/paired-run
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_h1_development_audit_v01.py \
  --paired-run "$PAIRED" --bootstrap-reps 2000 \
  --output "$HOME/vln_h1_train_dev_audit_v01.json"
```

The output is create-only and contains no image data or simulator state
privileged inputs. Optionally set `--label-dir` to the already audited
source label package; the original source auditing is mandatory regardless.

Check `N_scene_groups=10`, `N_contexts=20`,
`N_actual_rollouts=80`, `N_actual_effects_vs_keep=60`.
An unmet or mismatched schema is an error: do not silently drop a case.

### Interpret before deciding next GPU stage

Proceed to a separate pre-registered confirmatory cohort ONLY if development
data demonstrate meaningful nontrivial operation heterogeneity and plausible
headroom beyond fixed rules. The outcome selected posthoc three-scene
mechanism arm never enters that confirmatory denominator.

Otherwise report a negative H1 engineering result instead of changing the
topic, inventing new intervention arms, or presenting Oracle as an achieved
model.

## Future confirmation specification (not yet executed)

Freeze before seeing any future outcomes:
1. A list of previously uninspected, scene-disjoint R2R TRAIN episodes (and a
   genuinely held-out scene split later for any trained method).
2. Outcome-blind sample ordering and hash seed; one episode per scene,
   preallocated eligible first ordinary idle S2 boundary at 20/40/60,
   minimum 1m prefix translation, 30 remaining main-loop steps, level RGB.
   Declare unreachable checkpoints as ineligible; do not replace based on
   outcome.
3. Four arm operations and source files/weights SHA256, same model baseline
   RNG and pose, charged physical actions. Predeclare any optional placebo
   comparison as a separate mechanism study, NOT an outcome chosen add-on.
4. Successful pair audit, native System-2 event offsets, matched
   pre-intervention inputs and no privileged online features.
5. SR primary; HARM/RESCUE and SPL, action/compute costs secondary, scene-level
   clusters and fixed-action baselines. Never let same-scene data leak between
   train/validation/test.

The new cohort may be constructed only after H1 exploration decisions are
documented and the confirmatory protocol is frozen. No sample size is claimed
sufficient based on this current N=10 development pilot.
