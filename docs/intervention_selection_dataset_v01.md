# Intervention Selection Dataset v0.1 — locked design and CPU acquisition plan

## Research question and limits

Train a risk-aware choice among KEEP, SEEK_LEFT45, SEEK_RIGHT45 and HISTORY_RESET from **only pre-intervention** instruction, RGB/history and actual past action features. Neither a LOST label nor an offline oracle action can be used as a substitute for **real paired potential outcomes**.

This branch implements CPU-only, create-only **pre-registration**. It does **not** implement or execute any new Habitat checkpoint capture, perturbation, paired rollouts, post-hoc reward labels, task-evidence annotations, or Q training. All 96 evaluations begin with `outcome: null`. Do not present an audited plan as a completed dataset.

Episode13 and previous six-scene experiments are development evidence, not independent evaluation. Their source scenes must be excluded before new TRAIN scenes are chosen.

## Fixed v0.1 design

| Item | Rule |
|---|---|
| Episode selection | First instruction-bearing Episode from each of 12 **new** distinct TRAIN scenes; existing source selector, no SR/NE/future filtering |
| Source cases | 12 new independent scene/episode identities |
| Natural context | Unmodified native first-eligible ordinary System-2 decision at the predeclared tier |
| Perturbed context | Same native gate followed by **three actual 15° Habitat yaw turns**, +45° or -45° as locked, NO model S2 in between; re-capture current RGB |
| Timing tiers | minimum Step 20, 40, 60; exactly four cases assigned to each tier via deterministic seed/episode-key hash |
| Perturb direction | Two +45° and two -45° per timing tier, predeclared without looking at outcomes |
| Eligibility | first idle ordinary System-2 gate at/after step threshold, >=1m measured prefix translation, >=30 remaining native steps, level RGB sensor; no future length |
| No eligible checkpoint | `INELIGIBLE_RETAINED_NO_REPLACEMENT`: do not silently swap scenes or lower thresholds |
| Candidate actions | KEEP, SEEK_LEFT45, SEEK_RIGHT45, HISTORY_RESET |
| SEEK | ±45° physically scan out and return (6 actual yaw actions), obtain outward RGB and add at exactly one S2 query with matched placeholder semantics |
| RESET | Filter S2 sampled history indices post-gate, not persistent weights/KV/hidden memory removal |
| Evaluation | 24 states x 4 operations = 96 **planned** potential outcomes; 12 natural KEEP outcomes may be reused after verified native capture and gate replay |
| Additional execution | 84 new full native-prefix-replayed continuations after complete source controls: 12 perturbed KEEP + 72 other arms |
| Feature safety | no goal distances/pathfinder/depth/GPS/future actions/intervention outcome as online selector inputs |
| Data partition | TRAIN engineering/development pilot; held-out scene splits must be chosen separately before future evaluation |

Natural and perturbed states are correlated: independent scene-level statistical sample count is 12, not 24. The perturbation cost of three real turns is common to all four operations of a *perturbed* context, whereas each SEEK action adds six yaw actions of its own. Report both the prefix disturbance cost and per-candidate intervention cost.

### Important state consistency requirement

The **perturbed** context is a new, actual simulator state. After yaw perturbation, all four treatments must start at the same pose/heading, sensor RGB, history representation, RNG and frozen model weights. In particular, replace the current RGB using the actual post-perturbation observation; do not treat the old pre-turn current RGB as current. The original history stays as it was at the natural gate, without inserting a fictitious additional S2 history step. The no-perturb natural state and perturbed state are **not** directly matched as the same state: compare candidate actions *within* each context.

## Commands — CPU-only preparation

Use the original source-run paths on the **server**, not a guessed absolute path. The example below requires a new directory for each run. If you already selected a new, fully excluded 12-scene native source, set NEW_NATIVE to that directory and skip the first native-source preparation.

```bash
cd ~/InternNav
git fetch origin
git switch exp/intervention-selection-dataset-v01
git pull --ff-only
conda activate internnav_habitat

# Prior inspected TRAIN native source; change to its actual server location.
OLD_NATIVE=data/train_disjoint_native_12_01
NEW_NATIVE=data/intervention_v01_native_12_01
PLAN=data/intervention_selection_v01_plan_01

# CPU: choose 12 entirely new TRAIN scenes, excluding all old 12 scenes.
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_train_disjoint_acquire.py prepare \
  --episodes 12 \
  --exclude-source-run "$OLD_NATIVE" \
  --run-dir "$NEW_NATIVE"

# CPU: lock stage and yaw allocations before viewing native outcomes.
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_dataset_v01.py prepare \
  --native-source "$NEW_NATIVE" \
  --exclude-source-run "$OLD_NATIVE" \
  --seed 20261031 \
  --run-dir "$PLAN"

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_dataset_v01.py audit \
  --run-dir "$PLAN"
```

Expected CPU result (only if all source hashes and scene exclusions pass):

- `INTERVENTION_DATASET_V01_PREPARE=PASS 12 episodes 24 contexts 96 planned potential outcomes`
- `CPU_PLAN_AUDITED_NO_ROLLOUTS`
- step counts `{20:4,40:4,60:4}` and each tier has two positive and two negative yaw plans
- `outcomes_available: false`, `training_examples_available: false`

CPU unit tests:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_dataset_v01.py' -v
```

Files generated by the CPU-only preparation:

- `manifest.json`: frozen source/config/script and plan SHA-256 values.
- `preregistered_plan.json`: immutable episode IDs, stage assignment, perturb direction and 96 action placeholders, without results.

## Next implementation boundary (NOT in this branch)

1. **Native capture:** run the existing native TRAIN acquisition on NEW_NATIVE only when ready for GPU work; it records source instructions, actual S2 calls and real actions. It is not an intervention-selection dataset yet.
2. **Generalized gates:** reuse `recovery_independent_checkpoint`'s full action/pose and actual RGB snapshot validations, but make minimum step threshold per-case (20/40/60) rather than its existing hard-coded 20. Do not edit the old frozen script or re-use the old reports as new checkpoints.
3. **Locked natural + perturbed paired gates:** for every case, verify natural gate and post-perturbation RGB/pose; each arm must reproduce exact source prefix and perturbed gate by native replay, with RNG/weight/historical RGB fingerprints.
4. **Paired arms:** reuse existing physical scan/history RESET probes, generalize the cohort size and source constraints, introduce left45 scan without changing native evaluator. Collect actual episode outcomes; the four arms must start at precisely the same context.
5. **Separate data tables:** `contexts_predecision.jsonl` (only legal online features), `annotations.jsonl` (reviewed evidence with provenance), `outcomes.jsonl` (offline intervention results + audit hashes), `preferences.jsonl` (derived only after complete paired groups).
6. **Utilities and evaluation:** retain SR, NE, SPL, actual action costs; predeclare reward weights and ties on TRAIN development only. No leakage from future reward into online inputs or human evidence labels; reserve new scenes for a separate held-out test.

### Acceptance rules

Prepare and audit alone are NOT enough to call the dataset ready. A completed context must have: (a) legitimate predecision RGB+instruction+history/action hashes, (b) four actual real rollouts and matching pre-arm pose/RNG/weights, (c) recorded SR/NE/SPL and costs, (d) no outcome-dependent replacement, and (e) a separate independent/unknown-allowed evidence review. Keep `INELIGIBLE`, `INCOMPLETE`, and `NOT_REVIEWED` as explicit categories instead of discarding inconvenient cases.

**Previous six-scene results and Episode13 do not enter this plan or count toward 12 new source scenes.**