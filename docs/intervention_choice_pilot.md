# Intervention-choice pilot v1 — six-arm comparability first

**Purpose:** verify matched physical state, input provenance, cost, history rules and frozen-parameter replay *before* analyzing intervention effectiveness. This is **not** a final recovery algorithm or a new held-out study. It reuses two already-inspected development contexts, NOT the six Full History images, which are descriptive annotation examples.

## Literature-derived design and limits

- SeekVLN already tests SEEK vs NAV with same-start counterfactual continuations. Real extra visual content alone is not a novel mechanism.
- MOON tests when history reset helps or harms; RESET itself is not a novelty.
- Dual-Anchoring already uses instruction-progress and landmark memory supervision.
- Our **unverified** question is whether online, task-relevant evidence features can eventually predict which of two information interventions has higher *conditional* benefit. Before attempting that, compare the interventions rigorously.

### Six conditions (fixed right45 scan; two originally inspected cases)

| Label | Where results come from | Actual extra yaw actions | History selection | Extra RGB |
|---|---|---:|---|---|
| KEEP | Original frozen checkpoint control | 0 | native | none |
| RESET | Original independent factorial history_reset | 0 | post-gate indices only | none |
| PLACEBO_KEEP | Existing evidence isolation right45 placebo | 6 | native | duplicate current RGB |
| SEEK_KEEP | Existing evidence isolation right45 evidence | 6 | native | actual rotated RGB |
| PLACEBO_RESET | **NEW** joint arm, same placebo scan | 6 | post-gate indices only | duplicate current RGB |
| SEEK_RESET | **NEW** joint arm, same evidence scan | 6 | post-gate indices only | actual rotated RGB |

The visual pair is matched **within each history condition**. Comparisons between scan/no-scan have real different motion and inference costs; do not interpret them as a pure image-information causal effect. Joint conditions use the same previously validated *history index filtering* as independent factorial RESET, **not** a model KV reset or perfect erasure of all earlier context. These distinctions are mandatory when interpreting results.

### Mechanism acceptance gate (fail closed)

Every new arm must reproduce the exact locked checkpoint prefix including pose and actions, same original instruction, frozen model parameters and RNG. In every scan arm:

1. Exactly three right + three left native 15-degree yaw actions.
2. Exactly six extra environmental actions charged against remaining budget.
3. Return to the same position and heading; scan action-and-pose trace matches prior evidence-isolation control.
4. Current RGB after returning matches the prior right45 scan across four scan conditions.
5. The **same** new image slot and prompt wording are used for evidence/placebo within a history stratum, and supplementary RGB hashes match declared sources.
6. For RESET arms, no native pre-checkpoint history index may enter any subsequent ordinary S2 historical-image selection. The first intervention S2 query has no pre-gate history image.
7. Both joint rollouts must finish; outcomes cannot be compared until all gate conditions pass.

This is a cost/provenance/mechanism audit, not proof that a view is task-relevant or a larger causal mediation result.

## CPU preflight (safe: no model loaded, no GPU)

The old evidence-isolation report **must** exist and say `MECHANISM_PILOT_VALIDATED`.
The previously discussed source directory was `data/recovery_evidence_isolation_02_02`; verify it on your server before using this illustrative path. The program automatically follows its manifest to the original checkpoint and factorial.

```bash
cd ~/InternNav
git fetch origin
git switch exp/intervention-choice-pilot-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_intervention_choice_pilot.py' -v

# Require an existing complete, verified mechanism source:
test -f data/recovery_evidence_isolation_02_02/mechanism_report.json

# Create only; DO NOT overwrite a prior run:
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_intervention_choice_pilot.py prepare \
  --mechanism-run data/recovery_evidence_isolation_02_02 \
  --run-dir data/intervention_choice_02_01
```

Expected CPU pass:
`INTERVENTION_CHOICE_PREPARE=PASS 2 contexts; new GPU arms=2; existing controls=4`.
If this fails, **stop** and inspect the error instead of running GPU or changing the source selection. The prepare manifest locks hashes of inputs, original comparator arms, configs and old experiment scripts.

## GPU only after CPU acceptance

```bash
set -o pipefail

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_intervention_choice_pilot.py run \
  --run-dir data/intervention_choice_02_01 \
  --arm lookback_right45_placebo \
  2>&1 | tee intervention_choice_placebo_reset_02_01.log

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_intervention_choice_pilot.py run \
  --run-dir data/intervention_choice_02_01 \
  --arm lookback_right45_evidence \
  2>&1 | tee intervention_choice_seek_reset_02_01.log
```

Only two new arms; each arm replays two original scenes. GPU work is sequential and no successful old arm is re-run.

Then execute the **strict comparability report**:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_intervention_choice_pilot.py report \
  --run-dir data/intervention_choice_02_01
```

The report saves:
`data/intervention_choice_02_01/six_arm_comparability_report.json`.

Successful gate status:
`TWO_SCENE_MECHANISM_COMPARABLE`.

## How to interpret outcomes, once comparability is established

The report lists six paired success indicators and final NE for each of two contexts, as well as **descriptive** contrasts:

`SEEK_KEEP − PLACEBO_KEEP`: effect of true additional RGB vs duplicate under KEEP history.

`SEEK_RESET − PLACEBO_RESET`: equivalent extra-RGB contrast under RESET.

`RESET − KEEP`: effect of RESET when no physical scan occurs.

`PLACEBO_RESET − PLACEBO_KEEP`: history effect within equal-cost placebo scans.

`SEEK_RESET − SEEK_KEEP`: history effect within equal-cost real-image scans.

`(SEEK_RESET − PLACEBO_RESET) − (SEEK_KEEP − PLACEBO_KEEP)`: descriptive effect interaction (not established causal mediation or generalization).

**N=2 is too small** to infer reliable average effects or label-dependent benefit. This study is not a 12-state or 72-rollout test. The fixed 45-degree view also cannot verify task relevance. Do **not** select the best arm retrospectively as an online agent.

### What happens next

If the two joint arms pass, plan a separate **train-split, scene-disjoint checkpoint acquisition** using the existing native decision recorder with predeclared checkpoint criteria. Reserve other train scenes (or a separate evaluation split) for evaluation and never train a predictor on the two previously inspected val_unseen/development scenes. Only after generating multiple independently grouped conditions should we attach the v2 **per-landmark** evidence labels and test their predictive relationship to paired intervention benefits.

If any replay/pose/history/image audit fails, fix the mechanism and re-run in a *new create-only directory*. Keep failed results as records and do not manufacture missing arms.
