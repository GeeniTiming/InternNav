# Focused three-scene mechanism ablation (v0.1)

**Research status:** posthoc development mechanism study, not a held-out test.
The three cases are i5noydFURQK/70 (right45), r1Q1Z4BcV1o/106 (right45), and PX4nDJXEHrG/79 (left45), all perturbed.

## Four arms per case

KEEP: reproduce old perturbed KEEP with no extra scan.
SCAN_NO_INJECTION: the same six out/return physical actions as the other scan arms, but no extra image or token.
PLACEBO: the same scan, with an additional duplicate current image and one extra image token.
REAL_SEEK: the same scan, with the actual outward view and one extra image token.

The complete study contains 12 new continuations. A within-scene scan direction is locked before this new experiment, but its scene was chosen AFTER observing previous outcomes. Do not claim unbiased recovery rate or train on these outcomes.

All original v0.1 paired/perturbed KEEP scripts and the evaluator remain unchanged. Preparation verifies original source manifests and complete source hashes. New System-2 call enter/exit anchors are logged for EVERY arm, including KEEP.

## Step 1: CPU-only tests and source checkout

    cd /home/geenitiming/InternNav
    git fetch origin
    git worktree add --detach ../InternNav-focused-mechanism origin/exp/intervention-v01-focused-mechanism-ablation
    cd ../InternNav-focused-mechanism
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_focused_mechanism_v01.py' -v
    python -m py_compile scripts/eval/intervention_selection_focused_mechanism_v01.py

Passing CPU tests does not imply that Habitat and GPU replay are validated.

## Step 2: prepare/preflight without GPU

Specify the ACTUAL, previously audited perturbed KEEP source. Example:

    PERT_KEEP=/mnt/data8t/geenitiming/internnav_data/intervention_v01_perturbed_keep_01
    OUT=/mnt/data8t/geenitiming/internnav_data/intervention_v01_focus_mechanism_01
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python scripts/eval/intervention_selection_focused_mechanism_v01.py prepare --perturbed-keep-run "$PERT_KEEP" --run-dir "$OUT"
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python scripts/eval/intervention_selection_focused_mechanism_v01.py preflight --run-dir "$OUT"

The output directory is create-only. Missing focus checkpoints, source hash drift, or changed protocol identities cause a hard failure.

## Step 3: GPU runs only after CPU passes

First verify that a fresh KEEP really reproduces the historical action/pose trace:

    python scripts/eval/intervention_selection_focused_mechanism_v01.py run --run-dir "$OUT" --arm perturbed_keep

Retain a failed arm and its logs; do not delete and retry silently.
After verifying KEEP, run the three intervention arms:

    for arm in perturbed_scan_no_injection perturbed_placebo perturbed_real_seek; do
      python scripts/eval/intervention_selection_focused_mechanism_v01.py run --run-dir "$OUT" --arm "$arm" || exit 1
    done
    python scripts/eval/intervention_selection_focused_mechanism_v01.py report --run-dir "$OUT"

## Audit gates

The final create-only report verifies historical KEEP success/NE and complete actions/poses against the newly logged KEEP; exact per-arm prefix System-2 token fingerprints and observed action offsets; identical case-specific scanned yaw sequences; return pose; pre-arm state/observation/RNG/history; and PLACEBO versus REAL_SEEK actual supplementary input provenance.

Even if the final audit passes, its finding is DESCRIPTIVE. These three outcome-selected scenes cannot estimate held-out recovery efficacy, task-relevance mediation, or learned intervention policy generalization.
