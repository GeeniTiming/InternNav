# InternNav v0.1: action divergence and controlled mechanism experiments

Status: CPU trace export implemented; proposed matched GPU controls NOT YET RUN.
Work on the existing exp/intervention-dataset-native-gates-v01 branch. Preserve
all previously frozen v0.1 80-arm outcomes, 60 KEEP effects, 120 preference
records, and the audited evidence review.

## A. Extract actual full action traces (CPU only)

The previous evidence ZIP has predecision RGB, first post-treatment System-2
output, action counts, trace source paths and STOP summaries, but does NOT
include the full per-action trajectory. Comparing results alone cannot reveal
where actions begin to diverge.

~~~bash
cd ~/InternNav
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_trace_divergence_v01.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_trace_divergence_v01.py build \
  --review-dir data/intervention_v01_mechanism_review_01 \
  --out-dir data/intervention_v01_trace_divergence_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_trace_divergence_v01.py audit \
  --out-dir data/intervention_v01_trace_divergence_01
~~~

The target directory must be new. The resulting uploadable file is:

data/intervention_v01_trace_divergence_01/trace_divergence_share.zip

It bundles these four files:
1. trace_divergence_vs_keep.jsonl: first differing policy-continuation
   action code index for 24 same-context non-KEEP comparisons, relative
   to KEEP, excluding forced physical yaw and including outcome annotations.
2. continuation_action_traces.jsonl: full action sequence for all 32 arms,
   observed main_step if present, and OFFLINE_ONLY actual pose/termination.
3. decision_comparisons.jsonl: 16 pairs of genuine first post-intervention
   System-2 outputs (SEEK_LEFT vs SEEK_RIGHT; SEEK_RIGHT vs RESET).
4. trace_divergence_report.json: source SHA256s, output SHA256s and scope.

Expected AFTER successful server run/audit:

POSTHOC_CONTINUATION_TRACES_EXPORTED_NOT_CAUSALLY_IDENTIFIED
POSTHOC_TRACE_DIVERGENCE_AUDITED 4 scenes 8 contexts 32 arms ... policy action rows

The exporter verifies the saved evidence-review SHA256s and full source
provenance, original native prefix physical poses/actions, three real yaw
actions for perturbed contexts, exact six real 45-degree SEEK scan turns,
separation of forced intervention from autonomous policy continuation,
and a create-only output. Natural KEEP traces do not have recorded
main_step, and perturbed KEEP's first S2 message is also unavailable.
Neither datum may be reconstructed by borrowing another arm's values.

IMPORTANT: first differing continuation action index is NOT first
System-2 reasoning divergence and NOT a demonstrated cause of recovery.
Ground-truth robot poses and observed success are offline diagnostics only.

## B. Mechanism-isolation arms for the NEXT GPU experiment (not executed)

Two recovered TRAIN contexts (episode 70 and episode 106, both perturbed)
were selected AFTER outcomes were inspected: use ONLY for exploratory
mechanism review, never for a held-out or independent training claim.

Compare four SAME pre-arm states:

| Arm | Real 6-action RIGHT45 scan? | Extra prompt/image slot? | Extra image |
|---|---|---|---|
| Frozen KEEP | No | No | None |
| SCAN_ONLY / NO_INJECTION | Yes | No | None |
| PLACEBO_MATCHED | Yes | Yes | Duplicate CURRENT RGB |
| Original EVIDENCE | Yes | Yes | True outward-facing RGB |

All scanned arms must have the same real physical turn sequence, exact
return pose, pre-arm RGB/history/RNG/weights, action budget, model checkpoint
and first System-2 logging. EVIDENCE and PLACEBO additionally require
identical text prompt, image count, image size and insertion order: only the
supplementary image CONTENT may differ. SCAN_ONLY must have neither
extra prompt nor injected image. Never edit the existing hashed v0.1 GPU
runner or manifests; create a separate trial root with a new source contract.

Interpret comparisons as follows:
- EVIDENCE vs PLACEBO: conditional outward image content effect.
- PLACEBO vs SCAN_ONLY: duplicate current image plus additional prompt.
- SCAN_ONLY vs KEEP: physical scan plus differences in action budget;
  not a pure motion effect absent a matched cost comparator.

One or two successes after hindsight case selection do not prove a general
mechanism; repeated seeds, negative controls and outcome-blind scene-disjoint
new TRAIN contexts are needed.

HISTORY_RESET's harms require inspecting exactly which historical landmark
or task-progress evidence is removed. A wrong first S2 output or a different
STOP action by itself cannot establish a history-drift diagnosis.

## Next criterion

Use the CPU trace export first to locate observed action divergences. Then
implement the above three genuinely matched GPU-control arms, separately
from the frozen 80-outcome v0.1 data. Only after these controls, start
training a task-evidence-conditioned, risk/cost-sensitive selector with
new outcome-blind training scenes.
