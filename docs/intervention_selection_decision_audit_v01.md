# Intervention selection — CPU posthoc decision-audit v0.1

## What the implementation actually establishes

This script takes the **four files** from `trace_divergence_share.zip` and performs a read-only, hash-verified audit. It recomputes, for each of the 24 treated-vs-KEEP pairs:

- First differing **relative policy-continuation action index** (not S2 time).
- First same-index orientation difference of >=1 degree (offline quaternion geodesic comparison).
- First same-index position difference of >=0.05 m (offline Euclidean comparison).
- Binary SR rescue/harm/no-change comparison and the original NE (offline only).
- Integrity of context identity, unique per-arm action indices, contiguous traces, all four arms in each context, count claims, first-S2 text comparison consistency, and three predeclared focus cases.

It writes four new files: `decision_aligned_events.jsonl`, `case_mechanism_dossiers.json`, `paired_ablation_manifest.json`, `mechanism_audit_report.json`.

**Despite the historic filename `decision_aligned_events.jsonl`, these records explicitly mark `system2_call_alignment=NOT_OBSERVABLE` and `first_system2_decision_difference=null`.** The supplied material does not contain verifiable System-2 call IDs anchored to KEEP and treated-arm policy-continuation actions. Neither `main_step` nor a change in action code can replace such an event. The 16 available first-S2 comparisons exclude KEEP and do not repair this gap. This script does not invent those events or claim decision alignment has been established.

Offline pose, outcome, and raw trace data are never used as policy or Q-network inputs. The four-arm ablation manifest describes **blocked, not executed** experiments: KEEP, SCAN_NO_INJECTION, PLACEBO, REAL_SEEK. It cannot prove same checkpoint, same RNG, return pose, or unchanged history from this ZIP.

The proposed physical scans have three 15-degree actions out and three back, consistent with the pre-registration for the existing models; action codes are *candidate* values and must be verified against the actual evaluator before using the manifest to drive Habitat. PLACEBO must match the S2 image-token format while substituting pre-gate current RGB. SCAN_NO_INJECTION tests a physically matched no-extra-token condition and therefore does not alone isolate the extra-image-token effect.

## Server-side CPU commands

Work on a clean branch checkout; keep local uncommitted changes safe. This feature branch is based on `exp/intervention-selection-dataset-v01`, because the shared ZIP was produced from later scripts that have not been published on this branch. **No Codex on the server.**

```bash
cd /home/geenitiming/InternNav
git fetch origin
# If your current checkout has uncommitted changes, preserve them before switching branches.
git switch --track origin/exp/decision-aligned-mechanism-audit-v01

# Use the directory with the four exported files, or the ZIP itself.
TRACE_SOURCE=/absolute/path/to/trace_divergence_share.zip
RUN_DIR=/absolute/path/to/new_decision_audit_v01_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_decision_audit_v01.py audit \
    --input "$TRACE_SOURCE" --run-dir "$RUN_DIR"

TRACE_AUDIT_BUNDLE="$TRACE_SOURCE" PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
    -p 'test_intervention_selection_decision_audit_v01.py' -v
```

Expected evidence size for the supplied ZIP: four distinct scene groups, eight contexts, 32 arm traces, 14,565 action rows, 24 treated-vs-KEEP comparisons, three predeclared focus cases, and 12 blocked ablation rows. **No simulation or GPU execution happens.**

CLI output is create-only: if `RUN_DIR` already exists, it fails rather than overwriting. `trace_divergence_report.json` SHA256 commitments are checked for its three sibling files. The report's references to *original server raw logs* and an external review report are **not independently rechecked**, because those files were not in the ZIP.

## What remains before a genuinely decision-aligned / causal experiment

1. Export raw KEEP and treated **System-2 calls** with stable per-call ID, real call entry/exit, arm/context ID, and an unambiguous anchor to the native policy action-log offset; preserve and hash the corresponding source paths. Do not reconstruct IDs from `main_step`.
2. Reproduce the same selected checkpoint under each treatment with identical prefix pose, current RGB, history fingerprint, RNG, model parameters and call schedule; verify real scan/return actions and return-pose/current RGB equality. If these checks fail, abort and report rather than silently running or reallocating cases.
3. Execute exactly the locked within-context four-arm comparisons for the three development cases; preserve independent outcome files, action costs and the source hashes. A scan-only / placebo success would constrain any claim that new pixels cause the rescue.
4. Once the mechanism is characterized, sample new TRAIN scenes under an outcome-blind rule and reserve fully independent scenes for held-out testing. These three posthoc cases are mechanism examples, not held-out efficacy evidence.

The feature branch does not alter the original dataset pre-registration, outcome labels, existing simulator, or historical reports.
