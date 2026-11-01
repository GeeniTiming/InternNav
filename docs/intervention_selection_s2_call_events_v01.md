# Passive S2 event capture v0.1 — new replay arms ONLY

## Status and scope

The original intervention-v0.1 action logs were all verified, but the S2 source probe returned `RAW_ACTION_LOGS_VERIFIED_NO_LINKABLE_S2_CALL_ANCHORS`. **This new recorder does not repair historical trajectories**. It records actual `model.generate` entry and exit events in new replay arms only, including their canonical episode key, per-arm generation call ID, per-episode call counter, exact per-episode executed action-prefix length, replay phase, main-loop step, token-input SHA256 and decoded S2 output.

These logs are **offline audit evidence only**, not online selector input. They do not identify `ordinary` vs `lookdown_followup` from the action index (that classification requires a native-loop capture hook). They do not record model input pixel content: previous outward RGB / injection provenance must still be independently verified from `evidence_input.json` and gate snapshots. Replay prefix/history/RNG/model-state matching, return pose, and image matching remain mandatory before a causal ablation can be considered paired.

## Integration

The recorder can be added to a new checkpoint replay driver that has a `probe` with canonical `current`, `records`, `phase`, `current_step`, and `evaluator`. The `recovery_train_six_arm.py` runner is wired to this recorder for **future fresh arm runs only**. It does not change previous six-arm outputs. Do not reuse an existing run directory or rerun previously audited pilot data just to create S2 logs.

The **three v0.1 focus cases use a different runner** (`intervention_v01_paired_all10_01` on the server), which is not in the currently published branch. This instrumentation must be integrated into that exact runner before running the 12 predeclared KEEP / SCAN_NO_INJECTION / PLACEBO / REAL_SEEK follow-ups; do **not** substitute the older six-arm pilot as evidence for those cases.

## CPU-only tests

```bash
cd /home/geenitiming/InternNav-decision-audit
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_s2_call_events_v01.py' -v
```

## Audit after future fresh, instrumented run

For a fresh six-arm-format run that contains same-arm `turn_actions.jsonl` and `s2_call_events.jsonl`:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_s2_call_events_v01.py audit \
  --run-dir /actual/fresh/six_arm_run --arm lookback_right45_evidence_keep
```

`S2_CALL_EVENTS_V01=PASS` means the per-arm call entry/exit pairs are structurally valid and action offsets fall within that arm's real action stream. Its status **does not claim** that a KEEP-vs-SEEK comparison is causally aligned; all arms must first pass the existing gate/protocol audits and include their own S2 events.
