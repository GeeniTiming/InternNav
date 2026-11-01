# S2 source-log readiness probe v0.1 (CPU only)

## Why this is needed

`DECISION_AUDIT_V01=PASS` with status
`CPU_POSTHOC_AUDITED_S2_BOUNDARIES_NOT_OBSERVABLE_ABLATION_BLOCKED`
means the 24 action-indexed KEEP comparisons are reproducible, **not** that
System-2 calls were aligned. The 16 text comparisons are between treated arms;
KEEP has no verifiable native per-call anchor in the trace bundle. A model
output string and a `main_step` are **not** a stable call ID.

The existing `recovery_task_evidence_acquire.py` can log native System-2
`call_id`, `native_action_prefix_count`, input image provenance and generated
text. Those logs can only anchor a matching native run, not a separate
physically perturbed or SEEK continuation. The existing paired intervention
code usually logs S2 call **counts**, not per-call action offsets. No attempt
should be made to splice unrelated native decisions into treated histories.

## Command (on server, no Codex and no GPU)

Use the same four-file ZIP/directory from the previous successful decision
audit; change the `TRACE_SOURCE` value to the actual server path. The probe
reads the original source paths already recorded in that bundle's
`trace_divergence_report.json` and verifies their SHA-256 bytes, then looks for
per-call JSONL logs in the same original run/arm directories.

```bash
cd /home/geenitiming/InternNav-decision-audit
# Safe when this dedicated audit worktree has no uncommitted changes:
git fetch origin
git switch --detach origin/exp/decision-aligned-mechanism-audit-v01

TRACE_SOURCE=/actual/path/to/trace_divergence_share.zip
OUT=/mnt/data8t/geenitiming/internnav_data/s2_source_probe_v01_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/intervention_selection_s2_source_probe_v01.py probe \
  --input "$TRACE_SOURCE" --run-dir "$OUT"

cat "$OUT/s2_readiness_report.json"
```

If the entire original data tree was **moved** (not modified), add
`--source-root /new/absolute/path/to/internnav_data`; the tool preserves paths
relative to the recorded common source root. Do not use this option to select
arbitrary unrelated runs. The output directory is create-only and must not
already exist.

CPU tests:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_intervention_selection_s2_source_probe_v01.py' -v
```

## Output interpretation

The probe produces three reports in `OUT`:

- `source_action_hash_audit.json` – SHA-256 of all eight ORIGINAL action-log
  sources, including missing or changed logs.
- `s2_call_candidate_inventory.json` – candidate same-run JSONL files with
  per-call IDs, prefix anchors and structural checks (only metadata, never
  prompts, RGB, model outputs or offline poses).
- `s2_readiness_report.json` – gate status and counts.

Statuses:

- `BLOCKED_ORIGINAL_ACTION_LOGS_MISSING`: original paths missing; restore the
  exact original source tree or use a correct `--source-root`.
- `BLOCKED_SOURCE_HASH_MISMATCH`: **STOP** – at least one original source
  differs from its frozen SHA-256. The CLI returns nonzero after writing the
  report; do not proceed with mismatched data.
- `RAW_ACTION_LOGS_VERIFIED_NO_LINKABLE_S2_CALL_ANCHORS`: source files match,
  but no per-call record is present; new instrumented replay will be required.
- `RAW_ACTION_LOGS_VERIFIED_S2_CANDIDATES_REQUIRE_EXACT_REPLAY_BINDING`: some
  candidate call logs exist; they **still cannot** be declared call-aligned
  without proving the same arm/episode/action offset, pre-gate history/RNG and
  replay lineage. In both last statuses, `N_SYSTEM2_call_aligned_pairs = 0`
  and the ablation remains blocked.

The probe does not execute Habitat, modify original labels, replay any
trajectory, export task outcomes as online inputs, or run a new ablation. Its
purpose is to avoid unnecessary GPU work and determine which source data must
be re-captured before causal mechanisms can be tested.

Once this report is available, choose the smallest necessary next step:
use verified native per-call logs if *actually* linked to native KEEP, or
instrument new **paired** checkpoint replays for missing arms, then run the
predeclared KEEP / SCAN_NO_INJECTION / PLACEBO / REAL_SEEK controls. Do not
interpret a Rescue by itself as the effect of new visual evidence.