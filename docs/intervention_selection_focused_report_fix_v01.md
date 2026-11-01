# v0.1 report-only KeyError('checks') fix

The completed 12-rollout experiment should NOT be rerun.

## Root cause

The original `report` uses:

    common.same_fields(detail["gate"], gate, tuple(gate), "... frozen S2 checkpoint")

Here `gate` is the immutable CheckpointProbe record containing both the
nine replay-observed state fields AND eligibility/protocol audit metadata
(`checks`, `eligible`, `remaining_native_main_steps`, etc). The newly
replayed `detail["gate"]` contains only nine runtime state fields.

Requiring every frozen metadata field in the runtime gate crashes with
`KeyError: 'checks'`. This is a schema comparison bug in the REPORT phase,
not evidence of divergent trajectories.

## Safety and source-hash compatibility

The completed experiment's preparation manifest pins the original focused
experiment script by its absolute filesystem path and SHA256. Thus DO NOT
`git pull` changes to that file in the original experimental checkout or
overwrite `manifest.json`. This fix is a SEPARATE report-only executable
and does not edit any frozen source file, the original driver or the evaluator.

For each of the three scan arms the patch verifies:
- The five predeclared eligibility checks in the source checkpoint are all true.
- All nine actual replay-gate fields match the original source exactly:
  main step, action-prefix count, pose, observation hashes, history length and
  hash, policy RNG hash, model weights fingerprint, and prefix trace SHA256.
- All other comparisons in the old `report` run WITHOUT modification.

The wrapper restores the original comparison function even if the report fails.
The report remains create-only: existing results are never overwritten.

## CPU-only execution (no model, no Habitat, no Codex)

Use a separate checkout to avoid changing the source path recorded in the
frozen manifest. The original `/home/geenitiming/InternNav` must remain
unchanged.

    cd /home/geenitiming/InternNav
    git fetch origin
    git worktree add --detach ../InternNav-focused-report-fix origin/exp/intervention-v01-focused-report-schema-fix
    cd ../InternNav-focused-report-fix
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_focused_report_fix_v01.py' -v
    python -m py_compile scripts/eval/intervention_selection_focused_report_fix_v01.py

Verify the original focused-driver file agrees with the frozen copy in the
fix worktree:

    sha256sum /home/geenitiming/InternNav/scripts/eval/intervention_selection_focused_mechanism_v01.py scripts/eval/intervention_selection_focused_mechanism_v01.py

Set OUT to the same directory containing the 12 completed rollouts:

    OUT=/mnt/data8t/geenitiming/internnav_data/intervention_v01_focus_mechanism_01
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python scripts/eval/intervention_selection_focused_report_fix_v01.py report --run-dir "$OUT"

Expected (ONLY if all original runtime audits pass):

    FOCUSED_MECHANISM_REPORT=PASS 3 scenes, 12 actual rollouts
    FOCUSED_REPORT_GATE_HOTFIX_V01=PASS

If another validation fails, **do not patch away it**: the error is the next
actual gate to inspect. No GPU reruns should be started merely because a
report-only schema check failed.

This experiment was selected posthoc and its report is descriptive, not
held-out causal evidence.

## New: restore exact frozen source after a later Git update

A report cannot run if the ORIGINAL source path recorded in manifest was
changed after the 12 GPU rollouts. A separate Git worktree alone does not
repair this: the original manifest still hashes
/home/geenitiming/InternNav/scripts/eval/intervention_selection_focused_mechanism_v01.py.
The new hotfix report loads that EXACT source after SHA256 verification;
it cannot safely substitute an older branch version.

The read-only inspector compares the run's expected source SHA256 with the
current original source and three known historical Git versions:

    OUT=/mnt/data8t/geenitiming/internnav_data/intervention_v01_focus_mechanism_01
    python scripts/eval/intervention_selection_frozen_source_restore_v01.py inspect --run-dir "$OUT"

If source_matches_manifest is false but matching_git_revisions is non-empty,
explicitly restore the exact matching historical bytes (not an approximate
similar version). This operation backs up the CURRENT original file into a
non-overwriting sidecar .bak file in its directory and replaces only that
single original script. No manifest edits, no model runs, no dataset changes:

    python scripts/eval/intervention_selection_frozen_source_restore_v01.py restore --run-dir "$OUT"

If NO_EXACT_GIT_MATCH occurs, STOP. The source may have contained uncommitted
changes at prepare. Recover the exact bytes from snapshots before proceeding;
never fabricate the hash or disable pilot.verify_manifest.

Run 11 CPU report regression tests plus 6 source recovery regression tests:

    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_focused_report_fix_v01.py' -v
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_frozen_source_restore_v01.py' -v

Then, still from the independent hotfix worktree:

    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python scripts/eval/intervention_selection_focused_report_fix_v01.py report --run-dir "$OUT"

The hotfix now loads the original SHA256-verified experiment driver from the
manifest source path, not the potentially different copy in its own worktree.
Any subsequent hash or actual rollout inconsistency remains a fatal error.

## 2026-11-01: correct four native camera actions before first continuation System-2

The 12 real episode/arm continuations all show an observed first S2
model.generate offset exactly FOUR actions after the original focused report
hardcoded expectation. Each measured action stream contains exactly four
`phase=continuation` env actions before the first S2 call.

The frozen InternNav habitat evaluator performs these actions when starting an
ordinary main-loop S2 decision:

    LOOKDOWN (action code 5)
    LOOKDOWN (action code 5)
    LOOKUP   (action code 4)
    LOOKUP   (action code 4)
    model.generate(...)

Thus the full strict first-call offset is:
`checkpoint_prefix_action_count + 3 (real prior yaw) + 6 (only scanned arms)
+ 4 (real native camera env actions)`.

This must NOT be implemented as a generic +4 tolerance. The report-only
hotfix now matches exactly one known original AST assertion and replaces it
with a check of exact recorded phase segments, native camera action codes,
main-step number, first real model.generate entry offset and per-episode S2
ordinal in all four arms. Every original gate/pose/RGB/RNG/history/source-SHA
audit is retained. A changed assertion AST triggers an error; old source and
data remain byte-for-byte untouched.

CPU tests:

    cd /home/geenitiming/InternNav-report-hotfix
    git fetch origin
    git switch --detach origin/exp/intervention-v01-focused-report-schema-fix
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/unit_test -p 'test_intervention_selection_focused_report_fix_v01.py' -v
    python -m py_compile scripts/eval/intervention_selection_focused_report_fix_v01.py

When tests pass and original source_matches_manifest remains true:

    OUT=/mnt/data8t/geenitiming/internnav_data/intervention_v01_focus_mechanism_01
    PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' python scripts/eval/intervention_selection_focused_report_fix_v01.py report --run-dir "$OUT"

No GPU reruns are required for this reporting issue. If a different audit
fails, do NOT change its gate without inspecting the actual source traces.
