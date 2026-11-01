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
