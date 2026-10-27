# Recovery candidate experiment (development cohort v1)

**Purpose:** After the completed fixed-heading diagnostic, run four *predeclared* recovery candidates at the exact same six native idle System-2 checkpoints. The original control is RELEASE (reused from \`orientation_checkpoint_01\`); no source results are overwritten. This is **not** an online recovery system, selector, or held-out validation.

## Candidate matrix

| Arm | Physical operation | History passed to future ordinary System-2 prompts |
| --- | --- | --- |
| RELEASE (prior locked control) | none | native full sampled history |
| history_reset | none | only images acquired after handoff, plus current RGB |
| left90_reset | 6 real left 15-degree body turns | post-handoff history + current |
| right90_reset | 6 real right 15-degree body turns | post-handoff history + current |
| backtrack_replan | return to most recent visited prefix pose >=0.75 m from handoff, using simulator shortest-path follower | native history (no reset) |

Both history and appearance are kept untouched up to handoff, and all interventions are made on a fresh native replay. **History reset means input-history filtering, not clearing model caches or deleting rgb_list.** The current frame and original instruction remain present. The current native prompt's image placeholders are generated from filtered history counts.

The backtrack target uses only observed *prefix positions*, not the reference path or goal. Its shortest-path follower is a privileged simulator execution mechanism, so the result is diagnostic rather than a deployable RGB-only recovery baseline. When no qualifying visited anchor exists, report \`not_eligible\` without selecting a different anchor. Never use target truth to decide where to backtrack.

Fixed boundaries are chosen offline from 60% of the completed baseline and are not online deviation detectors. No case-specific tuning, new angle, new anchor, or post-result intervention selection is permitted. The six cases are an already-inspected development cohort with three original successes and three failures.

## Run: CPU checks first, then GPU

Use the existing Habitat environment and checkout, in a foreground \`tmux\` session.

\`\`\`bash
cd /home/geenitiming/InternNav
git fetch origin
git switch exp/task-conditioned-recovery-candidates-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
    -p 'test_recovery_candidates.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/recovery_candidates.py prepare \
    --source-run data/orientation_checkpoint_01 \
    --run-dir data/recovery_candidates_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/recovery_candidates.py run-all \
    --run-dir data/recovery_candidates_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/recovery_candidates.py report \
    --run-dir data/recovery_candidates_01
\`\`\`

The four arms run sequentially, with a fresh model load per arm. If an arm fails, \`run-all\` stops and preserves all prior outputs. Runs are create-only: never overwrite an existing arm, source manifest, source action trace, or outcome report. To resume manually after an interrupted arm, inspect its output and use a **new** run directory rather than forging completeness.

## Outputs and decision criteria

- \`candidate_protocol.json\`: locked anchors, candidate definitions, source hashes, budget rule.
- \`ARM/turn_actions.jsonl\`: every actual Habitat action, including external correction, with phase and main-step labels.
- \`ARM/checkpoints/SCENE_EP/boundary.json\`, \`handoff.json\`, \`after_recovery.png\`: matched native prefix and real handoff.
- \`ARM/checkpoints/SCENE_EP/history_inputs.json\`: each post-handoff ordinary System-2 prompt's original and filtered history image indices.
- \`candidate_report.json\`: for **every fixed case and arm**, SR, OS, final NE, extra actions, number rescued vs harmed; backtrack eligibility must be disclosed.

Ineligible backtracks must not be counted as failures of a *valid* candidate or removed from the denominator without notice. A full native control action/pose mismatch or changed locked prompt/seed/code artifacts is \`NOT_COMPARABLE\`, never a treatment result. Simulator follower failures and exhausted budgets must be reported explicitly.

**Research gate:** Count restored failures **and** destroyed native successes; only a genuinely positive, repeatable SR benefit can justify learning an exit scorer. The after-the-fact best arm is oracle potential, never a deployable result. For any observed benefit, repeat with independent unseen route groups before claiming effectiveness.

## Known limits

- The cost rule charges each external body action against remaining native main-loop steps (an approximation, because a main-loop step is not necessarily one actual Habitat action; report both).
- The 6 cases are diagnostic development data and cannot estimate population recovery benefit.
- Filtering the history changes model inputs, but does not isolate a specific memory-failure mechanism.
- Model checkpoint weights, original evaluator source, and original locked experiments are not modified.
