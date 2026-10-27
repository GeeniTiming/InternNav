# Case15: fixed in-place prompt controls

This is a development diagnostic for the already inspected
`zsNo4HB9uLZ / 15` failure. It is not a new recovery algorithm, a
training cohort, or a held-out performance estimate.

## Question and locked design

At the previously verified idle System-2 boundary (main step 85,
446 actual native actions), does a fixed in-place prompt suffice?
The native loop already performs a new high-level query at this boundary.
Calling it again with identical input is not a distinct replanning treatment.

Three new runs are declared together, without selecting the best prompt later:

| Arm | Intervention in the next existing native query |
| --- | --- |
| control | No text change; must reproduce all 547 native actions and metrics |
| neutral | One generic instruction to reconsider the current observation and task |
| review | One instruction to review retained history, the current observation, and completed/remaining instruction parts |
| physical reference | Reuse the completed `case15_physical_recovery_02/repair` result; no new physical rollout |

The exact two strings are constants in the script and are copied into the
preparation protocol. They contain no case-specific direction, landmark,
waypoint, reference action or goal coordinate. They are different compound
prompts, not a clean intervention on a single memory mechanism.

Both reminders are inserted once, into the first user prompt at step 85.
If the model next chooses LOOKDOWN, its normal conversation continues; the
reminder is not inserted again. All later queries follow the native prompt.
No extra model call or environmental action is made by the insertion hook.

Policy weights, history contents, history sampling, observation handling,
seeds, action decoding and episode budgets stay native. This uses
cache-disabled inference. The source physical experiment is read-only.

## Run in the existing checkout, foreground inside tmux

```bash
cd /home/geenitiming/InternNav
git switch perf/dualvln-cache-pilot
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
  -p test_case15_reinstruction_probe.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/case15_reinstruction_probe.py prepare \
  --source-run data/case15_physical_recovery_02 \
  --run-dir data/case15_reinstruction_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/case15_reinstruction_probe.py run \
  --run-dir data/case15_reinstruction_01 --arm control

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/case15_reinstruction_probe.py run \
  --run-dir data/case15_reinstruction_01 --arm neutral

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/case15_reinstruction_probe.py run \
  --run-dir data/case15_reinstruction_01 --arm review

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/case15_reinstruction_probe.py report \
  --run-dir data/case15_reinstruction_01
```

Run sequentially. Stop on any nonzero exit; do not start the next arm after
a failed control. Existing run directories are never overwritten or resumed.
For a rerun, use a new experiment output directory, not a new project checkout.
This does not signal, pause or stop any ongoing evaluation or other GPU process.

The source is the successful physical experiment, not the malformed original
failure log. Its validated derived reference is reused as-is. No key repair,
quaternion normalization or other input rewriting occurs in this experiment.

## Checks and outputs

The fresh control must match the entire native source action/pose trace and
metrics. Each prompt arm must first match the source and fresh-control boundary:
scene/episode prefix, pose, raw-observation hashes, retained-history hashes and
RNG state. Before changing text it also matches the original prompt, selected
image bytes, history indices, pose, RNG and parameter-version fingerprints.
A failed gate is `NOT_COMPARABLE`, not a recovery failure.

The process-local AST hooks preserve the evaluator file. CPU tests include
the actual native loop's structure, an unchanged-prompt sham, refusal of a
changed prompt implementation, one-shot insertion, rejection of altered
prefix/context/RNG, source preservation, and unchanged budgets.

Results:
- `data/case15_reinstruction_01/reinstruction_report.json`
- `video/{control,neutral,review}/epoch_0/zsNo4HB9uLZ_0015.mp4`
- Each arm records `boundary_gate.json`, `prompt_event.json` (full before/after
  text), `reinstruction_summary.json`, native actions and pilot metrics.
- The physical reference's 28 correction actions and original handoff are
  included in the report. Its timing is not a new matched speed comparison.

## Predeclared interpretation

If either fixed in-place prompt succeeds, physical relocation was not
necessary for that observed continuation. If only review succeeds, the
result supports this particular history/task-review prompt, not memory
corruption as a cause. If both fail while the physical reference succeeds,
the two tested prompts were insufficient here; other in-place interventions
remain possible.

Use videos to check semantics separately from SR/NE. A single selected case
cannot establish general recovery efficacy, justify a trained selector, or
supply held-out evaluation. Keep the completed 24-case train investigation
closed and unchanged. Do not tune new prompts on this case after seeing results.
