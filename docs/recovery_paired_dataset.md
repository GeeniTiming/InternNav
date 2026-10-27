# Building the first paired 2×2 recovery dataset (development labels only)

The existing six locked checkpoints already have **four independently executed and
matched** continuations:

| Factor code | Physical turn | History rebuild | Existing result |
| --- | --- | --- | --- |
| 00 | no | no | orientation_checkpoint_01/control |
| 10 | RIGHT90 | no | orientation_turn_01/right90 |
| 01 | no | yes | recovery_candidates_01/history_reset |
| 11 | RIGHT90 | yes | recovery_candidates_01/right90_reset |

There is **no need to re-run Habitat on these six cases**. Do not confuse
RIGHT90+RESET with LEFT90+RESET, and do not mix in the backtrack arm.

## 1. Sync and run CPU tests

\`\`\`bash
cd /home/geenitiming/InternNav
git switch exp/task-conditioned-recovery-candidates-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover \
  -s tests/unit_test \
  -p 'test_recovery_dataset_builder.py' -v
\`\`\`

The builder is **stdlib-only** and does not initialize the navigation model.

## 2. Export the paired development labels

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_dataset_builder.py \
  --checkpoint-run data/orientation_checkpoint_01 \
  --turn-run data/orientation_turn_01 \
  --candidate-run data/recovery_candidates_01 \
  --out-dir data/recovery_paired_dataset_01
\`\`\`

The output path **must not already exist**. All input checks occur before any
output is created. The locked source experiment directories are read-only.

Expected if the previously reported six-case results are unchanged:

- Six contexts and 24 candidate rows, grouped by **scene + episode**.
- Positive observed interaction: 3 cases; zero: 2; negative: 1.
- RELEASE success: 3/6, RIGHT90: 3/6, HISTORY RESET: 1/6,
  RIGHT90+RESET: 3/6.
- The output report status is
  \`PAIRED_LABELS_READY_HISTORY_RGB_NOT_CAPTURED\`.

If any check fails, **do not edit old results or replace a case**. Preserve the
exception and report the mismatch before attempting another export. Fixing the
builder itself, if necessary, should not change historical action or success
labels. To re-export, use a new output directory instead of overwriting.

## 3. Outputs

- \`dataset_report.json\`: count, gamma histogram, rescue/harm for each arm,
  provenance and training-readiness warnings.
- \`paired_labels.jsonl\`: one complete 2×2 unit per checkpoint,
  \`Y00/Y10/Y01/Y11\`, observed gamma, and arm costs.
- \`candidate_rows.csv\`: long-form arm-labelled rows for checking, pivoting and
  plotting. Each row belongs to a checkpoint; **24 rows ≠ 24 independent
  training examples**.
- \`contexts.jsonl\`: instruction, runtime budget proxy, current RGB path,
  action-prefix path, explicit \`history_rgb_available=false\`.
- \`contexts/SCENE_EP/current_rgb.png\`: copied **actual checkpoint RGB**.
- \`contexts/SCENE_EP/prefix_actions.jsonl\`: recorded native action IDs and
  indices with **privileged poses/GPS/distance stripped**.
- \`input_provenance.json\`: source and exported file SHA-256 checksums.

The script verifies four report schemas, arm identities, locked checkpoint
metadata, actual per-arm native action/pose prefixes, instruction equality,
result/progress consistency, number of forced turn actions, and post-handoff
reset-history indices. It does not re-run any existing locked source experiment
or modify its manifest.

### The two training-readiness states are different

\`PAIRED_LABELS_READY_HISTORY_RGB_NOT_CAPTURED\` means **the labels and
current RGB are available**, not that a multimodal training dataset is
complete. Existing locked experiments saved hashes of native history images but
**not the original sampled historical RGB bytes**. Hashes, videos or future
observations must not be silently substituted for history frames.

Therefore, the exported six contexts are valid for **paired-label, interaction
statistics and baseline value-probe prototyping**, but not yet for a proper
instruction + current RGB + historic RGB multimodal selector. The next
collection extension must capture historical RGB (or deterministic frozen
encoder features) at the matched checkpoint during a new native replay, with
source identity, history order, frame hashes and observation consistency
verified. That capture is not implemented by this script.

### Training/evaluation caution

The six cases were all used to choose the candidate family. They are
**exploratory development cases only**, and this builder never creates a
train/validation/test split from them. All six may be correlated through scene
or route identity.

The observed interaction

\`\`\`text
Gamma = Y11 - Y10 - Y01 + Y00
\`\`\`

is a *paired contrast on these already-observed rollouts*, not by itself
evidence of population causal interaction. A selector trained using this data
cannot be independently evaluated on the same episodes.

**Next engineering milestone:** data collection for new route-disjoint
checkpoints with non-privileged RGB/history features and all four paired arms,
then train ordinary, additive and interaction-aware value predictors on
separate training and evaluation groups.
