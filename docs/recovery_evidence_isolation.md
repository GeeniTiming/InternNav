# Evidence-isolation mechanism pilot (experimental, NOT a trained recovery method)

Branch: \`exp/evidence-isolation-v1\`.

## Precisely stated hypothesis

In an identical native System-2 decision state, a scan-and-return to the
original heading followed by **one real supplemental lateral RGB view**
may change downstream navigation beyond the effect of performing the
**same six physical turn actions** and passing a duplicated current RGB
instead. This is a *mechanism pilot*, not evidence of task relevance or a
deployable active-view strategy.

This is deliberately distinct from claiming that \`RIGHT45\` is a reasonable
navigation policy: the angle is an **a priori controlled diagnostic factor**,
selected symmetrically left/right, not an inferred landmark bearing.

### Experimental groups (prior data reused, no control reruns)

| Group | Agent motion | Additional image | History |
|--|--|--|--|
| A \`release\` | no scan | no | KEEP |
| B \`left45_keep\` / \`right45_keep\` | real 45° turn, hold orientation | native current RGB only | KEEP |
| C \`lookback_{side}45_evidence\` | 45° out + 45° back (6×15° native turns) | **actual outward-view RGB**, resized exactly as native | KEEP |
| D \`lookback_{side}45_placebo\` | identical 6×15° out/back | **duplicate of current resized RGB** | KEEP |
| E \`history_reset\` | no scan | no | RESET |

The existing verified 5×2 factorial contains A/B/E. The script creates
only C/D, totaling 4 NEW arms × 2 existing contexts = 8 new rollouts.

**Comparison C−D** controls the physical turn sequence, final pose,
original prompt and added image-token count. It estimates the effect
of **supplementary image content in this prompt**. It does *not*
separately establish semantic goal grounding, prove that a landmark
was discovered, or support generalization from 2 contexts. B−C is
not a pure heading effect because intervention budgets and view
delivery differ.

A/D differ in both the six-turn scan and one additional image token.
Do not call this difference a pure physical effect.

### Invariants checked

- Native policy unchanged, no added model weights, cache disabled.
- Frozen, preselected native checkpoint; prefix actions/poses, RGB/depth/GPS
  hashes, native RGB history, RNG and parameter versions match.
- Six real Habitat 15° turns, no teleport, zero positional displacement,
  return to checkpoint heading within 0.05°.
- C and D within the same direction must have identical complete
  pre-intervention prefixes, six-action physical scan, and RGB on return.
- Supplementary image inserted **once** into the first ordinary System-2
  prompt, before its unchanged current image. The same text
  (\`Supplementary visual input: <image>.\`) and image count are used
  for C/D; no edits to persistent \`rgb_list\`.
- The original main-step cap is decreased by six in both C and D.
- Export captures the first System-2 response, real outward-view PNG,
  after-return PNG, image-pixel hashes and final SR/NE.
- Rejects incomplete/modified earlier reports and create-only outputs.

## Running in the existing InternNav experiment workspace

\`\`\`bash
cd /home/geenitiming/InternNav
git fetch origin
git switch exp/evidence-isolation-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_evidence_isolation.py' -v
\`\`\`

The new code does **not** change the source files hashed by your previous
\`recovery_new_checkpoint_02_01\` or
\`recovery_new_factorial_02_01\`.

CPU-only source, checkpoint and factorial validation:

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_evidence_isolation.py prepare \
  --source-run data/recovery_new_checkpoint_02_01 \
  --factorial-run data/recovery_new_factorial_02_01 \
  --run-dir data/recovery_evidence_isolation_02_01
\`\`\`

Expected: \`EVIDENCE_ISOLATION_PREPARE=PASS 2 contexts; new arms= 4\`.

**Start with the placebo arm**, so native pose and prompt control gets
verified before interpreting any visual benefit:

\`\`\`bash
set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_evidence_isolation.py run \
  --run-dir data/recovery_evidence_isolation_02_01 \
  --arm lookback_right45_placebo \
  2>&1 | tee recovery_evidence_right45_placebo.log
\`\`\`

A successful run should show \`SCAN_AND_RETURN=PASS\`,
\`SUPPLEMENTARY_VIEW_INJECTED=PASS\` and
\`pilot_summary.json\` / \`turn_summary.json\` both status \`complete\`.

Then run **the matched evidence arm**:

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_evidence_isolation.py run \
  --run-dir data/recovery_evidence_isolation_02_01 \
  --arm lookback_right45_evidence \
  2>&1 | tee recovery_evidence_right45_evidence.log
\`\`\`

Then \`lookback_left45_placebo\` and
\`lookback_left45_evidence\` via the same \`run --arm\` command.

Do **not** use \`run-all\` once any arm has already run: it is
intentionally create-only.

When all four arms complete, perform the CPU-only audit:

\`\`\`bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_evidence_isolation.py report \
  --run-dir data/recovery_evidence_isolation_02_01 \
  > recovery_evidence_isolation_report.log
\`\`\`

The output is
\`data/recovery_evidence_isolation_02_01/mechanism_report.json\`.
Send that JSON to analyze C−D and the first-System-2 response.

**If an arm fails, preserve the run directory and full log**.
Investigate before reusing any existing arm directory. Because the
input manifests hash the experiment script, after a code fix you must
prepare a NEW experiment root. Do not edit locked manifests.

## Follow-on mechanism identification (not implemented in this pilot)

After C−D works in the 2-scene engineering test:

1. Move acquisition to \`train\` episodes and independently predefine
   numerous checkpoint contexts (not selected by treatment outcomes).
2. Annotate the currently relevant landmark/sub-instruction and view
   visibility **without revealing this offline annotation to the policy**.
3. For each checkpoint derive candidate azimuth from *prior observed*
   landmark/heading or known unexplored sectors; verify that the chosen
   view actually contains task-relevant evidence rather than noise.
4. Hold the prompt and physical scan controlled; evaluate whether
   view-content gains correlate with landmark visibility/ambiguity.
5. Only then compare evidence-conditioned candidate selection with
   uniform scanning, KEEP, RESET and the existing seek/memory baselines.

**Success on these two smoke-test scenes does not establish a new VLN
failure mechanism, a trained Q function or a paper contribution.**
