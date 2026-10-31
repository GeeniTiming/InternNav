# Episode 13 mechanism diagnosis: two ablations plus two logged reference replays

## Research question

The six-scene matched TRAIN study already passed comparability and showed a specific conditional harm: `cV4RVeZvu5T/13`, checkpoint step 21, native KEEP/RESET/SEEK_KEEP success, but PLACEBO_RESET and SEEK_RESET failure with final NE about 11.67m. This is an **inspected development case**, not held-out evidence or an intervention-policy training example.

The hypothesis is NOT already proven to be progress drift or memory corruption. Separate three coupled components: (i) real right45 outward-and-return Habitat yaw scan, (ii) one extra image slot plus its fixed supplementary prompt, (iii) S2 sampled-history RESET.

## Four new single-Episode executions

| Run arm | Actual scan | Extra image/prompt | History | Role |
|---|---:|---|---|---|
| `virtual_placebo_reset` | no | duplicate current RGB | RESET | New ablation: extra input without scan |
| `scan_no_injection_reset` | six real yaw actions | none | RESET | New ablation: scan without extra input |
| `replay_placebo_reset` | six real yaw actions | duplicate current RGB | RESET | Exact logged replay of existing failed arm |
| `replay_seek_keep` | six real yaw actions | actual outward RGB | KEEP | Exact logged replay of existing successful arm |

Original source six-arm results, checkpoint traces, original annotation SHA and all source hashes are read-only. Control replays must match the frozen *complete* per-action path and final success/NE, or report fails. The first two are new treatments; four total full-native replays, **not six-scene repeats**.

Virtual vs scan treatments have different physical action costs (0 vs 6) and initial input forms, so these are **controlled component knockouts**, not a complete factorial nor proof of independent semantic causes. RESET only changes S2 history indices, not persistent model state or KV cache.

## Checkout

```bash
cd ~/InternNav
git fetch origin
git switch exp/episode13-mechanism-ablation-v1
git pull --ff-only
conda activate internnav_habitat
```

## CPU unit tests

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_episode13_ablation.py' -v
```

These tests do not execute Habitat; only your server GPU runs provide experimental results.

## Lock inputs and prepare (CPU only)

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_episode13_ablation.py prepare \
  --six-arm-run data/train_six_arm_06_01 \
  --run-dir data/episode13_mechanism_ablation_01
```

Expected: `EP13_ABLATION_PREPARE=PASS one frozen case; four single-episode runs`.

## Execute one GPU arm at a time

Start with the virtual-placebo input-only arm to validate that insertion works without an actual scan.

```bash
set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_episode13_ablation.py run \
  --run-dir data/episode13_mechanism_ablation_01 \
  --arm virtual_placebo_reset \
  2>&1 | tee episode13_virtual_placebo_reset.log
```

If that succeeds, run the other three sequentially:

```bash
for arm in scan_no_injection_reset replay_placebo_reset replay_seek_keep; do
  PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
    python -u scripts/eval/recovery_episode13_ablation.py run \
      --run-dir data/episode13_mechanism_ablation_01 --arm "$arm" \
      2>&1 | tee "episode13_$arm.log" || exit 1
done
```

All arms are create-only: on error, retain the partial arm and log for diagnosis. Never silently resume an existing arm over partial files.

## Audit and interpret (CPU only)

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_episode13_ablation.py report \
  --run-dir data/episode13_mechanism_ablation_01 \
  | tee episode13_mechanism_ablation_report.log
```

Only on complete paired checks should status be `EP13_MECHANISM_COMPARABLE`. Report output is `data/episode13_mechanism_ablation_01/diagnostic_report.json`.

Each run saves **all native decoded S2 outputs** and **S2 history-index selections** under:

`data/episode13_mechanism_ablation_01/<arm>/checkpoints/cV4RVeZvu5T_13/`

with files `system2_outputs.json`, `system2_history_selections.json`, and `diagnostic_inputs.json`. The decoded text is source-generation evidence, not ground truth about internal task progress. Do not assume a Step 33 motor-action divergence is necessarily generated at Step 33; compare the latest S2 outputs preceding the first action mismatch.

Check after the report: whether scan_no_injection_reset fails or succeeds, whether virtual_placebo_reset fails or succeeds, and whether the reference replays remain exactly reproducible. If any reference fails, **STOP**, not a causality conclusion.

## Predeclared interpretation table

| Virtual placebo + RESET | Scan without injection + RESET | Interpretation boundary |
|---|---|---|
| failure | success | extra input/prompt can be sufficient in the no-scan setting; not proof of semantic-history conflict |
| success | failure | physical scan/no-injection setting can be sufficient; investigate yaw and charged budget |
| failure | failure | multiple potentially harmful routes or shared effects; further controls may be needed |
| success | success | neither component alone sufficient; combined interaction remains candidate, not proven |

Proceed next to a larger independently selected TRAIN development cohort with some native failures (without post-outcome picking) and a distinct scene-held-out evaluation. The isolated case supports **mechanism diagnosis**, not training or generalization.