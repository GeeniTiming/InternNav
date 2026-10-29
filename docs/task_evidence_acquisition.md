# Task-evidence acquisition: native DualVLN, **TRAIN** data, no recovery

Branch: `exp/task-evidence-acquisition-v1`. This is a **data and
hypothesis-discovery pipeline**, not the recovery method. It does **not**
establish semantic evidence insufficiency or a novel mechanism.

## Why this exists (origin of the prospective method)

VLN decisions depend on the instruction `I`, current RGB `O_t`,
historical RGB and actions `H_t`: `pi(I,O_t,H_t)`. Our prior
fixed-turn/RESET experiments found that changing inputs can change
the next System-2 command; they did **not** reveal why an intervention
should be selected, nor whether a previously unseen landmark caused it.

Before any Q/value network or intervention-selector code, test:

1. Are missing/ambiguous/conflicting **task-relevant** visual
   observations identifiable in real pre-decision inputs?
2. Can at least two independent reviewers annotate those conditions
   consistently, with cited historical/current images?
3. Do separately predeclared, matching interventions later have
   *different* value depending on these labels?

Negative outcomes mean the proposed task-evidence method needs
reconsideration; don't explain them away by post-hoc example choice.

## Exact source/data policy

- Frozen InternVLA-N1 DualVLN policy; `cache_pilot.py` and native
  Habitat evaluator remain unchanged.
- Explicit R2R `train` split only; one first instruction-bearing
  episode per distinct scene, preserving dataset order. **No labels,
  goal distances or future path used for scene selection**.
- Every *actual* native System-2 generate is logged, not only
  a single checkpoint. Both ordinary and `LOOKDOWN` follow-up calls
  are recorded; only **ordinary** decisions enter first-pass annotation.
- The recorder taps images after the native processor prepares them
  and before the model generates its output. Captured prompt/images
  are exact native inputs. Historical RGB is never resampled for
  the capture; image assets are pixel-hash deduplicated.
- The original model still uses depth/GPS wherever its native
  execution requires them. Those are **NOT exported as task-evidence
  selector features**; pose and future success information are
  strictly isolated to offline diagnosis.
- No sidecar policy, no new RGB, no external actions, no RESET,
  no adjustment to sampling RNG, no privileged target inputs.
- All original val_unseen/factorial/evidence-isolation experiments
  remain frozen, and are NOT training examples.

## 1. Pull code and CPU-test first

```bash
cd /home/geenitiming/InternNav
git fetch origin
git switch exp/task-evidence-acquisition-v1
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_task_evidence_acquire.py' -v
```

Do not run GPU if the CPU test fails. Check that
`data/vln_ce/raw_data/r2r/train/train.json.gz`, the train scenes and
frozen model checkpoints exist. The CPU prepare verifies actual source
annotations and scene assets. Do not silently switch to val_unseen.

## Recovery from the initial scene-ID path mismatch (October 29)

The first GPU smoke run in `task_evidence_train_smoke_02_02`
failed at the first System-2 input because the recorder compared
the raw `ep.scene_id` with the raw manifest string. The native
Habitat path can have a different root/representation from the
annotation's scene path. The fixed recorder compares the project's
canonical `failure_case_probe.episode_key()` tuple
`(scene-file-stem, str(episode_id))`; it still rejects unexpected
keys and refuses duplicate canonical IDs.

This is a **capture identity bug**, not a navigator failure, and
the failed run cannot serve as an experimental outcome. Because
the new script has a different source hash and both `control/`
and `capture/` may already exist, the original `02_02`
directory is frozen. **Do not delete, edit, or resume it.**

For this first rerun, use the SAME source-selection parameters as
the old pilot with a fresh directory `task_evidence_train_smoke_02_03`:

```bash
git pull --ff-only
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_task_evidence_acquire.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_acquire.py prepare \
  --episodes 2 --run-dir data/task_evidence_train_smoke_02_03

set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_task_evidence_acquire.py run \
  --run-dir data/task_evidence_train_smoke_02_03 \
  2>&1 | tee task_evidence_train_smoke_02_03.log
```

Only consider a run complete if the `decision_report.json` status
is `TRAIN_NATIVE_DECISION_DATASET_READY`. This modification was
committed remotely; CPU and GPU checks must be run on the server.

## 2. Run a **2-TRAIN-scene smoke test** (not for method claims)

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_acquire.py prepare \
  --episodes 2 \
  --run-dir data/task_evidence_train_smoke_02_01

set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_task_evidence_acquire.py run \
  --run-dir data/task_evidence_train_smoke_02_01 \
  2>&1 | tee task_evidence_train_smoke_02_01.log
```

Expected at end: `status=TRAIN_NATIVE_DECISION_DATASET_READY`,
`N_episodes=2`, `N_decisions>2`. The report is
`data/task_evidence_train_smoke_02_01/capture/decision_report.json`.

Inspect the actual files and verify history/steps; do not immediately
scale to 20 if there is an exception or missing native images.
All directories are **create-only**; preserve failed runs and logs,
use `..._02_02` after any code fix.

## 3. Expand to 20 TRAIN scene-disjoint native trajectories

**Only after the smoke test passes**:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_acquire.py prepare \
  --episodes 20 \
  --run-dir data/task_evidence_train_20_01

set -o pipefail
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
python -u scripts/eval/recovery_task_evidence_acquire.py run \
  --run-dir data/task_evidence_train_20_01 \
  2>&1 | tee task_evidence_train_20_01.log
```

Raw decision/input data are in `capture/`:

| Asset | Content | Permitted online selector? |
|---|---|---|
| `online_decisions.jsonl` | each S2 input: instruction, exact rendered prompt, ordered input RGB image slots, sampled history indices, step/action count, S2 output | pre-generation fields only; output is available **after** S2 |
| `images/<pixel-sha>.png` | real PIL images actually submitted | Yes |
| `online_action_codes.jsonl` | native action codes only (no GPS/pose) | Prefix up to each decision count only |
| `OFFLINE_ONLY_poses.jsonl` | state/rotation for auditing trajectory | **No** |
| `control/actions.jsonl`, `control/progress.json` | native full trajectory/outcomes, generated by cache_pilot | **No; offline auditing only** |
| `capture/decision_report.json` | native completion and capture integrity audit | No policy feature |

The source `rendered_prompt` and `prompt_text` are captured without
asking the model to explain itself. Do not treat S2's decoded output
as an independent ground-truth evidence label.

## 4. Sample 50 states **without looking at outcomes**

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation.py sample \
  --source-run data/task_evidence_train_20_01 \
  --count 50 --seed 20261029 --min-gap 4 \
  --out-dir data/task_evidence_annotations_a_50_01
```

Selection is seeded, round-robin among distinct scenes, sampling
ordinary S2 decisions only and imposing >=4 main steps between
sampled decisions of the same episode. If there aren't 50 eligible
states, report the shortfall rather than replacing them by known
successes or failures.

Each `annotations.jsonl` record has the original task, selected
images, the exact preceding **action-code** history (no poses or future
actions), and an `annotation` dictionary initially all marked
`unreviewed/unknown`. These are **honest placeholders**.

To make manual review easier, render read-only comparison sheets
(current frame plus the four most recent historical inputs):

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation.py render \
  --annotation-dir data/task_evidence_annotations_a_50_01 \
  --history-panels 4
```

The PNGs are saved to `previews/` and are deliberately NOT classifier
features or evidence labels. Render is create-only.

For each decision, read the image files under the absolute
`source_capture_dir`. The ordered `historical_rgb` list is exactly
the frames in the native System-2 prompt; `current_rgb` is last.
Image *slot* is 0-based within
`historical_rgb + [current_rgb]`. This is distinct from a PNG
content hash: one image may appear at multiple time slots.

Fields to annotate:

- `current_subinstruction_span`: literal substring of original
  instruction, or mark `needs_review` if stage is unclear
- `progress`: `before_subgoal`, `at_subgoal`,
  `after_subgoal`, `unknown`
- `target_landmarks`: list of referenced visible/required objects
- `visibility`: `current`, `history_only`, `both`,
  `not_observed`, `uncertain`
- `ambiguity`: `one_candidate`, `multiple_candidates`,
  `none_visible`, `unknown`
- `history_consistency`: `consistent`, `conflicting`,
  `insufficient`, `unknown`
- `support_image_slots`: list of 0-based image slots supporting
  the interpretation
- `support_image_refs`: corresponding relative image filenames
  in the recorded input (no images from a future step)
- optional `evidence_regions`: entries such as
  `{"slot": 3, "landmark": "bed", "bbox_xyxy_norm": [0.2,0.1,0.6,0.7]}`
- `reason`, `confidence`, `annotator_id` and
  `status` (`unreviewed`, `needs_review`, `complete`)

For `complete` a nonempty annotator, reason, literal subinstruction
and, whenever a landmark is marked visible, corresponding cited
image **slots** are required. For conflicts cite at least two
predecision slots. The tool rejects unsupported references and
invalid normalized boxes, but cannot itself verify semantic truth.
Human adjudication is still necessary.

Example **illustrative structure**, not an observed episode:

```json
{
  "annotation": {
    "status": "complete",
    "annotator_id": "reviewer_A",
    "current_subinstruction_span": "walk next to the bed",
    "progress": "before_subgoal",
    "target_landmarks": ["bed"],
    "visibility": "current",
    "ambiguity": "one_candidate",
    "history_consistency": "insufficient",
    "support_image_refs": ["images/<actual-hash>.png"],
    "support_image_slots": [8],
    "evidence_regions": [],
    "reason": "A bed is identifiable in the current view; no reliable progress cue in supplied frames",
    "confidence": "medium"
  }
}
```

Replace `<actual-hash>` and `8` with actual file and slot from the
record. Do not add guessed evidence. If the current stage is genuinely
unknown, mark `needs_review` rather than invent a stage.

## 5. Validate human labels and measure reliability

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation.py validate \
  --annotation-dir data/task_evidence_annotations_a_50_01
```

For double annotation, copy the template `annotations.jsonl`
and `sampling_manifest.json` **unchanged initially** into a
new directory named `task_evidence_annotations_b_50_01`. Reviewer B
independently fills only `annotation`; do not send Reviewer A's
judgments to B.

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation.py compare \
  --review-a data/task_evidence_annotations_a_50_01 \
  --review-b data/task_evidence_annotations_b_50_01 \
  --out-json data/task_evidence_interrater_50_01.json
```

The comparison reports simple field-wise agreement and all
disagreements for subsequent adjudication; raw agreement alone does
not establish reliable labels for imbalanced classes. Consider
class-wise confusion and Cohen's kappa after the pilot.

## 6. Stop/go decision, and what we must NOT claim

Before a full recovery method:

1. Evidence categories must be reliably recognized from pre-decision
   images/instructions, without observing future correctness.
2. Uncertain cases remain uncertain; annotate multiple independent
   scenes and report disagreement/missingness.
3. Only then predeclare and run treatment arms driven by specific
   evidence deficits. Equalize physical actions, image count and
   inference prompt conditions when isolating visual information.
4. Compare against uniform SEEK, ordinary navigation, fixed scans,
   KEEP/RESET and existing evidence/memory baselines.
5. Split scenes/routes into development and held-out evaluation
   **before** fitting the intervention selector. No cherry-picked
   best treatment per episode.

**DO NOT** train Q or claim active semantic recovery from the
2-scene evidence-isolation development results. This acquisition
pipeline is a methodological prerequisite, not a finished innovation.
