# Task-evidence v2: per-landmark observational labels

**Status: annotation protocol pilot; NOT a trained detector or a causal recovery method.**

This change follows a manual audit of six TRAIN ordinary System-2
decisions from two original trajectories: scene `5LpN3gDmAk7`
(episode 4, call IDs 4/10/12) and `7y3sRwLe3Va` (episode 1, calls
16/21/23). Every sample had a readable current RGB and up to 8
historical RGBs. The prior v1 task-wide `visibility` scalar was
**not** well-defined for instructions with multiple landmarks.

## Why v2

- A central island may be visible while a target doorway, dining
  table, and mirror remain unconfirmed. Do **not** label the
  whole task as visible based on one intermediate reference.
- A bed may be visible in historical RGB while a closet is in
  current RGB. Each named landmark gets its own observation label.
- The *stage* of navigation (whether a subinstruction is complete)
  differs from landmark visibility. When insufficient evidence is
  available, `progress="unknown"` is a legitimate completed review.
- Actions constrain progress. Call 16 had 4 right-turn commands,
  **zero forward** commands: it cannot have traversed around the
  kitchen island during this episode before that decision. Other
  action totals cannot by themselves prove arrival at a doorway.
- Call IDs 4, 12 and 23 have the same pixel hash in the last
  historical and current RGB slots; this is **not two independent
  visual observations**. Explicitly track duplicate slots.
- No unambiguous **historical conflict** positive case was
  established by these six previews. Do not fabricate that category.

## V2 schema and support requirements

`scripts/eval/recovery_task_evidence_annotation_v2.py` retains
original source/episode, image order, pixel hashes, action-prefix
counts and no-outcome sampling. It does not mutate v1 templates or
the completed native acquisition run.

`annotation` is one human review per decision and contains:

- `progress`: before/at/after subgoal or unknown
- `stage_span`: exact substring of the instruction for a
  positively identified stage; empty for unknown
- `progress_basis`: visual, actions, visual_and_actions,
  or unknown; cite 0-based episode-local action indices if used
- `history_consistency`: consistent, conflicting,
  insufficient, unknown; conflicting claims require two
  distinct, nonidentical RGB slots
- `landmarks`: an array of **separate** landmark entries,
  each with `name`, literal `instruction_span`,
  `role`, `visibility`, `support_image_slots`,
  `regions` and `reason`
- `status`, `annotator_id`, `reason`, `confidence`

Landmark `role` is one of:
`route_anchor`, `intermediate_target`,
`goal_disambiguator`, `stop_reference`, `other`.

Landmark `visibility` is one of:
`current`, `history_only`, `both`, `not_observed`,
`uncertain`. The last two are **not interchangeable**:
`not_observed` means no verified relevant object appears
in the supplied native images; `uncertain` means there are
ambiguous candidates or image evidence is insufficient.

Example of two distinct objects, **illustrative only**:

```json
{
  "landmarks": [
    {
      "name": "center unit",
      "instruction_span": "center unit",
      "role": "route_anchor",
      "visibility": "current",
      "support_image_slots": [4],
      "regions": [],
      "reason": "Kitchen island identifiable in current input"
    },
    {
      "name": "mirror",
      "instruction_span": "mirror",
      "role": "goal_disambiguator",
      "visibility": "uncertain",
      "support_image_slots": [],
      "regions": [],
      "reason": "Cannot verify a task-matching mirror"
    }
  ]
}
```

0-based slot indices refer to the **entire** native System-2
`historical_rgb + [current_rgb]`, not the image's deduplicated
PNG basename. Example slot 4 applies only when the input has
four historical frames. Never invent image region boxes or
landmark identities to fill a missing annotation.

A completed review may keep `progress="unknown"`,
`progress_basis="unknown"` and `stage_span=""` if the
`reason` explains what cannot be inferred. The validator checks
structure/source provenance, **not semantic correctness**.

## CPU-only v2 dry run on the same six previously inspected states

```bash
cd /home/geenitiming/InternNav
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests/unit_test \
  -p 'test_recovery_task_evidence_annotation_v2.py' -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation_v2.py sample \
  --source-run data/task_evidence_train_smoke_02_03 \
  --count 6 --seed 20261029 --min-gap 4 \
  --out-dir data/task_evidence_annotations_v2_smoke_06_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation_v2.py validate \
  --annotation-dir data/task_evidence_annotations_v2_smoke_06_01
```

Expected sample call IDs: `[12,16,10,21,4,23]`.
Initially every `annotation.status` is `unreviewed`. The
validator returning `ANNOTATION_V2_AUDIT_PASS` means source
integrity and schema are correct, **NOT** that the human labels
have been completed or judged reliable.

For image review, the existing v1 `render` tool can render the v2
`annotations.jsonl` since the image references are unchanged:

```bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
python scripts/eval/recovery_task_evidence_annotation.py render \
  --annotation-dir data/task_evidence_annotations_v2_smoke_06_01 \
  --history-panels 8
```

The next acquisition step is a fresh, independently selected set
of 20 scene-disjoint TRAIN episodes from the frozen native agent,
followed by v2 evidence annotations. Those episodes are data for
protocol development, **not independent efficacy results**.
Reserve other scenes before testing any future learned selector.

The v1 six-sample archive and `task_evidence_train_smoke_02_03`
remain unchanged; no GPU re-run is required for v2.
