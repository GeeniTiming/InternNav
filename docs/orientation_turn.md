# Fixed cross-case left/right90 diagnostic

This stage reuses the verified native controls in `orientation_checkpoint_01`.
It runs both declared headings on every locked case, retaining original history.
There is no direction selector or training. Better-of-two selection after seeing
outcomes is an oracle diagnostic, not deployable performance.

```bash
cd /home/geenitiming/InternNav
conda activate internnav_habitat
git pull --ff-only

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
    -p test_orientation_turn.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/orientation_turn.py prepare \
    --source-run data/orientation_checkpoint_01 \
    --run-dir data/orientation_turn_01

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/orientation_turn.py run-all \
    --run-dir data/orientation_turn_01
```

Run preparation and tests successfully before starting `run-all`. Arms run
sequentially in the foreground, with one model load per direction, six cases per
load (12 rollouts total). Each rollout computes native actions until the captured
boundary; it does not force the recorded prefix or teleport to a saved pose.
An existing output directory/arm is rejected. If a child fails, run-all stops.
Individual commands `run --run-dir ... --arm left90` (or right90) and CPU `report`
are also available; do not overwrite an incomplete arm.

## Matched boundary and intervention

The saved capture report is fully revalidated without rewriting it. Source
configs, native traces, observations, checkpoints and code are hash-locked.
At the captured boundary require identical native prefix actions/poses, step,
current observation fingerprints, history length/content, policy RNG, parameter
versions and prefix digest. The policy must have no pending local or high-level
plan and its camera must be level. A mismatch is NOT_COMPARABLE and no turn is
applied. Cases and checkpoints cannot be replaced based on outcomes.

Apply six actual native15-degree LEFT actions for left90, or six RIGHT actions
for right90. Verify unchanged position and target yaw. Intermediate views are
saved only, never ingested into policy history; normally ingest the actual final
observation once on continuation. Preserve policy RNG across the external turn
and retain all historical images. No reset, GT waypoint, forced STOP, goal-distance
filter, or extra navigation-model query is introduced. Near-goal success controls
stay in the cohort.

## Budget and reporting

Actual turns consume Habitat environment steps. To charge their cost against the
native evaluation horizon, reduce the native loop limit by six after alignment,
restoring it at the start of each next episode and on exit. Do not increment
`step_id` or insert fake history images: native history sampling indexes images by
that counter. Thus the explicit cost rule is six fewer native loop steps plus
six external turn steps; camera operations and native plan-expiry iterations
mean this is not an assertion that loop steps equal all physical actions.

The report includes the charged limit, native main-step count, limit termination,
all actual environment actions, six forced turns, autonomous post-handoff body
actions, System-2 query count, final position/STOP/done, SR, SPL and NE. Pure turns
cost actions even if path-length-based SPL barely changes. Debug videos are under
`video/left90` and `video/right90`; the six turn views and handoff image are saved
separately. Native videos need not show every external turn continuously.

`turn_report.json` reports rescue and harm separately for each direction, keeping
all six cases and the original control denominators. It does not pick a winning
heading per case as a claimed method result. Failed gates or budget-related
terminations must be distinguished from recovery outcomes. Two arms on one route
are paired interventions, not independent samples. The cohort is development
material and cannot establish held-out efficacy or an isolated memory mechanism.

Checkpoint capture does not label a state as off-route. Target distance is not
reference-route deviation. The current protocol neither changes checkpoint
selection nor adds a history-clearing experiment after inspecting results.