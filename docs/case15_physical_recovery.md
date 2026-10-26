# Episode 15: one physical recovery, then native continuation

This is a diagnostic for the audited native failure in scene zsNo4HB9uLZ,
episode 15, trajectory 62. It is not a proposed recovery algorithm.

## Locked comparison

- Source: the completed native capture in data/failure_case15_01.
- Control: another native replay, with a non-mutating callback at the same boundary.
- Repair: identical native prefix, then one reference-informed physical correction.
- Boundary: top of native loop step 85, before observation ingestion or new S2 generation.
  The expected prefix contains exactly 446 actual Habitat actions, including the
  camera queries in step 84. The previous S1 plan has been exhausted; its discarded
  next action is not a live motion command.
- First recovery waypoint: the actual pre-step-76 position.
- Final recovery waypoint: exactly 1 m after reference_path[2], toward reference_path[3].
  It is about 5.53 m from the annotated goal in Euclidean distance.
- Both waypoints must be navigable and connected before any recovery motion.
  There is no snapping, retargeting, teleportation, or search for a successful point.
- A shortest-path follower executes at most 80 native body actions. Its STOP means
  "waypoint reached" and is never sent to the episode environment.
- All recovery motion contributes to Habitat path length, metrics and environment
  step budget. It does not advance the policy's main-loop step counter or create
  artificial policy decisions. All original maximum-step settings remain unchanged.

## History and handoff

Existing images and the original language instruction are retained. Intermediate
recovery images are saved under repair/recovery_frames but never ingested into
rgb_list. The final, true observation is returned to the evaluator at step 85,
where normal observation ingestion and fresh S2 generation continue.

Policy RNG streams are preserved around the deterministic external controller.
Parameter versions and existing image-history bytes are checked for changes.
No model weights, prompts, local action rules, cache settings, or source evaluator
files are edited. One callback is inserted into a process-local copy of the
native loop by AST transformation; the original method and class initializer are
restored when the diagnostic exits.

The current view, existing history, native action prefix, pose and decision
boundary must match the control before correction. A failed comparison is
NOT_COMPARABLE and applies no recovery action. The control must also reproduce
the entire original trace and metrics before the repair arm can run.

## Run in the existing checkout

Use the existing internnav_habitat environment and existing branch.
All commands run in the foreground and work inside tmux. No PR is needed.

~~~bash
cd /home/geenitiming/InternNav
git pull --ff-only
conda activate internnav_habitat

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/unit_test \
  -p test_physical_recovery_probe.py -v

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/physical_recovery_probe.py prepare \
  --source-run data/failure_case15_01 \
  --run-dir data/case15_physical_recovery_01
~~~

Then run one arm at a time. The control must finish with
NATIVE_CONTROL_REPRODUCTION=PASS.

~~~bash
set -o pipefail

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/physical_recovery_probe.py run \
  --run-dir data/case15_physical_recovery_01 --arm control \
  2>&1 | tee data/case15_physical_recovery_01/control_console.log
~~~

Only after that passes:

~~~bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/eval/physical_recovery_probe.py run \
  --run-dir data/case15_physical_recovery_01 --arm repair \
  2>&1 | tee data/case15_physical_recovery_01/repair_console.log

PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/physical_recovery_probe.py report \
  --run-dir data/case15_physical_recovery_01
~~~

Preparation and arm directories are create-only. On failure, preserve the output
and inspect the error; do not modify a locked target to make the result pass.
cache.py inherited from the earlier pilot preparation is unused; never run it
for this comparison. Existing source runs and any ongoing evaluation are untouched.

## Outputs and interpretation

- physical_recovery_report.json: control reproduction, final metrics, handoff,
  and the first 12 autonomous body actions.
- control/boundary_gate.json and repair/boundary_gate.json: matched pre-intervention state.
- repair/after_recovery.json and after_recovery.png: actual physical handoff.
- repair/probe_actions.jsonl: each actual action tagged native, recovery, or
  native_after_recovery.
- repair/recovery_frames/: actual intermediate correction views.
- video/control/ and video/repair/: native debug videos. The repair debug video
  jumps across external motion; use recovery_frames for that interval.

If the repaired arm completes the task, a simple physical correction may already
suffice for this case. If it leaves the corrected route again, further work must
separate target selection, local control and historical context. Neither outcome
isolates a memory cause. Benchmark success does not establish complete language
compliance. This selected val_unseen case is exploratory; it is not a development
cohort or an aggregate efficacy result.

CPU tests cover boundary instrumentation, exact-prefix refusal, fresh-observation
handoff, history invariance, RNG restoration, follower STOP/budget handling and
annotation locks. The real Habitat/GPU run must still be verified on the server.

## Recovery from the reported reference-key typo

The source action file was reported to contain one malformed key at zero-based
record 10: "o rientation_wxyz". New preparation validates every reference record
on CPU before model loading. No missing quaternion is filled and no quaternion
is renormalized.

To use the existing value under the correctly spelled key, explicitly enable
--repair-known-key-typo. The only allowed repair is that exact key at record 10;
a duplicate canonical key, other missing fields, malformed/nonfinite values or
an unrecognized typo fail preparation. The original source file and all old
manifests are preserved.

Preparation writes reference_actions.jsonl and reference_audit.json in the new
run directory and locks their hashes. Run and report compare against this
derived reference, while original source bytes remain hash-locked as provenance.
The audit records the old/new key, unchanged quaternion, and source/derived
hashes. A changed script is not run against an old locked manifest.

For the failed attempt in data/case15_physical_recovery_01, use:

~~~bash
PYTHONPATH="$PWD" USE_TF=0 CUDA_VISIBLE_DEVICES='' \
  python scripts/eval/physical_recovery_probe.py prepare \
  --source-run data/failure_case15_01 \
  --run-dir data/case15_physical_recovery_02 \
  --repair-known-key-typo \
  --previous-attempt data/case15_physical_recovery_01
~~~

This also checks the corrected source prefix against the independent 446-action
control/probe_actions.jsonl from the failed attempt before loading a model.
Any disagreement aborts. Expect REFERENCE_SCHEMA_CHECK=PASS with key_repairs=1
and PREVIOUS_PREFIX_CHECK=PASS with actions=446.

Then run control, repair and report using data/case15_physical_recovery_02 in all
commands above. Keep the failed _01 directory. Its process stopped before the
boundary check completed, so it cannot be resumed as a completed control:
one fresh single-episode control is required.
