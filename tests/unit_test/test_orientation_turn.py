"""CPU checks for paired turn boundaries, real motion, costs and source immutability."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_orientation_checkpoint as fixtures
SPEC = importlib.util.spec_from_file_location('orientation_turn_test', ROOT / 'scripts/eval/orientation_turn.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class TurningEnv(fixtures.FakeEnv):
    def __init__(self, episode, drift=0, terminate=False):
        self.yaw, self.drift, self.terminate = 0., drift, terminate
        super().__init__(episode)

    def agent_state(self):
        state = super().agent_state()
        q = PROBE.pose.yaw_quaternion(self.yaw)
        state.rotation = SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3])
        state.sensor_states['rgb'].rotation = state.rotation
        return state

    def step(self, action):
        value = super().step(action)
        if action in (2, 3):
            self.yaw += 15 if action == 2 else -15
            self.position[0] += self.drift
            self.obs['rgb'] = self.obs['rgb'].copy()
            self.obs['rgb'][:] = int(abs(self.yaw))
        return value[0], value[1], value[2] or (self.terminate and action in (2,3)), value[3]


class TurnTests(fixtures.Base):
    def source(self):
        root = fixtures.ReportTests.build_capture(self)
        with contextlib.redirect_stdout(io.StringIO()):
            PROBE.checkpoint.report(SimpleNamespace(run_dir=str(root)))
        return root

    def make_turn(self, arm='left90', drift=0, terminate=False):
        source = self.source()
        _, report = PROBE.source_check(source)
        root = source.parent / 'turn'
        (root / arm).mkdir(parents=True)
        protocol = {'cases': report['cases'], 'native_main_step_limit': 200, 'charged_main_step_limit': 194}
        ds, _, references = fixtures.native_fixture()
        env = TurningEnv(ds['episodes'][1], drift=drift, terminate=terminate)
        evaluator = SimpleNamespace(env=env, model=SimpleNamespace(), max_steps_per_episode=200)
        weights = mock.patch.object(PROBE.common, 'parameter_versions', return_value='weights')
        rng = mock.patch.object(PROBE.common, 'rng_digest', return_value='rng')
        preserve = mock.patch.object(PROBE.physical, 'preserve_policy_rng', side_effect=contextlib.nullcontext)
        for patch in (weights, rng, preserve):
            patch.start()
            self.addCleanup(patch.stop)
        probe = PROBE.TurnProbe(evaluator, root, arm, protocol, references)
        self.addCleanup(probe.close)
        env.step = probe.record
        self.boundary(probe, env, [], 0, action=None)
        for _ in range(12):
            env.step(1)
        return root, probe, env, source

    def test_left_and_right_execute_six_turns_preserve_history_and_charge_budget(self):
        for arm, action, yaw in [('left90', 2, 90), ('right90', 3, -90)]:
            root, probe, env, source = self.make_turn(arm)
            history = []
            with contextlib.redirect_stdout(io.StringIO()):
                observed, done = self.boundary(probe, env, history, 12)
            self.assertEqual(env.calls, [1]*12 + [action]*6)
            self.assertEqual(env.position, [3., 0., 0.])
            self.assertAlmostEqual(env.yaw, yaw)
            self.assertIs(observed, env.obs)
            self.assertFalse(done)
            self.assertEqual(history, [])
            self.assertEqual(probe.evaluator.max_steps_per_episode, 194)
            self.assertEqual(len(list((root / arm / 'frames').rglob('turn_*.png'))), 6)
            env.step(1)
            self.boundary(probe, env, history, 13)
            self.assertEqual(probe.forced_count, 6)  # no repeated intervention
            env.step(0)
            probe.finish()
            self.assertEqual(probe.finished[('F','1')]['autonomous_body_actions_after'], 1)
            probe.close()
            self.assertEqual(probe.evaluator.max_steps_per_episode, 200)

    def test_changed_current_image_fails_before_turning(self):
        _, probe, env, _ = self.make_turn()
        env.obs['rgb'][0,0,0] = 10
        with self.assertRaisesRegex(RuntimeError, 'observation_hashes'):
            self.boundary(probe, env, [], 12)
        self.assertEqual(len(env.calls), 12)

    def test_pending_plan_or_wrong_boundary_fails_before_turning(self):
        _, probe, env, _ = self.make_turn()
        with self.assertRaisesRegex(RuntimeError, 'idle boundary'):
            self.boundary(probe, env, [], 12, pixel_goal=[1,2])
        self.assertEqual(len(env.calls), 12)

    def test_changed_rng_fails_before_turning(self):
        _, probe, env, _ = self.make_turn()
        with mock.patch.object(PROBE.common, 'rng_digest', return_value='changed'):
            with self.assertRaisesRegex(RuntimeError, 'rng_sha256'):
                self.boundary(probe, env, [], 12)
        self.assertEqual(len(env.calls), 12)

    def test_translation_or_termination_during_turn_is_rejected(self):
        for kw in ({'drift': .1}, {'terminate': True}):
            _, probe, env, _ = self.make_turn(**kw)
            with self.assertRaises(RuntimeError):
                self.boundary(probe, env, [], 12)
            self.assertIsNone(probe.handoff)
            probe.close('rejected')
            self.assertEqual(probe.evaluator.max_steps_per_episode, 200)

    def test_episode_transition_restores_limit_and_counters(self):
        _, probe, env, _ = self.make_turn()
        with contextlib.redirect_stdout(io.StringIO()):
            self.boundary(probe, env, [], 12)
        env.step(0)
        ds, _, _ = fixtures.native_fixture()
        env.set_episode(ds['episodes'][2])
        env.yaw = 0.
        self.boundary(probe, env, [], 0, action=None)
        self.assertEqual(probe.evaluator.max_steps_per_episode, 200)
        self.assertEqual(probe.forced_count, 0)
        self.assertEqual(probe.phase, 'prefix')
        self.assertIn(('F','1'), probe.finished)

    def test_missing_handoff_is_not_reported_as_a_completed_case(self):
        _, probe, env, _ = self.make_turn()
        with self.assertRaisesRegex(RuntimeError, 'no validated intervention'):
            probe.finish()

    def test_source_validation_does_not_rewrite_source_files_and_restores_writer(self):
        source = self.source()
        paths = [p for p in source.rglob('*') if p.is_file()]
        before = {p:(p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        original = PROBE.pilot.write_json
        PROBE.source_check(source)
        self.assertIs(PROBE.pilot.write_json, original)
        self.assertEqual({p:(p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)
        report = PROBE.pilot.read_json(source / 'checkpoint_report.json')
        report['locked_cases'] = 999
        PROBE.pilot.write_json(source / 'checkpoint_report.json', report)
        with self.assertRaises(RuntimeError):
            PROBE.source_check(source)
        self.assertIs(PROBE.pilot.write_json, original)

    def test_prepare_locks_both_arms_and_preserves_native_config(self):
        source = self.source()
        root = source.parent / 'turn'
        cfg = fixtures.Config(source / 'habitat.yaml')
        with mock.patch.object(PROBE.pilot, 'load_config', return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(root)))
        _, _, protocol = PROBE.read_run(root)
        self.assertEqual(protocol['arms'], {'left90':90, 'right90':-90})
        self.assertEqual(protocol['charged_main_step_limit'],194)
        self.assertEqual(cfg.eval_settings['max_steps_per_episode'],200)
        self.assertFalse((root / 'control').exists())
        with self.assertRaises(FileExistsError):
            PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(root)))

    def test_run_all_stops_after_first_failure_and_refuses_existing_arm(self):
        root = self.temporary()
        args = SimpleNamespace(run_dir=str(root))
        with mock.patch.object(PROBE, 'read_run'), mock.patch.object(PROBE.subprocess, 'run', side_effect=RuntimeError('failed')) as child:
            with self.assertRaises(RuntimeError):
                PROBE.run_all(args)
            self.assertEqual(child.call_count,1)
        (root / 'left90').mkdir()
        with mock.patch.object(PROBE, 'read_run'), mock.patch.object(PROBE.subprocess, 'run') as child:
            with self.assertRaises(FileExistsError):
                PROBE.run_all(args)
            child.assert_not_called()


    def test_report_separates_rescue_harm_and_cost_without_selecting_the_best_heading(self):
        root, left, env, source = self.make_turn()
        ds, _, references = fixtures.native_fixture()
        cfg = fixtures.Config(source / 'habitat.yaml')
        # Create the same immutable manifest/protocol as the real prepare command.
        # The probe directory already exists, so prepare into another fresh root.
        prepared = source.parent / 'prepared'
        with mock.patch.object(PROBE.pilot, 'load_config', return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(prepared)))
        protocol = PROBE.pilot.read_json(prepared / 'turn_protocol.json')
        source_progress = PROBE.capture.read_progress(source / 'control/progress.json')
        runtime = PROBE.pilot.read_json(source / 'control/pilot_summary.json')
        for arm in PROBE.ARMS:
            (prepared / arm).mkdir()
            env2 = TurningEnv(ds['episodes'][1])
            evaluator = SimpleNamespace(env=env2, model=SimpleNamespace(), max_steps_per_episode=200)
            probe = PROBE.TurnProbe(evaluator, prepared, arm, protocol, references)
            env2.step = probe.record
            with contextlib.redirect_stdout(io.StringIO()):
                for ep in ds['episodes'][1:3]:
                    env2.set_episode(ep)
                    env2.yaw = 0.
                    self.boundary(probe, env2, [], 0, action=None)
                    for i in range(12):
                        env2.step(1)
                    self.boundary(probe, env2, [], 12)
                    env2.step(1)
                    env2.step(0)
                probe.finish()
                probe.status = 'complete'
                probe.close()
            tagged = PROBE.physical.read_lines(prepared / arm / 'turn_actions.jsonl')
            plain = [{k:r[k] for k in ('scene_id','episode_id','action','done','position','orientation_wxyz')} for r in tagged]
            (prepared / arm / 'actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plain))
            progress = copy.deepcopy(list(source_progress.values()))
            if arm == 'left90':
                progress[0]['success'] = 1.
            else:
                progress[1]['success'] = 0.
            (prepared / arm / 'progress.json').write_text(''.join(json.dumps(r)+'\n' for r in progress))
            PROBE.pilot.write_json(prepared / arm / 'pilot_summary.json',runtime)
        with contextlib.redirect_stdout(io.StringIO()):
            result=PROBE.report(SimpleNamespace(run_dir=str(prepared)))
        self.assertEqual(result['counts_by_direction']['left90'],{'rescued':1,'harmed':0})
        self.assertEqual(result['counts_by_direction']['right90'],{'rescued':0,'harmed':1})
        self.assertTrue(all(r['forced_turns']==6 for rows in result['outcomes'].values() for r in rows))
        # A forged saved heading is rejected even if the progress metrics still look good.
        path=prepared/'right90/turn_actions.jsonl'
        rows=PROBE.physical.read_lines(path)
        next(r for r in reversed(rows) if r['phase']=='alignment')['orientation_wxyz']=[1.,0.,0.,0.]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(RuntimeError):
            PROBE.report(SimpleNamespace(run_dir=str(prepared)))


if __name__ == '__main__':
    unittest.main()