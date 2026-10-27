"""CPU tests for the two-position/three-heading intervention and actual turn controller."""
import copy
import importlib.util
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import contextlib

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('pose_factorial_test', ROOT / 'scripts/eval/case15_pose_factorial.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def fixture():
    q = PROBE.yaw_quaternion(30)
    rows = [dict(scene_id='scene', episode_id='15', action=4, done=False,
                 position=[12.230002403259277, .17162801325321198, -3.893124580383301],
                 orientation_wxyz=list(q)) for _ in range(547)]
    for row in rows[:10]:
        row['orientation_wxyz'] = PROBE.yaw_quaternion(15)
    rows[10]['action'] = 2
    rows[393]['action'] = 1
    rows[393]['position'] = [12.738821983337402, .17162801325321198, -2.49418568611145]
    boundary = {'prefix_action_count': 446, 'trigger_step': 85, 'goal_radius_m': 3.,
                'handoff_goal_exclusion_m': 4., 'history_rule': 'Keep native history.'}
    return rows, boundary


class ProtocolTests(unittest.TestCase):
    def test_collapsed_original_headings_get_a_distinct_full_factorial(self):
        rows, boundary = fixture()
        before = copy.deepcopy(rows)
        protocol = PROBE.make_protocol(rows, boundary)
        self.assertAlmostEqual(protocol['original_anchor_vs_trigger_heading_difference_deg'], 0)
        self.assertEqual(len(protocol['candidates']), 6)
        for group in (('control', 'turn_left90', 'turn_right90'),
                      ('anchor_same_heading', 'anchor_left90', 'anchor_right90')):
            states = [protocol['candidates'][a] for a in group]
            self.assertEqual(states[0]['position'], states[1]['position'])
            self.assertEqual(states[0]['position'], states[2]['position'])
            self.assertAlmostEqual(PROBE.heading_error(states[0]['orientation_wxyz'],
                                                       states[1]['orientation_wxyz']), 90)
            self.assertAlmostEqual(PROBE.heading_error(states[0]['orientation_wxyz'],
                                                       states[2]['orientation_wxyz']), 90)
        self.assertEqual(rows, before)

    def test_targets_do_not_depend_on_future_rollout_or_new_reference_segment(self):
        rows, boundary = fixture()
        expected = PROBE.make_protocol(rows, boundary)
        for row in rows[446:]:
            row['position'] = [999., 999., 999.]
            row['orientation_wxyz'] = PROBE.yaw_quaternion(-120)
        self.assertEqual(PROBE.make_protocol(rows, boundary), expected)
        self.assertEqual(expected['candidates']['anchor_left90']['position'], rows[393]['position'])

    def test_changed_anchor_boundary_and_turn_grid_are_rejected(self):
        rows, boundary = fixture()
        changed = copy.deepcopy(rows)
        changed[393]['position'][0] += .1
        with self.assertRaisesRegex(ValueError, 'anchor changed'):
            PROBE.make_protocol(changed, boundary)
        with self.assertRaisesRegex(ValueError, '446-action'):
            PROBE.make_protocol(rows, dict(boundary, prefix_action_count=445))
        changed = copy.deepcopy(rows)
        changed[10]['orientation_wxyz'] = PROBE.yaw_quaternion(45)
        with self.assertRaisesRegex(ValueError, '15-degree'):
            PROBE.make_protocol(changed, boundary)

    def test_factorial_rejects_different_positions_or_wrong_heading(self):
        rows, boundary = fixture()
        protocol = PROBE.make_protocol(rows, boundary)
        states = {a: {k: copy.deepcopy(c[k]) for k in ('position', 'orientation_wxyz')}
                  for a, c in protocol['candidates'].items()}
        self.assertTrue(PROBE.check_factorial(states, protocol)['matched_headings_across_positions'])
        changed = copy.deepcopy(states)
        changed['anchor_left90']['position'][0] += .01
        with self.assertRaisesRegex(RuntimeError, 'positions differ'):
            PROBE.check_factorial(changed, protocol)
        changed = copy.deepcopy(states)
        changed['turn_right90']['orientation_wxyz'] = PROBE.yaw_quaternion(30)
        with self.assertRaisesRegex(RuntimeError, 'heading target'):
            PROBE.check_factorial(changed, protocol)

    def test_yaw_is_sign_invariant_and_wraps_across_180_degrees(self):
        for angle in (-179, -90, 0, 90, 179, 359):
            q = PROBE.yaw_quaternion(angle)
            self.assertAlmostEqual(PROBE.heading_error(q, [-v for v in q]), 0)
        self.assertAlmostEqual(PROBE.heading_error(PROBE.yaw_quaternion(179),
                                                   PROBE.yaw_quaternion(-179)), 2)
        with self.assertRaises(ValueError):
            PROBE.yaw_degrees([1, .1, 0, 0])


class TurnTests(unittest.TestCase):
    def controller(self, angle=30, actual_step=15, drift=0, done=False):
        pose = {'position': [1., 2., 3.], 'orientation_wxyz': PROBE.yaw_quaternion(angle)}
        actions = []

        def state():
            return copy.deepcopy(pose)

        def advance(action):
            actions.append(action)
            yaw = PROBE.yaw_degrees(pose['orientation_wxyz'])
            pose['orientation_wxyz'] = PROBE.yaw_quaternion(yaw + (actual_step if action == 2 else -actual_step))
            pose['position'][0] += drift
            return {'fresh': len(actions)}, 0, done, {}
        return pose, actions, state, advance

    def test_left_and_right_90_are_six_real_turns_without_translation_or_stop(self):
        for offset, action in ((90, 2), (-90, 3)):
            pose, actions, state, advance = self.controller()
            count = PROBE.turn_to_heading(PROBE.yaw_quaternion(30 + offset), state, advance, 10)
            self.assertEqual(count, 6)
            self.assertEqual(actions, [action] * 6)
            self.assertEqual(pose['position'], [1., 2., 3.])
            self.assertNotIn(0, actions)

    def test_half_turn_uses_bounded_native_steps_and_same_heading_needs_none(self):
        pose, actions, state, advance = self.controller(179)
        self.assertEqual(PROBE.turn_to_heading(PROBE.yaw_quaternion(-1), state, advance, 12), 12)
        self.assertLess(PROBE.heading_error(pose['orientation_wxyz'], PROBE.yaw_quaternion(-1)), .05)
        before = list(actions)
        self.assertEqual(PROBE.turn_to_heading(pose['orientation_wxyz'], state, advance, 0), 0)
        self.assertEqual(actions, before)

    def test_off_grid_target_fails_before_action_and_budget_is_enforced(self):
        _, actions, state, advance = self.controller()
        with self.assertRaisesRegex(RuntimeError, 'off the native turn grid'):
            PROBE.turn_to_heading(PROBE.yaw_quaternion(37), state, advance, 10)
        self.assertFalse(actions)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            PROBE.turn_to_heading(PROBE.yaw_quaternion(120), state, advance, 1)
        self.assertEqual(len(actions), 1)

    def test_translation_during_turn_wrong_angle_and_termination_are_rejected(self):
        for settings, message in (({'drift': .001}, 'changed position'),
                                  ({'actual_step': 30}, 'turn angle'),
                                  ({'done': True}, 'episode ended')):
            _, actions, state, advance = self.controller(**settings)
            with self.assertRaisesRegex(RuntimeError, message):
                PROBE.turn_to_heading(PROBE.yaw_quaternion(120), state, advance, 10)
            self.assertEqual(len(actions), 1)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.base = Path(self.temporary.name)
        self.root, self.source = self.base / 'new', self.base / 'source'
        for arm in PROBE.ARMS:
            (self.root / arm).mkdir(parents=True)
        (self.source / 'control').mkdir(parents=True)
        self.history = [Image.new('RGB', (3, 3), 'red')]
        self.obs = {'rgb': __import__('numpy').zeros((3, 3, 3), dtype='uint8')}
        self.state = {'position': [1., 2., 3.], 'orientation_wxyz': [1., 0., 0., 0.],
                      'distance_to_goal': 8.}
        self.row = dict(scene_id='scene', episode_id='15', action=4, done=False,
                        position=[1., 2., 3.], orientation_wxyz=[1., 0., 0., 0.])
        self.evaluator = SimpleNamespace(env=SimpleNamespace(step=mock.Mock()),
                                         model=SimpleNamespace(named_parameters=lambda: []))
        self.protocol = {'trigger_step': 85, 'prefix_action_count': 1,
                         'candidates': {a: {} for a in PROBE.ARMS}, 'history_rule': 'unchanged'}
        gate = {'step': 85, 'prefix_action_count': 1, 'state': self.state,
                'observation_hashes': PROBE.physical.observation_hashes(self.obs),
                'history_length': 1, 'history_sha256': PROBE.physical.history_hash(self.history),
                'rng_sha256': 'rng'}
        PROBE.pilot.write_json(self.source / 'control/boundary_gate.json', gate)
        self.probes = []
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(PROBE.prompt, 'rng_digest', return_value='rng'))
        self.stack.enter_context(mock.patch.object(PROBE.physical, 'preserve_policy_rng',
                                                  contextlib.nullcontext))

    def tearDown(self):
        for probe in self.probes:
            probe.close()
        self.stack.close()
        self.temporary.cleanup()

    def probe(self, arm='control'):
        result = PROBE.PoseProbe(self.evaluator, self.root, arm, self.protocol, [self.row], self.source)
        result.records = [copy.deepcopy(self.row)]
        result.state = lambda: copy.deepcopy(self.state)
        self.probes.append(result)
        return result

    def boundary(self, probe):
        return probe.on_boundary(85, self.obs, False, self.history, [], None, [], 1)

    def test_control_is_unchanged_and_correction_happens_once_with_fresh_observation(self):
        control = self.probe()
        control.recover = mock.Mock()
        self.assertIs(self.boundary(control)[0], self.obs)
        control.recover.assert_not_called()
        probe = self.probe('turn_left90')
        fresh = {'rgb': self.obs['rgb'] + 1}
        probe.recover = mock.Mock(return_value=fresh)
        before = PROBE.physical.history_hash(self.history)
        self.assertIs(self.boundary(probe)[0], fresh)
        self.assertEqual(probe.phase, 'native_after_recovery')
        self.boundary(probe)
        self.assertEqual(probe.recover.call_count, 1)
        self.assertEqual(PROBE.physical.history_hash(self.history), before)
        self.evaluator.env.step.assert_not_called()

    def test_native_prefix_or_view_mismatch_blocks_physical_intervention(self):
        self.boundary(self.probe())
        for arm, change in (('turn_left90', 'prefix'), ('turn_right90', 'view')):
            probe = self.probe(arm)
            probe.recover = mock.Mock()
            if change == 'prefix':
                probe.records[0]['action'] = 1
            else:
                self.obs['rgb'][0, 0, 0] = 1
            with self.assertRaisesRegex(RuntimeError, 'NOT_COMPARABLE'):
                self.boundary(probe)
            probe.recover.assert_not_called()

    def test_history_corruption_during_recovery_is_not_accepted(self):
        self.boundary(self.probe())
        probe = self.probe('turn_left90')

        def corrupt(obs):
            self.history.clear()
            return obs
        probe.recover = corrupt
        with self.assertRaisesRegex(RuntimeError, 'History changed'):
            self.boundary(probe)


class PipelineTests(unittest.TestCase):
    def temporary(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        result = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(result.cleanup)
        return Path(result.name).resolve()

    def test_prepare_is_create_only_and_invalid_source_creates_no_output(self):
        base = self.temporary()
        with mock.patch.object(PROBE, 'source_check') as check:
            with self.assertRaises(FileExistsError):
                PROBE.prepare(SimpleNamespace(source_run=str(base), run_dir=str(base)))
            check.assert_not_called()
        target = base / 'new'
        with mock.patch.object(PROBE, 'source_check', side_effect=ValueError('bad source')):
            with self.assertRaisesRegex(ValueError, 'bad source'):
                PROBE.prepare(SimpleNamespace(source_run=str(base), run_dir=str(target)))
        self.assertFalse(target.exists())

    def test_run_all_stops_after_first_child_failure(self):
        base = self.temporary()
        with mock.patch.object(PROBE, 'read_run'), mock.patch.object(PROBE, 'report') as report:
            failure = subprocess.CalledProcessError(1, 'fixture')
            with mock.patch.object(PROBE.subprocess, 'run', side_effect=[SimpleNamespace(returncode=0), failure]) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    PROBE.run_all(SimpleNamespace(run_dir=str(base)))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][-1], 'control')
        self.assertEqual(run.call_args_list[1].args[0][-1], 'turn_left90')
        self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
        report.assert_not_called()

    def test_run_all_rejects_existing_arm_without_starting_a_process(self):
        base = self.temporary()
        (base / 'control').mkdir()
        with mock.patch.object(PROBE, 'read_run'), mock.patch.object(PROBE.subprocess, 'run') as run:
            with self.assertRaises(FileExistsError):
                PROBE.run_all(SimpleNamespace(run_dir=str(base)))
        run.assert_not_called()

    def test_prepare_locks_six_configs_and_preserves_native_budgets_and_source_files(self):
        class Config:
            def __init__(self):
                self.agent = SimpleNamespace(model_settings={'mode': 'dual_system',
                    'kv_cache_continuation': False, 'model_path': 'unchanged'})
                self.eval_settings = {'max_steps_per_episode': 500, 'output_path': 'old'}
            def model_dump(self, **kwargs):
                return {'agent': {'model_settings': self.agent.model_settings},
                        'eval_settings': self.eval_settings}

        base = self.temporary()
        source, target = base / 'source', base / 'new'
        source.mkdir()
        (source / 'manifest.json').write_text('{}')
        (source / 'reinstruction_report.json').write_text('{}')
        for arm in PROBE.prompt.ARMS:
            (source / arm).mkdir()
            for name in ('actions.jsonl', 'probe_actions.jsonl', 'pilot_summary.json',
                         'boundary_gate.json', 'prompt_event.json', 'reinstruction_summary.json'):
                (source / arm / name).write_text('{}')
        snapshots = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        rows, boundary = fixture()
        old = {'seed_base': 20261024, 'episodes': [{'scene_id': 'scene', 'episode_id': '15'}],
               'source_hashes': {}, 'generated_hashes': {}, 'checkpoint_shard_sizes': {}}
        with mock.patch.object(PROBE, 'source_check', return_value=(old, boundary, rows, {}, {})), \
                mock.patch.object(PROBE.pilot, 'load_config', return_value=Config()):
            PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target)))
        PROBE.pilot.verify_manifest(target)
        protocol = PROBE.pilot.read_json(target / 'pose_protocol.json')
        self.assertEqual(protocol['arms'], list(PROBE.ARMS))
        import ast
        for arm in PROBE.ARMS:
            tree = ast.parse((target / (arm + '.py')).read_text(encoding='utf-8'))
            literal = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == 'loads')
            cfg = __import__('json').loads(literal)
            self.assertEqual(cfg['eval_settings']['max_steps_per_episode'], 500)
            self.assertFalse(cfg['agent']['model_settings']['kv_cache_continuation'])
        self.assertEqual({p: p.read_bytes() for p in snapshots}, snapshots)


if __name__ == '__main__':
    unittest.main()
