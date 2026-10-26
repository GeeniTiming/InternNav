"""CPU protocol tests; no model, Habitat, CUDA, or navigation rollout."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('physical_recovery_test', ROOT / 'scripts/eval/physical_recovery_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class LoopFixture:
    def native(self):
        step_id, done, observations = 0, False, {'native': True}
        rgb_list, action_seq, pixel_goal, local_actions, action = [], [], None, [], 2
        while (not done) and (step_id <= self.max_steps_per_episode):
            rgb_list.append(observations['native'])
            self.seen.append((step_id, observations['native']))
            step_id += 1
        return self.seen


class ProtocolTests(unittest.TestCase):
    def episode(self):
        return {'episode_id': 15, 'scene_id': 'mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb',
                'instruction': {'instruction_text': PROBE.INSTRUCTION},
                'goals': [{'position': [5.932779788970947, .17162801325321198, -2.3794898986816406],
                           'radius': 3.0}],
                'reference_path': [[17.45949935913086, .17162801325321198, .21568399667739868],
                                   [14.38700008392334, .17162801325321198, -.21611900627613068],
                                   [12.461600303649902, .17162801325321198, -2.3902199268341064],
                                   [9.91016960144043, .17162801325321198, -2.424109935760498],
                                   [7.576839923858643, .17162801325321198, -2.444770097732544],
                                   [5.932779788970947, .17162801325321198, -2.3794898986816406]]}

    def trace(self):
        return [dict(scene_id='mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb', episode_id='15',
                     action=4, done=False,
                     position=[12.230002403259277, .17162801325321198, -3.893124580383301])
                for _ in range(547)]

    def test_locked_target_is_one_metre_after_corner_and_not_at_goal(self):
        import math
        episode, trace = self.episode(), self.trace()
        before = copy.deepcopy((episode, trace))
        protocol = PROBE.make_protocol(episode, trace)
        self.assertAlmostEqual(math.dist(protocol['waypoints'][-1], episode['reference_path'][2]), 1.0)
        self.assertGreater(math.dist(protocol['waypoints'][-1], protocol['goal_position']), 4.0)
        self.assertEqual(protocol['prefix_action_count'], 446)
        self.assertEqual(protocol['trigger_step'], 85)
        self.assertEqual((episode, trace), before)

    def test_annotation_and_prefix_mismatch_are_rejected(self):
        episode, trace = self.episode(), self.trace()
        for field, value in [('episode_id', 16), ('scene_id', 'mp3d/other/other.glb')]:
            changed = copy.deepcopy(episode)
            changed[field] = value
            with self.assertRaises(ValueError):
                PROBE.make_protocol(changed, trace)
        changed = copy.deepcopy(episode)
        changed['instruction']['instruction_text'] = PROBE.INSTRUCTION.strip()
        with self.assertRaises(ValueError):
            PROBE.make_protocol(changed, trace)
        with self.assertRaises(ValueError):
            PROBE.make_protocol(episode, trace[:-1])
        trace[445]['position'][0] += 1
        with self.assertRaises(ValueError):
            PROBE.make_protocol(episode, trace)


class DriveTests(unittest.TestCase):
    def test_waypoint_stop_is_consumed_without_ending_the_episode(self):
        position, actions = [0., 0., 0.], []
        next_actions = iter([1, 0, 1, 0])

        def advance(action):
            actions.append(action)
            position[0] += .25
            return {}, 0, False, {}

        count = PROBE.drive_waypoints([[.25, 0, 0], [.5, 0, 0]], lambda goal: next(next_actions),
                                      advance, lambda: position, 4, .01, .31)
        self.assertEqual(count, 2)
        self.assertEqual(actions, [1, 1])
        self.assertNotIn(0, actions)

    def test_early_stop_and_no_path_do_not_execute_an_action(self):
        advance = mock.Mock()
        for action in (0, None, 4, 5):
            with self.assertRaises(RuntimeError):
                PROBE.drive_waypoints([[1, 0, 0]], lambda goal: action, advance,
                                      lambda: [0, 0, 0], 4, .35, .31)
        advance.assert_not_called()

    def test_budget_and_environment_termination_are_failures(self):
        advance = mock.Mock(return_value=({}, 0, False, {}))
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            PROBE.drive_waypoints([[1, 0, 0]], lambda goal: 1, advance,
                                  lambda: [0, 0, 0], 0, .35, .31)
        advance.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'terminated'):
            PROBE.drive_waypoints([[1, 0, 0]], lambda goal: 1,
                                  lambda action: ({}, 0, True, {}), lambda: [0, 0, 0], 4, .35, .31)

    def test_large_pose_jump_is_not_accepted_as_physical_motion(self):
        position = [0, 0, 0]

        def advance(action):
            position[0] += 2
            return {}, 0, False, {}

        with self.assertRaisesRegex(RuntimeError, 'one native action'):
            PROBE.drive_waypoints([[3, 0, 0]], lambda goal: 1, advance,
                                  lambda: position, 4, .35, .31)


class InstrumentationTests(unittest.TestCase):
    def test_noop_callback_preserves_native_execution(self):
        seen = []

        def boundary(*args):
            seen.append(args[0])
            return args[1], args[2]

        evaluator = SimpleNamespace(max_steps_per_episode=2, seen=[],
                                    _physical_recovery_probe=SimpleNamespace(on_boundary=boundary))
        native = LoopFixture.native(evaluator)
        evaluator.seen = []
        wrapped = PROBE.instrument_loop(LoopFixture.native)
        self.assertEqual(wrapped(evaluator), native)
        self.assertEqual(seen, [0, 1, 2])

    def test_callback_observation_is_used_in_same_iteration(self):
        def boundary(step, observations, done, *unused):
            return ({'native': False} if step == 1 else observations), done

        evaluator = SimpleNamespace(max_steps_per_episode=1, seen=[],
                                    _physical_recovery_probe=SimpleNamespace(on_boundary=boundary))
        self.assertEqual(PROBE.instrument_loop(LoopFixture.native)(evaluator), [(0, True), (1, False)])

    def test_actual_native_method_can_be_instrumented_without_editing_source(self):
        path = ROOT / 'internnav/habitat_extensions/vln/habitat_vln_evaluator.py'
        before = path.read_bytes()
        module = ast.parse(before.decode('utf-8'))
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'HabitatVLNEvaluator')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_run_eval_dual_system')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        instrumented = PROBE.instrument_loop(namespace[method.name])
        self.assertIn('_physical_recovery_probe', instrumented.__code__.co_names)
        self.assertEqual(path.read_bytes(), before)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'control').mkdir()
        (self.root / 'repair').mkdir()
        self.pilot = PROBE.sibling('cache_pilot')
        self.history = [Image.new('RGB', (4, 4), (10, 20, 30))]
        self.observations = {'rgb': __import__('numpy').zeros((4, 4, 3), dtype='uint8')}
        self.protocol = {'trigger_step': 85, 'prefix_action_count': 1, 'history_rule': 'unchanged'}
        self.entry = dict(scene_id='scene', episode_id='15', action=4, done=False,
                          position=[1., 2., 3.], orientation_wxyz=[1., 0., 0., 0.])
        self.evaluator = SimpleNamespace(env=SimpleNamespace(step=mock.Mock()),
                                         model=SimpleNamespace(named_parameters=lambda: []))
        self.probes = []

    def probe(self, arm='control'):
        probe = PROBE.BoundaryProbe(self.evaluator, self.root, arm, self.pilot,
                                    self.protocol, [self.entry])
        probe.records.append(dict(self.entry))
        probe.state = lambda: dict(position=[1., 2., 3.], orientation_wxyz=[1., 0., 0., 0.],
                                   distance_to_goal=8.)
        self.probes.append(probe)
        return probe

    def tearDown(self):
        for probe in self.probes:
            probe.stream.close()

    def boundary(self, probe, action=2):
        return probe.on_boundary(85, self.observations, False, self.history, [], None, [], action)

    def test_control_is_a_sham_and_discarded_previous_plan_action_is_allowed(self):
        probe = self.probe()
        probe.recover = mock.Mock()
        observations, done = self.boundary(probe)
        self.assertIs(observations, self.observations)
        self.assertFalse(done)
        self.assertTrue(probe.triggered)
        probe.recover.assert_not_called()
        self.evaluator.env.step.assert_not_called()
        self.assertTrue((self.root / 'control/boundary_gate.json').exists())

    def test_prefix_mismatch_and_active_plan_fail_before_motion(self):
        probe = self.probe()
        probe.records[0] = dict(self.entry, action=1)
        probe.recover = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'native prefix'):
            self.boundary(probe)
        probe.recover.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'decision boundary'):
            probe.on_boundary(85, self.observations, False, self.history, [], [20, 20], [], 1)

    def test_repair_requires_exact_same_current_view_and_history(self):
        control = self.probe()
        self.boundary(control)
        repair = self.probe('repair')
        repair.recover = mock.Mock()
        self.observations['rgb'][0, 0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, 'observation_hashes'):
            self.boundary(repair)
        repair.recover.assert_not_called()

    def test_repair_returns_fresh_observation_and_never_clears_history(self):
        control = self.probe()
        self.boundary(control)
        repair = self.probe('repair')
        fresh = {'rgb': self.observations['rgb'] + 20}
        repair.recover = mock.Mock(return_value=fresh)
        before = PROBE.history_hash(self.history)
        with mock.patch.object(PROBE, 'preserve_policy_rng', __import__('contextlib').nullcontext):
            observations, done = self.boundary(repair)
        self.assertIs(observations, fresh)
        self.assertFalse(done)
        self.assertEqual(PROBE.history_hash(self.history), before)
        self.assertEqual(repair.phase, 'native_after_recovery')
        self.assertEqual(repair.post_boundary['observation_hashes'], PROBE.observation_hashes(fresh))
        # A second callback at the same step must not perform another correction.
        self.boundary(repair)
        self.assertEqual(repair.recover.call_count, 1)

    def test_history_mutation_during_correction_is_rejected(self):
        self.boundary(self.probe())
        repair = self.probe('repair')

        def corrupt(observations):
            self.history.clear()
            return observations

        repair.recover = corrupt
        with mock.patch.object(PROBE, 'preserve_policy_rng', __import__('contextlib').nullcontext):
            with self.assertRaisesRegex(RuntimeError, 'History was mutated'):
                self.boundary(repair)

    def test_observation_fingerprint_includes_dtype_shape_and_bytes(self):
        import numpy as np
        first = {'x': np.array([1, 2], dtype=np.int32)}
        self.assertNotEqual(PROBE.observation_hashes(first),
                            PROBE.observation_hashes({'x': np.array([1, 2], dtype=np.float32)}))
        self.assertNotEqual(PROBE.observation_hashes(first),
                            PROBE.observation_hashes({'x': np.array([[1, 2]], dtype=np.int32)}))

    def test_nested_instruction_observation_is_supported_without_dropping_it(self):
        one = {'instruction': {'text': 'Walk.', 'tokens': [1, 2, 3]}}
        two = {'instruction': {'text': 'Stop.', 'tokens': [1, 2, 3]}}
        self.assertNotEqual(PROBE.observation_hashes(one), PROBE.observation_hashes(two))
        self.assertEqual(PROBE.observation_hashes(one), PROBE.observation_hashes(copy.deepcopy(one)))


class RngTests(unittest.TestCase):
    def test_all_policy_rng_streams_are_restored_even_on_error(self):
        import random
        import numpy as np
        import sys
        state = {'cpu': 10, 'cuda': [20, 30]}
        fake = SimpleNamespace(
            get_rng_state=lambda: state['cpu'],
            set_rng_state=lambda value: state.update(cpu=value),
            cuda=SimpleNamespace(
                get_rng_state_all=lambda: list(state['cuda']),
                set_rng_state_all=lambda value: state.update(cuda=list(value))))
        random.seed(7)
        np.random.seed(7)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        with mock.patch.dict(sys.modules, {'torch': fake}):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with PROBE.preserve_policy_rng():
                    random.random()
                    np.random.rand()
                    state.update(cpu=100, cuda=[200, 300])
                    raise RuntimeError('fixture')
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before_numpy[0])
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        self.assertEqual(after_numpy[2:], before_numpy[2:])
        self.assertEqual(state, {'cpu': 10, 'cuda': [20, 30]})


class ReferenceSchemaTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(scene_id='scene', episode_id='15', action=4, done=False,
                     position=[1., 2., 3.], orientation_wxyz=[.5, .5, .5, .5]) for _ in range(12)]

    def test_reported_typo_requires_explicit_flag_and_preserves_all_values(self):
        rows = self.rows()
        expected = copy.deepcopy(rows)
        rows[10]['o rientation_wxyz'] = rows[10].pop('orientation_wxyz')
        before = copy.deepcopy(rows)
        with self.assertRaisesRegex(ValueError, 'repair-known-key-typo'):
            PROBE.normalize_reference(rows)
        normalized, edits = PROBE.normalize_reference(rows, True)
        self.assertEqual(normalized, expected)
        self.assertEqual(rows, before)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]['record_index_zero_based'], 10)
        self.assertEqual(edits[0]['quaternion_value_unchanged'], expected[10]['orientation_wxyz'])

    def test_clean_reference_has_no_repairs(self):
        rows = self.rows()
        normalized, edits = PROBE.normalize_reference(rows)
        self.assertEqual(normalized, rows)
        self.assertEqual(edits, [])
        normalized[0]['position'][0] = 99
        self.assertEqual(rows[0]['position'][0], 1.)

    def test_other_indices_ambiguous_keys_and_other_missing_keys_are_rejected(self):
        for index, field in ((9, 'o rientation_wxyz'), (10, 'rotation_wxyz')):
            rows = self.rows()
            rows[index][field] = rows[index].pop('orientation_wxyz')
            with self.assertRaises(ValueError):
                PROBE.normalize_reference(rows, True)
        rows = self.rows()
        rows[10]['o rientation_wxyz'] = list(rows[10]['orientation_wxyz'])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            PROBE.normalize_reference(rows, True)

    def test_missing_nonfinite_malformed_or_nonunit_pose_is_not_filled(self):
        for bad in ([1., 0., 0.], [0., 0., 0., 0.], [float('nan'), 0., 0., 1.],
                    [1., 0., 0., float('inf')], [True, 0., 0., 0.]):
            rows = self.rows()
            rows[10]['o rientation_wxyz'] = bad
            rows[10].pop('orientation_wxyz')
            with self.assertRaises(ValueError):
                PROBE.normalize_reference(rows, True)
        rows = self.rows()
        rows[10].pop('orientation_wxyz')
        with self.assertRaisesRegex(ValueError, 'missing'):
            PROBE.normalize_reference(rows, True)

    def test_invalid_source_is_rejected_during_prepare_before_output_or_capture(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp).resolve()
            source, target = root / 'source', root / 'new'
            (source / 'control').mkdir(parents=True)
            rows = self.rows()
            rows[10].pop('orientation_wxyz')
            path = source / 'control/actions.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            before = path.read_bytes()
            pilot = PROBE.sibling('cache_pilot')
            pilot.verify_manifest = lambda path: {}
            pilot.read_json = lambda path: ({'status': 'complete', 'stage_synchronized': False}
                                            if path.name == 'pilot_summary.json'
                                            else {'episodes': [ProtocolTests().episode()]})
            capture = SimpleNamespace(prepare=mock.Mock())
            with mock.patch.object(PROBE, 'sibling', side_effect=lambda name: pilot if name == 'cache_pilot' else capture):
                with self.assertRaisesRegex(ValueError, 'missing'):
                    PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target)))
            capture.prepare.assert_not_called()
            self.assertFalse(target.exists())
            self.assertEqual(path.read_bytes(), before)

    def test_previous_prefix_verifies_repaired_quaternion_against_independent_record(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp).resolve()
            source, previous = root / 'source', root / 'previous'
            (source / 'control').mkdir(parents=True)
            (previous / 'control').mkdir(parents=True)
            rows = self.rows()
            raw = source / 'control/actions.jsonl'
            raw.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            pilot = PROBE.sibling('cache_pilot')
            pilot.write_json(previous / 'manifest.json', {
                'physical_recovery_probe': {'source_run': str(source)},
                'source_hashes': {str(raw): pilot.sha256(raw)}})
            captured = previous / 'control/probe_actions.jsonl'
            captured.write_text(json.dumps(dict(rows[0], phase='native')) + '\n')
            result = PROBE.check_previous_prefix(previous, source, rows,
                                                  {'prefix_action_count': 1}, pilot)
            self.assertTrue(result['comparison']['actions_and_poses_match'])
            changed = dict(rows[0], phase='native', orientation_wxyz=[1., 0., 0., 0.])
            captured.write_text(json.dumps(changed) + '\n')
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                PROBE.check_previous_prefix(previous, source, rows, {'prefix_action_count': 1}, pilot)
            # A later source edit cannot bypass the original failed-attempt hash.
            raw.write_text(raw.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Source bytes differ'):
                PROBE.check_previous_prefix(previous, source, rows, {'prefix_action_count': 1}, pilot)


if __name__ == '__main__':
    unittest.main()
