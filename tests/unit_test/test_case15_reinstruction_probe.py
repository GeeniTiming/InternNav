"""CPU tests for prompt-only intervention, native-loop compatibility and provenance."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'case15_reinstruction_test', ROOT / 'scripts/eval/case15_reinstruction_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class LoopFixture:
    def native(self):
        step_id, done, observations = 0, False, {'native': True}
        rgb_list, action_seq, pixel_goal, local_actions, action = [], [], None, [], 2
        while (not done) and (step_id <= self.max_steps_per_episode):
            sources = [{'value': 'Original instruction.'}]
            history_id, input_images = [], []
            prompt_instruction = copy.deepcopy(sources[0]["value"])
            self.seen.append(prompt_instruction)
            rgb_list.append(observations)
            step_id += 1
        return self.seen

    def changed(self):
        step_id = 0
        while step_id <= self.max_steps_per_episode:
            prompt_instruction = 'A changed native prompt implementation.'
            step_id += 1


class InstrumentationTests(unittest.TestCase):
    def test_sham_hooks_preserve_native_loop_and_prompt(self):
        boundary = mock.Mock(side_effect=lambda step, obs, done, *rest: (obs, done))
        prompt = mock.Mock(side_effect=lambda step, text, *rest: text)
        one = SimpleNamespace(max_steps_per_episode=2, seen=[])
        two = SimpleNamespace(max_steps_per_episode=2, seen=[],
                              _reinstruction_probe=SimpleNamespace(on_boundary=boundary, amend_prompt=prompt))
        self.assertEqual(LoopFixture.native(one), PROBE.instrument_loop(LoopFixture.native)(two))
        self.assertEqual(boundary.call_count, 3)
        self.assertEqual(prompt.call_count, 3)

    def test_actual_native_dual_system_loop_is_supported_without_importing_habitat(self):
        path = ROOT / 'internnav/habitat_extensions/vln/habitat_vln_evaluator.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == '_run_eval_dual_system')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        instrumented = PROBE.instrument_loop(namespace[node.name])
        self.assertEqual(instrumented.__code__.co_filename, '<case15-reinstruction>')
        self.assertNotIn('recover', instrumented.__code__.co_names)

    def test_native_prompt_shape_drift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape changed'):
            PROBE.instrument_loop(LoopFixture.changed)


class PromptTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        base = Path(self.temporary.name).resolve()
        self.root, self.source = base / 'new', base / 'source'
        for arm in PROBE.ARMS:
            (self.root / arm).mkdir(parents=True)
        (self.source / 'control').mkdir(parents=True)
        self.history = [Image.new('RGB', (3, 3), 'red')]
        self.images = [self.history[0], Image.new('RGB', (3, 3), 'blue')]
        self.observation = {'instruction': {'text': 'Walk.', 'tokens': [1, 2]},
                            'rgb': __import__('numpy').zeros((3, 3, 3), dtype='uint8')}
        self.row = {'scene_id': 'scene', 'episode_id': '15', 'action': 4,
                    'done': False, 'position': [1., 2., 3.],
                    'orientation_wxyz': [1., 0., 0., 0.]}
        sim = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(
            position=[1., 2., 3.], rotation=SimpleNamespace(w=1., x=0., y=0., z=0.)))
        self.step = mock.Mock(return_value=(self.observation, 0, False, {}))
        self.evaluator = SimpleNamespace(
            env=SimpleNamespace(step=self.step, _env=SimpleNamespace(sim=sim),
                                get_metrics=lambda: {'distance_to_goal': 5.},
                                get_current_episode=lambda: SimpleNamespace(scene_id='scene', episode_id='15')),
            model=SimpleNamespace(named_parameters=lambda: iter([])))
        self.boundary = {'trigger_step': 85, 'prefix_action_count': 1}
        self.probes = []
        self.rng_patch = mock.patch.object(PROBE, 'rng_digest', return_value='rng')
        self.rng_patch.start()
        source_gate = {
            'step': 85, 'prefix_action_count': 1,
            'state': {'position': [1., 2., 3.], 'orientation_wxyz': [1., 0., 0., 0.],
                      'distance_to_goal': 5.},
            'observation_hashes': PROBE.physical.observation_hashes(self.observation),
            'history_length': 1, 'history_sha256': PROBE.physical.history_hash(self.history),
        }
        PROBE.pilot.write_json(self.source / 'control/boundary_gate.json', source_gate)

    def tearDown(self):
        for probe in self.probes:
            probe.close()
        self.rng_patch.stop()
        self.temporary.cleanup()

    def probe(self, arm):
        result = PROBE.PromptProbe(self.evaluator, self.root, arm, self.boundary, [self.row], self.source)
        result.records = [copy.deepcopy(self.row)]
        self.probes.append(result)
        return result

    def reach(self, probe, **changed):
        values = dict(step=85, observations=self.observation, done=False, history=self.history,
                      action_seq=[], pixel_goal=None, local_actions=[], action=1)
        values.update(changed)
        return probe.on_boundary(**values)

    def control(self):
        probe = self.probe('control')
        self.reach(probe)
        text = probe.amend_prompt(85, 'Native prompt.', [0], self.images)
        self.assertEqual(text, 'Native prompt.')
        return probe

    def test_two_reminders_are_one_shot_and_execute_no_environment_action(self):
        control = self.control()
        before_history = PROBE.physical.history_hash(self.history)
        source_bytes = (self.source / 'control/boundary_gate.json').read_bytes()
        for arm in ('neutral', 'review'):
            probe = self.probe(arm)
            obs, done = self.reach(probe)
            self.assertIs(obs, self.observation)
            self.assertFalse(done)
            result = probe.amend_prompt(85, 'Native prompt.', [0], self.images)
            self.assertEqual(result, 'Native prompt. ' + PROBE.PROMPTS[arm])
            self.assertEqual(probe.amend_prompt(85, 'Look-down continuation.', [0], self.images),
                             'Look-down continuation.')
            self.assertEqual(probe.amend_prompt(86, 'Next native prompt.', [0], self.images),
                             'Next native prompt.')
            self.assertEqual(probe.recovery_actions, 0)
            self.assertEqual(probe.prompt_event['extra_env_actions_at_insertion'], 0)
        self.step.assert_not_called()
        self.assertEqual(PROBE.physical.history_hash(self.history), before_history)
        self.assertEqual((self.source / 'control/boundary_gate.json').read_bytes(), source_bytes)
        self.assertEqual(control.prompt_event['reminder'], '')

    def test_prompt_image_or_history_context_mismatch_is_rejected(self):
        self.control()
        probe = self.probe('review')
        self.reach(probe)
        for field in ('text', 'images', 'history'):
            text = 'Changed.' if field == 'text' else 'Native prompt.'
            images = [Image.new('RGB', (3, 3), 'green')] if field == 'images' else self.images
            ids = [1] if field == 'history' else [0]
            with self.assertRaisesRegex(RuntimeError, 'pre-prompt context'):
                probe.amend_prompt(85, text, ids, images)
            self.assertIsNone(probe.prompt_event)
        self.step.assert_not_called()

    def test_changed_native_prefix_or_pending_local_plan_is_rejected(self):
        probe = self.probe('neutral')
        for field, value in [('action_seq', [1]), ('pixel_goal', [20, 30]),
                             ('local_actions', [1]), ('done', True)]:
            with self.assertRaisesRegex(RuntimeError, 'idle'):
                self.reach(probe, **{field: value})
        probe = self.probe('review')
        probe.records[0]['action'] = 1
        with self.assertRaisesRegex(RuntimeError, 'prefix differs'):
            self.reach(probe)
        self.step.assert_not_called()

    def test_missing_gate_and_changed_rng_are_rejected(self):
        self.control()
        probe = self.probe('review')
        with self.assertRaisesRegex(RuntimeError, 'before boundary'):
            probe.amend_prompt(85, 'Native prompt.', [0], self.images)
        with mock.patch.object(PROBE, 'rng_digest', return_value='different-rng'):
            with self.assertRaisesRegex(RuntimeError, 'rng_sha256'):
                self.reach(probe)
        self.step.assert_not_called()

    def test_parameter_change_invalidates_an_otherwise_completed_arm(self):
        probe = self.control()
        probe.status = 'complete'
        self.evaluator.model.named_parameters = lambda: iter([
            ('changed', SimpleNamespace(shape=(1,), _version=2))])
        probe.close()
        check = PROBE.pilot.read_json(self.root / 'control/reinstruction_summary.json')
        self.assertEqual(check['status'], 'invalid')
        self.assertFalse(check['parameter_versions_unchanged'])

    def test_source_observation_mismatch_fails_before_prompting(self):
        probe = self.probe('neutral')
        with self.assertRaisesRegex(RuntimeError, 'observation_hashes'):
            self.reach(probe, observations={'rgb': __import__('numpy').ones((3, 3, 3))})
        self.assertFalse(probe.triggered)
        self.step.assert_not_called()


class RngDigestTests(unittest.TestCase):
    def test_digest_uses_state_values_is_read_only_and_detects_changed_streams(self):
        import numpy as np
        import random
        import sys
        cpu, cuda = [10], [20]

        class Tensor:
            def __init__(self, values):
                self.values = np.array(values, dtype=np.uint8)
            def cpu(self):
                return self
            def numpy(self):
                return self.values

        fake = SimpleNamespace(get_rng_state=lambda: Tensor(cpu),
                               cuda=SimpleNamespace(get_rng_state_all=lambda: [Tensor(cuda)]))
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with mock.patch.dict(sys.modules, {'torch': fake}):
                first = PROBE.rng_digest()
                self.assertEqual(first, PROBE.rng_digest())  # new wrappers, same underlying values
                self.assertEqual(random.getstate(), python_state)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
                cuda[0] += 1
                self.assertNotEqual(first, PROBE.rng_digest())
                cuda[0] -= 1
                cpu[0] += 1
                self.assertNotEqual(first, PROBE.rng_digest())
                cpu[0] -= 1
                random.random()
                self.assertNotEqual(first, PROBE.rng_digest())
                random.setstate(python_state)
                np.random.rand()
                self.assertNotEqual(first, PROBE.rng_digest())
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class PrepareTests(unittest.TestCase):
    def test_existing_run_is_not_overwritten(self):
        with mock.patch.object(PROBE, 'source_check') as check:
            with self.assertRaises(FileExistsError):
                PROBE.prepare(SimpleNamespace(source_run='unused', run_dir=str(ROOT)))
        check.assert_not_called()

    def test_invalid_physical_source_creates_no_output(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            target = Path(tmp) / 'new'
            with mock.patch.object(PROBE, 'source_check', side_effect=ValueError('invalid source')):
                with self.assertRaisesRegex(ValueError, 'invalid source'):
                    PROBE.prepare(SimpleNamespace(source_run=tmp, run_dir=str(target)))
            self.assertFalse(target.exists())

    def test_preparation_preserves_sources_and_locks_all_three_configs(self):
        class Config:
            def __init__(self):
                self.agent = SimpleNamespace(model_settings={'mode': 'dual_system',
                    'kv_cache_continuation': False, 'model_path': 'unchanged-checkpoint'})
                self.eval_settings = {'output_path': 'old', 'max_steps_per_episode': 500}
            def model_dump(self, **kwargs):
                return {'agent': {'model_settings': self.agent.model_settings},
                        'eval_settings': self.eval_settings}

        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            source, target = Path(tmp) / 'source', Path(tmp) / 'new'
            for arm in ('control', 'repair'):
                folder = source / arm
                folder.mkdir(parents=True)
                for name in ('actions.jsonl', 'probe_actions.jsonl', 'pilot_summary.json',
                             'boundary_gate.json', 'recovery_summary.json'):
                    (folder / name).write_text('{}', encoding='utf-8')
            (source / 'repair/after_recovery.json').write_text('{}')
            (source / 'manifest.json').write_text('{}')
            snapshots = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            old = {'seed_base': 20261024, 'episodes': [{'scene_id': 'scene', 'episode_id': '15'}],
                   'source_hashes': {}, 'generated_hashes': {}, 'checkpoint_shard_sizes': {}}
            result = (old, {'trigger_step': 85, 'prefix_action_count': 446}, [], {}, {}, {})
            with mock.patch.object(PROBE, 'source_check', return_value=result), \
                    mock.patch.object(PROBE.pilot, 'load_config', return_value=Config()):
                PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target)))
            manifest = PROBE.pilot.verify_manifest(target)
            protocol = PROBE.pilot.read_json(target / 'reinstruction_protocol.json')
            self.assertEqual(protocol['prompts'], PROBE.PROMPTS)
            self.assertEqual(protocol['arms'], list(PROBE.ARMS))
            for arm in PROBE.ARMS:
                tree = ast.parse((target / (arm + '.py')).read_text(encoding='utf-8'))
                literal = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Attribute) and n.func.attr == 'loads')
                cfg = json.loads(literal)
                self.assertFalse(cfg['agent']['model_settings']['kv_cache_continuation'])
                self.assertEqual(cfg['eval_settings']['max_steps_per_episode'], 500)
                self.assertEqual(cfg['agent']['model_settings']['model_path'], 'unchanged-checkpoint')
                self.assertEqual(cfg['eval_settings']['output_path'], str(target.resolve() / arm))
            self.assertTrue(manifest['source_hashes'])
            self.assertEqual({p: p.read_bytes() for p in snapshots}, snapshots)


if __name__ == '__main__':
    unittest.main()
