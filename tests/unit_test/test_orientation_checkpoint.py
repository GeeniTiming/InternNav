"""CPU tests for unmodified replay, first-boundary capture and result gates."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('orientation_checkpoint_test', ROOT / 'scripts/eval/orientation_checkpoint.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def yaml_stub():
    return SimpleNamespace(safe_load=lambda text: json.loads('\n'.join(
        line for line in text.splitlines() if not line.startswith('#'))),
        safe_dump=lambda value, **kw: json.dumps(value))


class Config:
    def __init__(self, habitat):
        self.eval_type = 'habitat_vln'
        self.agent = SimpleNamespace(model_settings={'mode': 'dual_system', 'kv_cache_continuation': False})
        self.env = SimpleNamespace(env_settings={'config_path': str(habitat)})
        self.eval_settings = {'max_steps_per_episode': 200}

    def model_dump(self, **kw):
        return {'eval_type': self.eval_type, 'agent': {'model_settings': self.agent.model_settings},
                'env': {'env_settings': self.env.env_settings}, 'eval_settings': self.eval_settings}


def native_fixture():
    episodes, progress, traces = [], {}, {}
    specs = [('zsNo4HB9uLZ', 15, 1, 0), ('F', 1, 2, 0), ('S', 2, 3, 1)]
    specs += [('zsNo4HB9uLZ', 100+i, 1, 1) for i in range(17)]
    for scene, eid, trajectory, success in specs:
        ep = {'scene_id': 'mp3d/' + scene + '/' + scene + '.glb', 'episode_id': eid,
              'trajectory_id': trajectory, 'instruction': {'instruction_text': 'Walk ' + str(eid) + '.'},
              'start_position': [0., 0., 0.], 'reference_path': [[trajectory, 0., 0.], [trajectory+1., 0., 0.]]}
        key = scene, str(eid)
        episodes.append(ep)
        progress[key] = {'scene_id': scene, 'episode_id': eid, 'episode_instruction': ep['instruction']['instruction_text'],
                         'success': float(success), 'spl': float(success), 'os': float(success),
                         'ne': 1. if success else 8., 'steps': 20}
        traces[key] = [dict(scene_id=ep['scene_id'], episode_id=str(eid), action=1, done=False,
                            position=[.25*(i+1), 0., 0.], orientation_wxyz=[1., 0., 0., 0.]) for i in range(20)]
        traces[key].append(dict(traces[key][-1], action=0, done=True))
    return {'episodes': episodes, 'instruction_vocab': {'word_list': ['walk']}}, progress, traces


class FakeEnv:
    def __init__(self, episode):
        self.calls = []
        self.pitch = 0.
        self._env = SimpleNamespace(sim=SimpleNamespace(get_agent_state=self.agent_state))
        self.set_episode(episode)

    def set_episode(self, episode):
        self.episode = SimpleNamespace(scene_id=episode['scene_id'], episode_id=episode['episode_id'])
        self.position = [0., 0., 0.]
        self.obs = {'rgb': np.zeros((3, 4, 3), dtype=np.uint8), 'depth': np.zeros((3, 4, 1), dtype=np.float32)}

    def agent_state(self):
        half = math.radians(self.pitch)/2
        return SimpleNamespace(position=self.position,
                               rotation=SimpleNamespace(w=1., x=0., y=0., z=0.),
                               sensor_states={'rgb': SimpleNamespace(rotation=SimpleNamespace(
                                   w=math.cos(half), x=math.sin(half), y=0., z=0.))})

    def get_current_episode(self):
        return self.episode

    def get_metrics(self):
        return {'distance_to_goal': .1}  # Goal proximity must not select/reject a checkpoint.

    def step(self, action):
        self.calls.append(action)
        if action == 1:
            self.position[0] += .25
        return self.obs, 0., action == 0, {}


class HelpersTests(unittest.TestCase):
    def test_eligibility_uses_locked_body_threshold_budget_translation_idle_and_level_view(self):
        case = {'native_nonstop_body_actions': 20, 'checkpoint_body_threshold': 12}
        def check(**changes):
            args = dict(case=case, body_count=12, translation=3., done=False, action_seq=[],
                        pixel_goal=None, local_actions=[], action=1, camera_balance=0, camera_pitch=0.)
            args.update(changes)
            return PROBE.eligibility(**args)['eligible']
        self.assertTrue(check())
        for values in ({'body_count': 11}, {'body_count': 15}, {'translation': .99},
                       {'done': True}, {'action_seq': [1]}, {'pixel_goal': [2, 3]},
                       {'local_actions': [1]}, {'action': 5}, {'camera_balance': -2},
                       {'camera_pitch': 15}, {'camera_pitch': None}):
            self.assertFalse(check(**values), values)
        self.assertTrue(check(body_count=14, translation=1.))

    def test_sensor_pitch_is_sign_invariant_and_rejects_nonunit_values(self):
        for angle in (-30, 0, 30):
            q = [math.cos(math.radians(angle)/2), math.sin(math.radians(angle)/2), 0., 0.]
            self.assertAlmostEqual(PROBE.camera_pitch_degrees(q), angle)
            self.assertAlmostEqual(PROBE.camera_pitch_degrees([-v for v in q]), angle)
        with self.assertRaises(ValueError):
            PROBE.camera_pitch_degrees([0., 0., 0., 0.])


class Base(unittest.TestCase):
    def temporary(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        return Path(temp.name).resolve()

    def make_source(self):
        base = self.temporary()
        source, cohort_root = base / 'source', base / 'cohort'
        (source / 'control').mkdir(parents=True)
        cohort_root.mkdir()
        dataset, progress, traces = native_fixture()
        with gzip.open(source / 'episodes.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(dataset, stream)
        for name, rows in (('progress.json', list(progress.values())),
                           ('actions.jsonl', [row for ep in dataset['episodes']
                            for row in traces[PROBE.capture.episode_key(ep)]])):
            (source / 'control' / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        runtime = {'seed_base': 123, 'stage_synchronized': False, 'torch': 'fixture', 'cuda': 'fixture',
                   'transformers': 'fixture', 'device': 'fixture'}
        PROBE.pilot.write_json(source / 'control/pilot_summary.json', dict(runtime, status='complete', metrics={'length': 20}))
        PROBE.pilot.write_json(source / 'manifest.json', {
            'seed_base': 123, 'source_hashes': {}, 'generated_hashes': {}, 'checkpoint_shard_sizes': {}})
        (source / 'control.py').write_text('# fixture config')
        PROBE.pilot.write_json(source / 'habitat.yaml', {'habitat': {'dataset': {'split': 'val_unseen', 'data_path': 'unused'}}})
        plan = PROBE.cohort.select_plan(dataset, progress, traces)
        plan['source_run'] = str(source)
        PROBE.pilot.write_json(cohort_root / 'plan.json', plan)
        watched = list(source.rglob('*')) + [cohort_root / 'plan.json']
        PROBE.pilot.write_json(cohort_root / 'cohort_manifest.json', {
            'schema': 'crosscase_orientation_inputs.v1',
            'hashes': {str(p): PROBE.pilot.sha256(p) for p in watched if p.is_file()}, 'checkpoint_shard_sizes': {}})
        return base, source, cohort_root, Config(source / 'habitat.yaml'), dataset, progress, traces, runtime

    def make_probe(self):
        root = self.temporary()
        (root / 'control').mkdir()
        dataset, progress, traces = native_fixture()
        cases = PROBE.cohort.select_plan(dataset, progress, traces)['cases']
        env = FakeEnv(dataset['episodes'][1])
        evaluator = SimpleNamespace(env=env, model=SimpleNamespace())
        self.parameters = 'weights'
        weights = mock.patch.object(PROBE.common, 'parameter_versions', side_effect=lambda model: self.parameters)
        rng = mock.patch.object(PROBE.common, 'rng_digest', return_value='rng')
        weights.start()
        rng.start()
        self.addCleanup(weights.stop)
        self.addCleanup(rng.stop)
        probe = PROBE.CheckpointProbe(evaluator, root, cases, traces)
        self.addCleanup(probe.close)
        return root, probe, env, dataset, traces

    def boundary(self, probe, env, history, step, **kw):
        args = dict(step=step, observations=env.obs, done=False, history=history,
                    action_seq=[], pixel_goal=None, local_actions=[], action=1)
        args.update(kw)
        return probe.on_boundary(**args)


class PipelineTests(Base):
    def test_prepare_preserves_sources_seed_budget_and_native_mode_and_locks_inputs(self):
        base, source, cohort_root, cfg, *rest = self.make_source()
        before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        root = base / 'capture'
        with mock.patch.dict(sys.modules, {'yaml': yaml_stub()}), \
                mock.patch.object(PROBE.pilot, 'load_config', return_value=cfg), \
                contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(SimpleNamespace(cohort_dir=str(cohort_root), run_dir=str(root)))
        manifest, data = PROBE.read_run(root)
        self.assertEqual(manifest['seed_base'], 123)
        self.assertEqual(len(manifest['episodes']), 2)
        self.assertEqual(PROBE.pilot.read_json(root / 'episodes.json.gz')['instruction_vocab'], {'word_list': ['walk']})
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertEqual(cfg.eval_settings['max_steps_per_episode'], 200)
        self.assertFalse(cfg.agent.model_settings['kv_cache_continuation'])
        self.assertFalse(any((root / arm).exists() for arm in ('control', 'left90', 'right90')))
        self.assertIn(str(Path(PROBE.__file__).resolve()), manifest['source_hashes'])

    def test_changed_source_or_plan_and_existing_run_are_rejected_before_output(self):
        base, source, cohort_root, cfg, *_ = self.make_source()
        with self.assertRaises(FileExistsError):
            PROBE.prepare(SimpleNamespace(cohort_dir='unused', run_dir=str(base)))
        plan = PROBE.pilot.read_json(cohort_root / 'plan.json')
        plan['cases'][0]['checkpoint_body_threshold'] -= 1
        PROBE.pilot.write_json(cohort_root / 'plan.json', plan)
        with self.assertRaisesRegex(RuntimeError, 'Locked cohort input'):
            PROBE.prepare(SimpleNamespace(cohort_dir=str(cohort_root), run_dir=str(base / 'new')))
        self.assertFalse((base / 'new').exists())

    def test_rehashed_modified_rule_still_cannot_change_locked_selection(self):
        base, source, cohort_root, *_ = self.make_source()
        path = cohort_root / 'plan.json'
        plan = PROBE.pilot.read_json(path)
        plan['cases'][0]['checkpoint_body_threshold'] -= 1
        PROBE.pilot.write_json(path, plan)
        lock = PROBE.pilot.read_json(cohort_root / 'cohort_manifest.json')
        lock['hashes'][str(path)] = PROBE.pilot.sha256(path)
        PROBE.pilot.write_json(cohort_root / 'cohort_manifest.json', lock)
        with self.assertRaisesRegex(ValueError, 'selection/rules'):
            PROBE.verify_cohort(cohort_root)

class ObserverTests(Base):
    def test_capture_is_a_sham_at_first_eligible_boundary_and_goal_distance_is_not_used(self):
        root, probe, env, dataset, references = self.make_probe()
        history = [Image.new('RGB', (4, 3), 'white')]
        history_before = PROBE.physical.history_hash(history)
        self.boundary(probe, env, history, 0, action=None)
        for i in range(20):
            probe.record(1)
            observed, done = self.boundary(probe, env, history, i+1)
            self.assertIs(observed, env.obs)
            self.assertFalse(done)
        probe.record(0)
        probe.finish()
        row = probe.finished[('F', '1')]
        self.assertEqual(row['checkpoint']['prefix_action_count'], 12)
        self.assertEqual(row['checkpoint']['prefix_body_actions'], 12)
        self.assertEqual(row['checkpoint']['remaining_baseline_body_actions'], 8)
        self.assertEqual(row['checkpoint']['state']['distance_to_goal_OFFLINE_ONLY'], .1)
        self.assertEqual(env.calls, [1]*20 + [0])
        self.assertEqual(PROBE.physical.history_hash(history), history_before)
        self.assertEqual(len(list((root / 'control/checkpoints').rglob('checkpoint.json'))), 1)
        self.assertEqual(row['native_trace_comparison']['max_position_error_m'], 0.)

    def test_active_plan_defers_capture_to_first_idle_boundary_without_tuning(self):
        _, probe, env, _, _ = self.make_probe()
        history = [Image.new('RGB', (4, 3))]
        self.boundary(probe, env, history, 0, action=None)
        for i in range(20):
            probe.record(1)
            self.boundary(probe, env, history, i+1, pixel_goal=[1, 2] if i+1 in (12, 13) else None)
        probe.record(0)
        probe.finish()
        self.assertEqual(probe.finished[('F', '1')]['checkpoint']['prefix_action_count'], 14)

    def test_no_idle_boundary_stays_ineligible_and_not_replaced(self):
        _, probe, env, _, _ = self.make_probe()
        history = []
        self.boundary(probe, env, history, 0, action=None)
        for i in range(20):
            probe.record(1)
            self.boundary(probe, env, history, i+1, pixel_goal=[1, 2])
        probe.record(0)
        probe.finish()
        self.assertEqual(probe.finished[('F', '1')]['status'], 'NOT_ELIGIBLE')
        self.assertIsNone(probe.finished[('F', '1')]['checkpoint'])

    def test_tilted_sensor_prevents_level_camera_capture(self):
        _, probe, env, _, _ = self.make_probe()
        self.boundary(probe, env, [], 0, action=None)
        env.pitch = -30
        for i in range(20):
            probe.record(1)
            self.boundary(probe, env, [], i+1)
        probe.record(0)
        probe.finish()
        self.assertIsNone(probe.finished[('F', '1')]['checkpoint'])

    def test_native_action_divergence_fails_before_a_checkpoint_is_created(self):
        root, probe, env, _, _ = self.make_probe()
        self.boundary(probe, env, [], 0, action=None)
        with self.assertRaisesRegex(RuntimeError, 'native action/pose'):
            probe.record(3)
        self.assertIsNone(probe.gate)
        self.assertTrue((root / 'control/native_mismatch.json').is_file())

    def test_history_or_parameter_mutation_is_rejected(self):
        _, probe, env, _, _ = self.make_probe()
        history = [Image.new('RGB', (4, 3), 'white')]
        self.boundary(probe, env, history, 0, action=None)
        for i in range(20):
            probe.record(1)
            self.boundary(probe, env, history, i+1)
        probe.record(0)
        history[0].putpixel((0, 0), (1, 2, 3))
        with self.assertRaisesRegex(RuntimeError, 'history prefix changed'):
            probe.finish()
        history[0].putpixel((0, 0), (255, 255, 255))
        self.parameters = 'modified weights'
        with self.assertRaisesRegex(RuntimeError, 'parameter versions'):
            probe.finish()

    def test_rng_change_during_capture_is_rejected(self):
        _, probe, env, _, _ = self.make_probe()
        self.boundary(probe, env, [], 0, action=None)
        for i in range(12):
            probe.record(1)
        with mock.patch.object(PROBE.common, 'rng_digest', side_effect=['before', 'after']):
            with self.assertRaisesRegex(RuntimeError, 'Capture changed'):
                self.boundary(probe, env, [], 12)

    def test_multi_episode_observer_resets_counts_and_preserves_each_complete_trace(self):
        _, probe, env, dataset, references = self.make_probe()
        for ep in dataset['episodes'][1:3]:
            env.set_episode(ep)
            history = [Image.new('RGB', (4, 3))]
            self.boundary(probe, env, history, 0, action=None)
            for i in range(20):
                probe.record(1)
                self.boundary(probe, env, history, i+1)
            probe.record(0)
        probe.finish()
        self.assertEqual(set(probe.finished), {('F', '1'), ('S', '2')})
        self.assertTrue(all(row['checkpoint']['prefix_action_count'] == 12 for row in probe.finished.values()))
        self.assertEqual(len(env.calls), 42)


class ReportTests(Base):
    def build_capture(self, no_eligible=False):
        base, source, cohort_root, cfg, dataset, progress, references, runtime = self.make_source()
        root = base / 'capture'
        with mock.patch.dict(sys.modules, {'yaml': yaml_stub()}), \
                mock.patch.object(PROBE.pilot, 'load_config', return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(SimpleNamespace(cohort_dir=str(cohort_root), run_dir=str(root)))
        (root / 'control').mkdir()
        cases = PROBE.pilot.read_json(root / 'capture_protocol.json')['cases']
        lookup = {PROBE.capture.episode_key(ep): ep for ep in dataset['episodes']}
        env = FakeEnv(lookup[tuple(cases[0]['key'])])
        with mock.patch.object(PROBE.common, 'parameter_versions', return_value='weights'), \
                mock.patch.object(PROBE.common, 'rng_digest', return_value='rng'), contextlib.redirect_stdout(io.StringIO()):
            probe = PROBE.CheckpointProbe(SimpleNamespace(env=env, model=SimpleNamespace()), root, cases, references)
            for case in cases:
                env.set_episode(lookup[tuple(case['key'])])
                self.boundary(probe, env, [], 0, action=None)
                for i in range(20):
                    probe.record(1)
                    self.boundary(probe, env, [], i+1, pixel_goal=[1, 2] if no_eligible else None)
                probe.record(0)
            probe.finish()
            probe.status = 'complete'
            probe.close()
        selected = [tuple(case['key']) for case in cases]
        (root / 'control/progress.json').write_text(''.join(json.dumps(progress[key]) + '\n' for key in selected))
        (root / 'control/actions.jsonl').write_text(''.join(json.dumps(row) + '\n' for key in selected for row in references[key]))
        PROBE.pilot.write_json(root / 'control/pilot_summary.json', dict(runtime, status='complete', metrics={'length': 2}))
        return root

    def test_report_accepts_only_full_native_reproduction_and_declares_no_intervention(self):
        root = self.build_capture()
        with contextlib.redirect_stdout(io.StringIO()):
            result = PROBE.report(SimpleNamespace(run_dir=str(root)))
        self.assertEqual(result['eligible_counts'], {'failure': 1, 'success': 1})
        self.assertEqual(result['status'], 'CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION')
        self.assertFalse(result['orientation_interventions_executed'])
        self.assertEqual(result['extra_env_actions'], 0)
        self.assertTrue(all(row['native_trace_comparison']['actions_and_poses_match'] for row in result['cases']))
        self.assertFalse((root / 'left90').exists())

    def test_report_retains_ineligible_cases_and_stops_if_a_stratum_has_no_checkpoint(self):
        root = self.build_capture(no_eligible=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = PROBE.report(SimpleNamespace(run_dir=str(root)))
        self.assertEqual(result['locked_cases'], 2)
        self.assertEqual(result['status'], 'INSUFFICIENT_ELIGIBLE_STRATA_STOP')
        self.assertTrue(all(row['status'] == 'NOT_ELIGIBLE_KEEP_LOCKED' for row in result['cases']))

    def test_report_rejects_changed_outcome_even_if_all_poses_match(self):
        root = self.build_capture()
        path = root / 'control/progress.json'
        rows = PROBE.physical.read_lines(path)
        rows[0]['success'] = 1.
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(RuntimeError, 'native outcome differs'):
            PROBE.report(SimpleNamespace(run_dir=str(root)))

    def test_report_rejects_a_later_checkpoint_or_changed_observation_metadata(self):
        root = self.build_capture()
        path = root / 'control/capture_summary.json'
        diagnostic = PROBE.pilot.read_json(path)
        diagnostic['cases'][0]['checkpoint']['prefix_action_count'] += 1
        PROBE.pilot.write_json(path, diagnostic)
        with self.assertRaisesRegex(RuntimeError, 'first qualifying'):
            PROBE.report(SimpleNamespace(run_dir=str(root)))


class WiringTests(Base):
    def test_registry_class_is_used_and_hooks_are_restored_after_rollout_failure(self):
        base, source, cohort_root, cfg, dataset, progress, references, runtime = self.make_source()
        root = base / 'capture'
        root.mkdir()
        data = PROBE.verify_cohort(cohort_root)
        evaluators = []

        class Native:
            def __init__(self, config):
                self.env = FakeEnv(dataset['episodes'][1])
                self.model = SimpleNamespace()
                evaluators.append(self)

            def _run_eval_dual_system(self):
                return None

        saved_init, saved_loop = Native.__init__, Native._run_eval_dual_system
        modules = {}
        for name in ('internnav', 'internnav.habitat_extensions', 'internnav.habitat_extensions.vln'):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        evaluator_module = ModuleType('internnav.evaluator')
        evaluator_module.Evaluator = SimpleNamespace(evaluators={'habitat_vln': Native})
        modules['internnav.evaluator'] = evaluator_module
        native_module = ModuleType('internnav.habitat_extensions.vln.habitat_vln_evaluator')
        native_module.HabitatVLNEvaluator = None  # Upstream decorator intentionally replaces this symbol.
        modules[native_module.__name__] = native_module

        def fail_after_initialize(args):
            (root / 'control').mkdir()
            Native(cfg)
            raise RuntimeError('simulated rollout failure')

        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(PROBE, 'read_run', return_value=({}, data)), \
                mock.patch.object(PROBE.physical, 'instrument_loop', return_value=saved_loop), \
                mock.patch.object(PROBE.pilot, 'run_arm', side_effect=fail_after_initialize), \
                mock.patch.object(PROBE.common, 'parameter_versions', return_value='weights'):
            with self.assertRaisesRegex(RuntimeError, 'simulated rollout failure'):
                PROBE.run(SimpleNamespace(run_dir=str(root)))
        self.assertIs(Native.__init__, saved_init)
        self.assertIs(Native._run_eval_dual_system, saved_loop)
        self.assertEqual(evaluators[0].env.step.__func__, FakeEnv.step)
        summary = PROBE.pilot.read_json(root / 'control/capture_summary.json')
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('simulated rollout failure', summary['error'])


if __name__ == '__main__':
    unittest.main()