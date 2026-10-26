"""CPU selection/provenance/report tests; no simulator, model or GPU."""
import ast
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('train_probe_test', ROOT / 'scripts/eval/train_failure_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def dataset_fixture(scenes=6, routes=4):
    episodes = []
    for scene in range(scenes):
        for route in range(routes):
            for variant in range(3):
                episodes.append({
                    'episode_id': scene * 100 + route * 3 + variant + 1,
                    'scene_id': 'mp3d/scene%d/scene%d.glb' % (scene, scene),
                    'trajectory_id': scene * 100 + route,
                    'start_position': [float(route), 0., 0.],
                    'reference_path': [[float(route), 0., 0.], [float(route) + .5, 0., -1.]],
                    'instruction': {'instruction_text': 'Route %d variant %d.' % (route, variant)},
                })
    return {'episodes': episodes, 'instruction_vocab': {'word_list': ['walk']}}


class SelectionTests(unittest.TestCase):
    def test_fixed_plan_deduplicates_variants_and_balances_batches(self):
        dataset = dataset_fixture()
        before = copy.deepcopy(dataset)
        plan = PROBE.select_plan(dataset)
        rows = plan['planned_cases']
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['route_group'] for r in rows}), 24)
        self.assertEqual(len({tuple(r['key']) for r in rows}), 24)
        for batch in (1, 2):
            selected = [r for r in rows if r['batch'] == batch]
            self.assertEqual(len(selected), 12)
            self.assertEqual(len({r['key'][0] for r in selected}), 6)
            self.assertTrue(all(sum(r['key'][0] == scene for r in selected) == 2
                                for scene in plan['scenes']))
        self.assertTrue(all(r['instruction_variant_count'] == 3 for r in rows))
        self.assertEqual(dataset, before)
        chosen = PROBE.subset(dataset, rows[:12])
        self.assertEqual(chosen['instruction_vocab'], dataset['instruction_vocab'])
        self.assertEqual([list(PROBE.key(ep)) for ep in chosen['episodes']], [r['key'] for r in rows[:12]])

    def test_selection_is_independent_of_input_order_and_outcome_fields(self):
        source = dataset_fixture(scenes=7, routes=6)
        expected = PROBE.select_plan(source)
        random.Random(4).shuffle(source['episodes'])
        for ep in source['episodes']:
            ep['success'] = ep['episode_id'] % 2
            ep['recovery_success'] = 1 - ep['success']
        self.assertEqual(PROBE.select_plan(source), expected)

    def test_trajectory_and_geometry_aliases_are_merged_transitively(self):
        dataset = dataset_fixture()
        a, b, c = dataset['episodes'][:3]
        b['reference_path'] = [[0., 0., 0.], [.5, 0., -1.05]]
        c['reference_path'] = copy.deepcopy(b['reference_path'])
        c['trajectory_id'] = 999
        rows = PROBE.select_plan(dataset)['planned_cases']
        merged = [r for r in rows if '999' in r['trajectory_ids']]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['instruction_variant_count'], 3)

    def test_duplicate_or_insufficient_routes_fail_without_shrinking_plan(self):
        source = dataset_fixture()
        source['episodes'].append(copy.deepcopy(source['episodes'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            PROBE.select_plan(source)
        with self.assertRaisesRegex(ValueError, 'six scenes'):
            PROBE.select_plan(dataset_fixture(scenes=5))
        with self.assertRaisesRegex(ValueError, 'six scenes'):
            PROBE.select_plan(dataset_fixture(routes=3))

    def test_missing_instruction_or_nonfinite_reference_is_rejected(self):
        for kind in ('instruction', 'reference'):
            source = dataset_fixture()
            if kind == 'instruction':
                source['episodes'][0]['instruction']['instruction_text'] = ' '
            else:
                source['episodes'][0]['reference_path'][0][0] = float('nan')
            with self.assertRaises(ValueError):
                PROBE.select_plan(source)

    def test_review_selection_uses_plan_order_and_stops_after_fixed_reserve(self):
        def outcome(order, success):
            return {'order': order, 'metrics': {'success': success}}
        rows = [outcome(i, 0) for i in range(9)]
        random.Random(2).shuffle(rows)
        status, selected, failures = PROBE.choose_review(rows, 1)
        self.assertEqual(status, 'READY_FOR_FAILURE_REVIEW')
        self.assertEqual([r['order'] for r in selected], list(range(5)))
        self.assertEqual(len(failures), 9)
        self.assertEqual(PROBE.choose_review([outcome(0, 1)], 1)[0], 'NEED_DECLARED_RESERVE')
        self.assertEqual(PROBE.choose_review([outcome(0, 1)], 2)[0], 'INSUFFICIENT_FAILURES_STOP')


class Config(SimpleNamespace):
    def model_dump(self, **unused):
        return {'eval_type': self.eval_type, 'agent': {'model_settings': self.agent.model_settings},
                'env': {'env_settings': self.env.env_settings}, 'eval_settings': self.eval_settings}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.target = self.root / 'cohort'
        self.dataset = dataset_fixture()
        self.pilot = PROBE.load_pilot()
        self.pilot.ROOT = self.root
        self.source = self.root / 'train.json.gz'
        with gzip.open(self.source, 'wt', encoding='utf-8') as stream:
            json.dump(self.dataset, stream)
        self.habitat = self.root / 'habitat.yaml'
        self.habitat.write_text(json.dumps({'habitat': {'dataset': {
            'split': 'val_unseen', 'data_path': '{split}.json.gz', 'scenes_dir': 'scenes'}}}))
        for ep in self.dataset['episodes']:
            path = self.root / 'scenes' / ep['scene_id']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture-scene')
        self.config_path = self.root / 'native.py'
        self.config_path.write_text('native-config-fixture')
        self.cfg = Config(
            eval_type='habitat_vln',
            agent=SimpleNamespace(model_settings={'mode': 'dual_system', 'num_history': 8,
                                                  'kv_cache_continuation': False, 'vis_debug': False}),
            env=SimpleNamespace(env_settings={'config_path': str(self.habitat)}),
            eval_settings={'max_steps_per_episode': 500, 'save_video': False})
        self.calls = []

        def load_config(path):
            path = Path(path)
            if path == self.config_path:
                return copy.deepcopy(self.cfg)
            tree = ast.parse(path.read_text())
            call = tree.body[-1].value.args[0]
            data = json.loads(ast.literal_eval(call.args[0]))
            return Config(eval_type=data['eval_type'],
                          agent=SimpleNamespace(model_settings=data['agent']['model_settings']),
                          env=SimpleNamespace(env_settings=data['env']['env_settings']),
                          eval_settings=data['eval_settings'])

        self.pilot.load_config = load_config

        def native_prepare(args):
            cfg = self.pilot.load_config(args.config)
            selected = self.pilot.select_episodes(self.dataset, args.episodes)
            directory = Path(args.run_dir)
            directory.mkdir()
            with gzip.open(directory / 'episodes.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(selected, stream)
            PROBE.write_config(directory / 'control.py', cfg)
            self.pilot.write_json(directory / 'manifest.json', {
                'source_hashes': {}, 'generated_hashes': {
                    str(directory / 'control.py'): self.pilot.sha256(directory / 'control.py'),
                    str(directory / 'episodes.json.gz'): self.pilot.sha256(directory / 'episodes.json.gz')},
                'checkpoint_shard_sizes': {}})
            self.calls.append((copy.deepcopy(args), cfg, selected))
        self.pilot.prepare = native_prepare
        self.selector = self.pilot.select_episodes
        self.yaml = SimpleNamespace(safe_load=json.loads,
                                    safe_dump=lambda value, **kwargs: json.dumps(value))
        self.args = SimpleNamespace(run_dir=str(self.target), config=str(self.config_path))

    def prepare(self):
        with mock.patch.dict('sys.modules', {'yaml': self.yaml}), contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(self.args, self.pilot)
        return self.pilot.read_json(self.target / 'plan.json')

    def batch(self, number, failed=(), missing_video=False):
        plan = self.pilot.read_json(self.target / 'plan.json')
        rows = [r for r in plan['planned_cases'] if r['batch'] == number]
        directory = PROBE.batch_root(self.target, number)
        (directory / 'control').mkdir()
        annotations = {PROBE.key(e): e for e in self.pilot.read_json(directory / 'episodes.json.gz')['episodes']}
        progress, actions = [], []
        for i, row in enumerate(rows):
            episode_key = tuple(row['key'])
            progress.append(dict(scene_id=episode_key[0], episode_id=episode_key[1], success=0. if i in failed else 1.,
                                 spl=0. if i in failed else .8, os=0. if i in failed else 1., ne=8. if i in failed else .1,
                                 steps=1, episode_instruction=annotations[episode_key]['instruction']['instruction_text']))
            actions.append(dict(scene_id=row['scene_id'], episode_id=row['episode_id'],
                                action=0, done=True, position=[0., 0., 0.], orientation_wxyz=[1., 0., 0., 0.]))
            video = directory / 'video/epoch_0' / ('%s_%04d.mp4' % (episode_key[0], int(episode_key[1])))
            video.parent.mkdir(parents=True, exist_ok=True)
            if not missing_video:
                video.write_bytes(b'fixture-video-not-a-real-video')
        for name, content in [('progress.json', progress), ('actions.jsonl', actions)]:
            (directory / 'control' / name).write_text(''.join(json.dumps(r) + '\n' for r in content))
        self.pilot.write_json(directory / 'control/pilot_summary.json', {
            'status': 'complete', 'stage_synchronized': False, 'seed_base': PROBE.POLICY_SEED,
            'torch': 'fixture', 'cuda': 'fixture', 'transformers': 'fixture', 'device': 'fixture',
            'metrics': {'length': 12}})

    def report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            PROBE.report(self.args, self.pilot)
        paths = sorted(self.target.glob('report_batch_*.json'))
        return self.pilot.read_json(paths[-1])

    def test_prepare_uses_train_only_keeps_native_budget_and_restores_selector(self):
        original_yaml = self.habitat.read_bytes()
        original_data = self.source.read_bytes()
        plan = self.prepare()
        self.assertEqual(len(self.calls), 2)
        self.assertIs(self.pilot.select_episodes, self.selector)
        self.assertEqual(self.habitat.read_bytes(), original_yaml)
        self.assertEqual(self.source.read_bytes(), original_data)
        for batch, (args, cfg, selected) in enumerate(self.calls, 1):
            self.assertEqual(args.seed, PROBE.POLICY_SEED)
            self.assertIsNone(args.max_steps)
            self.assertEqual(cfg.eval_settings['max_steps_per_episode'], 500)
            self.assertEqual(cfg.agent.model_settings['num_history'], 8)
            self.assertFalse(cfg.agent.model_settings['kv_cache_continuation'])
            self.assertTrue(cfg.agent.model_settings['vis_debug'])
            self.assertEqual(len(selected['episodes']), 12)
            self.assertEqual([list(PROBE.key(ep)) for ep in selected['episodes']],
                             [r['key'] for r in plan['planned_cases'] if r['batch'] == batch])
            source_habitat = json.loads((self.target / 'train_source_habitat.yaml').read_text().split('\n', 1)[1])
            self.assertEqual(source_habitat['habitat']['dataset']['split'], 'train')
        self.assertEqual(PROBE.verify_root(self.target, self.pilot), plan)


    def test_prepare_error_restores_selector_and_preserves_original_sources(self):
        original_yaml, original_data = self.habitat.read_bytes(), self.source.read_bytes()
        self.pilot.prepare = mock.Mock(side_effect=RuntimeError('native prepare fixture'))
        with self.assertRaisesRegex(RuntimeError, 'native prepare'):
            self.prepare()
        self.assertIs(self.pilot.select_episodes, self.selector)
        self.assertEqual(self.habitat.read_bytes(), original_yaml)
        self.assertEqual(self.source.read_bytes(), original_data)

    def test_create_only_and_changed_plan_are_rejected(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        path = self.target / 'plan.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            PROBE.verify_root(self.target, self.pilot)

    def test_report_keeps_denominator_and_exports_only_predeclared_failures(self):
        self.prepare()
        self.batch(1, failed=(1, 5, 9))
        result = self.report()
        self.assertEqual(result['status'], 'READY_FOR_FAILURE_REVIEW')
        self.assertEqual(result['episodes_completed'], 12)
        self.assertEqual(result['native_successes'], 9)
        self.assertEqual(result['native_failures'], 3)
        self.assertEqual([r['order'] for r in result['cases']], [1, 5, 9])
        self.assertEqual(len(result['all_native_outcomes']), 12)
        self.assertEqual(result['distinct_reference_route_groups'], 12)
        for row in result['cases']:
            exported = Path(row['case_directory'])
            self.assertTrue((exported / 'episode.json').is_file())
            self.assertTrue((exported / 'actions.json').is_file())
        self.assertEqual(self.report(), result)
        self.pilot.run_arm = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'Stop rule'):
            PROBE.run(SimpleNamespace(run_dir=str(self.target), batch=2), self.pilot)
        self.pilot.run_arm.assert_not_called()

    def test_fixed_reserve_runs_only_when_needed_and_never_expands_beyond_24(self):
        self.prepare()
        self.batch(1, failed=(3,))
        self.assertEqual(self.report()['status'], 'NEED_DECLARED_RESERVE')
        self.pilot.run_arm = mock.Mock()
        PROBE.run(SimpleNamespace(run_dir=str(self.target), batch=2), self.pilot)
        self.pilot.run_arm.assert_called_once()
        self.batch(2, failed=())
        result = self.report()
        self.assertEqual(result['status'], 'INSUFFICIENT_FAILURES_STOP')
        self.assertEqual(result['episodes_completed'], 24)
        self.assertEqual(result['native_failures'], 1)
        self.assertEqual(result['selected_for_review'], 1)

    def test_missing_video_instruction_mismatch_and_trace_schema_fail_report(self):
        self.prepare()
        self.batch(1, failed=(0, 1, 2), missing_video=True)
        with self.assertRaises(FileNotFoundError):
            self.report()
        # Supply files, then independently corrupt the instruction and pose key.
        directory = self.target / 'batch_01'
        progress_path = directory / 'control/progress.json'
        original = progress_path.read_bytes()
        progress = PROBE.read_lines(progress_path)
        progress[0]['episode_instruction'] = 'different instruction'
        progress_path.write_text(''.join(json.dumps(r) + '\n' for r in progress))
        with self.assertRaisesRegex(ValueError, 'Instruction mismatch'):
            self.report()
        progress_path.write_bytes(original)
        trace_path = directory / 'control/actions.jsonl'
        records = PROBE.read_lines(trace_path)
        records[0]['o rientation_wxyz'] = records[0].pop('orientation_wxyz')
        trace_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.report()

    def test_existing_case_export_is_preserved_if_it_differs(self):
        path = self.root / 'export.json'
        PROBE.preserve_export(path, {'original': 1})
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'preserving'):
            PROBE.preserve_export(path, {'different': 2})
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
