"""CPU-only cohort selection tests. No Habitat/model or intervention execution."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('orientation_cohort_test', ROOT / 'scripts/eval/orientation_cohort.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def fixture():
    episodes, progress, traces = [], {}, {}

    def add(scene, eid, geometry, trajectory, success, body=20):
        key = scene, str(eid)
        scene_id = 'mp3d/' + scene + '/' + scene + '.glb'
        instruction = 'Instruction ' + str(eid) + '.'
        episodes.append({'episode_id': eid, 'scene_id': scene_id, 'trajectory_id': trajectory,
                         'instruction': {'instruction_text': instruction},
                         'reference_path': [[geometry, 0., 0.], [geometry + 1, 0., 0.]]})
        progress[key] = {'scene_id': scene_id, 'episode_id': eid, 'episode_instruction': instruction,
                         'success': success, 'spl': float(success), 'os': float(success), 'ne': 1. if success else 8.}
        rows = [dict(scene_id=scene_id, episode_id=str(eid), action=1, done=False,
                     position=[.25 * i, 0., 0.], orientation_wxyz=[1., 0., 0., 0.])
                for i in range(body)]
        rows.append(dict(scene_id=scene_id, episode_id=str(eid), action=0, done=True,
                         position=[.25 * max(0, body-1), 0., 0.], orientation_wxyz=[1., 0., 0., 0.]))
        traces[key] = rows
    add('zsNo4HB9uLZ', 15, 0., 1, 0)
    add('zsNo4HB9uLZ', 16, 0., 1, 1)
    add('A', 20, 10., 2, 0)
    add('A', 21, 10., 3, 1)
    add('A', 22, 11., 3, 1)  # transitive trajectory alias to the failed route
    add('A', 23, 20., 4, 0)
    add('B', 24, 30., 5, 0)
    add('C', 25, 40., 6, 0)
    add('A', 26, 50., 7, 1)
    add('D', 27, 60., 8, 1)
    add('E', 28, 70., 9, 1)
    return {'episodes': episodes}, progress, traces, add


class SelectionTests(unittest.TestCase):
    def test_whole_case15_route_and_transitive_failed_siblings_are_excluded_from_controls(self):
        dataset, progress, traces, _ = fixture()
        before = copy.deepcopy((dataset, progress, traces))
        plan = PROBE.select_plan(dataset, progress, traces)
        self.assertEqual(plan['selected_counts'], {'failure': 3, 'success': 3})
        self.assertEqual([r['key'] for r in plan['cases'] if r['stratum'] == 'failure'],
                         [['A', '20'], ['B', '24'], ['C', '25']])
        self.assertEqual([r['key'] for r in plan['cases'] if r['stratum'] == 'success'],
                         [['A', '26'], ['D', '27'], ['E', '28']])
        self.assertEqual(len({r['route_group'] for r in plan['cases']}), 6)
        self.assertEqual(plan['cases'][0]['route_member_keys'], [['A', '20'], ['A', '21'], ['A', '22']])
        self.assertEqual(plan['excluded_case15_route_groups'][0]['keys'],
                         [['zsNo4HB9uLZ', '15'], ['zsNo4HB9uLZ', '16']])
        self.assertFalse(plan['navigation_executed'])
        self.assertEqual((dataset, progress, traces), before)

    def test_recovery_fields_do_not_affect_selection_or_thresholds(self):
        dataset, progress, traces, _ = fixture()
        expected = PROBE.select_plan(dataset, progress, traces)
        for ep in dataset['episodes']:
            ep['recovery_success'] = 999
        for row in progress.values():
            row['left90_success'], row['right90_success'] = 1, 0
        actual = PROBE.select_plan(dataset, progress, traces)
        self.assertEqual(actual, expected)
        self.assertTrue(all(r['checkpoint_body_threshold'] == 12 for r in actual['cases']))
        self.assertEqual(actual['checkpoint_rule']['minimum_remaining_nonstop_body_actions'], 6)

    def test_shortage_stays_locked_and_zero_stratum_stops(self):
        dataset, progress, traces, _ = fixture()
        for key in list(progress):
            if key[0] in ('D', 'E'):
                del progress[key]
                del traces[key]
        dataset['episodes'] = [ep for ep in dataset['episodes']
                               if PROBE.capture.episode_key(ep) in progress]
        plan = PROBE.select_plan(dataset, progress, traces)
        self.assertEqual(plan['selected_counts']['success'], 1)
        self.assertEqual(plan['status'], 'LOCKED_FEWER_GROUPS_PENDING_CHECKPOINT_CAPTURE')
        for key in list(progress):
            if key == ('A', '26'):
                del progress[key]
                del traces[key]
        dataset['episodes'] = [ep for ep in dataset['episodes']
                               if PROBE.capture.episode_key(ep) in progress]
        self.assertEqual(PROBE.select_plan(dataset, progress, traces)['status'], 'INSUFFICIENT_STRATA_STOP')

    def test_short_native_episode_is_retained_as_ineligible_not_replaced(self):
        dataset, progress, traces, _ = fixture()
        traces[('A', '26')] = [traces[('A', '26')][-1]]
        plan = PROBE.select_plan(dataset, progress, traces)
        row = next(r for r in plan['cases'] if r['key'] == ['A', '26'])
        self.assertEqual(row['checkpoint_body_threshold'], 0)
        self.assertEqual(row['checkpoint_status'], 'INSUFFICIENT_BASELINE_BODY_BUDGET_KEEP_LOCKED')

    def test_duplicate_bad_annotation_instruction_and_trace_are_rejected(self):
        dataset, progress, traces, _ = fixture()
        duplicate = copy.deepcopy(dataset)
        duplicate['episodes'].append(copy.deepcopy(duplicate['episodes'][0]))
        with self.assertRaises(ValueError):
            PROBE.select_plan(duplicate, progress, traces)
        changed = copy.deepcopy(dataset)
        changed['episodes'][2]['reference_path'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            PROBE.select_plan(changed, progress, traces)
        changed = copy.deepcopy(progress)
        changed[('A', '20')]['episode_instruction'] = 'Different instruction.'
        with self.assertRaisesRegex(ValueError, 'Instruction'):
            PROBE.select_plan(dataset, changed, traces)
        changed = copy.deepcopy(traces)
        changed[('A', '20')][0].pop('orientation_wxyz')
        with self.assertRaisesRegex(ValueError, 'missing'):
            PROBE.select_plan(dataset, progress, changed)

    def test_incomplete_or_concatenated_trace_is_rejected(self):
        dataset, progress, traces, _ = fixture()
        traces[('A', '20')][-1]['done'] = False
        with self.assertRaisesRegex(ValueError, 'final done'):
            PROBE.select_plan(dataset, progress, traces)
        traces[('A', '20')][-1]['done'] = True
        traces[('A', '20')][0]['done'] = True
        with self.assertRaisesRegex(ValueError, 'final done'):
            PROBE.select_plan(dataset, progress, traces)


class PrepareTests(unittest.TestCase):
    def temporary(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name).resolve()

    def build_source(self, base):
        source = base / 'source'
        (source / 'control').mkdir(parents=True)
        dataset, progress, traces, add = fixture()
        for i in range(9):
            add('E', 100+i, 70., 9, 1)
        self.assertEqual(len(dataset['episodes']), 20)
        with gzip.open(source / 'episodes.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(dataset, stream)
        (source / 'control/progress.json').write_text(
            ''.join(json.dumps(progress[PROBE.capture.episode_key(ep)]) + '\n' for ep in dataset['episodes']))
        (source / 'control/actions.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for ep in dataset['episodes']
                    for row in traces[PROBE.capture.episode_key(ep)]))
        PROBE.pilot.write_json(source / 'control/pilot_summary.json', {
            'status': 'complete', 'stage_synchronized': False, 'seed_base': 20261024,
            'metrics': {'length': 20}})
        PROBE.pilot.write_json(source / 'manifest.json', {
            'seed_base': 20261024, 'source_hashes': {}, 'generated_hashes': {}, 'checkpoint_shard_sizes': {}})
        (source / 'control.py').write_text('# fixture')
        (source / 'habitat.yaml').write_text('# fixture')
        cfg = SimpleNamespace(agent=SimpleNamespace(model_settings={'kv_cache_continuation': False}),
                              env=SimpleNamespace(env_settings={'config_path': str(source / 'habitat.yaml')}))
        return source, cfg

    def test_prepare_is_read_only_for_source_and_hash_locks_the_new_plan(self):
        base = self.temporary()
        source, cfg = self.build_source(base)
        before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        target = base / 'new'
        yaml = SimpleNamespace(safe_load=lambda s: {'habitat': {'dataset': {'split': 'val_unseen'}}})
        with mock.patch.dict(sys.modules, {'yaml': yaml}), \
                mock.patch.object(PROBE.pilot, 'load_config', return_value=cfg), \
                contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target)))
        plan = PROBE.pilot.read_json(target / 'plan.json')
        lock = PROBE.pilot.read_json(target / 'cohort_manifest.json')
        self.assertEqual(plan['source_native_episodes'], 20)
        self.assertEqual(plan['selected_counts'], {'failure': 3, 'success': 3})
        self.assertEqual(lock['hashes'][str(target / 'plan.json')], PROBE.pilot.sha256(target / 'plan.json'))
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertFalse(any((target / arm).exists() for arm in ('control', 'left90', 'right90')))

    def test_existing_directory_train_source_and_incomplete_source_are_rejected(self):
        base = self.temporary()
        with self.assertRaises(FileExistsError):
            PROBE.prepare(SimpleNamespace(source_run='unused', run_dir=str(base)))
        source, cfg = self.build_source(base)
        target = base / 'new'
        yaml = SimpleNamespace(safe_load=lambda s: {'habitat': {'dataset': {'split': 'train'}}})
        with mock.patch.dict(sys.modules, {'yaml': yaml}), mock.patch.object(PROBE.pilot, 'load_config', return_value=cfg):
            with self.assertRaisesRegex(ValueError, 'closed train cohort'):
                PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target)))
        self.assertFalse(target.exists())
        summary = PROBE.pilot.read_json(source / 'control/pilot_summary.json')
        summary['status'] = 'running'
        PROBE.pilot.write_json(source / 'control/pilot_summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'completed'):
            PROBE.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target)))
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
