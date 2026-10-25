"""CPU-only capture/provenance tests; no torch, simulator or model weights."""

import contextlib
import copy
import gzip
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('failure_probe_test', ROOT / 'scripts/eval/failure_case_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FailureProbeTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'test_scratch'
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.tmp.name).resolve()
        if not self.root.is_relative_to(scratch.resolve()):
            raise RuntimeError('Test temporary directory escaped the workspace')
        self.addCleanup(self.tmp.cleanup)
        self.source, self.target = self.root / 'source', self.root / 'capture'
        (self.source / 'control').mkdir(parents=True)
        self.pilot = PROBE.load_pilot()
        self.summary = dict(status='complete', seed_base=20261024, stage_synchronized=False,
                            torch='2.6', cuda='12.4', transformers='4.51', device='A800')
        self.pilot.write_json(self.source / 'control/pilot_summary.json', self.summary)
        self.pilot.write_json(self.source / 'manifest.json', {
            'seed_base': 20261024, 'source_hashes': {}, 'generated_hashes': {}, 'checkpoint_shard_sizes': {}})
        self.row = dict(scene_id='room', episode_id=15, success=0.0, spl=0.0, os=0.0,
                        ne=9.0, episode_instruction='Walk past the refrigerator.')
        (self.source / 'control/progress.json').write_text(json.dumps(self.row) + '\n')
        self.dataset = dict(instruction_vocab={'word_list': ['walk']}, episodes=[
            dict(scene_id='mp3d/room/room.glb', episode_id=15,
                 instruction={'instruction_text': self.row['episode_instruction']}),
            dict(scene_id='mp3d/other/other.glb', episode_id=16, instruction={'instruction_text': 'Other.'})])
        with gzip.open(self.source / 'episodes.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(self.dataset, stream)
        self.trace = [self.entry(15, 1), self.entry(15, 0)]
        records = [self.entry(16, 1)] * 3 + self.trace
        (self.source / 'control/actions.jsonl').write_text('\n'.join(map(json.dumps, records)) + '\n')
        self.cfg = SimpleNamespace(
            agent=SimpleNamespace(model_settings={'kv_cache_continuation': False, 'vis_debug': False,
                                                  'num_history': 8, 'resize_w': 384}),
            eval_settings={'max_steps_per_episode': 500, 'save_video': False})
        self.loader = lambda path: copy.deepcopy(self.cfg)
        self.pilot.load_config = self.loader
        self.selector = self.pilot.select_episodes
        self.captured = {}

        def native_prepare(args):
            cfg = self.pilot.load_config(args.config)
            subset = self.pilot.select_episodes(self.dataset, args.episodes)
            self.captured.update(cfg=cfg, subset=subset, args=args)
            self.target.mkdir()
            self.pilot.write_json(self.target / 'manifest.json', {
                'seed_base': args.seed, 'source_hashes': {}, 'generated_hashes': {},
                'checkpoint_shard_sizes': {}, 'episodes': [
                    {'scene_id': ep['scene_id'], 'episode_id': str(ep['episode_id'])}
                    for ep in subset['episodes']]})
        self.pilot.prepare = native_prepare
        self.args = SimpleNamespace(source_run=str(self.source), run_dir=str(self.target),
                                    episode='15', scene='room')

    @staticmethod
    def entry(episode, action):
        return dict(scene_id='data/scene_data/mp3d_ce/mp3d/room/room.glb', episode_id=str(episode),
                    action=action, done=action == 0, position=[0.0, 0.0, 0.0],
                    orientation_wxyz=[1.0, 0.0, 0.0, 0.0])

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            PROBE.prepare(self.args, self.pilot)

    def replay(self, video=True):
        (self.target / 'control').mkdir()
        self.pilot.write_json(self.target / 'control/pilot_summary.json', self.summary)
        (self.target / 'control/progress.json').write_text(json.dumps(self.row) + '\n')
        (self.target / 'control/actions.jsonl').write_text('\n'.join(map(json.dumps, self.trace)) + '\n')
        if video:
            (self.target / 'video').mkdir()
            (self.target / 'video/fixture.mp4').write_bytes(b'nonempty-video-file-fixture')

    def report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            PROBE.report(SimpleNamespace(run_dir=str(self.target)), self.pilot)
        return self.pilot.read_json(self.target / 'failure_review_report.json')

    def test_selection_preserves_vocab_source_and_requires_unique_case(self):
        before = copy.deepcopy(self.dataset)
        selected = PROBE.select_case(self.dataset, '15', 'room')
        selected['episodes'][0]['instruction']['instruction_text'] = 'Changed.'
        self.assertEqual(before, self.dataset)
        self.assertEqual(selected['instruction_vocab'], self.dataset['instruction_vocab'])
        duplicate = copy.deepcopy(self.dataset)
        duplicate['episodes'].append(copy.deepcopy(duplicate['episodes'][0]))
        with self.assertRaisesRegex(ValueError, 'one matching'):
            PROBE.select_case(duplicate, '15')
        self.assertEqual(PROBE.episode_key({'scene_id': r'data\room\room.glb', 'episode_id': 15}), ('room', '15'))

    def test_prepare_keeps_native_parameters_and_seed_and_restores_hooks(self):
        self.prepare()
        cfg = self.captured['cfg']
        expected = copy.deepcopy(self.cfg.agent.model_settings)
        expected.update(vis_debug=True, vis_debug_path=str(self.target / 'video'))
        self.assertEqual(cfg.agent.model_settings, expected)
        self.assertEqual(cfg.eval_settings, self.cfg.eval_settings)
        self.assertEqual(self.captured['args'].seed, 20261024)
        self.assertIsNone(self.captured['args'].max_steps)
        self.assertEqual(self.captured['subset']['episodes'][0]['episode_id'], 15)
        self.assertIs(self.pilot.load_config, self.loader)
        self.assertIs(self.pilot.select_episodes, self.selector)
        manifest = self.pilot.verify_manifest(self.target)
        self.assertIn('not an evaluation cohort', manifest['selection'])
        self.assertFalse(self.cfg.agent.model_settings['vis_debug'])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_prepare_error_restores_helpers_without_output_creation(self):
        def fail(args):
            self.pilot.load_config(args.config)
            raise RuntimeError('prepare failed')
        self.pilot.prepare = fail
        with self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            self.prepare()
        self.assertIs(self.pilot.load_config, self.loader)
        self.assertIs(self.pilot.select_episodes, self.selector)
        self.assertFalse(self.target.exists())

    def test_nonfailure_instruction_mismatch_and_cache_are_rejected(self):
        row = {**self.row, 'success': 1.0}
        (self.source / 'control/progress.json').write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'native failure'):
            self.prepare()
        row = {**self.row, 'episode_instruction': 'Different instruction.'}
        (self.source / 'control/progress.json').write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'instruction'):
            self.prepare()
        (self.source / 'control/progress.json').write_text(json.dumps(self.row) + '\n')
        self.cfg.agent.model_settings['kv_cache_continuation'] = True
        with self.assertRaisesRegex(ValueError, 'cache disabled'):
            self.prepare()
        self.assertFalse(self.target.exists())

    def test_report_aligns_one_episode_and_requires_video(self):
        self.prepare()
        self.replay(video=False)
        result = self.report()
        self.assertEqual(result['status'], 'NOT_COMPARABLE_OR_MISSING_VIDEO')
        self.assertTrue(result['native_trace_vs_reference']['actions_and_poses_match'])
        (self.target / 'video').mkdir()
        (self.target / 'video/fixture.mp4').write_bytes(b'nonempty-file-fixture')
        self.assertEqual(self.report()['status'], 'READY_FOR_VISUAL_REVIEW')

    def test_report_rejects_runtime_mismatch_and_detects_changed_action(self):
        self.prepare()
        self.replay()
        changed = [self.entry(15, 3), self.entry(15, 0)]
        (self.target / 'control/actions.jsonl').write_text('\n'.join(map(json.dumps, changed)) + '\n')
        self.assertEqual(self.report()['status'], 'NOT_COMPARABLE_OR_MISSING_VIDEO')
        self.pilot.write_json(self.target / 'control/pilot_summary.json', {**self.summary, 'seed_base': 99})
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            self.report()

    def test_changed_reference_and_duplicate_progress_are_rejected(self):
        self.prepare()
        (self.source / 'control/actions.jsonl').write_text('changed\n')
        with self.assertRaisesRegex(RuntimeError, 'changed since preparation'):
            self.pilot.verify_manifest(self.target)
        (self.source / 'control/progress.json').write_text((json.dumps(self.row) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate progress'):
            PROBE.read_progress(self.source / 'control/progress.json')


if __name__ == '__main__':
    unittest.main()
