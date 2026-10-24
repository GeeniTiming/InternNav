"""Pilot isolation and comparison contracts; no simulator, checkpoints or GPU."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot():
    path = Path(__file__).resolve().parents[2] / 'scripts/eval/cache_pilot.py'
    spec = importlib.util.spec_from_file_location('cache_pilot_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_is_fixed_and_does_not_modify_source(pilot):
    source = {'instruction_vocab': {'word_list': []}, 'episodes': [
        {'scene_id': 'mp3d/a/a.glb', 'episode_id': i, 'success': int(i % 2 == 0)} for i in range(5)
    ]}
    original = copy.deepcopy(source)
    selected = pilot.select_episodes(source, 3)
    assert [e['episode_id'] for e in selected['episodes']] == [0, 1, 2]
    selected['episodes'][0]['episode_id'] = 100
    assert source == original


@pytest.mark.parametrize('count', [0, -1, 3])
def test_invalid_subset_count_is_rejected(pilot, count):
    with pytest.raises(ValueError):
        pilot.select_episodes({'episodes': [{'scene_id': 'a', 'episode_id': 1}]}, count)


def test_duplicate_episode_keys_are_rejected(pilot):
    source = {'episodes': [{'scene_id': 'a', 'episode_id': 1}] * 2}
    with pytest.raises(ValueError, match='not unique'):
        pilot.select_episodes(source, 2)


def trace_entry(action=1):
    return {'scene_id': 'mp3d/a/a.glb', 'episode_id': '1', 'action': action,
            'position': [0.0, 0.0, 0.0], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0], 'done': False}


def test_report_checks_camera_actions_poses_and_quaternion_sign(pilot):
    reference = [trace_entry(5), trace_entry(4), trace_entry(1), trace_entry(0)]
    same = copy.deepcopy(reference)
    same[0]['orientation_wxyz'] = [-1.0, 0.0, 0.0, 0.0]
    assert pilot.compare_traces(reference, same)['actions_and_poses_match']
    changed_camera = copy.deepcopy(same)
    changed_camera[0]['action'] = 4
    assert not pilot.compare_traces(reference, changed_camera)['actions_and_poses_match']
    changed_pose = copy.deepcopy(same)
    changed_pose[1]['position'][0] = 0.25
    result = pilot.compare_traces(reference, changed_pose)
    assert not result['actions_and_poses_match']
    assert result['max_position_error_m'] == 0.25
    assert not pilot.compare_traces(reference, same[:-1])['actions_and_poses_match']


def test_per_episode_seed_is_stable_and_distinguishes_episodes(pilot):
    assert pilot.episode_seed(20261024, '10') == pilot.episode_seed(20261024, 10)
    assert pilot.episode_seed(20261024, '10') != pilot.episode_seed(20261024, '11')
    assert 0 <= pilot.episode_seed(2**40, 12) < 2**31


def test_empty_or_nonfinite_trace_is_not_a_successful_validation(pilot):
    assert not pilot.compare_traces([], [])['actions_and_poses_match']
    invalid = trace_entry()
    invalid['position'][0] = float('nan')
    assert not pilot.compare_traces([trace_entry()], [invalid])['actions_and_poses_match']


def test_negative_step_cap_is_rejected_before_writing(pilot, tmp_path):
    with pytest.raises(ValueError, match='positive'):
        pilot.prepare(SimpleNamespace(run_dir=str(tmp_path / 'unused'), max_steps=-1))
    assert not (tmp_path / 'unused').exists()


def test_metrics_comparison_does_not_hide_failure_or_missing_fields(pilot):
    reference = {'length': 2, 'sucs_all': 0.5, 'nes_all': 3.0}
    assert pilot.metrics_match(reference, dict(reference))
    assert not pilot.metrics_match(reference, {**reference, 'sucs_all': 0.0})
    assert not pilot.metrics_match(reference, {'length': 2, 'sucs_all': 0.5})
    assert not pilot.metrics_match(reference, {**reference, 'nes_all': float('nan')})


def test_prepare_keeps_baseline_parameters_and_detects_input_changes(pilot, tmp_path, monkeypatch):
    from internnav.configs.agent import AgentCfg
    from internnav.configs.evaluator import EnvCfg, EvalCfg

    root = tmp_path / 'checkout'
    root.mkdir()
    monkeypatch.setattr(pilot, 'ROOT', root)
    cfg = EvalCfg(eval_type='habitat_vln',
                  agent=AgentCfg(model_name='internvla_n1', model_settings={
                      'mode': 'dual_system', 'model_path': 'checkpoints/model', 'num_history': 8,
                      'resize_w': 384, 'resize_h': 384}),
                  env=EnvCfg(env_type='habitat', env_settings={'config_path': 'scripts/eval/configs/base.yaml'}),
                  eval_settings={'output_path': 'logs/ongoing', 'max_steps_per_episode': 500, 'save_video': False})
    original = cfg.model_dump(mode='json')
    monkeypatch.setattr(pilot, 'load_config', lambda path: copy.deepcopy(cfg))
    watched = ['scripts/eval/configs/base.py', 'scripts/eval/cache_pilot.py',
               'internnav/habitat_extensions/vln/habitat_vln_evaluator.py',
               'internnav/model/basemodel/internvla_n1/internvla_n1.py']
    for name in watched:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# fixture\n')
    (root / 'scripts/eval/configs/base.yaml').write_text(
        '# @package _global_\nhabitat:\n  dataset:\n    split: val_unseen\n'
        '    data_path: data/{split}.json.gz\n    scenes_dir: data/scenes\n')
    annotation = root / 'data/val_unseen.json.gz'
    annotation.parent.mkdir(parents=True)
    import gzip
    import json
    with gzip.open(annotation, 'wt') as stream:
        json.dump({'instruction_vocab': {'word_list': []}, 'episodes': [
            {'scene_id': 'mp3d/a/a.glb', 'episode_id': 1},
            {'scene_id': 'mp3d/a/a.glb', 'episode_id': 2}]}, stream)
    scene = root / 'data/scenes/mp3d/a/a.glb'
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b'scene-fixture')
    model = root / 'checkpoints/model'
    model.mkdir(parents=True)
    (model / 'model.safetensors.index.json').write_text('{"weight_map":{"a":"model.safetensors"}}')
    (model / 'model.safetensors').write_bytes(b'checkpoint-fixture')
    (model / 'config.json').write_text('{}')
    run_dir = root / 'data/pilot'
    args = SimpleNamespace(run_dir=str(run_dir), config='scripts/eval/configs/base.py',
                           episodes=2, seed=20261024, max_steps=None)
    pilot.prepare(args)
    manifest = pilot.verify_manifest(run_dir)
    assert manifest['baseline_config'] == original
    assert manifest['max_steps_override'] is None
    assert cfg.model_dump(mode='json') == original
    # Generated configs load through the same mechanism as the official entrypoint.

    def read_generated(arm):
        namespace = {}
        exec((run_dir / f'{arm}.py').read_text(), namespace)
        return namespace['eval_cfg']
    control, cache = read_generated('control'), read_generated('cache')
    assert control.eval_settings['max_steps_per_episode'] == 500
    assert control.agent.model_settings['num_history'] == 8
    assert control.agent.model_settings['kv_cache_continuation'] is False
    assert cache.agent.model_settings['kv_cache_continuation'] is True
    cache.agent.model_settings['kv_cache_continuation'] = False
    cache.eval_settings['output_path'] = control.eval_settings['output_path']
    assert cache == control
    with pytest.raises(FileExistsError):
        pilot.prepare(args)
    annotation.write_bytes(annotation.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        pilot.verify_manifest(run_dir)
