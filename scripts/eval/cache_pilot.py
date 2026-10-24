"""Bounded, seeded A/B pilot for the opt-in DualVLN cache readout.

Prepare is CPU-only. Run arms sequentially, in an isolated checkout. Report
compares every actual Habitat action (including camera actions) and pose.
The optimization itself comes from InternRobotics/InternNav PR #368.
"""

import argparse
import copy
import functools
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('_cache_pilot_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def select_episodes(dataset, count):
    """Take a declared prefix, without consulting outcomes or goal distances."""
    if count < 1 or count > len(dataset['episodes']):
        raise ValueError('episodes must be between 1 and the source episode count')
    selected = copy.deepcopy(dataset)
    selected['episodes'] = selected['episodes'][:count]
    keys = [(e['scene_id'], str(e['episode_id'])) for e in selected['episodes']]
    if len(keys) != len(set(keys)):
        raise ValueError('Selected episode keys are not unique')
    return selected


def episode_seed(base, episode_id):
    return (int(base) + int(episode_id)) % (2**31)


def prepare(args):
    import yaml

    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('max-steps must be positive')
    run_dir = Path(args.run_dir).resolve()
    cfg_path = (ROOT / args.config).resolve()
    cfg = load_config(cfg_path)
    if cfg.eval_type != 'habitat_vln' or cfg.agent.model_settings.get('mode') != 'dual_system':
        raise ValueError('The pilot requires the native Habitat dual_system config')
    habitat_path = (ROOT / cfg.env.env_settings['config_path']).resolve()
    habitat_cfg = yaml.safe_load(habitat_path.read_text(encoding='utf-8'))
    ds_cfg = habitat_cfg['habitat']['dataset']
    source_path = (ROOT / ds_cfg['data_path'].format(split=ds_cfg['split'])).resolve()
    selected = select_episodes(read_json(source_path), args.episodes)
    scenes_dir = ROOT / ds_cfg['scenes_dir']
    for ep in selected['episodes']:
        scene = scenes_dir / ep['scene_id'].removeprefix('data/scene_datasets/')
        if not scene.is_file():
            raise FileNotFoundError(scene)
    model_path = (ROOT / cfg.agent.model_settings['model_path']).resolve()
    index_path = model_path / 'model.safetensors.index.json'
    index = read_json(index_path)
    shards = sorted(set(index['weight_map'].values()))
    for name in shards:
        if not (model_path / name).is_file():
            raise FileNotFoundError(model_path / name)
    watched = [cfg_path, habitat_path, source_path, model_path / 'config.json', index_path]
    watched += [ROOT / p for p in (
        'scripts/eval/cache_pilot.py',
        'internnav/habitat_extensions/vln/habitat_vln_evaluator.py',
        'internnav/model/basemodel/internvla_n1/internvla_n1.py',
    )]
    manifest = {
        'schema': 'dualvln_cache_pilot.v1', 'seed_base': args.seed,
        'revision': subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, check=False).stdout.strip(),
        'selection': 'first N source episodes; no outcome filtering',
        'episodes': [{'scene_id': e['scene_id'], 'episode_id': str(e['episode_id'])}
                     for e in selected['episodes']],
        'source_hashes': {str(p): sha256(p) for p in watched},
        'checkpoint_shard_sizes': {str(model_path / n): (model_path / n).stat().st_size for n in shards},
        'baseline_config': cfg.model_dump(mode='json'),
        'max_steps_override': args.max_steps,
    }
    # Create-only: an existing pilot is never silently replaced or resumed.
    run_dir.mkdir(parents=True, exist_ok=False)
    subset_path = run_dir / 'episodes.json.gz'
    with subset_path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            zipped.write(json.dumps(selected, ensure_ascii=False).encode('utf-8'))
    ds_cfg['data_path'] = str(subset_path)
    pilot_yaml = run_dir / 'habitat.yaml'
    pilot_yaml.write_text('# @package _global_\n' + yaml.safe_dump(habitat_cfg, sort_keys=False), encoding='utf-8')
    for arm in ('control', 'cache'):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.env.env_settings['config_path'] = str(pilot_yaml)
        arm_cfg.eval_settings['output_path'] = str(run_dir / arm)
        arm_cfg.agent.model_settings['kv_cache_continuation'] = arm == 'cache'
        if args.max_steps is not None:
            arm_cfg.eval_settings['max_steps_per_episode'] = args.max_steps
        # Upstream models have unvalidated None defaults on some non-Optional
        # fields. Omit those defaults so Pydantic can reconstruct the same config.
        serialized = json.dumps(arm_cfg.model_dump(mode='json', exclude_none=True))
        (run_dir / f'{arm}.py').write_text(
            'import json\nfrom internnav.configs.evaluator import EvalCfg\n'
            f'eval_cfg = EvalCfg.model_validate(json.loads({serialized!r}))\n', encoding='utf-8')
    generated = [subset_path, pilot_yaml, run_dir / 'control.py', run_dir / 'cache.py']
    manifest['generated_hashes'] = {str(p): sha256(p) for p in generated}
    write_json(run_dir / 'manifest.json', manifest)
    print('PILOT_PREPARE=PASS', 'episodes=', len(selected['episodes']), 'run_dir=', run_dir)


def verify_manifest(run_dir):
    manifest = read_json(run_dir / 'manifest.json')
    for group in ('source_hashes', 'generated_hashes'):
        for filename, expected in manifest[group].items():
            if sha256(filename) != expected:
                raise RuntimeError('Pilot input/code changed since preparation: ' + filename)
    for filename, expected_size in manifest['checkpoint_shard_sizes'].items():
        if Path(filename).stat().st_size != expected_size:
            raise RuntimeError('Checkpoint shard size changed: ' + filename)
    return manifest


def process_snapshot():
    command = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid', '--format=csv']
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def compare_traces(control, cache, pose_tolerance=1e-5):
    mismatches = []
    max_position_error = 0.0
    max_orientation_error = 0.0
    for i, (a, b) in enumerate(zip(control, cache)):
        keys = ('scene_id', 'episode_id', 'action', 'done')
        position_error = math.dist(a['position'], b['position'])
        qa, qb = a['orientation_wxyz'], b['orientation_wxyz']
        orientation_error = min(math.dist(qa, qb), math.dist(qa, [-v for v in qb]))
        max_position_error = max(max_position_error, position_error)
        max_orientation_error = max(max_orientation_error, orientation_error)
        finite = math.isfinite(position_error) and math.isfinite(orientation_error)
        if (not finite or any(a[k] != b[k] for k in keys)
                or position_error > pose_tolerance or orientation_error > pose_tolerance):
            if len(mismatches) < 5:
                mismatches.append({'index': i, 'control': a, 'cache': b})
    matched = bool(control) and len(control) == len(cache) and not mismatches
    return {'actions_and_poses_match': matched, 'control_action_count': len(control),
            'cache_action_count': len(cache), 'max_position_error_m': max_position_error,
            'max_quaternion_error_sign_invariant': max_orientation_error, 'first_mismatches': mismatches}


def metrics_match(control, cache):
    return (control.keys() == cache.keys()
            and all(math.isclose(float(control[k]), float(cache[k]), rel_tol=1e-6, abs_tol=1e-6)
                    for k in control))


def run_arm(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This pilot is single-process; do not launch it with multi-rank torchrun')
    run_dir = Path(args.run_dir).resolve()
    manifest = verify_manifest(run_dir)
    arm_dir = run_dir / args.arm
    arm_dir.mkdir(exist_ok=False)
    # Force this checkout ahead of an editable install pointing at the ongoing run.
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import numpy as np
    import torch
    import transformers
    import internnav
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401

    if not Path(internnav.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError('The Python process imported a different InternNav checkout')
    if not torch.cuda.is_available():
        raise RuntimeError('The navigation pilot requires CUDA; prepare and CPU tests do not')
    cfg = load_config(run_dir / f'{args.arm}.py')
    before_processes = process_snapshot()
    evaluator = Evaluator.init(cfg)
    phases = {}
    restorations = []

    def replace(obj, name, wrapped):
        old = getattr(obj, name)
        restorations.append((obj, name, old))
        setattr(obj, name, wrapped)

    def timed(label, function):
        @functools.wraps(function)
        def call(*positional, **keywords):
            torch.cuda.synchronize(evaluator.device)
            start = time.perf_counter()
            try:
                return function(*positional, **keywords)
            finally:
                torch.cuda.synchronize(evaluator.device)
                entry = phases.setdefault(label, {'calls': 0, 'seconds': 0.0})
                entry['calls'] += 1
                entry['seconds'] += time.perf_counter() - start
        return call

    if args.profile_stages:
        for name, label in [('generate', 's2_generate'), ('generate_latents', 'latent_readout'),
                            ('generate_latents_from_cache', 'latent_readout'), ('generate_traj', 's1_trajectory')]:
            replace(evaluator.model, name, timed(label, getattr(evaluator.model, name)))
        replace(evaluator.env, 'step', timed('habitat_step', evaluator.env.step))
    original_reset = evaluator.env.reset

    def seeded_reset():
        env = evaluator.env
        if 0 <= env._current_episode_index < len(env.episodes):
            episode = env.episodes[env._current_episode_index]
            seed = episode_seed(manifest['seed_base'], episode.episode_id)
            env._env.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return original_reset()

    replace(evaluator.env, 'reset', seeded_reset)
    original_step = evaluator.env.step
    trace = []
    action_file = (arm_dir / 'actions.jsonl').open('w', encoding='utf-8')

    def recorded_step(action):
        result = original_step(action)
        episode = evaluator.env.get_current_episode()
        state = evaluator.env._env.sim.get_agent_state()
        q = state.rotation
        entry = {'scene_id': episode.scene_id, 'episode_id': str(episode.episode_id),
                 'action': int(action), 'done': bool(result[2]),
                 'position': [float(v) for v in state.position],
                 'orientation_wxyz': [float(q.w), float(q.x), float(q.y), float(q.z)]}
        trace.append(entry)
        action_file.write(json.dumps(entry) + '\n')
        return result

    replace(evaluator.env, 'step', recorded_step)
    summary = {'arm': args.arm, 'status': 'running', 'stage_synchronized': args.profile_stages,
               'pid': os.getpid(), 'revision': manifest['revision'],
               'processes_before': before_processes, 'python': sys.version,
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
               'device': torch.cuda.get_device_name(evaluator.device), 'source_checkout': str(ROOT),
               'seed_base': manifest['seed_base'], 'init_time_excluded': True}
    torch.cuda.reset_peak_memory_stats(evaluator.device)
    torch.cuda.synchronize(evaluator.device)
    start = time.perf_counter()
    try:
        summary['metrics'] = evaluator.eval()
        if summary['metrics']['length'] != len(manifest['episodes']):
            raise RuntimeError('Completed episode count differs from the locked manifest')
        summary['status'] = 'complete'
    finally:
        torch.cuda.synchronize(evaluator.device)
        summary['wall_seconds'] = time.perf_counter() - start
        summary['phases'] = phases
        if args.profile_stages:
            summary['unattributed_seconds'] = summary['wall_seconds'] - sum(p['seconds'] for p in phases.values())
        summary['actual_action_count'] = len(trace)
        summary['peak_torch_allocated_mib'] = torch.cuda.max_memory_allocated(evaluator.device) / 1024**2
        summary['processes_after'] = process_snapshot()
        action_file.close()
        for obj, name, old in reversed(restorations):
            setattr(obj, name, old)
        write_json(arm_dir / 'pilot_summary.json', summary)
    print('PILOT_ARM_COMPLETE:', args.arm, 'wall_seconds=', summary['wall_seconds'])


def report(args):
    run_dir = Path(args.run_dir).resolve()
    verify_manifest(run_dir)
    control, cache = [read_json(run_dir / arm / 'pilot_summary.json') for arm in ('control', 'cache')]
    for key in ('status', 'stage_synchronized', 'torch', 'cuda', 'transformers', 'device', 'seed_base'):
        if control[key] != cache[key] or (key == 'status' and control[key] != 'complete'):
            raise RuntimeError('NOT_COMPARABLE: ' + key)
    traces = []
    for arm in ('control', 'cache'):
        with (run_dir / arm / 'actions.jsonl').open(encoding='utf-8') as stream:
            traces.append([json.loads(line) for line in stream if line.strip()])
    result = compare_traces(*traces)
    result.update({'control_metrics': control['metrics'], 'cache_metrics': cache['metrics'],
                   'metrics_match': metrics_match(control['metrics'], cache['metrics']),
                   'control_seconds': control['wall_seconds'], 'cache_seconds': cache['wall_seconds'],
                   'observed_time_ratio': control['wall_seconds'] / cache['wall_seconds'],
                   'interpretation': 'Bounded pilot, not a full SR/SPL validation. Inspect GPU process snapshots; '
                   'shared-card timings do not establish exclusive-card speed. Stage profiling adds synchronization overhead.'})
    write_json(run_dir / 'comparison.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prep = subparsers.add_parser('prepare', help='Create a locked subset/config; no GPU or model initialization')
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--config', default='scripts/eval/configs/habitat_dual_system_cfg.py')
    prep.add_argument('--episodes', type=int, default=2)
    prep.add_argument('--seed', type=int, default=20261024)
    prep.add_argument('--max-steps', type=int, default=None, help='Optional diagnostic cap; default preserves baseline')
    prep.set_defaults(function=prepare)
    arm = subparsers.add_parser('run', help='Run exactly one arm using the native evaluator')
    arm.add_argument('--run-dir', required=True)
    arm.add_argument('--arm', choices=('control', 'cache'), required=True)
    arm.add_argument('--profile-stages', action='store_true', help='Add per-stage CUDA synchronization for diagnosis')
    arm.set_defaults(function=run_arm)
    compare = subparsers.add_parser('report')
    compare.add_argument('--run-dir', required=True)
    compare.set_defaults(function=report)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == '__main__':
    main()
