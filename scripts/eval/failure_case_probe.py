"""Capture one selected native failure using the existing debug-video path.

No policy hooks, physical intervention, cache changes or training are introduced.
This is a diagnostic case, not an evaluation cohort.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


def episode_key(row):
    return PurePosixPath(str(row['scene_id']).replace('\\', '/')).stem, str(row['episode_id'])


def read_progress(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            key = episode_key(row)
            if key in rows:
                raise ValueError('Duplicate progress key: ' + str(key))
            rows[key] = row
    return rows


def select_case(dataset, episode_id, scene=None):
    matches = [ep for ep in dataset['episodes'] if str(ep['episode_id']) == str(episode_id)
               and (scene is None or episode_key(ep)[0] == PurePosixPath(scene.replace('\\', '/')).stem)]
    if len(matches) != 1:
        raise ValueError('Expected one matching episode; specify --scene if ambiguous')
    result = copy.deepcopy(dataset)
    result['episodes'] = copy.deepcopy(matches)
    return result


def load_pilot():
    path = Path(__file__).with_name('cache_pilot.py')
    spec = importlib.util.spec_from_file_location('_failure_case_pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args, pilot=None):
    pilot = pilot or load_pilot()
    source, target = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if target.exists():
        raise FileExistsError('Use a new run-dir; existing results are preserved')
    manifest = pilot.verify_manifest(source)
    summary = pilot.read_json(source / 'control/pilot_summary.json')
    if summary['status'] != 'complete' or summary['seed_base'] != manifest['seed_base']:
        raise ValueError('Reference must be complete with its original seed base')
    if summary['stage_synchronized']:
        raise ValueError('Reference must have stage profiling disabled')
    subset = select_case(pilot.read_json(source / 'episodes.json.gz'), args.episode, args.scene)
    episode = subset['episodes'][0]
    key = episode_key(episode)
    row = read_progress(source / 'control/progress.json').get(key)
    if row is None or float(row['success']) != 0:
        raise ValueError('The selected case must be a recorded native failure')
    instruction = episode['instruction']['instruction_text']
    if row['episode_instruction'] != instruction:
        raise ValueError('Reference instruction and annotation differ')
    native_load, native_select = pilot.load_config, pilot.select_episodes

    def capture_config(path):
        cfg = native_load(path)
        if cfg.agent.model_settings.get('kv_cache_continuation', False):
            raise ValueError('Reference control must have cache disabled')
        cfg.agent.model_settings['vis_debug'] = True
        cfg.agent.model_settings['vis_debug_path'] = str(target / 'video')
        return cfg

    def capture_case(data, count):
        if count != 1:
            raise ValueError('Capture requires one episode')
        return select_case(data, args.episode, args.scene)

    pilot.load_config, pilot.select_episodes = capture_config, capture_case
    try:
        pilot.prepare(SimpleNamespace(run_dir=str(target), config=str(source / 'control.py'),
                                      episodes=1, seed=manifest['seed_base'], max_steps=None))
    finally:
        pilot.load_config, pilot.select_episodes = native_load, native_select
    generated = pilot.read_json(target / 'manifest.json')
    generated['selection'] = 'One native failure selected for diagnosis; not an evaluation cohort'
    generated['failure_probe'] = {
        'schema': 'native_failure_capture.v1', 'source_run': str(source), 'key': list(key),
        'instruction': instruction, 'source_metrics': {k: row[k] for k in ('success', 'spl', 'os', 'ne')},
        'observation_overrides': {'vis_debug': True, 'vis_debug_path': str(target / 'video')},
    }
    for path in (Path(__file__).resolve(), source / 'manifest.json', source / 'control/actions.jsonl',
                 source / 'control/progress.json', source / 'control/pilot_summary.json'):
        generated['source_hashes'][str(path)] = pilot.sha256(path)
    pilot.write_json(target / 'manifest.json', generated)
    print('FAILURE_PROBE_PREPARE=PASS', 'case=', key, 'seed_base=', manifest['seed_base'])


def case_trace(path, key):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
               if line.strip()]
    return [row for row in records if episode_key(row) == key], len(records)


def report(args, pilot=None):
    pilot = pilot or load_pilot()
    target = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(target)
    probe = manifest['failure_probe']
    source, key = Path(probe['source_run']), tuple(probe['key'])
    a = pilot.read_json(source / 'control/pilot_summary.json')
    b = pilot.read_json(target / 'control/pilot_summary.json')
    if a['status'] != 'complete' or b['status'] != 'complete':
        raise ValueError('Reference and replay must be complete')
    for field in ('torch', 'cuda', 'transformers', 'device', 'seed_base', 'stage_synchronized'):
        if a[field] != b[field]:
            raise ValueError('Replay runtime differs: ' + field)
    reference, _ = case_trace(source / 'control/actions.jsonl', key)
    replay, total = case_trace(target / 'control/actions.jsonl', key)
    if len(replay) != total or len(manifest['episodes']) != 1:
        raise ValueError('Replay contains unexpected episodes')
    comparison = pilot.compare_traces(reference, replay)
    old_rows, new_rows = [read_progress(path / 'control/progress.json') for path in (source, target)]
    if set(new_rows) != {key}:
        raise ValueError('Replay progress has an unexpected episode set')
    old_metrics, new_metrics = [{k: rows[key][k] for k in ('success', 'spl', 'os', 'ne')}
                                for rows in (old_rows, new_rows)]
    matched_metrics = pilot.metrics_match(old_metrics, new_metrics)
    videos = [{'path': str(p), 'bytes': p.stat().st_size}
              for p in sorted((target / 'video').rglob('*.mp4')) if p.stat().st_size > 0]
    ready = comparison['actions_and_poses_match'] and matched_metrics and bool(videos)
    result = {'status': 'READY_FOR_VISUAL_REVIEW' if ready else 'NOT_COMPARABLE_OR_MISSING_VIDEO',
              'key': list(key), 'instruction': probe['instruction'],
              'native_trace_vs_reference': comparison, 'metrics_match': matched_metrics,
              'source_metrics': old_metrics, 'replay_metrics': new_metrics,
              'last_actual_action': replay[-1]['action'] if replay else None,
              'videos': videos, 'console_log': str(target / 'console.log'),
              'scope': 'Native failure reproduction only; no recovery mechanism has been established.'}
    pilot.write_json(target / 'failure_review_report.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--episode', required=True)
    prep.add_argument('--scene')
    prep.set_defaults(function=prepare)
    check = commands.add_parser('report')
    check.add_argument('--run-dir', required=True)
    check.set_defaults(function=report)
    arguments = parser.parse_args()
    if arguments.command == 'prepare':
        import subprocess
        import imageio_ffmpeg

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-version'], capture_output=True, check=True)
        print('VIDEO_ENCODER_CHECK=PASS')
    arguments.function(arguments)
