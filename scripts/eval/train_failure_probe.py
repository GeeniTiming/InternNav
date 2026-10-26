"""Fixed train-split native-failure discovery; no physical intervention or training.

Lock 24 distinct reference-route groups across six scenes before any rollout.
Run 12 first; use the fixed 12-case reserve only if fewer than three fail.
Keep every native outcome. Review at most the first five failures in plan order.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

SCENES, ROUTES_PER_SCENE, BATCH_SIZE = 6, 4, 12
SELECTION_SEED, POLICY_SEED = 20261026, 20261024
MIN_FAILURES, MAX_REVIEW = 3, 5
SCOPE = ('Exploratory train-split diagnosis, selected before outcomes. Different reference-route groups '
         'are not a claim of statistical independence. Native failure selection is not recovery '
         'efficacy, a held-out benchmark result, or evidence for a memory cause.')


def load_pilot():
    path = Path(__file__).with_name('cache_pilot.py')
    spec = importlib.util.spec_from_file_location('_train_failure_pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(row):
    scene = PurePosixPath(str(row['scene_id']).replace('\\', '/')).stem
    return scene, str(row['episode_id'])


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def select_plan(dataset):
    """Union same-trajectory siblings and equal reference paths, then hash-select."""
    episodes = dataset['episodes']
    if len({key(ep) for ep in episodes}) != len(episodes):
        raise ValueError('Duplicate scene/episode keys')
    parents, owners, geometries = list(range(len(episodes))), {}, []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for index, ep in enumerate(episodes):
        scene, episode_id = key(ep)
        int(episode_id)  # native per-episode seed requires numeric R2R IDs
        if not ep.get('instruction', {}).get('instruction_text', '').strip():
            raise ValueError('Missing instruction: ' + str(key(ep)))
        path = ep.get('reference_path', [])
        if (len(path) < 2 or any(len(p) != 3 or any(not math.isfinite(float(v)) for v in p) for p in path)):
            raise ValueError('Invalid reference path: ' + str(key(ep)))
        geometry = digest([[round(float(v), 4) for v in p] for p in path])
        geometries.append(geometry)
        aliases = [(scene, 'geometry', geometry)]
        if ep.get('trajectory_id') is not None:
            aliases.append((scene, 'trajectory', str(ep['trajectory_id'])))
        for alias in aliases:
            if alias in owners:
                union(index, owners[alias])
            else:
                owners[alias] = index
    groups = {}
    for index in range(len(episodes)):
        groups.setdefault(find(index), []).append(index)
    by_scene = {}
    for members in groups.values():
        scene = key(episodes[members[0]])[0]
        group_id = digest([scene, sorted({geometries[i] for i in members})])
        representative = min(members, key=lambda i: (
            digest([SELECTION_SEED, 'instruction', list(key(episodes[i]))]), key(episodes[i])))
        row = {'key': list(key(episodes[representative])), 'route_group': group_id,
               'scene_id': episodes[representative]['scene_id'],
               'episode_id': str(episodes[representative]['episode_id']),
               'trajectory_ids': sorted({str(episodes[i]['trajectory_id']) for i in members
                                         if episodes[i].get('trajectory_id') is not None}),
               'instruction_variant_count': len(members)}
        by_scene.setdefault(scene, []).append(row)
    eligible = [scene for scene, rows in by_scene.items() if len(rows) >= ROUTES_PER_SCENE]
    if len(eligible) < SCENES:
        raise ValueError('Need six scenes with at least four distinct route groups each')
    scenes = sorted(eligible, key=lambda scene: (digest([SELECTION_SEED, 'scene', scene]), scene))[:SCENES]
    chosen = {scene: sorted(by_scene[scene], key=lambda row: (
        digest([SELECTION_SEED, 'route', row['route_group']]), row['route_group']))[:ROUTES_PER_SCENE]
              for scene in scenes}
    plan = []
    for route_number in range(ROUTES_PER_SCENE):
        for scene in scenes:
            row = copy.deepcopy(chosen[scene][route_number])
            row.update(order=len(plan), batch=len(plan) // BATCH_SIZE + 1)
            plan.append(row)
    return {'schema': 'train_failure_cohort.v1', 'split': 'train',
            'selection_seed': SELECTION_SEED, 'policy_seed_base': POLICY_SEED,
            'scenes': scenes, 'planned_cases': plan, 'batch_size': BATCH_SIZE,
            'minimum_failures_for_review': MIN_FAILURES, 'maximum_review_cases': MAX_REVIEW,
            'reserve_rule': 'Run batch 2 only if completed batch 1 has fewer than three failures. '
                            'Stop after the two declared batches even if fewer than three fail.',
            'scope': SCOPE}


def subset(dataset, rows):
    selected = {tuple(row['key']) for row in rows}
    mapping = {key(ep): ep for ep in dataset['episodes']}
    if not selected <= mapping.keys() or len(selected) != len(rows):
        raise ValueError('Locked selection is missing or duplicated in the source dataset')
    result = copy.deepcopy(dataset)
    result['episodes'] = [copy.deepcopy(mapping[tuple(row['key'])]) for row in rows]
    return result


def batch_root(root, batch):
    return root / ('batch_%02d' % batch)


def verify_root(root, pilot):
    manifest = pilot.read_json(root / 'cohort_manifest.json')
    for filename, expected in manifest['hashes'].items():
        if pilot.sha256(filename) != expected:
            raise RuntimeError('Cohort input/code changed: ' + filename)
    return pilot.read_json(root / 'plan.json')


def write_config(path, cfg):
    serialized = json.dumps(cfg.model_dump(mode='json', exclude_none=True))
    path.write_text('import json\nfrom internnav.configs.evaluator import EvalCfg\n'
                    + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n',
                    encoding='utf-8')


def prepare(args, pilot=None):
    import yaml
    pilot = pilot or load_pilot()
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Use a new run-dir; existing experiments are preserved')
    config_path = (pilot.ROOT / args.config).resolve()
    cfg = pilot.load_config(config_path)
    if cfg.eval_type != 'habitat_vln' or cfg.agent.model_settings.get('mode') != 'dual_system':
        raise ValueError('Require the native Habitat dual_system config')
    habitat_path = (pilot.ROOT / cfg.env.env_settings['config_path']).resolve()
    habitat = yaml.safe_load(habitat_path.read_text(encoding='utf-8'))
    ds = habitat['habitat']['dataset']
    if '{split}' not in ds['data_path']:
        raise ValueError('Use the original split-templated config, not an old one-episode pilot config')
    train_path = (pilot.ROOT / ds['data_path'].format(split='train')).resolve()
    dataset = pilot.read_json(train_path)
    plan = select_plan(dataset)
    for ep in subset(dataset, plan['planned_cases'])['episodes']:
        scene_path = pilot.ROOT / ds['scenes_dir'] / ep['scene_id'].removeprefix('data/scene_datasets/')
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
    root.mkdir(parents=True, exist_ok=False)
    plan_path = root / 'plan.json'
    pilot.write_json(plan_path, plan)
    ds['split'], ds['data_path'] = 'train', str(train_path)
    train_habitat = root / 'train_source_habitat.yaml'
    train_habitat.write_text('# @package _global_\n' + yaml.safe_dump(habitat, sort_keys=False), encoding='utf-8')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('cache_pilot.py').resolve(),
               config_path, habitat_path, train_path, plan_path, train_habitat]
    original_selector = pilot.select_episodes
    try:
        for batch in (1, 2):
            directory = batch_root(root, batch)
            rows = [row for row in plan['planned_cases'] if row['batch'] == batch]
            batch_cfg = copy.deepcopy(cfg)
            batch_cfg.env.env_settings['config_path'] = str(train_habitat)
            batch_cfg.agent.model_settings['kv_cache_continuation'] = False
            batch_cfg.agent.model_settings['vis_debug'] = True
            batch_cfg.agent.model_settings['vis_debug_path'] = str(directory / 'video')
            source_config = root / ('batch_%02d_source.py' % batch)
            write_config(source_config, batch_cfg)
            watched.append(source_config)

            def locked_selector(data, count, selected=rows):
                if count != BATCH_SIZE:
                    raise ValueError('Unexpected batch size')
                return subset(data, selected)

            pilot.select_episodes = locked_selector
            pilot.prepare(SimpleNamespace(run_dir=str(directory), config=str(source_config),
                                          episodes=BATCH_SIZE, seed=POLICY_SEED, max_steps=None))
    finally:
        pilot.select_episodes = original_selector
    cohort_manifest = root / 'cohort_manifest.json'
    pilot.write_json(cohort_manifest, {'schema': 'train_failure_inputs.v1',
                                      'hashes': {str(p): pilot.sha256(p) for p in watched}})
    for batch in (1, 2):
        directory = batch_root(root, batch)
        manifest = pilot.read_json(directory / 'manifest.json')
        manifest['selection'] = SCOPE
        for path in (cohort_manifest, plan_path, Path(__file__).resolve()):
            manifest['source_hashes'][str(path)] = pilot.sha256(path)
        pilot.write_json(directory / 'manifest.json', manifest)
    print('TRAIN_COHORT_PREPARE=PASS planned=24 batch_size=12 scenes=6 split=train cache=False')
    print('RUN_DIR:', root)
    print('SOURCE_DATASET:', train_path)
    print('FIRST_BATCH_KEYS:', json.dumps([r['key'] for r in plan['planned_cases'] if r['batch'] == 1]))
    print('No model, GPU rollout, recovery, or training was run.')


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_actions(records, expected):
    grouped = {episode_key: [] for episode_key in expected}
    required = {'scene_id', 'episode_id', 'action', 'done', 'position', 'orientation_wxyz'}
    for index, row in enumerate(records):
        missing = required - row.keys()
        if missing:
            raise ValueError('Action record %d missing %s' % (index, sorted(missing)))
        episode_key = key(row)
        if episode_key not in grouped:
            raise ValueError('Unexpected action episode: ' + str(episode_key))
        for field, size in (('position', 3), ('orientation_wxyz', 4)):
            values = row[field]
            if len(values) != size or any(not math.isfinite(float(v)) for v in values):
                raise ValueError('Invalid action pose at record %d' % index)
        grouped[episode_key].append(row)
    if any(not rows for rows in grouped.values()):
        raise ValueError('A planned episode has no actual action trace')
    return grouped


def collect_batch(root, batch, pilot, plan):
    directory = batch_root(root, batch)
    manifest = pilot.verify_manifest(directory)
    summary = pilot.read_json(directory / 'control/pilot_summary.json')
    if summary['status'] != 'complete' or summary['stage_synchronized']:
        raise RuntimeError('Require a completed, unprofiled native batch: ' + str(batch))
    if summary['seed_base'] != POLICY_SEED or int(summary['metrics']['length']) != BATCH_SIZE:
        raise RuntimeError('Unexpected native seed or completed episode count')
    planned = [r for r in plan['planned_cases'] if r['batch'] == batch]
    annotations = {key(ep): ep for ep in pilot.read_json(directory / 'episodes.json.gz')['episodes']}
    outcomes = {}
    for row in read_lines(directory / 'control/progress.json'):
        episode_key = key(row)
        if episode_key in outcomes:
            raise ValueError('Duplicate progress episode: ' + str(episode_key))
        outcomes[episode_key] = row
    expected = {tuple(row['key']) for row in planned}
    if set(outcomes) != expected or set(annotations) != expected:
        raise ValueError('Progress/annotation episode set differs from the locked plan')
    records = validate_actions(read_lines(directory / 'control/actions.jsonl'), expected)
    result = []
    for row in planned:
        episode_key = tuple(row['key'])
        progress, annotation = outcomes[episode_key], annotations[episode_key]
        if progress['episode_instruction'] != annotation['instruction']['instruction_text']:
            raise ValueError('Instruction mismatch: ' + str(episode_key))
        if float(progress['success']) not in (0., 1.):
            raise ValueError('Invalid binary success outcome')
        for field in ('spl', 'os', 'ne'):
            if not math.isfinite(float(progress[field])):
                raise ValueError('Nonfinite outcome: ' + field)
        epoch = manifest.get('baseline_config', {}).get('eval_settings', {}).get('epoch', 0)
        video = directory / 'video' / ('epoch_%s' % epoch) / ('%s_%04d.mp4' % (episode_key[0], int(episode_key[1])))
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError('Missing native debug video: ' + str(video))
        result.append(dict(row, instruction=annotation['instruction']['instruction_text'],
                           metrics={f: progress[f] for f in ('success', 'spl', 'os', 'ne')},
                           logged_main_steps=progress.get('steps'),
                           actual_action_count=len(records[episode_key]),
                           last_actual_action=records[episode_key][-1]['action'],
                           last_env_done=records[episode_key][-1]['done'],
                           source_run=str(directory), video=str(video)))
    return result, summary, annotations, records


def choose_review(outcomes, batches_completed):
    failures = sorted([row for row in outcomes if float(row['metrics']['success']) == 0.],
                      key=lambda row: row['order'])
    if len(failures) >= MIN_FAILURES:
        status = 'READY_FOR_FAILURE_REVIEW'
    elif batches_completed == 1:
        status = 'NEED_DECLARED_RESERVE'
    else:
        status = 'INSUFFICIENT_FAILURES_STOP'
    return status, failures[:MAX_REVIEW], failures


def run(args, pilot=None):
    pilot = pilot or load_pilot()
    root = Path(args.run_dir).resolve()
    plan = verify_root(root, pilot)
    if args.batch == 2:
        outcomes, _, _, _ = collect_batch(root, 1, pilot, plan)
        if sum(float(row['metrics']['success']) == 0. for row in outcomes) >= MIN_FAILURES:
            raise ValueError('Stop rule: batch 1 already has three failures; do not run the reserve')
    cfg = pilot.load_config(batch_root(root, args.batch) / 'control.py')
    if cfg.agent.model_settings.get('kv_cache_continuation', False):
        raise ValueError('Native failure discovery requires cache-disabled inference')
    pilot.run_arm(SimpleNamespace(run_dir=str(batch_root(root, args.batch)), arm='control',
                                  profile_stages=False))


def preserve_export(path, content):
    payload = (json.dumps(content, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Existing case export differs; preserving it: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)


def report(args, pilot=None):
    pilot = pilot or load_pilot()
    root = Path(args.run_dir).resolve()
    plan = verify_root(root, pilot)
    outcomes, first_summary, first_annotations, first_records = collect_batch(root, 1, pilot, plan)
    resources = {1: (first_annotations, first_records)}
    completed = 1
    second_output = batch_root(root, 2) / 'control'
    if second_output.exists():
        if sum(float(row['metrics']['success']) == 0. for row in outcomes) >= MIN_FAILURES:
            raise ValueError('Reserve was started despite the predeclared stop rule')
        second, summary, annotations, records = collect_batch(root, 2, pilot, plan)
        for field in ('torch', 'cuda', 'transformers', 'device', 'seed_base', 'stage_synchronized'):
            if first_summary[field] != summary[field]:
                raise ValueError('Batch runtime differs: ' + field)
        resources[2] = (annotations, records)
        outcomes += second
        completed = 2
    status, selected, all_failures = choose_review(outcomes, completed)
    selected_exports = []
    for row in selected:
        episode_key = tuple(row['key'])
        annotations, records = resources[row['batch']]
        folder = root / 'cases' / ('%s_%s' % episode_key)
        preserve_export(folder / 'episode.json', annotations[episode_key])
        preserve_export(folder / 'outcome.json', row)
        preserve_export(folder / 'actions.json', records[episode_key])
        selected_exports.append(dict(row, case_directory=str(folder)))
    result = {'status': status, 'split': 'train', 'completed_batches': completed,
              'episodes_completed': len(outcomes), 'native_successes': len(outcomes) - len(all_failures),
              'native_failures': len(all_failures), 'selected_for_review': len(selected),
              'distinct_reference_route_groups': len({r['route_group'] for r in outcomes}),
              'scenes_completed': len({r['key'][0] for r in outcomes}),
              'review_selection_rule': 'First at most five native failures in the predeclared plan order; '
                                       'no physical-recovery outcome was consulted.',
              'cases': selected_exports, 'all_native_outcomes': outcomes,
              'next_step': ('Run fixed batch 2 once.' if status == 'NEED_DECLARED_RESERVE'
                            else 'Inspect recorded failures; do not expand or resample this cohort.'),
              'scope': SCOPE}
    pilot.write_json(root / ('report_batch_%02d.json' % completed), result)
    # Compact console output; complete success and failure denominators remain in the report.
    print(json.dumps({k: v for k, v in result.items() if k != 'all_native_outcomes'},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--config', default='scripts/eval/configs/habitat_dual_system_cfg.py')
    prep.set_defaults(function=prepare)
    execute = sub.add_parser('run')
    execute.add_argument('--run-dir', required=True)
    execute.add_argument('--batch', type=int, choices=(1, 2), required=True)
    execute.set_defaults(function=run)
    summary = sub.add_parser('report')
    summary.add_argument('--run-dir', required=True)
    summary.set_defaults(function=report)
    args = parser.parse_args()
    if args.command == 'prepare':
        import subprocess
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-version'], capture_output=True, check=True)
    args.function(args)


if __name__ == '__main__':
    main()
