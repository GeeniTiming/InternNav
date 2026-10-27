"""Lock a small cross-case orientation diagnostic from an existing native pilot.

CPU preparation only. No checkpoint has yet been captured and no recovery arm
is executed. Native outcomes define the diagnostic strata, never intervention
outcomes. Existing train-cohort and case15 results remain read-only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_pilot as pilot
import physical_recovery_probe as physical
import failure_case_probe as capture

MAX_PER_STRATUM = 3
EXCLUDED_KEY = ('zsNo4HB9uLZ', '15')
SCOPE = (
    'A bounded development diagnostic selected from the existing 20-episode '
    'val-unseen native pilot. Known native success/failure is used for stratification; '
    'no recovery outcome is consulted. Not a prevalence estimate, training set, '
    'held-out benchmark, or online deviation detector. The completed 24-case train '
    'investigation stays closed.'
)
CHECKPOINT_RULE = {
    'body_fraction': 0.60, 'minimum_prefix_translation_m': 1.0,
    'minimum_remaining_nonstop_body_actions': 6,
    'eligibility': 'First idle, level-camera System-2 boundary at or after the declared '
                   'native non-STOP body-action threshold, after at least 1 m of prefix '
                   'translation, with at least six baseline non-STOP body actions remaining.',
    'fallback': 'If no eligible boundary exists, keep the case as NOT_ELIGIBLE; do not replace it.',
    'timing_privilege': 'The threshold uses the completed baseline length. This is offline '
                        'checkpoint timing for diagnosis, not an online trigger.',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def route_groups(episodes):
    keys = [capture.episode_key(ep) for ep in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate scene/episode keys')
    parents, owners, geometries = list(range(len(episodes))), {}, []

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for i, ep in enumerate(episodes):
        if not ep.get('instruction', {}).get('instruction_text', '').strip():
            raise ValueError('Missing instruction')
        path = ep.get('reference_path', [])
        if len(path) < 2 or any(len(p) != 3 or any(not math.isfinite(float(v)) for v in p) for p in path):
            raise ValueError('Invalid reference path')
        geometry = digest([[round(float(v), 4) for v in p] for p in path])
        geometries.append(geometry)
        aliases = [(keys[i][0], 'geometry', geometry)]
        if ep.get('trajectory_id') is not None:
            aliases.append((keys[i][0], 'trajectory', str(ep['trajectory_id'])))
        for alias in aliases:
            if alias in owners:
                union(i, owners[alias])
            else:
                owners[alias] = i
    groups = {}
    for i in range(len(episodes)):
        groups.setdefault(find(i), []).append(i)
    return [{'indices': members,
             'route_group': digest([keys[members[0]][0], sorted({geometries[i] for i in members})])}
            for members in sorted(groups.values(), key=min)]


def balanced_prefix(rows):
    chosen, scenes = [], set()
    for row in rows:
        if row['key'][0] not in scenes and len(chosen) < MAX_PER_STRATUM:
            chosen.append(row)
            scenes.add(row['key'][0])
    for row in rows:
        if row not in chosen and len(chosen) < MAX_PER_STRATUM:
            chosen.append(row)
    return chosen


def select_plan(dataset, progress, traces):
    episodes = dataset['episodes']
    keys = [capture.episode_key(ep) for ep in episodes]
    if EXCLUDED_KEY not in keys:
        raise ValueError('Source must include the audited case15 for whole-route exclusion')
    if set(keys) != set(progress) or set(keys) != set(traces):
        raise ValueError('Annotation, native progress and trace episode sets differ')
    for ep, key in zip(episodes, keys):
        row = progress[key]
        if row['episode_instruction'] != ep['instruction']['instruction_text']:
            raise ValueError('Instruction differs between annotation and native outcome')
        if float(row['success']) not in (0., 1.):
            raise ValueError('Expected binary native success')
        physical.validate_trace(traces[key], str(key))
        if not traces[key][-1]['done'] or any(r['done'] for r in traces[key][:-1]):
            raise ValueError('Native trace must contain exactly one final done record')
        if any(capture.episode_key(r) != key for r in traces[key]):
            raise ValueError('Trace contains another episode')
    buckets, excluded = {'failure': [], 'success': []}, []
    for group in route_groups(episodes):
        members = group['indices']
        member_keys = [keys[i] for i in members]
        if EXCLUDED_KEY in member_keys:
            excluded.append({'route_group': group['route_group'], 'keys': [list(k) for k in member_keys]})
            continue
        failed = [i for i in members if float(progress[keys[i]]['success']) == 0.]
        stratum = 'failure' if failed else 'success'
        i = (failed or members)[0]  # first source-annotation member; failure priority within a route
        key = keys[i]
        body = sum(r['action'] in (1, 2, 3) for r in traces[key])
        row = {
            'key': list(key), 'route_group': group['route_group'], 'stratum': stratum,
            'source_annotation_order': i, 'scene_id': episodes[i]['scene_id'],
            'episode_id': str(episodes[i]['episode_id']),
            'trajectory_id': episodes[i].get('trajectory_id'),
            'route_member_keys': [list(k) for k in member_keys],
            'instruction': episodes[i]['instruction']['instruction_text'],
            'native_metrics': {k: progress[key][k] for k in ('success', 'spl', 'os', 'ne')},
            'native_actual_action_count': len(traces[key]),
            'native_nonstop_body_actions': body,
            'checkpoint_body_threshold': math.ceil(CHECKPOINT_RULE['body_fraction'] * body),
            'checkpoint_status': ('PENDING_NATIVE_REPLAY'
                                  if body - math.ceil(CHECKPOINT_RULE['body_fraction'] * body)
                                  >= CHECKPOINT_RULE['minimum_remaining_nonstop_body_actions']
                                  else 'INSUFFICIENT_BASELINE_BODY_BUDGET_KEEP_LOCKED'),
            'baseline_trace_sha256': digest(traces[key]),
        }
        buckets[stratum].append(row)
    selected = {name: balanced_prefix(rows) for name, rows in buckets.items()}
    cases = []
    for i in range(MAX_PER_STRATUM):
        for name in ('failure', 'success'):
            if i < len(selected[name]):
                row = dict(selected[name][i], order=len(cases))
                cases.append(row)
    counts = {name: len(rows) for name, rows in selected.items()}
    if not all(counts.values()):
        status = 'INSUFFICIENT_STRATA_STOP'
    elif all(n == MAX_PER_STRATUM for n in counts.values()):
        status = 'LOCKED_SELECTION_PENDING_CHECKPOINT_CAPTURE'
    else:
        status = 'LOCKED_FEWER_GROUPS_PENDING_CHECKPOINT_CAPTURE'
    return {
        'schema': 'crosscase_orientation_plan.v1', 'status': status,
        'split': 'val_unseen', 'source_native_episodes': len(episodes),
        'native_successes': sum(float(progress[k]['success']) == 1 for k in keys),
        'native_failures': sum(float(progress[k]['success']) == 0 for k in keys),
        'excluded_case15_route_groups': excluded,
        'available_route_groups': {name: len(rows) for name, rows in buckets.items()},
        'selected_counts': counts, 'maximum_per_stratum': MAX_PER_STRATUM,
        'selection_rule': 'Merge same-scene trajectory aliases and equal reference geometry '
                          'transitively. Exclude the whole case15 route. Routes with any native '
                          'failure enter the failure stratum. Prefer distinct scenes, then fill '
                          'in source annotation order. Choose at most three per stratum; no resampling.',
        'checkpoint_rule': CHECKPOINT_RULE,
        'arms': {'control': 0, 'left90': 90, 'right90': -90},
        'future_execution_constraints': {
            'position': 'Same physical checkpoint; rotations only, no external translation.',
            'motion': 'Use native turn actions; verify angle and unchanged position, no teleportation.',
            'history': 'Preserve policy history; log intermediate turn views without ingesting them. '
                       'Ingest the actual final observation once, normally.',
            'controls': 'Restore policy RNG and retain frozen parameters and the original action budget. '
                        'Require complete native-control reproduction before interpreting interventions.',
            'selection': 'Run all three declared headings for every eligible locked case. '
                         'Do not use intervention outcomes to choose cases or checkpoints.',
        },
        'cases': cases, 'navigation_executed': False, 'scope': SCOPE,
    }


def prepare(args):
    source, root = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Use a new run-dir; existing results are preserved')
    old = pilot.verify_manifest(source)
    summary = pilot.read_json(source / 'control/pilot_summary.json')
    dataset = pilot.read_json(source / 'episodes.json.gz')
    if (summary['status'] != 'complete' or summary['stage_synchronized']
            or summary['seed_base'] != old['seed_base'] or len(dataset['episodes']) != 20
            or int(summary['metrics']['length']) != 20):
        raise ValueError('Use the completed, unprofiled 20-episode native control pilot')
    cfg = pilot.load_config(source / 'control.py')
    if cfg.agent.model_settings.get('kv_cache_continuation', False):
        raise ValueError('Source must be the native cache-disabled control')
    habitat_path = (pilot.ROOT / cfg.env.env_settings['config_path']).resolve()
    import yaml
    habitat = yaml.safe_load(habitat_path.read_text(encoding='utf-8'))
    if habitat['habitat']['dataset']['split'] != 'val_unseen':
        raise ValueError('This predeclared development source is val_unseen, not the closed train cohort')
    progress = capture.read_progress(source / 'control/progress.json')
    traces = {capture.episode_key(ep): [] for ep in dataset['episodes']}
    for row in physical.read_lines(source / 'control/actions.jsonl'):
        key = capture.episode_key(row)
        if key not in traces:
            raise ValueError('Unexpected native trace episode')
        traces[key].append(row)
    plan = select_plan(dataset, progress, traces)
    plan['source_run'] = str(source)
    root.mkdir(parents=True, exist_ok=False)
    path = root / 'plan.json'
    pilot.write_json(path, plan)
    watched = [source / 'manifest.json', source / 'control/pilot_summary.json',
               source / 'control/progress.json', source / 'control/actions.jsonl',
               source / 'control.py', source / 'episodes.json.gz', habitat_path,
               Path(__file__).resolve(), Path(pilot.__file__).resolve(),
               Path(physical.__file__).resolve(), Path(capture.__file__).resolve(), path]
    hashes = dict(old['source_hashes'])
    hashes.update(old['generated_hashes'])
    hashes.update({str(p): pilot.sha256(p) for p in watched})
    pilot.write_json(root / 'cohort_manifest.json', {
        'schema': 'crosscase_orientation_inputs.v1', 'hashes': hashes,
        'checkpoint_shard_sizes': old['checkpoint_shard_sizes'], 'scope': SCOPE})
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.set_defaults(function=prepare)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
