#!/usr/bin/env python3
"""CPU-only, read-only audit of completed InternNav native trajectories.

Inputs: two locked H1 cohort protocols and their previously captured native runs.
Writes a CREATE-ONLY audit directory, never mutates originals or selects episodes
by intervention outcomes. Anomaly events are retrospective *candidates*, not
online labels, mechanistic diagnoses, or MOON geodesic LOST judgments.

Action codes (verified from InternNav checkpoint instrumentation):
  0: termination; 1: forward; 2,3: body turns; 4,5: camera pitch motions.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_BASE = Path('/mnt/data8t/geenitiming/internnav_data')
COHORTS = {
    'development': 'intervention_v01_paired_all10_01',
    'validation': 'h1_validation_paired_all11_01',
}
STILL_EPS = 1e-4  # m per atomic action: strict position stasis
MIN_STILL_ACTIONS = 30
MIN_BLOCKED_FORWARD = 5
MIN_TURNS = 12
NEAR_GOAL_EUCLID_M = 1.0
ROUTE_PROXY_M = 2.0


def canonical_key(scene_id, episode_id):
    p = Path(str(scene_id))
    s = p.stem if p.suffix == '.glb' else p.name
    return (s, str(episode_id))


def xyz(value):
    if isinstance(value, dict):
        value = value['position']
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'Expected XYZ point: {value!r}')
    p = [float(x) for x in value]
    if not all(math.isfinite(x) for x in p):
        raise ValueError('Nonfinite XYZ point')
    return p


def euclid_route_proxy(p, route):
    """Distance to straight 3D interpolation; NOT a geodesic metric."""
    best = float('inf')
    for a, b in zip(route, route[1:]):
        ab = [b[i] - a[i] for i in range(3)]
        ap = [p[i] - a[i] for i in range(3)]
        denom = sum(x*x for x in ab)
        t = max(0., min(1., sum(ap[i]*ab[i] for i in range(3))/denom)) if denom else 0.
        q = [a[i] + t*ab[i] for i in range(3)]
        best = min(best, math.dist(p, q))
    return best


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{no}: invalid JSON') from exc


def collect_by_episode(path, field='scene_id'):
    out = defaultdict(list)
    for row in read_jsonl(path):
        if field == 'scene_id':
            k = canonical_key(row['scene_id'], row['episode_id'])
        elif field == 'episode_key':
            k = canonical_key(*row['episode_key'])
        elif field == 'key':
            k = canonical_key(*row['key'])
        else:
            raise ValueError('Unsupported identity field')
        out[k].append(row)
    return dict(out)


def stationary_intervals(deltas, eps=STILL_EPS):
    """1-indexed action-inclusive maximal intervals with no positional change."""
    start = None
    for i in range(1, len(deltas)+1):
        is_still = i < len(deltas) and deltas[i] <= eps
        if is_still and start is None:
            start = i
        if not is_still and start is not None:
            yield start, i-1
            start = None


def preceding(records, action_index, prefix_name):
    candidates = [r for r in records if int(r[prefix_name]) <= action_index]
    return max(candidates, key=lambda r: int(r[prefix_name])) if candidates else None


def following(records, action_index, prefix_name):
    candidates = [r for r in records if int(r[prefix_name]) >= action_index]
    return min(candidates, key=lambda r: int(r[prefix_name])) if candidates else None


def link_record(rec, prefix_name, s2=False):
    if rec is None:
        return None
    out = {'action_prefix_count': int(rec[prefix_name]), 'main_step': int(rec['main_step'] if s2 else rec['step'])}
    if s2:
        out.update(call_id=rec['call_id'], call_type=rec['call_type'])
    else:
        out.update(eligible=rec.get('eligible'))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite / modify experimental outputs

    summary = []
    events = []
    audit_meta = {'schema':'vln_native_anomaly_audit.v1',
                  'method':'Retrospective geometric/behavioral candidate mining',
                  'no_frozen_source_modified':True,
                  'no_new_rollouts':True,
                  'privileged_fields':'OFFLINE_ONLY',
                  'distance_metric':'3D Euclidean to interpolated reference_path; NOT geodesic',
                  'thresholds': {'still_eps_m':STILL_EPS, 'min_still_actions':MIN_STILL_ACTIONS,
                                 'min_blocked_forward':MIN_BLOCKED_FORWARD,'min_turns':MIN_TURNS,
                                 'near_goal_euclid_m':NEAR_GOAL_EUCLID_M,
                                 'route_euclid_proxy_m':ROUTE_PROXY_M},
                  'sources':{}}

    for cohort, folder in COHORTS.items():
        root = args.base / folder
        protocol = json.loads((root/'paired_protocol.json').read_text())
        native = Path(protocol['source_native_run'])
        gates = Path(protocol['source_natural_gate_run'])
        with gzip.open(native/'episodes.json.gz', 'rt', encoding='utf-8') as f:
            episodes = json.load(f)['episodes']
        epis = {canonical_key(e['scene_id'],e['episode_id']):e for e in episodes}
        if len(epis) != len(episodes):
            raise ValueError('Repeated canonical episode key in '+str(native))
        actions = collect_by_episode(native/'control/actions.jsonl')
        report = json.loads((gates/'natural_checkpoint_report.json').read_text())
        cases = {canonical_key(*c['episode_key']):c for c in report['cases']}
        if not set(epis) == set(actions) == set(cases):
            raise ValueError(f'{cohort} episode ID sets do not match across episodes/actions/gates')

        boundary_path = gates/'control/boundary_events.jsonl'
        if not boundary_path.is_file():
            raise FileNotFoundError(f'Missing recorded main-loop boundary evidence: {boundary_path}')
        boundaries = collect_by_episode(boundary_path, field='key')
        s2_path = native/'capture/online_decisions.jsonl'
        has_s2 = s2_path.is_file()
        s2_events = collect_by_episode(s2_path, field='episode_key') if has_s2 else {}
        audit_meta['sources'][cohort]={'paired':str(root),'native':str(native),'gate':str(gates),
                                       'true_s2_calls_captured':bool(has_s2)}
        if len(episodes) != 12:
            raise ValueError(f'Expected 12 pre-registered episodes in {cohort}, saw {len(episodes)}')

        for k in sorted(epis):
            ep = epis[k]
            rows = actions[k]
            if not rows:
                raise ValueError(f'Empty native trace {k}')
            path = [xyz(p) for p in ep['reference_path']]
            if len(path)<2:
                raise ValueError(f'Invalid reference path {k}')
            goal = xyz(ep['goals'][0])
            poses = [xyz(ep['start_position'])] + [xyz(r['position']) for r in rows]
            delta = [0.] + [math.dist(a,b) for a,b in zip(poses,poses[1:])]
            goal_dist = [math.dist(x,goal) for x in poses]
            proxy_dist = [euclid_route_proxy(x,path) for x in poses]
            nearest_i = min(range(len(poses)),key=goal_dist.__getitem__)
            gate = cases[k]['natural_checkpoint']
            prefix = gate.get('prefix_action_count') if gate else None
            bnd = boundaries.get(k,[])
            s2 = s2_events.get(k,[])
            for seq, prefix_name in [(bnd,'prefix_action_count'),(s2,'native_action_prefix_count')]:
                if any(int(x[prefix_name])<0 or int(x[prefix_name])>len(rows) for x in seq):
                    raise ValueError(f'Out-of-range boundary action prefix: {cohort} {k}')
            bnd.sort(key=lambda r:int(r['prefix_action_count']))
            s2.sort(key=lambda r:int(r['native_action_prefix_count']))

            def register(kind, start, end, *, evidence):
                if not (0<=start<=end<=len(rows)):
                    raise ValueError('Bad event indices')
                segment = rows[max(0,start-1):end]
                counts = Counter(int(r['action']) for r in segment)
                item = {
                    'cohort':cohort,'scene':k[0],'episode':k[1],
                    'type':kind,'start_action':start,'end_action':end,
                    'length_actions':end-start+1,
                    'forward_attempts':counts[1], 'turn_actions':counts[2]+counts[3],
                    'camera_actions':counts[4]+counts[5],
                    'translation_m':round(sum(delta[max(1,start):end+1]),6),
                    'start_goal_euclidean_m':round(goal_dist[start],4),
                    'end_goal_euclidean_m':round(goal_dist[end],4),
                    'start_route_proxy_m':round(proxy_dist[start],4),
                    'end_route_proxy_m':round(proxy_dist[end],4),
                    'native_success':int(cases[k]['baseline_native_success']),
                    'gate_action_count':prefix,
                    'preceding_main_loop_boundary':link_record(preceding(bnd,start,'prefix_action_count'),'prefix_action_count'),
                    'following_main_loop_boundary':link_record(following(bnd,start,'prefix_action_count'),'prefix_action_count'),
                    'preceding_actual_s2':link_record(preceding(s2,start,'native_action_prefix_count'),'native_action_prefix_count',True),
                    'following_actual_s2':link_record(following(s2,start,'native_action_prefix_count'),'native_action_prefix_count',True),
                    'true_s2_log_available':has_s2,
                    'evidence_OFFLINE_ONLY':evidence,
                }
                events.append(item)

            for start,end in stationary_intervals(delta):
                if end-start+1 < MIN_STILL_ACTIONS:
                    continue
                codes=Counter(int(r['action']) for r in rows[start-1:end])
                attempts = codes[1]
                turns = codes[2]+codes[3]
                evidence={'stationary_actions':end-start+1,
                          'actions':{str(i):codes[i] for i in range(6)},
                          'goal_euclid_min_m':round(min(goal_dist[start-1:end+1]),4)}
                if attempts>=MIN_BLOCKED_FORWARD:
                    # Intervening turns/camera steps permitted; full pose remains stationary.
                    register('forward_without_translation_candidate',start,end,evidence=evidence)
                if turns>=MIN_TURNS and attempts==0:
                    register('in_place_turning_candidate',start,end,evidence=evidence)
                if min(goal_dist[start-1:end+1])<=NEAR_GOAL_EUCLID_M and turns>=MIN_TURNS:
                    register('near_goal_stationary_turning_candidate',start,end,evidence=evidence)

            # These are sequence-level retrospective outcomes, NOT online triggers.
            if goal_dist[nearest_i] <= NEAR_GOAL_EUCLID_M and goal_dist[-1] > 3.0 and not cases[k]['baseline_native_success']:
                register('near_goal_then_departed_candidate',nearest_i,len(rows),evidence={
                    'minimum_goal_euclidean_m':round(goal_dist[nearest_i],4),
                    'final_goal_euclidean_m':round(goal_dist[-1],4),
                    'path_length_after_nearest_m':round(sum(delta[nearest_i+1:]),4)})

            # A line-segment Euclidean *proxy*, NOT the geodesic-distance threshold in MOON.
            start=None
            for i in range(len(proxy_dist)+1):
                high=i<len(proxy_dist) and proxy_dist[i]>ROUTE_PROXY_M
                if high and start is None: start=i
                if not high and start is not None:
                    register('route_euclidean_proxy_over_2m',start,i-1,evidence={
                        'maximum_proxy_m':round(max(proxy_dist[start:i]),4),
                        'NOT_A_MOON_GEODESIC_LOST_LABEL':True})
                    start=None

            counts = Counter(int(r['action']) for r in rows)
            summary.append({
                'cohort':cohort,'scene':k[0],'episode':k[1],
                'native_success':int(cases[k]['baseline_native_success']),
                'native_termination':cases[k].get('native_termination'),
                'native_action_count':len(rows),'body_actions':counts[1]+counts[2]+counts[3],
                'forward_actions':counts[1],'turn_actions':counts[2]+counts[3],
                'camera_actions':counts[4]+counts[5],
                'final_done':bool(rows[-1]['done']),
                'total_translation_m':round(sum(delta),4),
                'minimum_goal_euclidean_m':round(min(goal_dist),4),
                'minimum_goal_action':nearest_i,
                'final_goal_euclidean_m':round(goal_dist[-1],4),
                'max_route_euclidean_proxy_m':round(max(proxy_dist),4),
                'original_gate_action_prefix':prefix,
                'S2_call_record_count':len(s2),
                'main_loop_boundaries_recorded':len(bnd),
            })

    with (out/'episode_summary.csv').open('x',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with (out/'event_candidates.jsonl').open('x',encoding='utf-8') as f:
        for event in events:
            f.write(json.dumps(event,ensure_ascii=False)+'\n')
    audit_meta.update(N_episodes=len(summary),N_event_candidates=len(events),
                      candidate_counts=dict(Counter(x['type'] for x in events)),
                      native_failures=sum(1 for x in summary if not x['native_success']))
    (out/'audit_manifest.json').write_text(json.dumps(audit_meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('NATIVE_ANOMALY_AUDIT=PASS')
    print('EPISODES:',len(summary),'NATIVE_FAILURES:',audit_meta['native_failures'])
    print('CANDIDATE_EVENTS:',len(events))
    for k,v in sorted(audit_meta['candidate_counts'].items()):
        print(f'  {k}: {v}')
    print('RESULT_DIR:',out)
    print('WARNING: all candidates are retrospective; geodesic LOST / semantic root cause not established')


if __name__=='__main__':
    main()
