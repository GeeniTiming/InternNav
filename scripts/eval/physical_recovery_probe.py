"""One reference-informed physical recovery in a locked native failure.

The policy remains frozen and cache-disabled. This diagnostic does not propose
an autonomous recovery algorithm or identify a memory cause. Intermediate
recovery views are recorded, not ingested into the policy's history.
"""
import argparse
import ast
import contextlib
import copy
import functools
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import textwrap
from types import SimpleNamespace

SCENE = 'zsNo4HB9uLZ'
INSTRUCTION = ('Walk straight past the table and turn right to go between the kitchen counters '
               'and walk straight past the refrigerator into the pantry and stop halfway '
               'between the two shelves on the right. ')
SCOPE = ('Single selected failure; reference-informed physical correction followed by frozen-policy '
         'continuation. Not an autonomous method, aggregate efficacy estimate, semantic-completion '
         'assessment, or isolated memory-causality test.')


def sibling(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('_physical_probe_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def validate_trace(rows, label):
    required = {'scene_id', 'episode_id', 'action', 'done', 'position', 'orientation_wxyz'}
    if not rows:
        raise ValueError(label + ': empty action trace')
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError('%s: record %d missing %s; keys=%s'
                             % (label, index, sorted(missing), sorted(row)))
        for field, length in (('position', 3), ('orientation_wxyz', 4)):
            values = row[field]
            if (not isinstance(values, list) or len(values) != length
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in values)):
                raise ValueError('%s: record %d has invalid %s' % (label, index, field))
        norm = math.sqrt(sum(v * v for v in row['orientation_wxyz']))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError('%s: record %d has a non-unit quaternion' % (label, index))
        if (not isinstance(row['scene_id'], str) or not row['scene_id']
                or isinstance(row['action'], bool) or not isinstance(row['action'], int)
                or row['action'] not in range(6) or not isinstance(row['done'], bool)):
            raise ValueError('%s: record %d has invalid action metadata' % (label, index))
    return rows


def normalize_reference(rows, repair_known_key_typo=False):
    """Rename only the reported key at record 10; never infer or change pose values."""
    records = copy.deepcopy(rows)
    edits = []
    alias, canonical = 'o rientation_wxyz', 'orientation_wxyz'
    for index, row in enumerate(records):
        if alias in row:
            if index != 10 or canonical in row:
                raise ValueError('Unexpected or ambiguous orientation key typo at record ' + str(index))
            if not repair_known_key_typo:
                raise ValueError('Source record 10 has key "o rientation_wxyz"; use '
                                 '--repair-known-key-typo to create an audited derived reference')
            value = row.pop(alias)
            row[canonical] = value
            edits.append({'record_index_zero_based': index, 'old_key': alias, 'new_key': canonical,
                          'quaternion_value_unchanged': copy.deepcopy(value)})
    validate_trace(records, 'reference')
    return records, edits


def check_previous_prefix(previous, source, reference, protocol, pilot):
    previous = Path(previous).resolve()
    manifest = pilot.read_json(previous / 'manifest.json')
    old_source = Path(manifest['physical_recovery_probe']['source_run']).resolve()
    if old_source != source:
        raise ValueError('Previous attempt used a different source run')
    raw = source / 'control/actions.jsonl'
    if manifest['source_hashes'].get(str(raw)) != pilot.sha256(raw):
        raise ValueError('Source bytes differ from those locked by the failed attempt')
    path = previous / 'control/probe_actions.jsonl'
    rows = validate_trace(read_lines(path), 'previous control prefix')
    if any(row.get('phase') != 'native' for row in rows):
        raise ValueError('Previous prefix includes a non-native phase')
    count = protocol['prefix_action_count']
    if len(rows) != count:
        raise ValueError('Previous attempt must contain exactly the declared 446-action prefix')
    comparison = pilot.compare_traces(reference[:count], rows)
    if not comparison['actions_and_poses_match']:
        raise ValueError('Derived reference disagrees with the independently captured prefix: '
                         + json.dumps(comparison))
    return {'path': str(path), 'sha256': pilot.sha256(path), 'comparison': comparison}


def locked_reference(root, pilot):
    return validate_trace(read_lines(root / 'reference_actions.jsonl'), 'locked derived reference')


def make_protocol(episode, trace):
    if (str(episode['episode_id']) != '15' or SCENE not in episode['scene_id']
            or episode['instruction']['instruction_text'] != INSTRUCTION):
        raise ValueError('This protocol is locked to scene zsNo4HB9uLZ, episode 15 and its exact instruction')
    if len(trace) != 547 or any(str(x['episode_id']) != '15' or SCENE not in x['scene_id'] for x in trace):
        raise ValueError('Expected the verified single-episode 547-action native trace')
    prefix_count = 446  # boundary before loop step 85; includes auxiliary camera actions in step 84
    trigger = [12.230002403259277, 0.17162801325321198, -3.893124580383301]
    anchor = [12.738821983337402, 0.17162801325321198, -2.49418568611145]
    if math.dist(trace[prefix_count - 1]['position'], trigger) > 1e-5:
        raise ValueError('Recorded intervention boundary does not match the audited case')
    if trace[prefix_count - 1]['action'] != 4 or trace[prefix_count - 1]['done']:
        raise ValueError('Expected an unfinished, camera-level native prefix')
    ref = episode['reference_path']
    if len(ref) != 6 or math.dist(ref[2], [12.461600303649902, trigger[1], -2.3902199268341064]) > 1e-5:
        raise ValueError('Reference corner changed')
    direction = [ref[3][i] - ref[2][i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in direction))
    target = [ref[2][i] + direction[i] / length for i in range(3)]
    goal = episode['goals'][0]
    exclusion = float(goal.get('radius') or 3.0) + 1.0
    if math.dist(target, goal['position']) <= exclusion:
        raise ValueError('Handoff must remain more than one metre outside the goal radius')
    return {'schema': 'case15_physical_recovery.v1', 'scene': SCENE, 'episode': '15',
            'trigger_step': 85, 'prefix_action_count': prefix_count,
            'trigger_position': trigger, 'waypoints': [anchor, target],
            'goal_position': goal['position'], 'goal_radius_m': float(goal.get('radius') or 3.0),
            'handoff_goal_exclusion_m': exclusion, 'follower_radius_m': 0.30,
            'endpoint_tolerance_m': 0.35, 'max_recovery_actions': 80,
            'pose_tolerance': 1e-5, 'max_single_action_displacement_m': 0.31,
            'history_rule': 'Preserve existing rgb_list. Recovery views are logged only. '
                            'At the next native iteration ingest the true final observation once, normally.',
            'scope': SCOPE}


def prepare(args):
    pilot, capture = sibling('cache_pilot'), sibling('failure_case_probe')
    source, target = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if target.exists():
        raise FileExistsError('Use a new run-dir; no overwrite or resume')
    manifest = pilot.verify_manifest(source)
    source_summary = pilot.read_json(source / 'control/pilot_summary.json')
    if source_summary['status'] != 'complete' or source_summary['stage_synchronized']:
        raise ValueError('Require a completed, unprofiled native source')
    dataset = pilot.read_json(source / 'episodes.json.gz')
    if len(dataset['episodes']) != 1:
        raise ValueError('Use the verified single-episode failure capture, not a multi-episode pilot')
    raw_reference = source / 'control/actions.jsonl'
    trace, edits = normalize_reference(read_lines(raw_reference),
                                       getattr(args, 'repair_known_key_typo', False))
    protocol = make_protocol(dataset['episodes'][0], trace)
    previous = getattr(args, 'previous_attempt', None)
    previous_check = check_previous_prefix(previous, source, trace, protocol, pilot) if previous else None
    audit = {'schema': 'case15_reference_derivation.v1', 'source_path': str(raw_reference),
             'source_sha256': pilot.sha256(raw_reference), 'record_count': len(trace),
             'edits': edits, 'previous_prefix_check': previous_check,
             'scope': 'Derived copy; original file and manifests remain unchanged. '
                      'Only an explicitly allowed key is renamed. No pose value is filled or modified.'}
    # Reuse native config/data/seed preparation without modifying either helper.
    capture.prepare(SimpleNamespace(source_run=str(source), run_dir=str(target),
                                    episode='15', scene=SCENE), pilot)
    generated = pilot.read_json(target / 'manifest.json')
    reference_path = target / 'reference_actions.jsonl'
    reference_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in trace),
                              encoding='utf-8')
    audit['derived_path'] = str(reference_path)
    audit['derived_sha256'] = pilot.sha256(reference_path)
    audit_path = target / 'reference_audit.json'
    pilot.write_json(audit_path, audit)
    for path in (reference_path, audit_path):
        generated['generated_hashes'][str(path)] = pilot.sha256(path)
    if previous_check:
        previous_path = Path(previous_check['path'])
        generated['source_hashes'][str(previous_path)] = previous_check['sha256']
    native = pilot.load_config(target / 'control.py')
    for arm in ('control', 'repair'):
        cfg = copy.deepcopy(native)
        cfg.eval_settings['output_path'] = str(target / arm)
        cfg.agent.model_settings['kv_cache_continuation'] = False
        cfg.agent.model_settings['vis_debug'] = True
        cfg.agent.model_settings['vis_debug_path'] = str(target / 'video' / arm)
        serialized = json.dumps(cfg.model_dump(mode='json', exclude_none=True))
        config_file = target / (arm + '.py')
        config_file.write_text(
            'import json\nfrom internnav.configs.evaluator import EvalCfg\n'
            + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n',
            encoding='utf-8')
        generated['generated_hashes'][str(config_file)] = pilot.sha256(config_file)
    protocol_path = target / 'recovery_protocol.json'
    pilot.write_json(protocol_path, protocol)
    generated['generated_hashes'][str(protocol_path)] = pilot.sha256(protocol_path)
    for path in (Path(__file__).resolve(), Path(__file__).with_name('failure_case_probe.py')):
        generated['source_hashes'][str(path)] = pilot.sha256(path)
    generated['physical_recovery_probe'] = {
        'source_run': str(source), 'scope': SCOPE,
        'unused_cache_config': 'cache.py is inherited from preparation and is never run',
        'arms': ['control', 'repair']}
    pilot.write_json(target / 'manifest.json', generated)
    print('PHYSICAL_RECOVERY_PREPARE=PASS', 'step=85', 'run_dir=', target)
    print('REFERENCE_SCHEMA_CHECK=PASS', 'records=', len(trace), 'key_repairs=', len(edits))
    if previous_check:
        print('PREVIOUS_PREFIX_CHECK=PASS', 'actions=', protocol['prefix_action_count'])
    print('LOCKED_WAYPOINTS:', json.dumps(protocol['waypoints']))


def instrument_loop(function):
    """Add one process-local callback before observation ingestion; edit no source file."""
    if function.__code__.co_freevars:
        raise ValueError('Unexpected evaluator closure')
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.While)
                  and any(isinstance(n, ast.Attribute) and n.attr == 'max_steps_per_episode'
                          for n in ast.walk(node.test))]
    if len(candidates) != 1:
        raise ValueError('Evaluator loop shape changed; refusing to instrument')
    hook = ast.parse(
        'observations, done = self._physical_recovery_probe.on_boundary('
        'step_id, observations, done, rgb_list, action_seq, pixel_goal, local_actions, action)'
    ).body[0]
    candidates[0].body.insert(0, hook)
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, '<case15-loop-boundary>', 'exec'), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


def observation_hashes(observations):
    import numpy as np

    def feed(digest, value):
        if isinstance(value, dict):
            digest.update(b'dict')
            for key in sorted(value):
                digest.update(json.dumps(key, ensure_ascii=False).encode('utf-8'))
                feed(digest, value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for item in value:
                feed(digest, item)
        elif isinstance(value, (str, bytes)) or value is None:
            digest.update(type(value).__name__.encode())
            digest.update(value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8'))
        else:
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject:
                raise ValueError('Cannot fingerprint object-valued observation')
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())

    result = {}
    for name, value in sorted(observations.items()):
        digest = hashlib.sha256()
        feed(digest, value)
        result[name] = digest.hexdigest()
    return result

def history_hash(images):
    digest = hashlib.sha256()
    for image in images:
        digest.update(str((image.mode, image.size)).encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def preserve_policy_rng():
    import numpy as np
    import torch
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        yield
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        torch.cuda.set_rng_state_all(before[3])


def drive_waypoints(waypoints, next_action, advance, position, budget, tolerance, max_displacement):
    """Never execute the follower's STOP; it terminates the VLN episode."""
    count = 0
    for goal in waypoints:
        while True:
            action = next_action(goal)
            if action is None:
                raise RuntimeError('RECOVERY_FAILED: follower returned no action')
            action = int(action)
            if action == 0:
                if math.dist(position(), goal) > tolerance:
                    raise RuntimeError('RECOVERY_FAILED: follower stopped before reaching the waypoint')
                break
            if action not in (1, 2, 3):
                raise RuntimeError('RECOVERY_FAILED: unexpected non-motion follower action')
            if count >= budget:
                raise RuntimeError('RECOVERY_FAILED: declared action budget exhausted')
            before = list(position())
            result = advance(action)
            count += 1
            if math.dist(before, position()) > max_displacement:
                raise RuntimeError('RECOVERY_FAILED: motion exceeds one native action')
            if bool(result[2]):
                raise RuntimeError('RECOVERY_FAILED: environment terminated during correction')
    return count


class BoundaryProbe:
    def __init__(self, evaluator, run_dir, arm, pilot, protocol, reference):
        self.evaluator, self.root, self.arm, self.pilot = evaluator, run_dir, arm, pilot
        self.protocol, self.reference = protocol, reference
        self.directory = run_dir / arm
        self.records = []
        self.phase, self.current_step, self.triggered = 'native', None, False
        self.status = 'running'
        self.stream = (self.directory / 'probe_actions.jsonl').open('x', encoding='utf-8')
        self.original_step = evaluator.env.step
        self.gate = None
        self.recovery_actions = 0
        self.post_boundary = None

    def record(self, action):
        result = self.original_step(action)
        state = self.evaluator.env._env.sim.get_agent_state()
        episode = self.evaluator.env.get_current_episode()
        q = state.rotation
        entry = {'scene_id': episode.scene_id, 'episode_id': str(episode.episode_id),
                 'action': int(action), 'done': bool(result[2]),
                 'position': [float(v) for v in state.position],
                 'orientation_wxyz': [float(q.w), float(q.x), float(q.y), float(q.z)],
                 'phase': self.phase, 'main_step': self.current_step}
        self.records.append(entry)
        self.stream.write(json.dumps(entry) + '\n')
        self.stream.flush()
        if self.phase == 'recovery':
            self.recovery_actions += 1
            from PIL import Image
            folder = self.directory / 'recovery_frames'
            folder.mkdir(exist_ok=True)
            Image.fromarray(result[0]['rgb']).save(folder / ('%03d.jpg' % self.recovery_actions))
            distance = float(self.evaluator.env.get_metrics()['distance_to_goal'])
            if not math.isfinite(distance) or distance <= self.protocol['goal_radius_m']:
                raise RuntimeError('RECOVERY_FAILED: physical correction reached the goal region or invalid distance')
        return result

    def state(self):
        state = self.evaluator.env._env.sim.get_agent_state()
        q = state.rotation
        distance = float(self.evaluator.env.get_metrics()['distance_to_goal'])
        return {'position': [float(v) for v in state.position],
                'orientation_wxyz': [float(q.w), float(q.x), float(q.y), float(q.z)],
                'distance_to_goal': distance}

    def on_boundary(self, step, observations, done, history, action_seq, pixel_goal, local_actions, action):
        self.current_step = int(step)
        if step != self.protocol['trigger_step'] or self.triggered:
            return observations, done
        if done or action_seq or pixel_goal is not None or local_actions or action not in (0, 1, 2, 3):
            raise RuntimeError('NOT_COMPARABLE: expected an idle, level-camera S2 decision boundary')
        count = self.protocol['prefix_action_count']
        comparison = self.pilot.compare_traces(self.reference[:count], self.records)
        if len(self.records) != count or not comparison['actions_and_poses_match']:
            self.pilot.write_json(self.directory / 'prefix_mismatch.json', comparison)
            raise RuntimeError('NOT_COMPARABLE: native prefix differs; no correction was applied')
        self.gate = {'step': step, 'prefix_action_count': count,
                     'state': self.state(), 'observation_hashes': observation_hashes(observations),
                     'history_length': len(history), 'history_sha256': history_hash(history),
                     'prefix_comparison': comparison}
        if self.arm == 'repair':
            control = self.pilot.read_json(self.root / 'control/boundary_gate.json')
            for field in ('step', 'prefix_action_count', 'state', 'observation_hashes',
                          'history_length', 'history_sha256'):
                if self.gate[field] != control[field]:
                    raise RuntimeError('NOT_COMPARABLE: control/repair boundary differs: ' + field)
        self.pilot.write_json(self.directory / 'boundary_gate.json', self.gate)
        from PIL import Image
        Image.fromarray(observations['rgb']).save(self.directory / 'before_recovery.png')
        self.triggered = True
        print('BOUNDARY_GATE=PASS', self.arm, 'step=', step, 'native_actions=', count)
        if self.arm == 'control':
            return observations, done
        before_history = history_hash(history)
        parameters = [(name, parameter._version) for name, parameter in self.evaluator.model.named_parameters()]
        with preserve_policy_rng():
            observations = self.recover(observations)
        if history_hash(history) != before_history:
            raise RuntimeError('History was mutated during physical correction')
        if parameters != [(name, parameter._version) for name, parameter in self.evaluator.model.named_parameters()]:
            raise RuntimeError('Model parameter versions changed')
        self.phase = 'native_after_recovery'
        self.post_boundary = {'state': self.state(), 'recovery_actions': self.recovery_actions,
                              'history_unchanged': True, 'parameter_versions_unchanged': True,
                              'policy_rng_restored': True,
                              'observation_hashes': observation_hashes(observations),
                              'history_rule': self.protocol['history_rule']}
        self.pilot.write_json(self.directory / 'after_recovery.json', self.post_boundary)
        Image.fromarray(observations['rgb']).save(self.directory / 'after_recovery.png')
        print('PHYSICAL_RECOVERY=PASS', json.dumps(self.post_boundary['state']))
        return observations, False

    def recover(self, observations):
        import numpy as np
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        import habitat_sim
        sim = self.evaluator.env._env.sim
        waypoints = [np.asarray(p, dtype=np.float32) for p in self.protocol['waypoints']]
        start = sim.get_agent_state().position
        for target in waypoints:
            if not sim.pathfinder.is_navigable(target):
                raise RuntimeError('RECOVERY_FAILED: locked waypoint is not navigable; do not move or retarget')
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = start, target
            if not sim.pathfinder.find_path(path) or not math.isfinite(float(path.geodesic_distance)):
                raise RuntimeError('RECOVERY_FAILED: locked recovery path is unreachable')
            start = target
        follower = ShortestPathFollower(sim, self.protocol['follower_radius_m'], return_one_hot=False)
        latest = [observations]
        self.phase = 'recovery'

        def advance(action):
            result = self.evaluator.env.step(action)
            latest[0] = result[0]
            return result

        drive_waypoints(waypoints, follower.get_next_action, advance,
                        lambda: sim.get_agent_state().position,
                        self.protocol['max_recovery_actions'],
                        self.protocol['endpoint_tolerance_m'],
                        self.protocol['max_single_action_displacement_m'])
        state = self.state()
        if (math.dist(state['position'], self.protocol['waypoints'][-1]) > self.protocol['endpoint_tolerance_m']
                or not math.isfinite(state['distance_to_goal'])
                or state['distance_to_goal'] <= self.protocol['handoff_goal_exclusion_m']):
            raise RuntimeError('RECOVERY_FAILED: final handoff position is invalid or too close to the goal')
        return latest[0]

    def close(self, error=None):
        self.stream.close()
        self.pilot.write_json(self.directory / 'recovery_summary.json', {
            'status': self.status, 'error': error, 'arm': self.arm, 'trigger_reached': self.triggered,
            'recovery_actions': self.recovery_actions, 'handoff': self.post_boundary, 'scope': SCOPE})


def control_check(root, pilot, manifest):
    source = Path(manifest['physical_recovery_probe']['source_run'])
    a, b = [pilot.read_json(p / 'control/pilot_summary.json') for p in (source, root)]
    for field in ('status', 'seed_base', 'stage_synchronized', 'torch', 'cuda', 'transformers', 'device'):
        if a[field] != b[field] or (field == 'status' and b[field] != 'complete'):
            raise RuntimeError('NOT_COMPARABLE: control runtime differs: ' + field)
    comparison = pilot.compare_traces(locked_reference(root, pilot),
                                     validate_trace(read_lines(root / 'control/actions.jsonl'), 'control'))
    if not comparison['actions_and_poses_match'] or not pilot.metrics_match(a['metrics'], b['metrics']):
        raise RuntimeError('NOT_COMPARABLE: control must reproduce the entire native source')
    diagnostic = pilot.read_json(root / 'control/recovery_summary.json')
    if diagnostic['status'] != 'complete' or not diagnostic['trigger_reached']:
        raise RuntimeError('NOT_COMPARABLE: control did not reach the declared boundary')
    return comparison


def run(args):
    pilot = sibling('cache_pilot')
    root = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(root)
    protocol = pilot.read_json(root / 'recovery_protocol.json')
    source = Path(manifest['physical_recovery_probe']['source_run'])
    reference = locked_reference(root, pilot)
    if args.arm == 'repair':
        control_check(root, pilot, manifest)
    import sys
    sys.path.insert(0, str(pilot.ROOT))
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator
    cls = Evaluator.evaluators['habitat_vln']  # decorator's module-level class symbol is None
    native_init, native_loop = cls.__init__, cls._run_eval_dual_system
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get('kv_cache_continuation', False):
            raise RuntimeError('Physical recovery diagnostic requires native cache-disabled inference')
        native_init(evaluator, cfg)
        probe = BoundaryProbe(evaluator, root, args.arm, pilot, protocol, reference)
        evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record
        probes.append(probe)

    wrapped_loop = instrument_loop(native_loop)
    cls.__init__ = initialize
    cls._run_eval_dual_system = wrapped_loop
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1 or not probes[0].triggered:
            raise RuntimeError('NOT_COMPARABLE: expected one evaluator and one declared boundary')
        if args.arm == 'repair' and probes[0].post_boundary is None:
            raise RuntimeError('RECOVERY_FAILED: handoff was not completed')
        probes[0].status = 'complete'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = native_init, native_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.close(error)
    if args.arm == 'control':
        control_check(root, pilot, manifest)
        print('NATIVE_CONTROL_REPRODUCTION=PASS')


def report(args):
    pilot = sibling('cache_pilot')
    root = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(root)
    comparison = control_check(root, pilot, manifest)
    summaries = {arm: pilot.read_json(root / arm / 'pilot_summary.json') for arm in ('control', 'repair')}
    diagnostic = pilot.read_json(root / 'repair/recovery_summary.json')
    for arm, summary in summaries.items():
        if summary['status'] != 'complete':
            raise RuntimeError('Incomplete rollout: ' + arm)
    for field in ('seed_base', 'stage_synchronized', 'torch', 'cuda', 'transformers', 'device'):
        if summaries['control'][field] != summaries['repair'][field]:
            raise RuntimeError('NOT_COMPARABLE: arm runtime differs: ' + field)
    if diagnostic['status'] != 'complete' or not diagnostic['handoff']:
        raise RuntimeError('Physical recovery did not produce a valid autonomous handoff')
    records = read_lines(root / 'repair/probe_actions.jsonl')
    future = [row for row in records if row['phase'] == 'native_after_recovery' and row['action'] in (0, 1, 2, 3)]
    handoff = diagnostic['handoff']
    protocol = pilot.read_json(root / 'recovery_protocol.json')
    goal = protocol['goal_position']
    before = handoff['state']['position']
    followups = []
    for row in future[:12]:
        followups.append({'step': row['main_step'], 'action': row['action'], 'position': row['position'],
                          'distance_to_handoff_xyz_m': math.dist(row['position'], before),
                          'euclidean_distance_to_goal_m_OFFLINE': math.dist(row['position'], goal),
                          'done': row['done']})
    result = {'status': 'READY_FOR_RECOVERY_REVIEW', 'control_exact_reproduction': comparison,
              'control_metrics': summaries['control']['metrics'], 'repair_metrics': summaries['repair']['metrics'],
              'physical_recovery': handoff, 'autonomous_body_action_count': len(future),
              'first_autonomous_actions': followups,
              'videos': [str(p) for p in sorted((root / 'video').rglob('*.mp4'))],
              'recovery_frames': str(root / 'repair/recovery_frames'), 'scope': SCOPE}
    pilot.write_json(root / 'physical_recovery_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--repair-known-key-typo', action='store_true',
                      help='Allow only the reported spaced key at source record 10, in a derived copy')
    prep.add_argument('--previous-attempt',
                      help='CPU-check the corrected reference against an existing failed control prefix')
    prep.set_defaults(function=prepare)
    execute = commands.add_parser('run')
    execute.add_argument('--run-dir', required=True)
    execute.add_argument('--arm', choices=('control', 'repair'), required=True)
    execute.set_defaults(function=run)
    summary = commands.add_parser('report')
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
