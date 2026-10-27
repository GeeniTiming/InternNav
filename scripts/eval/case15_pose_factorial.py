"""Case15: a predeclared two-position by three-heading physical diagnostic.

Return targets come only from the recorded native prefix. No new reference-route
segment is supplied. A simulator shortest-path follower is privileged execution,
so this is not an autonomous or deployable recovery method.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case15_reinstruction_probe as prompt
physical, pilot = prompt.physical, prompt.pilot

ARMS = {
    'control': ('trigger', 0),
    'turn_left90': ('trigger', 90),
    'turn_right90': ('trigger', -90),
    'anchor_same_heading': ('anchor', 0),
    'anchor_left90': ('anchor', 90),
    'anchor_right90': ('anchor', -90),
}
ANCHOR_INDEX = 393  # recorded post-action position, before the audited main-step-76 decision
SCOPE = (
    'One previously inspected val-unseen development case, with predeclared physical '
    'pose interventions. Anchor selection is case-specific; execution uses privileged '
    'simulator geometry. No new reference-route segment, autonomous recovery method, '
    'aggregate efficacy, semantic-completion or isolated memory-causality claim. '
    'Heading changes also change visual input and incur actual action cost.'
)


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def yaw_degrees(q):
    if len(q) != 4 or abs(q[1]) > 1e-5 or abs(q[3]) > 1e-5:
        raise ValueError('Expected a level-body wxyz yaw quaternion')
    if not all(math.isfinite(v) for v in q) or math.hypot(q[0], q[2]) < 0.99:
        raise ValueError('Invalid body quaternion')
    return wrap_degrees(math.degrees(2.0 * math.atan2(q[2], q[0])))


def yaw_quaternion(angle):
    half = math.radians(angle) / 2.0
    return [math.cos(half), 0.0, math.sin(half), 0.0]


def heading_error(a, b):
    return abs(wrap_degrees(yaw_degrees(a) - yaw_degrees(b)))


def native_turn_angle(rows):
    angles = [abs(wrap_degrees(yaw_degrees(row['orientation_wxyz'])
                              - yaw_degrees(rows[i - 1]['orientation_wxyz'])))
              for i, row in enumerate(rows) if i and row['action'] in (2, 3)]
    if not angles or any(abs(a - 15.0) > 0.01 for a in angles):
        raise ValueError('The locked native trace must use 15-degree body turns')
    return 15.0


def make_protocol(reference, boundary):
    count = boundary['prefix_action_count']
    if count != 446 or boundary['trigger_step'] != 85 or len(reference) != 547:
        raise ValueError('Expected the locked 446-action/step-85 native boundary')
    anchor, trigger = reference[ANCHOR_INDEX], reference[count - 1]
    expected = [12.738821983337402, 0.17162801325321198, -2.49418568611145]
    if anchor['action'] != 1 or math.dist(anchor['position'], expected) > 1e-5:
        raise ValueError('Recorded historical anchor changed')
    turn = native_turn_angle(reference[:count])
    yaw = yaw_degrees(trigger['orientation_wxyz'])
    candidates = {}
    for arm, (location, offset) in ARMS.items():
        candidates[arm] = {
            'location': location, 'heading_offset_deg': offset,
            'position': list(anchor['position'] if location == 'anchor' else trigger['position']),
            'orientation_wxyz': yaw_quaternion(yaw + offset),
        }
    return {
        'schema': 'case15_pose_factorial.v1', 'trigger_step': 85,
        'prefix_action_count': count, 'arms': list(ARMS), 'candidates': candidates,
        'anchor_source': {'raw_action_index_zero_based': ANCHOR_INDEX,
                          'position': anchor['position'], 'orientation_wxyz': anchor['orientation_wxyz']},
        'trigger_source': {'position': trigger['position'], 'orientation_wxyz': trigger['orientation_wxyz']},
        'original_anchor_vs_trigger_heading_difference_deg': heading_error(
            anchor['orientation_wxyz'], trigger['orientation_wxyz']),
        'native_turn_degrees': turn, 'yaw_tolerance_deg': 0.05,
        'pose_tolerance': 1e-5, 'max_recovery_actions': 80,
        'follower_radius_m': 0.30, 'endpoint_tolerance_m': 0.35,
        'max_single_action_displacement_m': 0.31,
        'goal_radius_m': boundary['goal_radius_m'],
        'handoff_goal_exclusion_m': boundary['handoff_goal_exclusion_m'],
        'history_rule': boundary['history_rule'], 'scope': SCOPE,
    }


def turn_to_heading(target, state, advance, budget, step_degrees=15.0,
                    tolerance=0.05, position_tolerance=1e-5):
    """Use real LEFT/RIGHT steps only; never set a simulator pose."""
    count = 0
    origin = list(state()['position'])
    while True:
        before = state()
        delta = wrap_degrees(yaw_degrees(target) - yaw_degrees(before['orientation_wxyz']))
        if abs(delta) <= tolerance:
            return count
        if abs(delta / step_degrees - round(delta / step_degrees)) > tolerance / step_degrees:
            raise RuntimeError('POSE_RECOVERY_FAILED: target heading is off the native turn grid')
        if count >= budget:
            raise RuntimeError('POSE_RECOVERY_FAILED: turn action budget exhausted')
        action = 2 if delta > 0 else 3
        result = advance(action)
        count += 1
        after = state()
        if math.dist(after['position'], origin) > position_tolerance:
            raise RuntimeError('POSE_RECOVERY_FAILED: in-place turn changed position')
        observed = wrap_degrees(yaw_degrees(after['orientation_wxyz'])
                                - yaw_degrees(before['orientation_wxyz']))
        expected = step_degrees if action == 2 else -step_degrees
        if abs(wrap_degrees(observed - expected)) > tolerance:
            raise RuntimeError('POSE_RECOVERY_FAILED: unexpected actual turn angle')
        if result[2]:
            raise RuntimeError('POSE_RECOVERY_FAILED: episode ended during alignment')


def source_check(source):
    manifest, physical_source, boundary, reference = prompt.read_run(source)
    comparison = prompt.control_check(source, manifest, physical_source, reference)
    report = pilot.read_json(source / 'reinstruction_report.json')
    if report['status'] != 'READY_FOR_INTERPRETATION' or report['key'] != [physical.SCENE, '15']:
        raise ValueError('Require the completed case15 reinstruction report')
    control = pilot.read_json(source / 'control/pilot_summary.json')
    for arm in prompt.ARMS:
        summary = pilot.read_json(source / arm / 'pilot_summary.json')
        check = pilot.read_json(source / arm / 'reinstruction_summary.json')
        if summary['status'] != 'complete' or check['status'] != 'complete':
            raise ValueError('Prior prompt arm is incomplete: ' + arm)
        prompt.same_fields(summary, control, prompt.RUNTIME_FIELDS, 'prior prompt runtime')
        if not pilot.metrics_match(summary['metrics'], report['outcomes'][arm]['metrics']):
            raise ValueError('Prior report and metrics disagree: ' + arm)
    return manifest, boundary, reference, comparison, report


def prepare(args):
    source, root = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Use a new run-dir; no overwrite or resume')
    old, boundary, reference, _, _ = source_check(source)
    protocol = make_protocol(reference, boundary)
    native = pilot.load_config(source / 'control.py')
    if native.agent.model_settings.get('kv_cache_continuation', False):
        raise ValueError('Native cache-disabled inference is required')
    if native.agent.model_settings.get('mode') != 'dual_system':
        raise ValueError('Expected native dual_system inference')
    manifest = {
        'schema': protocol['schema'], 'seed_base': old['seed_base'], 'episodes': old['episodes'],
        'revision': subprocess.run(['git', '-C', str(pilot.ROOT), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, check=False).stdout.strip(),
        'source_hashes': dict(old['source_hashes']),
        'checkpoint_shard_sizes': dict(old['checkpoint_shard_sizes']),
        'generated_hashes': {}, 'pose_probe': {'source_run': str(source), 'scope': SCOPE},
    }
    manifest['source_hashes'].update(old['generated_hashes'])
    watched = [Path(__file__).resolve(), Path(prompt.__file__).resolve(),
               Path(physical.__file__).resolve(), Path(pilot.__file__).resolve(),
               source / 'manifest.json', source / 'reinstruction_report.json']
    for arm in prompt.ARMS:
        watched += [source / arm / name for name in (
            'actions.jsonl', 'probe_actions.jsonl', 'pilot_summary.json',
            'boundary_gate.json', 'prompt_event.json', 'reinstruction_summary.json')]
    for path in watched:
        manifest['source_hashes'][str(path)] = pilot.sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    path = root / 'pose_protocol.json'
    pilot.write_json(path, protocol)
    manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    for arm in ARMS:
        cfg = copy.deepcopy(native)
        cfg.eval_settings['output_path'] = str(root / arm)
        cfg.agent.model_settings['vis_debug'] = True
        cfg.agent.model_settings['vis_debug_path'] = str(root / 'video' / arm)
        cfg.agent.model_settings['kv_cache_continuation'] = False
        serialized = json.dumps(cfg.model_dump(mode='json', exclude_none=True))
        path = root / (arm + '.py')
        path.write_text(
            'import json\nfrom internnav.configs.evaluator import EvalCfg\n'
            + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n', encoding='utf-8')
        manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    pilot.write_json(root / 'manifest.json', manifest)
    print('POSE_PREPARE=PASS', 'arms=', ','.join(ARMS), 'run_dir=', root)
    print('ANCHOR_TRIGGER_HEADING_DIFFERENCE_DEG=',
          protocol['original_anchor_vs_trigger_heading_difference_deg'])
    print('Targets: historical position and fixed 0/+90/-90-degree offsets; no new GT route segment.')


class PoseProbe(prompt.PromptProbe):
    """Reuse the already tested exact-boundary gate and native action recorder."""
    def __init__(self, evaluator, root, arm, protocol, reference, source):
        super().__init__(evaluator, root, arm, protocol, reference, source)
        self.segments = []
        self.return_end_state = None
        self.return_comparison = None

    def on_boundary(self, step, observations, done, history, action_seq, pixel_goal, local_actions, action):
        was_triggered = self.triggered
        observations, done = super().on_boundary(
            step, observations, done, history, action_seq, pixel_goal, local_actions, action)
        if was_triggered or not self.triggered or self.arm == 'control':
            return observations, done
        before_history = physical.history_hash(history)
        before_rng = prompt.rng_digest()
        with physical.preserve_policy_rng():
            observations = self.recover(observations)
        if before_rng != prompt.rng_digest():
            raise RuntimeError('Policy RNG was not restored')
        if physical.history_hash(history) != before_history:
            raise RuntimeError('History changed during pose correction')
        if prompt.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError('Parameter versions changed during pose correction')
        self.phase = 'native_after_recovery'
        self.post_boundary = {
            'state': self.state(), 'candidate': self.protocol['candidates'][self.arm],
            'recovery_actions': self.recovery_actions, 'segments': self.segments,
            'return_end_state': self.return_end_state, 'return_trace_comparison': self.return_comparison,
            'history_unchanged': True, 'parameter_versions_unchanged': True, 'policy_rng_restored': True,
            'observation_hashes': physical.observation_hashes(observations),
            'history_rule': self.protocol['history_rule'],
        }
        pilot.write_json(self.directory / 'after_pose_recovery.json', self.post_boundary)
        from PIL import Image
        Image.fromarray(observations['rgb']).save(self.directory / 'after_pose_recovery.png')
        print('POSE_HANDOFF=PASS', self.arm, json.dumps(self.post_boundary['state']))
        return observations, False

    def recover(self, observations):
        candidate = self.protocol['candidates'][self.arm]
        latest = [observations]
        self.phase = 'recovery'

        def advance(action):
            result = self.evaluator.env.step(action)
            latest[0] = result[0]
            return result

        if candidate['location'] == 'anchor':
            import numpy as np
            import habitat_sim
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
            sim = self.evaluator.env._env.sim
            target = np.asarray(candidate['position'], dtype=np.float32)
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = sim.get_agent_state().position, target
            if not sim.pathfinder.is_navigable(target) or not sim.pathfinder.find_path(path):
                raise RuntimeError('POSE_RECOVERY_FAILED: recorded anchor is unreachable')
            follower = ShortestPathFollower(sim, self.protocol['follower_radius_m'], return_one_hot=False)
            start = len(self.records)
            physical.drive_waypoints(
                [target], follower.get_next_action, advance, lambda: self.state()['position'],
                self.protocol['max_recovery_actions'], self.protocol['endpoint_tolerance_m'],
                self.protocol['max_single_action_displacement_m'])
            self.return_end_state = self.state()
            self.segments.append({'stage': 'return', 'begin': start, 'end': len(self.records)})
            if self.arm != 'anchor_same_heading':
                base = pilot.read_json(self.root / 'anchor_same_heading/after_pose_recovery.json')
                segment = next(s for s in base['segments'] if s['stage'] == 'return')
                rows = physical.read_lines(self.root / 'anchor_same_heading/probe_actions.jsonl')
                self.return_comparison = pilot.compare_traces(
                    rows[segment['begin']:segment['end']], self.records[start:])
                if not self.return_comparison['actions_and_poses_match']:
                    raise RuntimeError('NOT_COMPARABLE: anchor return motions differ before alignment')
                if math.dist(self.state()['position'], base['return_end_state']['position']) > self.protocol['pose_tolerance']:
                    raise RuntimeError('NOT_COMPARABLE: anchor return endpoints differ')
        else:
            self.return_end_state = self.state()
        turn_start = len(self.records)
        turn_origin = list(self.state()['position'])
        turn_to_heading(
            candidate['orientation_wxyz'], self.state, advance,
            self.protocol['max_recovery_actions'] - self.recovery_actions,
            self.protocol['native_turn_degrees'], self.protocol['yaw_tolerance_deg'],
            self.protocol['pose_tolerance'])
        self.segments.append({'stage': 'alignment', 'begin': turn_start, 'end': len(self.records)})
        end = self.state()
        if math.dist(end['position'], turn_origin) > self.protocol['pose_tolerance']:
            raise RuntimeError('POSE_RECOVERY_FAILED: alignment changed the handoff position')
        position_tolerance = (self.protocol['endpoint_tolerance_m']
                              if candidate['location'] == 'anchor' else self.protocol['pose_tolerance'])
        if math.dist(end['position'], candidate['position']) > position_tolerance:
            raise RuntimeError('POSE_RECOVERY_FAILED: handoff position differs from the locked target')
        if heading_error(end['orientation_wxyz'], candidate['orientation_wxyz']) > self.protocol['yaw_tolerance_deg']:
            raise RuntimeError('POSE_RECOVERY_FAILED: handoff heading differs from the locked target')
        if not math.isfinite(end['distance_to_goal']) or end['distance_to_goal'] <= self.protocol['handoff_goal_exclusion_m']:
            raise RuntimeError('POSE_RECOVERY_FAILED: handoff is in or too close to the goal region')
        return latest[0]

    def close(self, error=None):
        self.stream.close()
        history_ok = bool(self.gate) and physical.history_hash(self.history_prefix) == self.gate['history_sha256']
        parameters_ok = self.parameters is not None and prompt.parameter_versions(self.evaluator.model) == self.parameters
        handoff_ok = self.arm == 'control' or self.post_boundary is not None
        if self.status == 'complete' and not (history_ok and parameters_ok and handoff_ok):
            self.status = 'invalid'
        pilot.write_json(self.directory / 'pose_summary.json', {
            'status': self.status, 'error': error, 'arm': self.arm, 'boundary_reached': self.triggered,
            'external_recovery_actions': self.recovery_actions, 'handoff': self.post_boundary,
            'history_prefix_unchanged': history_ok, 'parameter_versions_unchanged': parameters_ok,
            'scope': SCOPE})


def read_run(root):
    manifest = pilot.verify_manifest(root)
    source = Path(manifest['pose_probe']['source_run'])
    _, boundary, reference, _, prior_report = source_check(source)
    protocol = pilot.read_json(root / 'pose_protocol.json')
    if protocol != make_protocol(reference, boundary):
        raise ValueError('The locked pose protocol differs')
    return manifest, source, protocol, reference, prior_report


def control_check(root, source, reference):
    old, new = [pilot.read_json(p / 'control/pilot_summary.json') for p in (source, root)]
    if new['status'] != 'complete':
        raise RuntimeError('NOT_COMPARABLE: fresh control is incomplete')
    prompt.same_fields(new, old, prompt.RUNTIME_FIELDS, 'fresh control runtime')
    rows = physical.validate_trace(physical.read_lines(root / 'control/actions.jsonl'), 'fresh control')
    comparison = pilot.compare_traces(reference, rows)
    check = pilot.read_json(root / 'control/pose_summary.json')
    if (not comparison['actions_and_poses_match'] or not pilot.metrics_match(new['metrics'], old['metrics'])
            or check['status'] != 'complete' or not check['boundary_reached']
            or check['external_recovery_actions']):
        raise RuntimeError('NOT_COMPARABLE: fresh control must reproduce the entire native failure')
    return comparison


def run(args):
    root = Path(args.run_dir).resolve()
    _, source, protocol, reference, _ = read_run(root)
    if args.arm not in ARMS:
        raise ValueError('Unknown arm')
    if args.arm != 'control':
        control_check(root, source, reference)
    if args.arm in ('anchor_left90', 'anchor_right90'):
        check = pilot.read_json(root / 'anchor_same_heading/pose_summary.json')
        if check['status'] != 'complete' or not check['handoff']:
            raise RuntimeError('Complete anchor_same_heading first')
    sys.path.insert(0, str(pilot.ROOT))
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators['habitat_vln']
    native_init, native_loop = cls.__init__, cls._run_eval_dual_system
    wrapped = physical.instrument_loop(native_loop)
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get('kv_cache_continuation', False):
            raise RuntimeError('Pose diagnostic requires native, cache-disabled inference')
        native_init(evaluator, cfg)
        probe = PoseProbe(evaluator, root, args.arm, protocol, reference, source)
        probes.append(probe)
        evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record

    cls.__init__, cls._run_eval_dual_system = initialize, wrapped
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1 or not probes[0].triggered:
            raise RuntimeError('NOT_COMPARABLE: expected one evaluator at the locked boundary')
        if args.arm != 'control' and probes[0].post_boundary is None:
            raise RuntimeError('POSE_RECOVERY_FAILED: no valid handoff')
        probes[0].status = 'complete'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = native_init, native_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.close(error)
    if any(p.status != 'complete' for p in probes):
        raise RuntimeError('NOT_COMPARABLE: history/parameter/handoff check failed')
    if args.arm == 'control':
        control_check(root, source, reference)
        print('NATIVE_CONTROL_REPRODUCTION=PASS')


def check_factorial(states, protocol):
    tolerance = protocol['pose_tolerance']
    for names in (('control', 'turn_left90', 'turn_right90'),
                  ('anchor_same_heading', 'anchor_left90', 'anchor_right90')):
        for name in names[1:]:
            if math.dist(states[name]['position'], states[names[0]]['position']) > tolerance:
                raise RuntimeError('NOT_COMPARABLE: positions differ inside a heading contrast')
    for arm, candidate in protocol['candidates'].items():
        if heading_error(states[arm]['orientation_wxyz'], candidate['orientation_wxyz']) > protocol['yaw_tolerance_deg']:
            raise RuntimeError('NOT_COMPARABLE: factorial heading target not achieved: ' + arm)
    for a, b in (('control', 'anchor_same_heading'), ('turn_left90', 'anchor_left90'),
                 ('turn_right90', 'anchor_right90')):
        if heading_error(states[a]['orientation_wxyz'], states[b]['orientation_wxyz']) > 2 * protocol['yaw_tolerance_deg']:
            raise RuntimeError('NOT_COMPARABLE: headings differ inside a position contrast')
    return {'same_position_within_each_heading_contrast': True,
            'matched_headings_across_positions': True,
            'heading_offsets_deg': [0, 90, -90]}


def report(args):
    root = Path(args.run_dir).resolve()
    _, source, protocol, reference, prior_report = read_run(root)
    comparison = control_check(root, source, reference)
    control = pilot.read_json(root / 'control/pilot_summary.json')
    states, outcomes = {}, {}
    for arm in ARMS:
        summary = pilot.read_json(root / arm / 'pilot_summary.json')
        diagnostic = pilot.read_json(root / arm / 'pose_summary.json')
        if summary['status'] != 'complete' or diagnostic['status'] != 'complete':
            raise RuntimeError('Incomplete pose arm: ' + arm)
        prompt.same_fields(summary, control, prompt.RUNTIME_FIELDS, arm + ' runtime')
        gate = pilot.read_json(root / arm / 'boundary_gate.json')
        prompt.same_fields(gate, pilot.read_json(root / 'control/boundary_gate.json'),
                           prompt.GATE_FIELDS + ('rng_sha256',), arm + ' boundary')
        rows = physical.validate_trace(physical.read_lines(root / arm / 'probe_actions.jsonl'), arm)
        count = protocol['prefix_action_count']
        if not pilot.compare_traces(reference[:count], rows[:count])['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: saved prefix differs: ' + arm)
        handoff = diagnostic['handoff']
        states[arm] = handoff['state'] if handoff else gate['state']
        future = [r for r in rows[count:] if r['phase'] != 'recovery' and r['action'] <= 3]
        outcomes[arm] = {
            'metrics': summary['metrics'], 'candidate': protocol['candidates'][arm],
            'handoff_state': states[arm], 'external_recovery_actions': diagnostic['external_recovery_actions'],
            'correction_motion_length_xyz_m': sum(
                math.dist(rows[i]['position'], rows[i-1]['position'])
                for i in range(count, len(rows)) if rows[i]['phase'] == 'recovery'),
            'segments': handoff['segments'] if handoff else [],
            'autonomous_body_actions_after_boundary': len(future),
            'first_autonomous_body_actions': future[:8],
            'last_actual_action': rows[-1]['action'], 'last_env_done': rows[-1]['done'],
            'videos': [str(p) for p in sorted((root / 'video' / arm).rglob('*.mp4'))],
            'recovery_frames': str(root / arm / 'recovery_frames') if arm != 'control' else None,
        }
    checks = check_factorial(states, protocol)
    result = {
        'schema': 'case15_pose_factorial_report.v1', 'status': 'READY_FOR_POSE_REVIEW',
        'key': [physical.SCENE, '15'], 'trigger_step': 85,
        'control_exact_reproduction': comparison, 'factorial_checks': checks, 'outcomes': outcomes,
        'prior_prompt_metrics': {a: prior_report['outcomes'][a]['metrics'] for a in prompt.ARMS},
        'physical_reference': prior_report['outcomes']['physical_reference'],
        'successful_pose_arms': [a for a in ARMS if a != 'control'
                                 and float(outcomes[a]['metrics']['sucs_all']) > 0],
        'timing_valid_for_speed_comparison': False, 'scope': SCOPE,
    }
    pilot.write_json(root / 'pose_factorial_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_all(args):
    root = Path(args.run_dir).resolve()
    read_run(root)
    if any((root / arm).exists() for arm in ARMS):
        raise FileExistsError('run-all is create-only; an arm already exists')
    for arm in ARMS:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                        '--run-dir', str(root), '--arm', arm], check=True)
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.set_defaults(function=prepare)
    execute = commands.add_parser('run')
    execute.add_argument('--run-dir', required=True)
    execute.add_argument('--arm', choices=tuple(ARMS), required=True)
    execute.set_defaults(function=run)
    batch = commands.add_parser('run-all')
    batch.add_argument('--run-dir', required=True)
    batch.set_defaults(function=run_all)
    summary = commands.add_parser('report')
    summary.add_argument('--run-dir', required=True)
    summary.set_defaults(function=report)
    args = parser.parse_args()
    if args.command == 'prepare':
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-version'], capture_output=True, check=True)
    args.function(args)


if __name__ == '__main__':
    main()
