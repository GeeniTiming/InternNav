"""Capture predeclared native checkpoints for the locked orientation cohort.

This stage executes native controls only. It records comparable replay boundaries;
it does not rotate the robot, edit history, or select interventions by outcomes.
"""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orientation_cohort as cohort
import case15_reinstruction_probe as common
pilot, physical, capture = cohort.pilot, cohort.physical, cohort.capture

SCOPE = (
    'Native replay and checkpoint capture on six or fewer locked development routes. '
    'Completed baseline lengths determine offline checkpoint timing. No orientation '
    'intervention, online detector, recovery effect, training, language-completion, '
    'held-out efficacy, or isolated memory-causality conclusion.'
)


def verify_cohort(root):
    lock = pilot.read_json(root / 'cohort_manifest.json')
    if lock['schema'] != 'crosscase_orientation_inputs.v1':
        raise ValueError('Unexpected cohort manifest')
    for path, expected in lock['hashes'].items():
        if pilot.sha256(path) != expected:
            raise RuntimeError('Locked cohort input changed: ' + path)
    for path, size in lock['checkpoint_shard_sizes'].items():
        if Path(path).stat().st_size != size:
            raise RuntimeError('Checkpoint shard size changed: ' + path)
    plan = pilot.read_json(root / 'plan.json')
    if plan['status'] not in ('LOCKED_SELECTION_PENDING_CHECKPOINT_CAPTURE',
                              'LOCKED_FEWER_GROUPS_PENDING_CHECKPOINT_CAPTURE'):
        raise ValueError('The locked cohort cannot proceed')
    source = Path(plan['source_run'])
    native = pilot.verify_manifest(source)
    dataset = pilot.read_json(source / 'episodes.json.gz')
    progress = capture.read_progress(source / 'control/progress.json')
    traces = {capture.episode_key(ep): [] for ep in dataset['episodes']}
    for row in physical.read_lines(source / 'control/actions.jsonl'):
        traces[capture.episode_key(row)].append(row)
    expected = cohort.select_plan(dataset, progress, traces)
    expected['source_run'] = str(source)
    if plan != expected:
        raise ValueError('Cohort selection/rules differ from the locked source')
    return plan, lock, source, native, dataset, progress, traces


def prepare(args):
    cohort_root, root = Path(args.cohort_dir).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Use a new run-dir; existing captures are preserved')
    plan, lock, source, native, dataset, _, _ = verify_cohort(cohort_root)
    cfg = copy.deepcopy(pilot.load_config(source / 'control.py'))
    if (cfg.eval_type != 'habitat_vln' or cfg.agent.model_settings.get('mode') != 'dual_system'
            or cfg.agent.model_settings.get('kv_cache_continuation', False)):
        raise ValueError('Require native cache-disabled Habitat dual_system inference')
    import yaml
    habitat_path = (pilot.ROOT / cfg.env.env_settings['config_path']).resolve()
    habitat = yaml.safe_load(habitat_path.read_text(encoding='utf-8'))
    if habitat['habitat']['dataset']['split'] != 'val_unseen':
        raise ValueError('Expected the locked development split')
    lookup = {capture.episode_key(ep): ep for ep in dataset['episodes']}
    subset = copy.deepcopy(dataset)
    subset['episodes'] = [copy.deepcopy(lookup[tuple(row['key'])]) for row in plan['cases']]
    manifest = {
        'schema': 'orientation_checkpoint_capture.v1', 'seed_base': native['seed_base'],
        'episodes': [{'scene_id': ep['scene_id'], 'episode_id': str(ep['episode_id'])}
                     for ep in subset['episodes']],
        'revision': subprocess.run(['git', '-C', str(pilot.ROOT), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=False).stdout.strip(),
        'source_hashes': dict(lock['hashes']), 'generated_hashes': {},
        'checkpoint_shard_sizes': dict(lock['checkpoint_shard_sizes']),
        'orientation_checkpoint': {'cohort_dir': str(cohort_root), 'source_run': str(source),
                                   'scope': SCOPE},
    }
    for path in (cohort_root / 'cohort_manifest.json', Path(__file__).resolve(),
                 Path(cohort.__file__).resolve(), Path(common.__file__).resolve()):
        manifest['source_hashes'][str(path)] = pilot.sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    data_path = root / 'episodes.json.gz'
    with data_path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            zipped.write(json.dumps(subset, ensure_ascii=False).encode('utf-8'))
    habitat['habitat']['dataset']['data_path'] = str(data_path)
    yaml_path = root / 'habitat.yaml'
    yaml_path.write_text('# @package _global_\n' + yaml.safe_dump(habitat, sort_keys=False), encoding='utf-8')
    cfg.env.env_settings['config_path'] = str(yaml_path)
    cfg.eval_settings['output_path'] = str(root / 'control')
    cfg.agent.model_settings['vis_debug'] = True
    cfg.agent.model_settings['vis_debug_path'] = str(root / 'video' / 'control')
    serialized = json.dumps(cfg.model_dump(mode='json', exclude_none=True))
    config_path = root / 'control.py'
    config_path.write_text(
        'import json\nfrom internnav.configs.evaluator import EvalCfg\n'
        + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n', encoding='utf-8')
    protocol_path = root / 'capture_protocol.json'
    pilot.write_json(protocol_path, {'schema': manifest['schema'], 'cases': plan['cases'],
                                    'checkpoint_rule': plan['checkpoint_rule'],
                                    'interventions_executed': False, 'scope': SCOPE})
    for path in (data_path, yaml_path, config_path, protocol_path):
        manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    pilot.write_json(root / 'manifest.json', manifest)
    print('CHECKPOINT_PREPARE=PASS cases=', len(plan['cases']), 'run_dir=', root)
    print('Native controls only; one model load for the locked cohort. No rotation arm is prepared.')


def read_run(root):
    manifest = pilot.verify_manifest(root)
    cohort_root = Path(manifest['orientation_checkpoint']['cohort_dir'])
    data = verify_cohort(cohort_root)
    protocol = pilot.read_json(root / 'capture_protocol.json')
    if (protocol['cases'] != data[0]['cases'] or protocol['checkpoint_rule'] != data[0]['checkpoint_rule']
            or protocol['interventions_executed']):
        raise ValueError('Capture protocol changed')
    return manifest, data


def quaternion_list(q):
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def camera_pitch_degrees(q):
    w, x, y, z = q
    if len(q) != 4 or not all(math.isfinite(v) for v in q) or abs(sum(v*v for v in q) - 1) > 1e-3:
        raise ValueError('Invalid RGB sensor quaternion')
    # Optical forward is local -Z; world Y is gravity-up. Body yaw does not affect pitch.
    return math.degrees(math.asin(max(-1., min(1., 2*(w*x - y*z)))))


def eligibility(case, body_count, translation, done, action_seq, pixel_goal, local_actions,
                action, camera_balance, camera_pitch):
    rule = cohort.CHECKPOINT_RULE
    remaining = case['native_nonstop_body_actions'] - body_count
    checks = {
        'threshold_reached': body_count >= case['checkpoint_body_threshold'],
        'translated_enough': translation + 1e-8 >= rule['minimum_prefix_translation_m'],
        'remaining_budget': remaining >= rule['minimum_remaining_nonstop_body_actions'],
        'idle_s2': not done and not action_seq and pixel_goal is None and not local_actions
                   and action in (0, 1, 2, 3),
        'level_camera': camera_balance == 0 and camera_pitch is not None and abs(camera_pitch) <= 1e-3,
    }
    return {'checks': checks, 'eligible': all(checks.values()), 'remaining_baseline_body_actions': remaining}


class CheckpointProbe:
    """Observe the native loop without changing observations, policy inputs or actions."""
    def __init__(self, evaluator, root, cases, references):
        self.evaluator, self.root = evaluator, root
        self.cases = {tuple(row['key']): row for row in cases}
        self.references = references
        self.current = None
        self.finished = {}
        self.parameters = common.parameter_versions(evaluator.model)
        self.action_stream = (root / 'control/capture_actions.jsonl').open('x', encoding='utf-8')
        self.event_stream = (root / 'control/boundary_events.jsonl').open('x', encoding='utf-8')
        self.original_step = evaluator.env.step
        self.status = 'running'

    def state(self):
        state = self.evaluator.env._env.sim.get_agent_state()
        return {'position': [float(v) for v in state.position],
                'orientation_wxyz': quaternion_list(state.rotation),
                'distance_to_goal_OFFLINE_ONLY': float(self.evaluator.env.get_metrics()['distance_to_goal'])}

    def begin(self, key):
        if key not in self.cases or key in self.finished:
            raise RuntimeError('NOT_COMPARABLE: unexpected/repeated episode: ' + str(key))
        self.current = key
        self.records, self.gate, self.history_prefix = [], None, ()
        self.body_count, self.translation, self.camera_balance = 0, 0., 0
        self.last_position = self.state()['position']
        self.current_step = 0

    def record(self, action):
        result = self.original_step(action)
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({'scene_id': ep.scene_id, 'episode_id': ep.episode_id})
        if key != self.current:
            raise RuntimeError('NOT_COMPARABLE: action outside its native episode boundary')
        state = self.evaluator.env._env.sim.get_agent_state()
        row = {'scene_id': ep.scene_id, 'episode_id': str(ep.episode_id), 'action': int(action),
               'done': bool(result[2]), 'position': [float(v) for v in state.position],
               'orientation_wxyz': quaternion_list(state.rotation)}
        index = len(self.records)
        reference = self.references[key]
        if index >= len(reference) or not pilot.compare_traces([reference[index]], [row])['actions_and_poses_match']:
            pilot.write_json(self.root / 'control/native_mismatch.json', {
                'key': list(key), 'index': index, 'actual': row,
                'expected': reference[index] if index < len(reference) else None})
            raise RuntimeError('NOT_COMPARABLE: native action/pose replay diverged at ' + str((key, index)))
        self.translation += math.dist(self.last_position, row['position'])
        self.last_position = row['position']
        self.body_count += int(row['action'] in (1, 2, 3))
        self.camera_balance += (1 if row['action'] == 4 else -1 if row['action'] == 5 else 0)
        self.records.append(row)
        self.action_stream.write(json.dumps(dict(row, main_step=self.current_step)) + '\n')
        self.action_stream.flush()
        return result

    def on_boundary(self, step, observations, done, history, action_seq, pixel_goal, local_actions, action):
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({'scene_id': ep.scene_id, 'episode_id': ep.episode_id})
        if key != self.current:
            self.finish()
            self.begin(key)
        self.current_step = int(step)
        idle = not done and not action_seq and pixel_goal is None and not local_actions and action in (0, 1, 2, 3)
        pitch = None
        if idle and self.camera_balance == 0:
            sensors = self.evaluator.env._env.sim.get_agent_state().sensor_states
            if 'rgb' not in sensors:
                raise RuntimeError('Cannot verify the level RGB sensor at a native boundary')
            pitch = camera_pitch_degrees(quaternion_list(sensors['rgb'].rotation))
        event = dict(eligibility(self.cases[key], self.body_count, self.translation, done, action_seq,
                                 pixel_goal, local_actions, action, self.camera_balance, pitch),
                     key=list(key), step=int(step), prefix_action_count=len(self.records),
                     prefix_body_actions=self.body_count, prefix_translation_m=self.translation,
                     camera_action_balance=self.camera_balance, camera_pitch_deg=pitch)
        self.event_stream.write(json.dumps(event) + '\n')
        self.event_stream.flush()
        if not event['eligible'] or self.gate is not None:
            return observations, done
        before = {'state': self.state(), 'observation_hashes': physical.observation_hashes(observations),
                  'history_sha256': physical.history_hash(history), 'rng_sha256': common.rng_digest()}
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError('Model parameter versions changed during native replay')
        self.gate = dict(event, **before, history_length=len(history),
                         parameters_sha256=self.parameters, prefix_trace_sha256=cohort.digest(self.records))
        self.history_prefix = tuple(history)
        directory = self.root / 'control/checkpoints' / (key[0] + '_' + key[1])
        directory.mkdir(parents=True, exist_ok=False)
        from PIL import Image
        Image.fromarray(observations['rgb']).save(directory / 'rgb.png')
        pilot.write_json(directory / 'checkpoint.json', self.gate)
        after = {'state': self.state(), 'observation_hashes': physical.observation_hashes(observations),
                 'history_sha256': physical.history_hash(history), 'rng_sha256': common.rng_digest()}
        if before != after or common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError('Capture changed native state, inputs, history, RNG or parameters')
        print('NATIVE_CHECKPOINT=CAPTURED key=', key, 'step=', step, 'actual_prefix=', len(self.records))
        return observations, done

    def finish(self):
        if self.current is None:
            return
        comparison = pilot.compare_traces(self.references[self.current], self.records)
        if not comparison['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: incomplete native episode trace: ' + str(self.current))
        if self.gate and physical.history_hash(self.history_prefix) != self.gate['history_sha256']:
            raise RuntimeError('Native history prefix changed after checkpoint capture')
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError('Model parameter versions changed')
        self.finished[self.current] = {
            'key': list(self.current), 'status': 'CAPTURED_PENDING_OUTCOME_CHECK' if self.gate else 'NOT_ELIGIBLE',
            'native_trace_comparison': comparison, 'checkpoint': self.gate,
            'history_prefix_unchanged': True, 'parameter_versions_unchanged': True}
        self.current = None

    def close(self, error=None):
        if error is not None:
            self.status = 'failed'
        self.action_stream.close()
        self.event_stream.close()
        pilot.write_json(self.root / 'control/capture_summary.json', {
            'status': self.status, 'error': error,
            'cases': [self.finished.get(key, {'key': list(key), 'status': 'NOT_COMPARABLE_OR_NOT_RUN'})
                      for key in self.cases], 'extra_env_actions': 0,
            'orientation_interventions_executed': False, 'scope': SCOPE})

def report(args):
    root = Path(args.run_dir).resolve()
    manifest, data = read_run(root)
    plan, _, source, _, _, source_progress, references = data
    summary = pilot.read_json(root / 'control/pilot_summary.json')
    source_summary = pilot.read_json(source / 'control/pilot_summary.json')
    diagnostic = pilot.read_json(root / 'control/capture_summary.json')
    if summary['status'] != 'complete' or diagnostic['status'] != 'complete':
        raise RuntimeError('NOT_COMPARABLE: native capture did not complete')
    common.same_fields(summary, source_summary, common.RUNTIME_FIELDS, 'native capture runtime')
    if diagnostic['extra_env_actions'] or diagnostic['orientation_interventions_executed']:
        raise RuntimeError('Native capture executed an intervention')
    expected_keys = {tuple(row['key']) for row in plan['cases']}
    progress = capture.read_progress(root / 'control/progress.json')
    traces = {key: [] for key in expected_keys}
    for row in physical.validate_trace(physical.read_lines(root / 'control/actions.jsonl'), 'native capture'):
        key = capture.episode_key(row)
        if key not in traces:
            raise RuntimeError('Unexpected capture trace episode')
        traces[key].append(row)
    observed = physical.read_lines(root / 'control/capture_actions.jsonl')
    if not pilot.compare_traces(physical.read_lines(root / 'control/actions.jsonl'), observed)['actions_and_poses_match']:
        raise RuntimeError('Native and observer action records differ')
    if set(progress) != expected_keys:
        raise RuntimeError('NOT_COMPARABLE: capture progress episode set differs')
    diagnostics = {tuple(row['key']): row for row in diagnostic['cases']}
    events = physical.read_lines(root / 'control/boundary_events.jsonl')
    rows = []
    for case in plan['cases']:
        key = tuple(case['key'])
        actual, old = progress[key], source_progress[key]
        if actual['episode_instruction'] != case['instruction'] or actual['steps'] != old['steps']:
            raise RuntimeError('NOT_COMPARABLE: instruction or native main-step count differs: ' + str(key))
        if not pilot.metrics_match({k: actual[k] for k in ('success', 'spl', 'os', 'ne')}, case['native_metrics']):
            raise RuntimeError('NOT_COMPARABLE: native outcome differs: ' + str(key))
        comparison = pilot.compare_traces(references[key], traces[key])
        if not comparison['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: full native trace differs: ' + str(key))
        row = diagnostics[key]
        if not row['history_prefix_unchanged'] or not row['parameter_versions_unchanged']:
            raise RuntimeError('NOT_COMPARABLE: history or parameters changed')
        eligible = [event for event in events if tuple(event['key']) == key and event['eligible']]
        gate = row['checkpoint']
        if gate:
            if not eligible or any(gate[k] != eligible[0][k] for k in eligible[0]):
                raise RuntimeError('Capture was not at the first qualifying native boundary')
            directory = root / 'control/checkpoints' / (key[0] + '_' + key[1])
            if pilot.read_json(directory / 'checkpoint.json') != gate or not (directory / 'rgb.png').is_file():
                raise RuntimeError('Captured checkpoint or RGB file differs')
            count = gate['prefix_action_count']
            if count <= 0 or count > len(references[key]) or gate['prefix_trace_sha256'] != cohort.digest(traces[key][:count]):
                raise RuntimeError('Captured prefix digest differs')
            row_status = 'COMPARABLE_CHECKPOINT_CAPTURED'
        else:
            if eligible or row['status'] != 'NOT_ELIGIBLE':
                raise RuntimeError('An eligible boundary was omitted')
            row_status = 'NOT_ELIGIBLE_KEEP_LOCKED'
        rows.append({'key': case['key'], 'stratum': case['stratum'], 'status': row_status,
                     'native_trace_comparison': comparison, 'native_metrics': case['native_metrics'],
                     'checkpoint': gate})
    counts = {name: sum(row['stratum'] == name and row['checkpoint'] is not None for row in rows)
              for name in ('failure', 'success')}
    result = {
        'schema': 'crosscase_orientation_checkpoints.v1',
        'status': ('CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION' if all(counts.values())
                   else 'INSUFFICIENT_ELIGIBLE_STRATA_STOP'),
        'locked_cases': len(rows), 'eligible_counts': counts, 'cases': rows,
        'native_controls_completed': True, 'extra_env_actions': 0,
        'orientation_interventions_executed': False,
        'snapshot_type': 'Metadata/RGB fingerprints for future fresh native replay; not a resumable model/env checkpoint.',
        'timing_valid_for_speed_comparison': False, 'scope': SCOPE,
    }
    pilot.write_json(root / 'checkpoint_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run(args):
    root = Path(args.run_dir).resolve()
    _, data = read_run(root)
    if (root / 'control').exists():
        raise FileExistsError('Native capture is create-only; control already exists')
    plan, _, _, _, _, _, references = data
    sys.path.insert(0, str(pilot.ROOT))
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators['habitat_vln']
    native_init, native_loop = cls.__init__, cls._run_eval_dual_system
    wrapped = physical.instrument_loop(native_loop)
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get('kv_cache_continuation', False):
            raise RuntimeError('Native checkpoint capture requires cache-disabled inference')
        native_init(evaluator, cfg)
        probe = CheckpointProbe(evaluator, root, plan['cases'], references)
        probes.append(probe)
        evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record

    cls.__init__, cls._run_eval_dual_system = initialize, wrapped
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm='control', profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError('Expected one native evaluator')
        probes[0].finish()
        if set(probes[0].finished) != {tuple(row['key']) for row in plan['cases']}:
            raise RuntimeError('NOT_COMPARABLE: not all locked cases completed')
        probes[0].status = 'complete'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = native_init, native_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.close(error)
    return report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--cohort-dir', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.set_defaults(function=prepare)
    for name, function in (('run', run), ('report', report)):
        command = commands.add_parser(name)
        command.add_argument('--run-dir', required=True)
        command.set_defaults(function=function)
    args = parser.parse_args()
    if args.command == 'prepare':
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-version'], capture_output=True, check=True)
    args.function(args)


if __name__ == '__main__':
    main()