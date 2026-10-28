"""Fixed left/right90 contrasts at captured native checkpoints; history is retained."""
import argparse
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_independent_checkpoint as checkpoint
import case15_pose_factorial as pose
pilot, physical, common, capture = checkpoint.pilot, checkpoint.physical, checkpoint.common, checkpoint.capture
ARMS = {'left90': 90, 'right90': -90}
SCOPE = ('Fixed development-case orientation contrasts, with native control reused from verified capture. '
         'Original history retained. No online trigger/direction selector, training, isolated memory cause '
         'or held-out efficacy claim. Choosing the better direction afterward is an oracle diagnostic only.')


def source_check(source):
    """Run existing full report validation without rewriting any source artifact."""
    saved = pilot.read_json(source / 'checkpoint_report.json')
    original = pilot.write_json
    def verify_write(path, value):
        if Path(path) != source / 'checkpoint_report.json' or value != saved:
            raise RuntimeError('Captured source report does not validate unchanged')
    pilot.write_json = verify_write
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            validated = checkpoint.report(SimpleNamespace(run_dir=str(source)))
    finally:
        pilot.write_json = original
    if (validated['status'] != 'CHECKPOINTS_READY_FOR_INTERVENTION_PREPARATION'
            or any(c['checkpoint'] is None for c in validated['cases'])):
        raise ValueError('Require comparable checkpoints for every locked case; do not replace cases')
    return pilot.verify_manifest(source), validated


def prepare(args):
    source, root = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Use a new run-dir')
    old, report = source_check(source)
    native = pilot.load_config(source / 'control.py')
    if native.agent.model_settings.get('kv_cache_continuation', False):
        raise ValueError('Require native cache-disabled inference')
    limit = int(native.eval_settings['max_steps_per_episode'])
    if any(c['checkpoint']['step'] + 6 > limit for c in report['cases']):
        raise ValueError('Insufficient declared native main-step budget for six charged turns')
    manifest = dict(schema='orientation_turn.v1', seed_base=old['seed_base'], episodes=old['episodes'],
                    revision=subprocess.run(['git', '-C', str(pilot.ROOT), 'rev-parse', 'HEAD'],
                                            capture_output=True, text=True, check=False).stdout.strip(),
                    source_hashes=dict(old['source_hashes']), generated_hashes={},
                    checkpoint_shard_sizes=old['checkpoint_shard_sizes'],
                    orientation_turn={'source_run': str(source), 'scope': SCOPE})
    manifest['source_hashes'].update(old['generated_hashes'])
    watched = [source / 'manifest.json', source / 'checkpoint_report.json', Path(__file__).resolve(),
               Path(checkpoint.__file__).resolve(), Path(pose.__file__).resolve(), Path(common.__file__).resolve()]
    watched += [p for p in (source / 'control').rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.png')]
    manifest['source_hashes'].update({str(p): pilot.sha256(p) for p in watched})
    root.mkdir(parents=True, exist_ok=False)
    protocol = {'schema': manifest['schema'], 'arms': ARMS, 'forced_turn_actions': 6,
                'native_main_step_limit': limit, 'charged_main_step_limit': limit - 6,
                'budget_rule': 'Six real turns consume Habitat steps and reduce the native loop limit by six. '
                               'Keep step_id/history indexing unchanged. Restore the original limit between episodes.',
                'cases': report['cases'], 'scope': SCOPE}
    path = root / 'turn_protocol.json'
    pilot.write_json(path, protocol)
    manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    for arm in ARMS:
        cfg = copy.deepcopy(native)
        cfg.eval_settings['output_path'] = str(root / arm)
        cfg.agent.model_settings['vis_debug'] = True
        cfg.agent.model_settings['vis_debug_path'] = str(root / 'video' / arm)
        path = root / (arm + '.py')
        serialized = json.dumps(cfg.model_dump(mode='json', exclude_none=True))
        path.write_text('import json\nfrom internnav.configs.evaluator import EvalCfg\n'
                        + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n', encoding='utf-8')
        manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    pilot.write_json(root / 'manifest.json', manifest)
    print('TURN_PREPARE=PASS cases=', len(report['cases']), 'arms=left90,right90 run_dir=', root)


def read_run(root):
    manifest = pilot.verify_manifest(root)
    source = Path(manifest['orientation_turn']['source_run'])
    _, report = source_check(source)
    protocol = pilot.read_json(root / 'turn_protocol.json')
    if protocol['cases'] != report['cases'] or protocol['arms'] != ARMS or protocol['forced_turn_actions'] != 6:
        raise ValueError('Locked turn protocol changed')
    return manifest, source, protocol


class TurnProbe(checkpoint.CheckpointProbe):
    def __init__(self, evaluator, root, arm, protocol, references):
        self.evaluator, self.root, self.arm, self.protocol = evaluator, root, arm, protocol
        self.directory = root / arm
        self.cases = {tuple(c['key']): c for c in protocol['cases']}
        self.references, self.current, self.finished = references, None, {}
        self.parameters = common.parameter_versions(evaluator.model)
        self.original_step = evaluator.env.step
        self.action_stream = (self.directory / 'turn_actions.jsonl').open('x', encoding='utf-8')
        self.original_limit = evaluator.max_steps_per_episode
        if self.original_limit != protocol['native_main_step_limit']:
            raise RuntimeError('Native main-step limit differs from the locked config')
        self.status = 'running'

    def begin(self, key):
        self.evaluator.max_steps_per_episode = self.original_limit
        super().begin(key)
        self.phase, self.handoff, self.forced_count = 'prefix', None, 0
        self.tagged, self.s2_calls = [], 0

    def record(self, action):
        result = self.original_step(action)
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({'scene_id': ep.scene_id, 'episode_id': ep.episode_id})
        if key != self.current:
            raise RuntimeError('Action occurred outside its locked episode')
        state = self.evaluator.env._env.sim.get_agent_state()
        row = dict(scene_id=ep.scene_id, episode_id=str(ep.episode_id), action=int(action), done=bool(result[2]),
                   position=[float(v) for v in state.position], orientation_wxyz=checkpoint.quaternion_list(state.rotation))
        if self.phase == 'prefix':
            index = len(self.records)
            count = self.cases[key]['checkpoint']['prefix_action_count']
            if index >= count or not pilot.compare_traces([self.references[key][index]], [row])['actions_and_poses_match']:
                raise RuntimeError('NOT_COMPARABLE: native prefix action or pose diverged')
        self.records.append(row)
        tagged = dict(row, phase=self.phase, main_step=self.current_step)
        self.tagged.append(tagged)
        self.action_stream.write(json.dumps(tagged) + '\n')
        self.action_stream.flush()
        if self.phase == 'alignment':
            self.forced_count += 1
            from PIL import Image
            folder = self.directory / 'frames' / (key[0] + '_' + key[1])
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result[0]['rgb']).save(folder / ('turn_%02d.png' % self.forced_count))
        return result

    def on_boundary(self, step, observations, done, history, action_seq, pixel_goal, local_actions, action):
        ep = self.evaluator.env.get_current_episode()
        key = capture.episode_key({'scene_id': ep.scene_id, 'episode_id': ep.episode_id})
        if key != self.current:
            self.finish()
            self.begin(key)
        self.current_step = int(step)
        expected = self.cases[key]['checkpoint']
        if self.gate is not None or step < expected['step']:
            return observations, done
        if (step != expected['step'] or len(self.records) != expected['prefix_action_count']
                or done or action_seq or pixel_goal is not None or local_actions or action not in (0, 1, 2, 3)):
            raise RuntimeError('NOT_COMPARABLE: locked idle boundary not reproduced')
        pitch = checkpoint.camera_pitch_degrees(checkpoint.quaternion_list(
            self.evaluator.env._env.sim.get_agent_state().sensor_states['rgb'].rotation))
        if abs(pitch) > 1e-3:
            raise RuntimeError('NOT_COMPARABLE: RGB camera is not level')
        gate = {'step': int(step), 'prefix_action_count': len(self.records), 'state': self.state(),
                'observation_hashes': physical.observation_hashes(observations),
                'history_length': len(history), 'history_sha256': physical.history_hash(history),
                'rng_sha256': common.rng_digest(), 'parameters_sha256': self.parameters,
                'prefix_trace_sha256': checkpoint.cohort.digest(self.records)}
        common.same_fields(gate, expected, tuple(gate), 'turn boundary')
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError('Model parameters changed')
        self.gate, self.history_prefix = gate, tuple(history)
        target = pose.yaw_quaternion(pose.yaw_degrees(gate['state']['orientation_wxyz']) + ARMS[self.arm])
        latest = [observations]
        self.phase = 'alignment'
        def advance(action):
            value = self.evaluator.env.step(action)
            latest[0] = value[0]
            return value
        with physical.preserve_policy_rng():
            count = pose.turn_to_heading(target, self.state, advance, budget=6)
        if count != 6 or self.forced_count != 6:
            raise RuntimeError('Expected exactly six real native turns')
        if (common.rng_digest() != gate['rng_sha256'] or physical.history_hash(history) != gate['history_sha256']
                or common.parameter_versions(self.evaluator.model) != self.parameters):
            raise RuntimeError('History, RNG or parameters changed during alignment')
        self.evaluator.max_steps_per_episode = self.protocol['charged_main_step_limit']
        self.phase = 'continuation'
        self.handoff = {'state': self.state(), 'target_orientation_wxyz': target,
                        'forced_actions': count, 'history_retained': True, 'policy_rng_restored': True,
                        'final_observation_hashes': physical.observation_hashes(latest[0]),
                        'charged_native_limit': self.evaluator.max_steps_per_episode}
        folder = self.directory / 'checkpoints' / (key[0] + '_' + key[1])
        folder.mkdir(parents=True, exist_ok=False)
        pilot.write_json(folder / 'boundary.json', gate)
        pilot.write_json(folder / 'handoff.json', self.handoff)
        from PIL import Image
        Image.fromarray(latest[0]['rgb']).save(folder / 'after_turn.png')
        print('TURN_HANDOFF=PASS', self.arm, key, 'step=', step, 'forced=', count)
        return latest[0], False

    def finish(self):
        if self.current is None:
            return
        if self.gate is None or self.handoff is None:
            raise RuntimeError('NOT_COMPARABLE: no validated intervention at the locked boundary')
        if (physical.history_hash(self.history_prefix) != self.gate['history_sha256']
                or common.parameter_versions(self.evaluator.model) != self.parameters):
            raise RuntimeError('History prefix or parameters changed after handoff')
        count = self.gate['prefix_action_count']
        comparison = pilot.compare_traces(self.references[self.current][:count], self.records[:count])
        if not comparison['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: prefix differs')
        after = self.tagged[count:]
        self.finished[self.current] = {'key': list(self.current), 'gate': self.gate, 'handoff': self.handoff,
                                      'prefix_comparison': comparison, 'actual_actions': len(self.records), 's2_queries': self.s2_calls,
                                      'autonomous_body_actions_after': sum(r['phase'] == 'continuation' and r['action'] in (1,2,3) for r in after),
                                      'last_actual_action': self.records[-1]['action'], 'last_env_done': self.records[-1]['done']}
        self.current = None

    def close(self, error=None):
        self.evaluator.max_steps_per_episode = self.original_limit
        self.action_stream.close()
        pilot.write_json(self.directory / 'turn_summary.json', {
            'status': 'failed' if error else self.status, 'error': error,
            'cases': list(self.finished.values()), 'scope': SCOPE})


def run(args):
    root = Path(args.run_dir).resolve()
    _, source, protocol = read_run(root)
    if (root / args.arm).exists():
        raise FileExistsError('Arm already exists')
    references = {tuple(c['key']): [] for c in protocol['cases']}
    for row in physical.read_lines(source / 'control/actions.jsonl'):
        references[capture.episode_key(row)].append(row)
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators['habitat_vln']
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    probes = []
    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get('kv_cache_continuation', False):
            raise RuntimeError('Require cache-disabled inference')
        original_init(evaluator, cfg)
        probe = TurnProbe(evaluator, root, args.arm, protocol, references)
        probes.append(probe)
        evaluator._physical_recovery_probe = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate
        def counted_generate(*args, **kwargs):
            probe.s2_calls += 1
            return probe.original_generate(*args, **kwargs)
        evaluator.model.generate = counted_generate
    cls.__init__, cls._run_eval_dual_system = initialize, physical.instrument_loop(original_loop)
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError('Expected one evaluator')
        probes[0].finish()
        if set(probes[0].finished) != set(references):
            raise RuntimeError('Not all locked cases completed')
        probes[0].status = 'complete'
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.evaluator.model.generate = probe.original_generate
            probe.close(error)


def report(args):
    root = Path(args.run_dir).resolve()
    _, source, protocol = read_run(root)
    control = capture.read_progress(source / 'control/progress.json')
    runtime = pilot.read_json(source / 'control/pilot_summary.json')
    outcomes = {}
    for arm in ARMS:
        summary = pilot.read_json(root / arm / 'pilot_summary.json')
        diagnostic = pilot.read_json(root / arm / 'turn_summary.json')
        if summary['status'] != 'complete' or diagnostic['status'] != 'complete':
            raise RuntimeError('Incomplete arm: ' + arm)
        common.same_fields(summary, runtime, common.RUNTIME_FIELDS, arm + ' runtime')
        progress = capture.read_progress(root / arm / 'progress.json')
        details = {tuple(c['key']): c for c in diagnostic['cases']}
        records = physical.validate_trace(physical.read_lines(root / arm / 'turn_actions.jsonl'), arm)
        plain = physical.validate_trace(physical.read_lines(root / arm / 'actions.jsonl'), arm)
        if not pilot.compare_traces(plain, records)['actions_and_poses_match']:
            raise RuntimeError('Observer/native action records differ')
        if set(progress) != set(control) or set(details) != set(control):
            raise RuntimeError('Arm cases differ from locked source')
        outcomes[arm] = []
        for case in protocol['cases']:
            key, gate = tuple(case['key']), case['checkpoint']
            info, item = details[tuple(case['key'])], progress[tuple(case['key'])]
            common.same_fields(info['gate'], gate, tuple(info['gate']), arm + ' gate')
            if item['episode_instruction'] != control[key]['episode_instruction']:
                raise RuntimeError('Instruction changed')
            rows = [r for r in records if capture.episode_key(r) == key]
            prefix = gate['prefix_action_count']
            turns = [r for r in rows if r['phase'] == 'alignment']
            if len(turns) != 6 or any(r['action'] != (2 if arm == 'left90' else 3) for r in turns):
                raise RuntimeError('Incorrect forced turn sequence')
            if any(math.dist(r['position'], gate['state']['position']) > 1e-5 for r in turns):
                raise RuntimeError('Forced turn translated the robot')
            target = pose.yaw_quaternion(pose.yaw_degrees(gate['state']['orientation_wxyz']) + ARMS[arm])
            if pose.heading_error(turns[-1]['orientation_wxyz'], target) > .05:
                raise RuntimeError('Forced heading not reached')
            references = [r for r in physical.read_lines(source / 'control/actions.jsonl') if capture.episode_key(r) == key]
            if not pilot.compare_traces(references[:prefix], rows[:prefix])['actions_and_poses_match']:
                raise RuntimeError('Saved prefix differs')
            base = control[key]
            outcomes[arm].append({'key': list(key), 'stratum': case['stratum'],
                'control_success': base['success'], 'success': item['success'],
                'rescued': base['success'] == 0 and item['success'] == 1,
                'harmed': base['success'] == 1 and item['success'] == 0,
                'control_spl': base['spl'], 'spl': item['spl'], 'spl_change': item['spl'] - base['spl'],
                'control_ne': base['ne'], 'ne': item['ne'], 'forced_turns': 6,
                'total_actual_actions': len(rows), 'control_actual_actions': len(references),
                'actual_action_change': len(rows) - len(references),
                's2_queries': info['s2_queries'], 'native_main_steps': item['steps'],
                'charged_native_main_step_limit': protocol['charged_main_step_limit'],
                'main_step_limit_reached': item['steps'] > protocol['charged_main_step_limit'] and not rows[-1]['done'],
                'autonomous_body_actions_after': info['autonomous_body_actions_after'],
                'last_actual_action': rows[-1]['action'], 'last_env_done': rows[-1]['done'],
                'final_position': rows[-1]['position'], 'handoff': info['handoff']})
    result = {'schema': 'orientation_turn_report.v1', 'status': 'READY_FOR_REVIEW',
              'outcomes': outcomes, 'counts_by_direction': {
                  arm: {'rescued': sum(r['rescued'] for r in rows), 'harmed': sum(r['harmed'] for r in rows)}
                  for arm, rows in outcomes.items()},
              'budget_rule': protocol['budget_rule'], 'scope': SCOPE}
    pilot.write_json(root / 'turn_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_all(args):
    root = Path(args.run_dir).resolve()
    read_run(root)
    if any((root / arm).exists() for arm in ARMS):
        raise FileExistsError('run-all is create-only')
    for arm in ARMS:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--run-dir', str(root), '--arm', arm], check=True)
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.set_defaults(function=prepare)
    for name, function in (('run', run), ('run-all', run_all), ('report', report)):
        command = commands.add_parser(name)
        command.add_argument('--run-dir', required=True)
        if name == 'run':
            command.add_argument('--arm', choices=tuple(ARMS), required=True)
        command.set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()