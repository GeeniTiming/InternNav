"""Locked, training-free recovery-candidate experiment on the orientation cohort.

Arms: history_reset, left90_reset, right90_reset, backtrack_replan.
The verified native control in the source checkpoint run supplies RELEASE.
This is an offline *diagnostic*, not an online trigger, trained selector, or
claim of held-out recovery efficacy.
"""
import argparse
import ast
import copy
import functools
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orientation_turn as turn

pilot = turn.pilot
physical = turn.physical
common = turn.common
checkpoint = turn.checkpoint
capture = turn.capture
pose = turn.pose

ARMS = ('history_reset', 'left90_reset', 'right90_reset', 'backtrack_replan')
RESET_ARMS = frozenset(('history_reset', 'left90_reset', 'right90_reset'))
ROTATIONS = {'left90_reset': 90, 'right90_reset': -90}
BACKTRACK_MIN_DISTANCE_M = 0.75
BACKTRACK_ACTION_BUDGET = 80
BACKTRACK_FOLLOWER_RADIUS_M = 0.30
BACKTRACK_ENDPOINT_TOLERANCE_M = 0.35
MAX_NATIVE_DISPLACEMENT_M = 0.31
SCOPE = ('Fixed six-case val-unseen development diagnostic with known native outcomes. '
         'The same predeclared checkpoint is replayed in each arm. Native CONTROL '
         'is reused without rewriting it. No outcome-conditional candidate tuning, '
         'online deviation trigger, trained selector, isolated memory mechanism '
         'or held-out efficacy claim. Simulator pathfinding is privileged execution.')


def choose_backtrack_anchor(rows, boundary_position, minimum_distance=BACKTRACK_MIN_DISTANCE_M):
    """Most recent distinct *visited* prefix position at least threshold away.

    Only positions observed before the locked decision boundary are used.
    No reference path, target point, success radius, future trace, or outcome.
    """
    if not rows:
        return None
    if not all(math.isfinite(float(x)) for x in boundary_position):
        raise ValueError('Invalid boundary position')
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        pos = row['position']
        if len(pos) != 3 or not all(math.isfinite(float(x)) for x in pos):
            raise ValueError('Invalid visited prefix position')
        if math.dist(pos, boundary_position) >= minimum_distance:
            return {'prefix_action_index': index, 'position': list(pos),
                    'distance_from_boundary_m': math.dist(pos, boundary_position)}
    return None


def post_reset_history_indices(cutoff, rgb_count, num_history):
    """Choose only *post-handoff* history; current RGB is appended separately.

    rgb_count includes the just-appended current image. Do NOT truncate rgb_list:
    native indexing continues to use main-loop step numbers.
    """
    if cutoff < 0 or rgb_count < cutoff + 1 or num_history < 0:
        raise ValueError('Invalid history-reset bookkeeping')
    end = rgb_count - 2
    if end < cutoff or num_history == 0:
        return []
    import numpy as np
    count = min(int(num_history), end - cutoff + 1)
    return np.unique(np.linspace(cutoff, end, count, dtype=np.int32)).tolist()


def instrument_loop(function):
    """Guarded process-local AST hooks at native loop head and history assembly.

    Keep the checked-in evaluator unchanged; reject any native source drift.
    """
    if function.__code__.co_freevars:
        raise ValueError('Unexpected evaluator closure')
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)
             and any(isinstance(n, ast.Attribute) and n.attr == 'max_steps_per_episode'
                     for n in ast.walk(node.test))]
    if len(loops) != 1:
        raise ValueError('Expected exactly one native episode loop')
    loop = loops[0]
    matches = []
    expected = ast.parse("history_id = np.unique(\n    np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)\n).tolist()").body[0]
    for node in ast.walk(loop):
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, child in enumerate(value):
                if isinstance(child, ast.Assign) and ast.dump(child) == ast.dump(expected):
                    matches.append((value, index))
    if len(matches) != 1:
        raise ValueError('Native history selection shape changed')
    statements, index = matches[0]
    statements.insert(index + 1, ast.parse(
        'history_id = self._recovery_candidate.filter_history_ids('
        'step_id, history_id, len(rgb_list))').body[0])
    loop.body.insert(0, ast.parse(
        'observations, done = self._recovery_candidate.on_boundary('
        'step_id, observations, done, rgb_list, action_seq, pixel_goal, local_actions, action)'
    ).body[0])
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, '<recovery-candidates-native-loop>', 'exec'), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


def locked_anchors(report, source):
    """Predeclare all backtrack candidates from *control prefixes* in prepare."""
    references = {}
    for row in physical.read_lines(source / 'control/actions.jsonl'):
        key = capture.episode_key(row)
        references.setdefault(key, []).append(row)
    result = {}
    for case in report['cases']:
        key = tuple(case['key'])
        gate = case['checkpoint']
        if gate is None:
            raise ValueError('All locked cases must have comparable checkpoints')
        prefix = references[key][:gate['prefix_action_count']]
        if checkpoint.cohort.digest(prefix) != gate['prefix_trace_sha256']:
            raise ValueError('Locked prefix differs: ' + repr(key))
        result['/'.join(key)] = choose_backtrack_anchor(prefix, gate['state']['position'])
    return result


def prepare(args):
    source, root = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Create-only experiment: use a new --run-dir')
    old, report = turn.source_check(source)
    native = pilot.load_config(source / 'control.py')
    if native.agent.model_settings.get('kv_cache_continuation', False):
        raise ValueError('Require cache-disabled native baseline')
    if native.agent.model_settings.get('mode') != 'dual_system':
        raise ValueError('Require dual_system native baseline')
    limit = int(native.eval_settings['max_steps_per_episode'])
    if limit < BACKTRACK_ACTION_BUDGET + 1:
        raise ValueError('Native episode budget is too small for declared actions')
    protocol = {
        'schema': 'recovery_candidates.v1', 'cases': report['cases'],
        'arms': list(ARMS), 'release_control_source': str(source / 'control'),
        'backtrack_anchors': locked_anchors(report, source),
        'backtrack_min_displacement_m': BACKTRACK_MIN_DISTANCE_M,
        'backtrack_budget': BACKTRACK_ACTION_BUDGET,
        'follower_radius_m': BACKTRACK_FOLLOWER_RADIUS_M,
        'endpoint_tolerance_m': BACKTRACK_ENDPOINT_TOLERANCE_M,
        'max_native_displacement_m': MAX_NATIVE_DISPLACEMENT_M,
        'native_main_step_limit': limit,
        'history_rule': ('At handoff, preserve rgb_list and all history bytes; '
                         'for each subsequent ordinary System-2 prompt, replace historical '
                         'image indices by an evenly spaced sample of images acquired AFTER '
                         'handoff, excluding the current image, which remains in cur_images. '
                         'Do not edit the instruction, camera/lookdown messages or model state.'),
        'budget_rule': ('Actual external turns/backtrack motions are Habitat actions and '
                        'reduce the permitted native main-loop horizon one-for-one; '
                        'the native step_id/history indexing is never artificially advanced.'),
        'scope': SCOPE,
    }
    manifest = {
        'schema': protocol['schema'], 'seed_base': old['seed_base'],
        'episodes': old['episodes'], 'checkpoint_shard_sizes': old['checkpoint_shard_sizes'],
        'revision': subprocess.run(['git', '-C', str(pilot.ROOT), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, check=False).stdout.strip(),
        'source_hashes': dict(old['source_hashes']),
        'generated_hashes': {},
        'recovery_candidates': {'source_run': str(source), 'scope': SCOPE},
    }
    manifest['source_hashes'].update(old['generated_hashes'])
    watched = [Path(__file__).resolve(), Path(turn.__file__).resolve(),
               Path(checkpoint.__file__).resolve(), Path(physical.__file__).resolve(),
               source / 'manifest.json', source / 'checkpoint_report.json']
    watched += [p for p in (source / 'control').rglob('*')
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.png')]
    manifest['source_hashes'].update({str(p): pilot.sha256(p) for p in watched})
    root.mkdir(parents=True, exist_ok=False)
    protocol_path = root / 'candidate_protocol.json'
    pilot.write_json(protocol_path, protocol)
    manifest['generated_hashes'][str(protocol_path)] = pilot.sha256(protocol_path)
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
            + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n',
            encoding='utf-8')
        manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    pilot.write_json(root / 'manifest.json', manifest)
    print('RECOVERY_CANDIDATES_PREPARE=PASS', len(report['cases']), 'cases', len(ARMS), 'new arms')
    print('Backtrack eligible:', sum(v is not None for v in protocol['backtrack_anchors'].values()))


def read_run(root):
    manifest = pilot.verify_manifest(root)
    source = Path(manifest['recovery_candidates']['source_run'])
    _, report = turn.source_check(source)
    protocol = pilot.read_json(root / 'candidate_protocol.json')
    if (protocol['schema'] != 'recovery_candidates.v1' or
            protocol['cases'] != report['cases'] or protocol['arms'] != list(ARMS) or
            protocol['backtrack_anchors'] != locked_anchors(report, source)):
        raise ValueError('Locked recovery protocol changed')
    return manifest, source, protocol


class CandidateProbe(turn.TurnProbe):
    """Use the proven strict prefix gates and native action recorder."""

    def begin(self, key):
        super().begin(key)
        self.history_cutoff = None
        self.history_events = []
        self.anchor = None
        self.external_actions = 0
        self.eligible = True
        self.reset_enabled = self.arm in RESET_ARMS

    def filter_history_ids(self, step, original_ids, rgb_count):
        if self.gate is None or self.history_cutoff is None:
            return original_ids
        selected = post_reset_history_indices(
            self.history_cutoff, rgb_count, int(self.evaluator.num_history))
        if any(i < self.history_cutoff or i >= rgb_count - 1 for i in selected):
            raise RuntimeError('Old history leaked into a reset System-2 prompt')
        self.history_events.append({
            'key': list(self.current), 'main_step': int(step),
            'rgb_count': int(rgb_count), 'cutoff': self.history_cutoff,
            'original_history_ids': [int(v) for v in original_ids],
            'selected_history_ids': selected,
        })
        return selected

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
                or done or action_seq or pixel_goal is not None or local_actions
                or action not in (0, 1, 2, 3)):
            raise RuntimeError('NOT_COMPARABLE: locked idle boundary not reproduced')
        pitch = checkpoint.camera_pitch_degrees(checkpoint.quaternion_list(
            self.evaluator.env._env.sim.get_agent_state().sensor_states['rgb'].rotation))
        if abs(pitch) > 1e-3:
            raise RuntimeError('NOT_COMPARABLE: RGB camera is not level')
        gate = {'step': int(step), 'prefix_action_count': len(self.records),
                'state': self.state(),
                'observation_hashes': physical.observation_hashes(observations),
                'history_length': len(history),
                'history_sha256': physical.history_hash(history),
                'rng_sha256': common.rng_digest(),
                'parameters_sha256': self.parameters,
                'prefix_trace_sha256': checkpoint.cohort.digest(self.records)}
        common.same_fields(gate, expected, tuple(gate), 'candidate boundary')
        if common.parameter_versions(self.evaluator.model) != self.parameters:
            raise RuntimeError('Model parameters changed before recovery')
        self.gate, self.history_prefix = gate, tuple(history)
        latest = [observations]
        self.phase = 'external_recovery'
        def advance(act):
            result = self.evaluator.env.step(act)
            latest[0] = result[0]
            return result
        with physical.preserve_policy_rng():
            if self.arm in ROTATIONS:
                target = pose.yaw_quaternion(
                    pose.yaw_degrees(gate['state']['orientation_wxyz']) + ROTATIONS[self.arm])
                turns = pose.turn_to_heading(target, self.state, advance, budget=6)
                if turns != 6:
                    raise RuntimeError('RECOVERY_FAILED: expected six actual native turns')
            elif self.arm == 'backtrack_replan':
                name = '/'.join(key)
                self.anchor = self.protocol['backtrack_anchors'][name]
                observed = choose_backtrack_anchor(self.records, gate['state']['position'])
                if observed != self.anchor:
                    raise RuntimeError('NOT_COMPARABLE: backtrack anchor changed')
                if self.anchor is None:
                    self.eligible = False  # no cherry-picking/replacement
                else:
                    self._backtrack(advance)
            elif self.arm != 'history_reset':
                raise ValueError('Unexpected candidate arm: ' + self.arm)
        self.external_actions = len(self.records) - expected['prefix_action_count']
        if self.arm in ROTATIONS and self.external_actions != 6:
            raise RuntimeError('RECOVERY_FAILED: incorrect turn count')
        if self.arm == 'history_reset' and self.external_actions != 0:
            raise RuntimeError('History reset must not move the agent')
        if (common.rng_digest() != gate['rng_sha256']
                or physical.history_hash(history) != gate['history_sha256']
                or common.parameter_versions(self.evaluator.model) != self.parameters):
            raise RuntimeError('Policy RNG, original history or weights changed')
        self.evaluator.max_steps_per_episode = self.original_limit - self.external_actions
        if self.evaluator.max_steps_per_episode <= step:
            raise RuntimeError('RECOVERY_FAILED: no charged native continuation budget')
        self.history_cutoff = len(history) if self.reset_enabled else None
        self.phase = 'continuation'
        self.handoff = {
            'state': self.state(), 'external_actions': self.external_actions,
            'anchor': self.anchor, 'eligible': self.eligible,
            'history_reset': self.reset_enabled,
            'history_cutoff': self.history_cutoff,
            'policy_rng_restored': True, 'model_unchanged': True,
            'original_history_unchanged': True,
            'charged_native_limit': self.evaluator.max_steps_per_episode,
            'final_observation_hashes': physical.observation_hashes(latest[0]),
        }
        folder = self.directory / 'checkpoints' / (key[0] + '_' + key[1])
        folder.mkdir(parents=True, exist_ok=False)
        pilot.write_json(folder / 'boundary.json', gate)
        pilot.write_json(folder / 'handoff.json', self.handoff)
        from PIL import Image
        Image.fromarray(latest[0]['rgb']).save(folder / 'after_recovery.png')
        print('RECOVERY_HANDOFF=PASS', self.arm, key,
              'external_actions=', self.external_actions, 'eligible=', self.eligible)
        return latest[0], False

    def _backtrack(self, advance):
        import numpy as np
        import habitat_sim
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        sim = self.evaluator.env._env.sim
        target = np.asarray(self.anchor['position'], dtype=np.float32)
        path = habitat_sim.ShortestPath()
        path.requested_start = sim.get_agent_state().position
        path.requested_end = target
        if not sim.pathfinder.is_navigable(target) or not sim.pathfinder.find_path(path):
            raise RuntimeError('RECOVERY_FAILED: visited anchor not navigable/connected')
        follower = ShortestPathFollower(sim, BACKTRACK_FOLLOWER_RADIUS_M, return_one_hot=False)
        physical.drive_waypoints(
            [target], follower.get_next_action, advance,
            lambda: self.state()['position'], BACKTRACK_ACTION_BUDGET,
            BACKTRACK_ENDPOINT_TOLERANCE_M, MAX_NATIVE_DISPLACEMENT_M)

    def finish(self):
        key = self.current
        if key is None:
            return
        super().finish()
        case = self.finished[key]
        case.update({
            'candidate_eligible': self.eligible,
            'external_recovery_actions': self.external_actions,
            'history_reset': self.reset_enabled,
            'history_prompt_events': len(self.history_events),
            'backtrack_anchor': self.anchor,
        })
        folder = self.directory / 'checkpoints' / (key[0] + '_' + key[1])
        pilot.write_json(folder / 'history_inputs.json', self.history_events)


def run(args):
    root = Path(args.run_dir).resolve()
    _, source, protocol = read_run(root)
    if args.arm not in ARMS:
        raise ValueError('Unknown arm')
    if (root / args.arm).exists():
        raise FileExistsError('Create-only arm; preserve the previous results')
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
        probe = CandidateProbe(evaluator, root, args.arm, protocol, references)
        probes.append(probe)
        evaluator._recovery_candidate = probe
        evaluator.env.step = probe.record
        probe.original_generate = evaluator.model.generate
        def counted_generate(*a, **kw):
            probe.s2_calls += 1
            return probe.original_generate(*a, **kw)
        evaluator.model.generate = counted_generate
    cls.__init__, cls._run_eval_dual_system = initialize, instrument_loop(original_loop)
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1:
            raise RuntimeError('Expected exactly one native evaluator')
        probes[0].finish()
        if set(probes[0].finished) != set(references):
            raise RuntimeError('NOT_COMPARABLE: incomplete locked cohort')
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
    controls = capture.read_progress(source / 'control/progress.json')
    runtime = pilot.read_json(source / 'control/pilot_summary.json')
    references = {}
    for row in physical.read_lines(source / 'control/actions.jsonl'):
        references.setdefault(capture.episode_key(row), []).append(row)
    rows = []
    counts = {}
    for arm in ARMS:
        summary = pilot.read_json(root / arm / 'pilot_summary.json')
        diagnostic = pilot.read_json(root / arm / 'turn_summary.json')
        if summary['status'] != 'complete' or diagnostic['status'] != 'complete':
            raise RuntimeError('Incomplete arm: ' + arm)
        common.same_fields(summary, runtime, common.RUNTIME_FIELDS, 'candidate runtime')
        progress = capture.read_progress(root / arm / 'progress.json')
        details = {tuple(c['key']): c for c in diagnostic['cases']}
        tagged = physical.validate_trace(
            physical.read_lines(root / arm / 'turn_actions.jsonl'), arm)
        plain = physical.validate_trace(
            physical.read_lines(root / arm / 'actions.jsonl'), arm)
        if not pilot.compare_traces(plain, tagged)['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: observer/native traces differ')
        if set(progress) != set(controls) or set(details) != set(controls):
            raise RuntimeError('NOT_COMPARABLE: cohort changed')
        counts[arm] = {'rescued': 0, 'harmed': 0, 'eligible': 0, 'not_eligible': 0}
        for case in protocol['cases']:
            key = tuple(case['key'])
            item, detail, base = progress[key], details[key], controls[key]
            gate = case['checkpoint']
            same = pilot.compare_traces(
                references[key][:gate['prefix_action_count']],
                [r for r in tagged if capture.episode_key(r) == key][:gate['prefix_action_count']])
            if not same['actions_and_poses_match']:
                raise RuntimeError('NOT_COMPARABLE: intervention prefix changed: ' + repr(key))
            if item['episode_instruction'] != base['episode_instruction']:
                raise RuntimeError('NOT_COMPARABLE: instruction changed')
            key_rows = [r for r in tagged if capture.episode_key(r) == key]
            external = [r for r in key_rows if r['phase'] == 'external_recovery']
            if len(external) != detail['external_recovery_actions']:
                raise RuntimeError('Recovery actions/summary disagree: ' + repr(key))
            if arm == 'history_reset' and external:
                raise RuntimeError('A history-only intervention performed motion')
            if arm in ROTATIONS:
                expected_action = 2 if arm == 'left90_reset' else 3
                if len(external) != 6 or any(r['action'] != expected_action for r in external):
                    raise RuntimeError('Turn sequence differs from declared arm')
            if arm == 'backtrack_replan' and any(r['action'] not in (1,2,3) for r in external):
                raise RuntimeError('Backtrack contains non-body action')
            expected_eligible = (protocol['backtrack_anchors']['/'.join(key)] is not None
                                 if arm == 'backtrack_replan' else True)
            if detail['candidate_eligible'] != expected_eligible:
                raise RuntimeError('Candidate eligibility changed')
            if arm in RESET_ARMS and detail['history_prompt_events'] <= 0:
                raise RuntimeError('No post-handoff history selection was observed')
            hist = pilot.read_json(root / arm / 'checkpoints' /
                                   (key[0] + '_' + key[1]) / 'history_inputs.json')
            if len(hist) != detail['history_prompt_events']:
                raise RuntimeError('History event count mismatch')
            if any(any(i < e['cutoff'] or i >= e['rgb_count'] - 1
                       for i in e['selected_history_ids']) for e in hist):
                raise RuntimeError('Pre-handoff history entered reset prompt')
            rescued = base['success'] == 0 and item['success'] == 1
            harmed = base['success'] == 1 and item['success'] == 0
            if expected_eligible:
                counts[arm]['eligible'] += 1
                counts[arm]['rescued'] += int(rescued)
                counts[arm]['harmed'] += int(harmed)
            else:
                counts[arm]['not_eligible'] += 1
            rows.append({
                'key': list(key), 'arm': arm, 'eligible': expected_eligible,
                'baseline_success': base['success'], 'success': item['success'],
                'baseline_os': base['os'], 'os': item['os'],
                'baseline_ne': base['ne'], 'ne': item['ne'],
                'rescued': rescued if expected_eligible else None,
                'harmed': harmed if expected_eligible else None,
                'external_actions': len(external),
                'total_actual_actions': len(key_rows),
                'main_steps': item['steps'],
                'stop_action': key_rows[-1]['action'],
                'final_env_done': key_rows[-1]['done'],
                'history_prompt_events': detail['history_prompt_events'],
            })
    result = {'schema': 'recovery_candidates_report.v1',
              'status': 'READY_FOR_INTERPRETATION',
              'source_control_reused': True, 'counts': counts,
              'outcomes': rows, 'scope': SCOPE}
    pilot.write_json(root / 'candidate_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_all(args):
    root = Path(args.run_dir).resolve()
    read_run(root)
    if any((root / arm).exists() for arm in ARMS):
        raise FileExistsError('run-all is create-only, no arm overwrite')
    for arm in ARMS:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                        '--run-dir', str(root), '--arm', arm], check=True)
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.set_defaults(function=prepare)
    for name, fn in (('run', run), ('run-all', run_all), ('report', report)):
        item = subs.add_parser(name)
        item.add_argument('--run-dir', required=True)
        if name == 'run':
            item.add_argument('--arm', choices=ARMS, required=True)
        item.set_defaults(function=fn)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
