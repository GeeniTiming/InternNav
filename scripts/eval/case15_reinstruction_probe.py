"""One-shot, in-place prompt controls at case15's verified recovery boundary.

A new native control and two fixed prompts reuse the next existing System-2
query. Compare their continuations with a completed reference-guided physical
correction. No new recovery policy, model training or history editing.
"""
import argparse
import ast
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import textwrap
from types import SimpleNamespace

# Existing helpers are imported without initializing Habitat, Torch or a model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physical_recovery_probe as physical
import cache_pilot as pilot

ARMS = ('control', 'neutral', 'review')
PROMPTS = {
    'control': '',
    'neutral': (
        'Before choosing the next navigation output, carefully reconsider the '
        'current observation and the original instruction. Return only the usual '
        'navigation output, without an explanation.'),
    'review': (
        'Before choosing the next navigation output, review the available historical '
        'observations together with the current observation and the original instruction. '
        'Reassess which parts of the instruction have been completed and which remain. '
        'Return only the usual navigation output, without an explanation.'),
}
SCOPE = (
    'One selected, previously inspected val-unseen development case and fixed prompt '
    'interventions. No autonomous recovery algorithm, held-out efficacy estimate or '
    'isolated memory-causality conclusion. Failure of these prompts does not rule out '
    'other in-place interventions. Physical reference is privileged and reference-informed.'
)
RUNTIME_FIELDS = ('seed_base', 'stage_synchronized', 'torch', 'cuda', 'transformers', 'device')
GATE_FIELDS = ('step', 'prefix_action_count', 'state', 'observation_hashes',
               'history_length', 'history_sha256')
PROMPT_FIELDS = ('step', 'actual_action_count', 'base_prompt_sha256', 'images_sha256',
                 'image_count', 'history_ids', 'state', 'rng_sha256', 'parameters_sha256')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parameter_versions(model):
    return digest(repr([(name, tuple(p.shape), p._version) for name, p in model.named_parameters()]))


def rng_digest():
    """Read values, not serialized Tensor storage identities; never sample/reset."""
    import numpy as np
    import torch
    value = hashlib.sha256(repr(random.getstate()).encode())
    state = np.random.get_state()
    value.update(repr((state[0], state[2:])).encode())
    value.update(state[1].tobytes())
    value.update(torch.get_rng_state().cpu().numpy().tobytes())
    for tensor in torch.cuda.get_rng_state_all():
        value.update(tensor.cpu().numpy().tobytes())
    return value.hexdigest()


def same_fields(actual, expected, fields, label):
    for field in fields:
        if actual[field] != expected[field]:
            raise RuntimeError('NOT_COMPARABLE: ' + label + ' differs: ' + field)


def source_check(source):
    manifest = pilot.verify_manifest(source)
    if 'physical_recovery_probe' not in manifest:
        raise ValueError('--source-run must be the completed case15 physical-recovery experiment')
    comparison = physical.control_check(source, pilot, manifest)
    protocol = pilot.read_json(source / 'recovery_protocol.json')
    reference = physical.locked_reference(source, pilot)
    dataset = pilot.read_json(source / 'episodes.json.gz')
    if len(dataset['episodes']) != 1 or physical.make_protocol(dataset['episodes'][0], reference) != protocol:
        raise ValueError('Physical-source case/protocol changed')
    summaries = {a: pilot.read_json(source / a / 'pilot_summary.json') for a in ('control', 'repair')}
    if any(s['status'] != 'complete' for s in summaries.values()):
        raise ValueError('Physical reference must be complete')
    same_fields(summaries['repair'], summaries['control'], RUNTIME_FIELDS, 'physical-source runtime')
    diagnostic = pilot.read_json(source / 'repair/recovery_summary.json')
    if diagnostic['status'] != 'complete' or not diagnostic['trigger_reached'] or not diagnostic['handoff']:
        raise ValueError('Physical reference has no completed handoff')
    same_fields(pilot.read_json(source / 'repair/boundary_gate.json'),
                pilot.read_json(source / 'control/boundary_gate.json'), GATE_FIELDS,
                'physical-source boundary')
    records = physical.validate_trace(physical.read_lines(source / 'repair/probe_actions.jsonl'),
                                      'physical reference')
    if not pilot.compare_traces(reference[:protocol['prefix_action_count']],
                                records[:protocol['prefix_action_count']])['actions_and_poses_match']:
        raise ValueError('Physical reference prefix changed')
    return manifest, protocol, reference, comparison, summaries, diagnostic


def prepare(args):
    source, root = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if root.exists():
        raise FileExistsError('Use a new run-dir; existing experiments are preserved')
    old, boundary, _, _, _, _ = source_check(source)
    native = pilot.load_config(source / 'control.py')
    if native.agent.model_settings.get('kv_cache_continuation', False):
        raise ValueError('Native cache-disabled source is required')
    if native.agent.model_settings.get('mode') != 'dual_system':
        raise ValueError('Expected native dual_system inference')
    protocol = {
        'schema': 'case15_reinstruction.v1', 'trigger_step': boundary['trigger_step'],
        'prefix_action_count': boundary['prefix_action_count'], 'arms': list(ARMS),
        'prompts': PROMPTS, 'insertion': 'Once, in the next existing native System-2 user prompt.',
        'history_rule': 'Keep all native history, sampling and action decoding unchanged.',
        'scope': SCOPE,
    }
    # Only new configs and a declared protocol are written. Source files are read-only.
    manifest = {
        'schema': protocol['schema'], 'seed_base': old['seed_base'], 'episodes': old['episodes'],
        'revision': subprocess.run(['git', '-C', str(pilot.ROOT), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, check=False).stdout.strip(),
        'source_hashes': dict(old['source_hashes']),
        'checkpoint_shard_sizes': dict(old['checkpoint_shard_sizes']),
        'generated_hashes': {}, 'reinstruction_probe': {'source_run': str(source), 'scope': SCOPE},
    }
    manifest['source_hashes'].update(old['generated_hashes'])
    watched = [source / 'manifest.json', Path(__file__).resolve(),
               Path(physical.__file__).resolve(), Path(pilot.__file__).resolve()]
    for arm in ('control', 'repair'):
        watched += [source / arm / p for p in (
            'actions.jsonl', 'probe_actions.jsonl', 'pilot_summary.json',
            'boundary_gate.json', 'recovery_summary.json')]
    watched += [source / 'repair/after_recovery.json']
    for path in watched:
        manifest['source_hashes'][str(path)] = pilot.sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    protocol_path = root / 'reinstruction_protocol.json'
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
            + 'eval_cfg = EvalCfg.model_validate(json.loads(' + repr(serialized) + '))\n', encoding='utf-8')
        manifest['generated_hashes'][str(path)] = pilot.sha256(path)
    pilot.write_json(root / 'manifest.json', manifest)
    print('REINSTRUCTION_PREPARE=PASS', 'step=85', 'arms=', ','.join(ARMS), 'run_dir=', root)
    print('No model/Habitat rollout. Physical results and the 24-case train cohort remain unchanged.')


def instrument_loop(function):
    """Two process-local hooks; refuse a changed native loop/prompt structure."""
    if function.__code__.co_freevars:
        raise ValueError('Unexpected evaluator closure')
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)
             and any(isinstance(a, ast.Attribute) and a.attr == 'max_steps_per_episode'
                     for a in ast.walk(n.test))]
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'prompt_instruction']
    expected = ast.parse('prompt_instruction = copy.deepcopy(sources[0]["value"])').body[0]
    if len(loops) != 1 or len(assignments) != 1 or ast.dump(assignments[0]) != ast.dump(expected):
        raise ValueError('Native loop/prompt shape changed; refusing to instrument')
    loops[0].body.insert(0, ast.parse(
        'observations, done = self._reinstruction_probe.on_boundary('
        'step_id, observations, done, rgb_list, action_seq, pixel_goal, local_actions, action)').body[0])
    prompt_hook = ast.parse(
        'prompt_instruction = self._reinstruction_probe.amend_prompt('
        'step_id, prompt_instruction, history_id, input_images)').body[0]
    inserted = 0
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and assignments[0] in value:
                value.insert(value.index(assignments[0]) + 1, prompt_hook)
                inserted += 1
    if inserted != 1:
        raise ValueError('Ambiguous prompt insertion')
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, '<case15-reinstruction>', 'exec'), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


class PromptProbe(physical.BoundaryProbe):
    # Reuse native action/pose recording only; no inherited recovery is invoked.
    def __init__(self, evaluator, root, arm, boundary, reference, source):
        super().__init__(evaluator, root, arm, pilot, boundary, reference)
        self.source = source
        self.prompt_event = None
        self.native_s2_queries = 0
        self.history_prefix = ()
        self.parameters = None

    def on_boundary(self, step, observations, done, history, action_seq, pixel_goal, local_actions, action):
        self.current_step = int(step)
        if step != self.protocol['trigger_step'] or self.triggered:
            return observations, done
        if done or action_seq or pixel_goal is not None or local_actions or action not in (0, 1, 2, 3):
            raise RuntimeError('NOT_COMPARABLE: expected the locked idle System-2 boundary')
        count = self.protocol['prefix_action_count']
        comparison = pilot.compare_traces(self.reference[:count], self.records)
        if len(self.records) != count or not comparison['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: native prefix differs before prompting')
        self.gate = {
            'step': int(step), 'prefix_action_count': count, 'state': self.state(),
            'observation_hashes': physical.observation_hashes(observations),
            'history_length': len(history), 'history_sha256': physical.history_hash(history),
            'rng_sha256': rng_digest(), 'prefix_comparison': comparison,
        }
        same_fields(self.gate, pilot.read_json(self.source / 'control/boundary_gate.json'),
                    GATE_FIELDS, 'source boundary')
        if self.arm != 'control':
            same_fields(self.gate, pilot.read_json(self.root / 'control/boundary_gate.json'),
                        GATE_FIELDS + ('rng_sha256',), 'in-place boundary')
        self.history_prefix = tuple(history)
        self.parameters = parameter_versions(self.evaluator.model)
        pilot.write_json(self.directory / 'boundary_gate.json', self.gate)
        self.triggered = True
        self.phase = 'native_after_boundary'
        print('REINSTRUCTION_BOUNDARY=PASS', self.arm, 'step=', step, 'actions=', count)
        return observations, done

    def amend_prompt(self, step, prompt, history_ids, input_images):
        self.native_s2_queries += 1
        if step != self.protocol['trigger_step'] or self.prompt_event is not None:
            return prompt
        if not self.triggered:
            raise RuntimeError('NOT_COMPARABLE: prompt reached before boundary validation')
        event = {
            'arm': self.arm, 'step': int(step), 'actual_action_count': len(self.records),
            'base_prompt_sha256': digest(prompt), 'base_prompt': prompt,
            'images_sha256': physical.history_hash(input_images), 'image_count': len(input_images),
            'history_ids': [int(v) for v in history_ids], 'state': self.state(),
            'rng_sha256': rng_digest(), 'parameters_sha256': parameter_versions(self.evaluator.model),
        }
        if self.arm != 'control':
            same_fields(event, pilot.read_json(self.root / 'control/prompt_event.json'),
                        PROMPT_FIELDS, 'pre-prompt context')
        reminder = PROMPTS[self.arm]
        amended = prompt + (' ' + reminder if reminder else '')
        event.update({'reminder': reminder, 'effective_prompt': amended,
                      'effective_prompt_sha256': digest(amended),
                      'extra_env_actions_at_insertion': len(self.records) - event['actual_action_count'],
                      'rng_unchanged_at_insertion': rng_digest() == event['rng_sha256']})
        if not event['rng_unchanged_at_insertion'] or event['extra_env_actions_at_insertion']:
            raise RuntimeError('Prompt insertion changed RNG or executed an environment action')
        if parameter_versions(self.evaluator.model) != event['parameters_sha256']:
            raise RuntimeError('Prompt insertion changed parameter versions')
        self.prompt_event = event
        pilot.write_json(self.directory / 'prompt_event.json', event)
        print('PROMPT_CONTROL=PASS', self.arm, 'insertions=', int(bool(reminder)))
        return amended

    def close(self, error=None):
        self.stream.close()
        history_ok = bool(self.gate) and physical.history_hash(self.history_prefix) == self.gate['history_sha256']
        parameters_ok = self.parameters is not None and parameter_versions(self.evaluator.model) == self.parameters
        if self.status == 'complete' and not (history_ok and parameters_ok and self.prompt_event):
            self.status = 'invalid'
        pilot.write_json(self.directory / 'reinstruction_summary.json', {
            'status': self.status, 'error': error, 'arm': self.arm,
            'boundary_reached': self.triggered, 'prompt_event_recorded': self.prompt_event is not None,
            'prompt_insertions': int(bool(self.prompt_event and PROMPTS[self.arm])),
            'external_recovery_actions': self.recovery_actions,
            'native_s2_queries': self.native_s2_queries,
            'history_prefix_unchanged': history_ok, 'parameter_versions_unchanged': parameters_ok,
            'scope': SCOPE})


def control_check(root, manifest, source, reference):
    old = pilot.read_json(source / 'control/pilot_summary.json')
    new = pilot.read_json(root / 'control/pilot_summary.json')
    if new['status'] != 'complete':
        raise RuntimeError('NOT_COMPARABLE: new control is incomplete')
    same_fields(new, old, RUNTIME_FIELDS, 'control runtime')
    rows = physical.validate_trace(physical.read_lines(root / 'control/actions.jsonl'), 'new control')
    comparison = pilot.compare_traces(reference, rows)
    if not comparison['actions_and_poses_match'] or not pilot.metrics_match(new['metrics'], old['metrics']):
        raise RuntimeError('NOT_COMPARABLE: control did not reproduce the complete native failure')
    check = pilot.read_json(root / 'control/reinstruction_summary.json')
    if check['status'] != 'complete' or not check['prompt_event_recorded']:
        raise RuntimeError('NOT_COMPARABLE: new control did not pass the prompt gate')
    return comparison


def read_run(root):
    manifest = pilot.verify_manifest(root)
    protocol = pilot.read_json(root / 'reinstruction_protocol.json')
    if (protocol['schema'] != 'case15_reinstruction.v1' or protocol['prompts'] != PROMPTS
            or protocol['arms'] != list(ARMS)):
        raise ValueError('The predeclared prompt protocol differs')
    source = Path(manifest['reinstruction_probe']['source_run'])
    _, boundary, reference, _, _, _ = source_check(source)
    return manifest, source, boundary, reference


def run(args):
    root = Path(args.run_dir).resolve()
    manifest, source, boundary, reference = read_run(root)
    if args.arm not in ARMS:
        raise ValueError('Unknown arm')
    if args.arm != 'control':
        control_check(root, manifest, source, reference)
    sys.path.insert(0, str(pilot.ROOT))
    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    cls = Evaluator.evaluators['habitat_vln']
    original_init, original_loop = cls.__init__, cls._run_eval_dual_system
    wrapped = instrument_loop(original_loop)
    probes = []

    def initialize(evaluator, cfg):
        if cfg.agent.model_settings.get('kv_cache_continuation', False):
            raise RuntimeError('This diagnostic requires native, cache-disabled inference')
        original_init(evaluator, cfg)
        probe = PromptProbe(evaluator, root, args.arm, boundary, reference, source)
        probes.append(probe)
        evaluator._reinstruction_probe = probe
        evaluator.env.step = probe.record

    cls.__init__, cls._run_eval_dual_system = initialize, wrapped
    error = None
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(root), arm=args.arm, profile_stages=False))
        if len(probes) != 1 or not probes[0].triggered or probes[0].prompt_event is None:
            raise RuntimeError('NOT_COMPARABLE: expected exactly one boundary and prompt event')
        probes[0].status = 'complete'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        cls.__init__, cls._run_eval_dual_system = original_init, original_loop
        for probe in probes:
            probe.evaluator.env.step = probe.original_step
            probe.close(error)
    if any(p.status != 'complete' for p in probes):
        raise RuntimeError('NOT_COMPARABLE: history/parameter check failed')
    if args.arm == 'control':
        control_check(root, manifest, source, reference)
        print('NATIVE_CONTROL_REPRODUCTION=PASS')


def report(args):
    root = Path(args.run_dir).resolve()
    manifest, source, boundary, reference = read_run(root)
    comparison = control_check(root, manifest, source, reference)
    control = pilot.read_json(root / 'control/pilot_summary.json')
    outcomes = {}
    for arm in ARMS:
        summary = pilot.read_json(root / arm / 'pilot_summary.json')
        check = pilot.read_json(root / arm / 'reinstruction_summary.json')
        if summary['status'] != 'complete' or check['status'] != 'complete' or check['external_recovery_actions']:
            raise RuntimeError('Incomplete or invalid prompt arm: ' + arm)
        same_fields(summary, control, RUNTIME_FIELDS, arm + ' runtime')
        same_fields(pilot.read_json(root / arm / 'boundary_gate.json'),
                    pilot.read_json(root / 'control/boundary_gate.json'),
                    GATE_FIELDS + ('rng_sha256',), arm + ' boundary')
        event = pilot.read_json(root / arm / 'prompt_event.json')
        same_fields(event, pilot.read_json(root / 'control/prompt_event.json'),
                    PROMPT_FIELDS, arm + ' original prompt context')
        if event['effective_prompt'] != event['base_prompt'] + (' ' + PROMPTS[arm] if PROMPTS[arm] else ''):
            raise RuntimeError('Effective prompt differs from the declared text: ' + arm)
        rows = physical.validate_trace(physical.read_lines(root / arm / 'actions.jsonl'), arm)
        count = boundary['prefix_action_count']
        if not pilot.compare_traces(reference[:count], rows[:count])['actions_and_poses_match']:
            raise RuntimeError('NOT_COMPARABLE: saved prompt-arm prefix differs: ' + arm)
        if (not event['rng_unchanged_at_insertion'] or event['extra_env_actions_at_insertion']
                or event['reminder'] != PROMPTS[arm]
                or check['prompt_insertions'] != int(bool(PROMPTS[arm]))):
            raise RuntimeError('Invalid prompt intervention record: ' + arm)
        outcomes[arm] = {
            'metrics': summary['metrics'], 'prompt_insertions': check['prompt_insertions'],
            'external_recovery_actions': 0, 'native_s2_queries': check['native_s2_queries'],
            'autonomous_body_actions_after_boundary': sum(r['action'] <= 3 for r in rows[boundary['prefix_action_count']:]),
            'video': [str(p) for p in sorted((root / 'video' / arm).rglob('*.mp4'))],
        }
    old = pilot.read_json(source / 'repair/pilot_summary.json')
    diag = pilot.read_json(source / 'repair/recovery_summary.json')
    outcomes['physical_reference'] = {
        'metrics': old['metrics'], 'external_recovery_actions': diag['recovery_actions'],
        'handoff': diag['handoff'], 'source_run': str(source),
        'reference_informed': True, 'new_rollout_executed': False,
    }
    successful_prompts = [a for a in ('neutral', 'review') if float(outcomes[a]['metrics']['sucs_all']) > 0]
    interpretation = (
        'At least one locked in-place prompt succeeded at this checkpoint. Relocation was not '
        'required for that observed continuation; no memory cause or general recovery efficacy follows.'
        if successful_prompts else
        'Neither locked in-place prompt succeeded here. This does not exclude other in-place '
        'interventions or establish an online physical recovery method.'
    )
    result = {
        'schema': 'case15_reinstruction_report.v1', 'status': 'READY_FOR_INTERPRETATION',
        'key': [physical.SCENE, '15'], 'trigger_step': boundary['trigger_step'],
        'control_exact_reproduction': comparison, 'outcomes': outcomes,
        'successful_prompt_arms': successful_prompts, 'interpretation': interpretation,
        'timing_valid_for_speed_comparison': False, 'scope': SCOPE,
    }
    pilot.write_json(root / 'reinstruction_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-run', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.set_defaults(function=prepare)
    execute = commands.add_parser('run')
    execute.add_argument('--run-dir', required=True)
    execute.add_argument('--arm', choices=ARMS, required=True)
    execute.set_defaults(function=run)
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
