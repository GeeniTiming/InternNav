"""Paired readout/S1 diagnosis while only the native full-replay policy acts.

This is a one-episode diagnostic, not a speed benchmark or a recovery method.
The existing pilot helper and production model/evaluator files stay unchanged.
"""

import argparse
import functools
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def capture_rng():
    import numpy as np
    import torch

    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def same_rng(a, b):
    import numpy as np
    import torch

    return (a['python'] == b['python'] and a['numpy'][0] == b['numpy'][0]
            and np.array_equal(a['numpy'][1], b['numpy'][1])
            and a['numpy'][2:] == b['numpy'][2:] and torch.equal(a['torch'], b['torch'])
            and len(a['cuda']) == len(b['cuda'])
            and all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


def tensor_difference(reference, candidate):
    import torch
    import torch.nn.functional as functional

    if reference.shape != candidate.shape:
        raise ValueError('Paired tensor shapes differ')
    a, b = reference.detach().float(), candidate.detach().float()
    difference = a - b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'shape': list(a.shape), 'dtype': str(reference.dtype), 'finite': finite,
            'bitwise_equal': torch.equal(reference, candidate),
            'max_abs_error': float(difference.abs().max()) if finite else None,
            'relative_l2_error': float(difference.norm() / a.norm().clamp_min(1e-12)) if finite else None,
            'min_last_dim_cosine': float(functional.cosine_similarity(a, b, dim=-1).min()) if finite else None}


def capture_cache_prefix(cache):
    # Keep old storage alive without duplicating the entire GPU cache. Appending
    # and cropping may legitimately replace storage, so data_ptr is not a gate.
    return cache.get_seq_length(), tuple((t, t._version) for t in cache.key_cache + cache.value_cache)


def cache_prefix_preserved(before, cache):
    import torch

    length, saved = before
    actual = cache.key_cache + cache.value_cache
    return (length == cache.get_seq_length() and len(saved) == len(actual)
            and all(old._version == version and old.shape == new.shape
                    and old.dtype == new.dtype and old.device == new.device and torch.equal(old, new)
                    for (old, version), new in zip(saved, actual)))


class PairedObserver:
    """Keep the native return values; run shadows with restored RNG states."""

    def __init__(self, model, decode_actions, context, emit):
        self.model = model
        self.decode_actions = decode_actions
        self.context = context
        self.emit = emit
        self.pending = None
        self.native_latents = None
        self.cache_latents = None
        self.readouts = []
        self.plans = []
        self.originals = {}

    def install(self):
        if self.originals:
            raise RuntimeError('Observer already installed')
        for name, wrapper in [('generate', self.generation), ('generate_latents', self.readout),
                              ('generate_traj', self.plan)]:
            original = getattr(self.model, name)
            self.originals[name] = original

            def forward(*args, _call=wrapper, **kwargs):
                return _call(*args, **kwargs)

            setattr(self.model, name, functools.wraps(original)(forward))

    def close(self):
        for name, original in self.originals.items():
            setattr(self.model, name, original)
        self.originals.clear()
        self.pending = self.native_latents = self.cache_latents = None

    def generation(self, *args, **kwargs):
        self.pending = self.native_latents = self.cache_latents = None
        outputs = self.originals['generate'](*args, **kwargs)
        self.pending = (outputs, kwargs.get('attention_mask'))
        return outputs

    def readout(self, input_ids, pixel_values, image_grid_thw):
        import torch

        if self.pending is None or not torch.equal(self.pending[0].sequences, input_ids):
            raise RuntimeError('Readout does not correspond to the captured generation')
        outputs, mask = self.pending
        original = self.originals['generate_latents']
        native = original(input_ids, pixel_values, image_grid_thw)
        after_native = capture_rng()
        rope = getattr(self.model, 'rope_deltas', None)
        cache = outputs.past_key_values
        before_cache = capture_cache_prefix(cache)
        try:
            repeat = original(input_ids, pixel_values, image_grid_thw)
            restore_rng(after_native)
            cached = self.model.generate_latents_from_cache(outputs, image_grid_thw, mask)
            record = {'readout_index': len(self.readouts), **self.context(),
                      'sequence_tokens': input_ids.shape[1], 'cache_tokens': before_cache[0],
                      'native_repeat': tensor_difference(native, repeat),
                      'cache_vs_native': tensor_difference(native, cached),
                      'cache_prefix_preserved': cache_prefix_preserved(before_cache, cache)}
        finally:
            self.model.rope_deltas = rope
            restore_rng(after_native)
        record['rng_restored'] = same_rng(after_native, capture_rng())
        if not record['cache_prefix_preserved'] or not record['rng_restored']:
            raise RuntimeError('Paired readout state-restoration check failed')
        self.readouts.append(record)
        self.emit('readouts', record)
        self.pending = None
        self.native_latents, self.cache_latents = native, cached
        return native

    def actions(self, trajectory):
        # The real decoder scales trajectory tensors in place. Never give it
        # the native return value or the cached tensor itself.
        return [int(v) for v in self.decode_actions(trajectory.detach().clone())]

    def plan(self, traj_latents, *args, **kwargs):
        if self.cache_latents is None or traj_latents is not self.native_latents:
            raise RuntimeError('S1 inputs are not linked to the paired readout')
        original = self.originals['generate_traj']
        before_native = capture_rng()
        native = original(traj_latents, *args, **kwargs)
        after_native = capture_rng()
        try:
            restore_rng(before_native)
            repeat = original(traj_latents, *args, **kwargs)
            restore_rng(before_native)
            cached = original(self.cache_latents, *args, **kwargs)
            native_actions = self.actions(native)
            repeat_actions = self.actions(repeat)
            cached_actions = self.actions(cached)
            # Native evaluator pads short plans with STOP before taking four
            # local actions. Preserve raw lists too, including an empty plan.

            def prefix(actions):
                return (actions + [0] * 4)[:4]

            record = {'plan_index': len(self.plans), 'readout_index': len(self.readouts) - 1,
                      **self.context(), 'native_repeat': tensor_difference(native, repeat),
                      'cache_vs_native': tensor_difference(native, cached),
                      'native_actions': native_actions, 'native_repeat_actions': repeat_actions,
                      'cache_actions': cached_actions,
                      'native_repeat_actions_match': native_actions == repeat_actions,
                      'cache_full_actions_match': native_actions == cached_actions,
                      'cache_first4_actions_match': prefix(native_actions) == prefix(cached_actions)}
        finally:
            restore_rng(after_native)
        record['rng_restored'] = same_rng(after_native, capture_rng())
        if not record['rng_restored']:
            raise RuntimeError('Paired S1 RNG-restoration check failed')
        self.plans.append(record)
        self.emit('plans', record)
        return native


def load_pilot():
    spec = importlib.util.spec_from_file_location('_paired_pilot', ROOT / 'scripts/eval/cache_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registered_evaluator(registry, eval_type):
    # Evaluator.register stores the class but its decorator returns None, so
    # the decorated module's class symbol is not the evaluator constructor.
    evaluator_class = registry.evaluators.get(eval_type)
    if not isinstance(evaluator_class, type):
        raise RuntimeError('No evaluator class registered for: ' + eval_type)
    return evaluator_class


def run(args):
    sys.path.insert(0, str(ROOT))
    pilot = load_pilot()
    run_dir = Path(args.run_dir).resolve()
    manifest = pilot.verify_manifest(run_dir)
    if len(manifest['episodes']) != 1:
        raise ValueError('This diagnostic is restricted to one prepared episode')
    if (run_dir / 'control').exists():
        raise FileExistsError('Use a freshly prepared run-dir; control must not already exist')
    reference = Path(args.reference_control).resolve()
    reference_summary = pilot.read_json(reference / 'pilot_summary.json')
    if reference_summary['status'] != 'complete' or reference_summary['seed_base'] != manifest['seed_base']:
        raise ValueError('Reference control must be complete and have the same seed base')
    if reference_summary['stage_synchronized']:
        raise ValueError('Reference control must have stage profiling disabled')
    native_cfg = pilot.load_config(run_dir / 'control.py')
    if native_cfg.agent.model_settings.get('kv_cache_continuation'):
        raise ValueError('The acting policy must use the native full replay')

    from internnav.evaluator import Evaluator
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401
    from internnav.model.utils.vln_utils import traj_to_actions

    evaluator_class = registered_evaluator(Evaluator, native_cfg.eval_type)
    if args.check_entrypoint:
        print('DIAGNOSTIC_ENTRYPOINT_CHECK=PASS',
              'evaluator=', evaluator_class.__module__ + '.' + evaluator_class.__qualname__)
        return

    directory = run_dir / 'paired_diagnostic'
    directory.mkdir(exist_ok=False)
    streams = {kind: (directory / f'{kind}.jsonl').open('x', encoding='utf-8')
               for kind in ('readouts', 'plans')}
    observers = []
    env_steps = []
    original_init = evaluator_class.__init__
    summary = {'status': 'running', 'schema': 'dualvln_paired_readout.v1',
               'acting_path': 'native_full_replay', 'timing_valid_for_speed_comparison': False,
               'diagnostic_sha256': pilot.sha256(__file__),
               'decoder_sha256': pilot.sha256(ROOT / 'internnav/model/utils/vln_utils.py'),
               'manifest_sha256': pilot.sha256(run_dir / 'manifest.json'),
               'reference_trace_sha256': pilot.sha256(reference / 'actions.jsonl')}

    def emit(kind, record):
        streams[kind].write(json.dumps(record, allow_nan=False) + '\n')
        streams[kind].flush()

    def instrumented_init(evaluator, config):
        original_init(evaluator, config)
        if not evaluator.model.get_system1_type().startswith('nextdit'):
            raise ValueError('Paired S1 replay is currently limited to the native nextdit System1')
        counter = [0]

        def context():
            episode = evaluator.env.get_current_episode()
            state = evaluator.env._env.sim.get_agent_state()
            return {'next_actual_action_index': counter[0], 'episode_id': str(episode.episode_id),
                    'scene_id': episode.scene_id, 'position': [float(v) for v in state.position]}

        observer = PairedObserver(evaluator.model, traj_to_actions, context, emit)
        observers.append(observer)
        observer.install()
        original_step = evaluator.env.step

        def step(*positional, **keywords):
            result = original_step(*positional, **keywords)
            counter[0] += 1
            return result

        evaluator.env.step = step
        env_steps.append((evaluator.env, original_step))

    evaluator_class.__init__ = instrumented_init
    try:
        pilot.run_arm(SimpleNamespace(run_dir=str(run_dir), arm='control', profile_stages=False))
        actual_summary = pilot.read_json(run_dir / 'control/pilot_summary.json')
        for key in ('torch', 'cuda', 'transformers', 'device', 'seed_base'):
            if actual_summary[key] != reference_summary[key]:
                raise RuntimeError('Reference runtime differs: ' + key)
        traces = []
        for folder in (reference, run_dir / 'control'):
            traces.append([json.loads(line) for line in (folder / 'actions.jsonl').read_text().splitlines()
                           if line.strip()])
        comparison = pilot.compare_traces(*traces)
        summary['native_trace_vs_reference'] = comparison
        summary['metrics_match_reference'] = pilot.metrics_match(
            reference_summary['metrics'], actual_summary['metrics'])
        observer = observers[0]
        summary.update({'status': 'complete', 'readout_count': len(observer.readouts),
                        'plan_count': len(observer.plans),
                        'native_readout_repeats_bitwise_equal': all(
                            r['native_repeat']['bitwise_equal'] for r in observer.readouts),
                        'native_s1_repeat_actions_match': all(
                            p['native_repeat_actions_match'] for p in observer.plans),
                        'native_s1_repeats_bitwise_equal': all(
                            p['native_repeat']['bitwise_equal'] for p in observer.plans),
                        'cache_s1_full_action_mismatches': sum(
                            not p['cache_full_actions_match'] for p in observer.plans),
                        'cache_s1_first4_action_mismatches': sum(
                            not p['cache_first4_actions_match'] for p in observer.plans),
                        'max_latent_abs_error': max((r['cache_vs_native']['max_abs_error']
                                                    for r in observer.readouts
                                                    if r['cache_vs_native']['finite']), default=None),
                        'first_cache_plan_difference': next(
                            (p for p in observer.plans if not p['cache_first4_actions_match']), None)})
        summary['paired_action_effect_evidence_valid'] = bool(
            observer.readouts and observer.plans and comparison['actions_and_poses_match']
            and summary['native_readout_repeats_bitwise_equal']
            and summary['native_s1_repeats_bitwise_equal']
            and summary['native_s1_repeat_actions_match']
            and all(r['cache_vs_native']['finite'] for r in observer.readouts)
            and all(p['cache_vs_native']['finite'] for p in observer.plans))
    except Exception as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        evaluator_class.__init__ = original_init
        for observer in observers:
            observer.close()
        for env, step in env_steps:
            env.step = step
        for stream in streams.values():
            stream.close()
        pilot.write_json(directory / 'summary.json', summary)
    print('PAIRED_READOUT_DIAGNOSTIC_COMPLETE')
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--reference-control', required=True)
    parser.add_argument('--check-entrypoint', action='store_true',
                        help='Resolve the registered evaluator without creating outputs or initializing a model')
    run(parser.parse_args())
