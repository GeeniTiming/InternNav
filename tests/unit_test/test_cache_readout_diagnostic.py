"""Diagnostic state isolation with stochastic S1 doubles; no Habitat or weights."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[2] / 'scripts/eval/cache_readout_diagnostic.py'
    spec = importlib.util.spec_from_file_location('_diagnostic_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCache:
    def __init__(self):
        self.key_cache = [torch.zeros(1, 1, 2, 1)]
        self.value_cache = [torch.zeros(1, 1, 2, 1)]

    def get_seq_length(self):
        return self.key_cache[0].shape[-2]


class StochasticModel:
    def __init__(self):
        self.rope_deltas = torch.tensor([3])
        self.fail_readout = False
        self.fail_shadow_plan = False
        self.corrupt_cache = False

    def generate(self, input_ids, **kwargs):
        return SimpleNamespace(sequences=input_ids.clone(), past_key_values=FakeCache())

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        torch.rand(1)
        return torch.zeros(1, 2, 3)

    def generate_latents_from_cache(self, outputs, image_grid_thw, mask):
        self.rope_deltas = torch.tensor([99])
        random.random()
        np.random.rand()
        torch.rand(3)
        if self.fail_readout:
            raise RuntimeError('shadow readout error')
        if self.corrupt_cache:
            outputs.past_key_values.key_cache[0].add_(1)
        return torch.full((1, 2, 3), 10.0)

    def generate_traj(self, latents, image, depth):
        random.random()
        np.random.rand()
        result = latents + torch.rand_like(latents)
        if self.fail_shadow_plan and latents.mean() > 5:
            raise RuntimeError('shadow trajectory error')
        return result


def seed():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)


def mutating_decoder(trajectory):
    trajectory.div_(2)
    return [1] if trajectory.mean() < 1 else [3]


def begin(model):
    ids = torch.tensor([[1, 2]])
    model.generate(input_ids=ids, attention_mask=torch.ones_like(ids))
    return model.generate_latents(ids, torch.ones(2), torch.tensor([[1, 1, 1]]))


def observe(diagnostic, model):
    emitted = []
    observer = diagnostic.PairedObserver(model, mutating_decoder,
                                         lambda: {'next_actual_action_index': 71},
                                         lambda kind, record: emitted.append((kind, record)))
    observer.install()
    return observer, emitted


def test_paired_shadows_preserve_native_returns_rng_rope_and_decoder_inputs(diagnostic):
    seed()
    baseline = StochasticModel()
    native_latents = begin(baseline)
    native_plans = [baseline.generate_traj(native_latents, None, None) for _ in range(2)]
    expected_rng = diagnostic.capture_rng()

    seed()
    model = StochasticModel()
    original_generate = model.generate
    observer, emitted = observe(diagnostic, model)
    latents = begin(model)
    plans = [model.generate_traj(latents, None, None) for _ in range(2)]
    assert torch.equal(latents, native_latents)
    assert all(torch.equal(a, b) for a, b in zip(plans, native_plans))
    assert diagnostic.same_rng(expected_rng, diagnostic.capture_rng())
    assert model.rope_deltas.item() == 3
    assert observer.readouts[0]['native_repeat']['bitwise_equal']
    assert observer.readouts[0]['cache_vs_native']['max_abs_error'] == 10
    assert all(p['native_repeat']['bitwise_equal'] for p in observer.plans)
    assert all(p['native_actions'] == [1] and p['cache_actions'] == [3] for p in observer.plans)
    assert all(not p['cache_first4_actions_match'] and p['rng_restored'] for p in observer.plans)
    assert [kind for kind, _ in emitted] == ['readouts', 'plans', 'plans']
    observer.close()
    assert model.generate == original_generate
    assert observer.cache_latents is None and observer.pending is None


def test_shadow_readout_exception_restores_rng_and_rope(diagnostic):
    seed()
    baseline = StochasticModel()
    begin(baseline)
    expected_rng = diagnostic.capture_rng()
    seed()
    model = StochasticModel()
    model.fail_readout = True
    observer, _ = observe(diagnostic, model)
    with pytest.raises(RuntimeError, match='shadow readout error'):
        begin(model)
    assert diagnostic.same_rng(expected_rng, diagnostic.capture_rng())
    assert model.rope_deltas.item() == 3
    observer.close()


def test_shadow_s1_exception_restores_post_native_rng(diagnostic):
    seed()
    baseline = StochasticModel()
    latents = begin(baseline)
    baseline.generate_traj(latents, None, None)
    expected_rng = diagnostic.capture_rng()
    seed()
    model = StochasticModel()
    model.fail_shadow_plan = True
    observer, _ = observe(diagnostic, model)
    latents = begin(model)
    with pytest.raises(RuntimeError, match='shadow trajectory error'):
        model.generate_traj(latents, None, None)
    assert diagnostic.same_rng(expected_rng, diagnostic.capture_rng())
    observer.close()


def test_cache_in_place_corruption_is_detected(diagnostic):
    model = StochasticModel()
    model.corrupt_cache = True
    observer, _ = observe(diagnostic, model)
    with pytest.raises(RuntimeError, match='state-restoration check failed'):
        begin(model)
    observer.close()


def test_real_dynamic_cache_append_crop_can_replace_storage(diagnostic):
    from transformers import DynamicCache

    cache = DynamicCache()
    cache.update(torch.zeros(1, 1, 2, 3), torch.ones(1, 1, 2, 3), 0)
    saved = diagnostic.capture_cache_prefix(cache)
    cache.update(torch.full((1, 1, 1, 3), 4.0), torch.full((1, 1, 1, 3), 5.0), 0)
    cache.crop(2)
    assert saved[1][0][0].data_ptr() != cache.key_cache[0].data_ptr()
    assert diagnostic.cache_prefix_preserved(saved, cache)
    cache.key_cache[0].add_(1)
    assert not diagnostic.cache_prefix_preserved(saved, cache)


def test_unpaired_generation_or_latents_are_rejected(diagnostic):
    model = StochasticModel()
    observer, _ = observe(diagnostic, model)
    with pytest.raises(RuntimeError, match='captured generation'):
        model.generate_latents(torch.tensor([[1, 2]]), None, None)
    latents = begin(model)
    with pytest.raises(RuntimeError, match='linked to the paired readout'):
        model.generate_traj(latents.clone(), None, None)
    with pytest.raises(RuntimeError, match='already installed'):
        observer.install()
    observer.close()


def test_nonfinite_or_differently_shaped_outputs_are_not_hidden(diagnostic):
    result = diagnostic.tensor_difference(torch.zeros(1, 2), torch.full((1, 2), float('nan')))
    assert not result['finite'] and result['max_abs_error'] is None
    with pytest.raises(ValueError, match='shapes differ'):
        diagnostic.tensor_difference(torch.zeros(1), torch.zeros(2))
