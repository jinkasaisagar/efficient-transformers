from types import SimpleNamespace

import torch

from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import _build_encoder_kv_valid_mask


def _make_cache(keys, values, layer_type):
    config = SimpleNamespace(layer_types=[layer_type])
    return QEffGemma4DynamicCache.from_legacy_cache(config, ((keys, values),))


def test_full_cache_read_only_uses_only_valid_encoder_positions():
    position_ids = torch.full((1, 256), -1, dtype=torch.int64)
    position_ids[:, :20] = torch.arange(20)
    keys = torch.arange(1024, dtype=torch.float32).view(1, 1, 1024, 1)
    values = keys.clone()
    cache = _make_cache(keys, values, "full_attention")

    gathered_keys, _ = cache.read_only(0, {"position_ids": position_ids})
    valid_mask = _build_encoder_kv_valid_mask(position_ids, torch.empty((1, 1024)))

    assert torch.equal(gathered_keys[0, 0, :20, 0], torch.arange(20, dtype=torch.float32))
    assert valid_mask.sum().item() == 20
    assert not valid_mask[:, 20:].any()


def test_full_cache_read_only_uses_latest_commit_position_to_include_history():
    position_ids = torch.full((1, 16), -1, dtype=torch.int64)
    position_ids[:, :8] = torch.arange(20, 28)
    keys = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1)
    values = keys.clone()
    cache = _make_cache(keys, values, "full_attention")

    gathered_keys, _ = cache.read_only(0, {"position_ids": position_ids})
    valid_mask = _build_encoder_kv_valid_mask(position_ids, torch.empty((1, 32)))

    assert torch.equal(gathered_keys[0, 0, :28, 0], torch.arange(28, dtype=torch.float32))
    assert valid_mask.sum().item() == 28
    assert not valid_mask[:, 28:].any()


def test_sliding_cache_read_only_uses_wrapped_physical_slots():
    position_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.int64)
    keys = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
    values = keys.clone()
    cache = _make_cache(keys, values, "sliding_attention")

    gathered_keys, _ = cache.read_only(0, {"position_ids": position_ids})
    valid_mask = _build_encoder_kv_valid_mask(position_ids, torch.empty((1, 4)))

    assert torch.equal(gathered_keys[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0, 0.0]))
    assert valid_mask.all()
