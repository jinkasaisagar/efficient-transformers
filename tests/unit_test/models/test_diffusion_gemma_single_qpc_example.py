# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import numpy as np
import torch

from examples.image_text_to_text.models.gemma_vision.diffusion_gemma.diffusion_gemma_single_qpc_example_utils import (
    DiffusionGemmaSingleQPCGenerator,
)


class _Binding:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class _Session:
    input_names = [
        "input_ids",
        "position_ids",
        "cache_position_ids",
        "full_attention_mask",
        "sliding_attention_mask",
        "vision_embeds",
        "image_idx",
        "mm_token_type_ids",
        "decoder_input_ids",
        "decoder_position_ids",
        "self_conditioning_logits",
        "is_encode",
        "use_self_conditioning",
        "past_key.0",
    ]
    output_names = ["canvas_logits", "image_idx_output", "past_key.0_RetainedState"]
    bindings = [
        _Binding("input_ids", (1, 4)),
        _Binding("decoder_input_ids", (1, 4)),
        _Binding("vision_embeds", (1, 6, 8)),
        _Binding("past_key.0", (1, 1, 16, 2)),
    ]

    def __init__(self):
        self.calls = []
        self.skipped = None
        self.retained_enabled = False

    def run(self, feed):
        self.calls.append((feed, self.retained_enabled))
        return {
            "image_idx_output": feed["image_idx"] + 1,
            "past_key.0_RetainedState": np.zeros((1,), dtype=np.float16),
        }

    def skip_buffers(self, buffers):
        self.skipped = buffers
        self.retained_enabled = True


def _generator(session):
    config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=32), canvas_length=4)
    return DiffusionGemmaSingleQPCGenerator(model_config=config, session=session, seed=7)


def test_example_prefill_chunks_prompt_by_canvas_length():
    session = _Session()
    generator = _generator(session)
    sequence_length = generator._prepare_prompt(
        {
            "input_ids": torch.arange(1, 11, dtype=torch.long).reshape(1, -1),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
            "mm_token_type_ids": torch.arange(10, dtype=torch.long).reshape(1, -1),
            "vision_embeds": torch.ones((1, 6, 8), dtype=torch.float32),
        },
        pad_token_id=0,
    )

    _, retained_kv_buffers = generator.prefill()

    assert sequence_length == 10
    assert retained_kv_buffers == 1
    assert len(session.calls) == 3
    np.testing.assert_array_equal(session.calls[0][0]["input_ids"], [[1, 2, 3, 4]])
    np.testing.assert_array_equal(session.calls[1][0]["input_ids"], [[5, 6, 7, 8]])
    np.testing.assert_array_equal(session.calls[2][0]["input_ids"], [[9, 10, 0, 0]])
    np.testing.assert_array_equal(session.calls[0][0]["position_ids"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(session.calls[1][0]["position_ids"], [[4, 5, 6, 7]])
    np.testing.assert_array_equal(session.calls[2][0]["position_ids"], [[8, 9, -1, -1]])
    np.testing.assert_array_equal(session.calls[2][0]["mm_token_type_ids"], [[8, 9, 0, 0]])
    assert [int(call[0]["image_idx"][0, 0]) for call in session.calls] == [0, 1, 2]
    assert [retained for _, retained in session.calls] == [False, True, True]
    assert generator._next_encoder_position() == 10
    assert "past_key.0" in session.skipped
    assert "past_key.0_RetainedState" in session.skipped


def test_example_prefill_does_not_add_chunk_for_exact_multiple():
    session = _Session()
    generator = _generator(session)
    generator._prepare_prompt(
        {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)},
        pad_token_id=0,
    )

    generator.prefill()

    assert len(session.calls) == 2
    np.testing.assert_array_equal(session.calls[-1][0]["position_ids"], [[4, 5, 6, 7]])


def test_host_slot_tracking_uses_block_level_sliding_rollover():
    generator = _generator(_Session())
    slots = np.full(8, -1, dtype=np.int64)

    projected_slots = generator._project_slot_positions(
        slots,
        np.array([[12, 13, 14, 15]], dtype=np.int64),
        sliding=True,
    )

    np.testing.assert_array_equal(projected_slots, [15, -1, -1, -1, -1, 12, 13, 14])


def test_canvas_commit_reuses_active_canvas_positions_and_updates_host_masks():
    session = _Session()
    generator = _generator(session)
    generator._prepare_prompt(
        {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
        pad_token_id=0,
    )
    generator.prefill()

    generator._active_canvas_position_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)
    generator._commit_canvas(np.array([[11, 12, 13, 14]], dtype=np.int64))

    commit_feed = session.calls[-1][0]
    np.testing.assert_array_equal(commit_feed["position_ids"], [[2, 3, 4, 5]])
    assert generator._retained_last_position == 5
    np.testing.assert_array_equal(generator._full_slot_positions[:6], [0, 1, 2, 3, 4, 5])

    generator.input_ids = np.zeros((1, 4), dtype=np.int64)
    generator.position_ids = np.array([[6, 7, 8, 9]], dtype=np.int64)
    decode_mask = generator._build_additive_mask(
        cache_length=generator.full_cache_length,
        sliding=False,
        cache_position_ids=np.full((1, 4), -1, dtype=np.int64),
        is_encode=False,
    )

    assert np.all(decode_mask[0, 0, :, :6] == 0.0)
    assert np.all(decode_mask[0, 0, :, 6 : generator.full_cache_length] == generator._MASK_VALUE)
    assert np.all(decode_mask[0, 0, :, generator.full_cache_length :] == 0.0)
