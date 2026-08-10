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
    ExampleChunkedDiffusionGemmaSingleQPCGenerator,
)


class _Binding:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class _Session:
    input_names = [
        "input_ids",
        "position_ids",
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
    return ExampleChunkedDiffusionGemmaSingleQPCGenerator(model_config=config, session=session, seed=7)


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
