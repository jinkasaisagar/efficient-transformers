# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from QEfficient.transformers.cloud_ai_100_diffusion_single_qpc_utils import (
    DiffusionGemmaSingleQPCGenerator,
)
from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC


def test_single_qpc_generate_creates_session_and_dispatches(monkeypatch):
    fake_self = object.__new__(_QEFFAutoModelForImageTextToTextSingleQPC)
    fake_self.model = SimpleNamespace(
        config=SimpleNamespace(architectures=["DiffusionGemmaForBlockDiffusion"]),
        generation_config=SimpleNamespace(pad_token_id=0, eos_token_id=1),
    )
    fake_self.qpc_path = None

    captured = {}
    session = object()

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return "ok"

    session_factory = MagicMock(return_value=session)
    monkeypatch.setattr("QEfficient.transformers.models.modeling_auto.QAICInferenceSession", session_factory)
    monkeypatch.setattr(
        "QEfficient.transformers.models.modeling_auto.diffusion_gemma_generate_single_qpc",
        _fake_generate,
    )

    output = fake_self.generate(
        inputs={"input_ids": torch.ones((1, 2), dtype=torch.long)},
        device_ids=[0],
        runtime_ai100=True,
        generation_len=32,
        qpc_path="/tmp/single.qpc",
    )

    assert output == "ok"
    session_factory.assert_called_once_with("/tmp/single.qpc", [0])
    assert captured["session"] is session
    assert captured["generation_len"] == 32
    assert captured["pad_token_id"] == 0
    assert captured["eos_token_id"] == 1


def test_single_qpc_runtime_prefill_decode_and_commit():
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
        output_names = ["canvas_logits", "past_key.0_RetainedState"]
        bindings = [
            _Binding("input_ids", (1, 4)),
            _Binding("decoder_input_ids", (1, 2)),
            _Binding("vision_embeds", (1, 3, 8)),
            _Binding("past_key.0", (1, 1, 8, 2)),
        ]

        def __init__(self):
            self.calls = []
            self.skipped = None
            self.deactivated = False

        def run(self, feed):
            self.calls.append(feed)
            if int(feed["is_encode"][0]) == 0:
                return {"canvas_logits": np.zeros((1, 2, 4), dtype=np.float32)}
            return {"past_key.0_RetainedState": np.zeros((1,), dtype=np.float16)}

        def skip_buffers(self, buffers):
            self.skipped = buffers

        def deactivate(self):
            self.deactivated = True

    session = _Session()
    config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=4), canvas_length=2)
    generator = DiffusionGemmaSingleQPCGenerator(model_config=config, session=session, seed=7)
    result = generator.generate(
        inputs={
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        generation_len=4,
        max_denoising_steps=1,
        ctx_len=8,
    )

    assert result.generated_ids.shape == (1, 4)
    assert result.retained_kv_buffers == 1
    assert result.total_steps == 2
    assert result.executed_blocks == 2
    assert result.canvas_length == 2
    assert result.ttft >= 0
    assert result.total_canvas_time >= 0
    assert [int(call["is_encode"][0]) for call in session.calls] == [1, 0, 1, 0]
    assert "past_key.0" in session.skipped
    assert "past_key.0_RetainedState" in session.skipped
    assert session.deactivated is True
