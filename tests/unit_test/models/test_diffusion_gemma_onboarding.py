# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    QEffDiffusionGemmaDecoderWrapper,
    QEffDiffusionGemmaEncoderWrapper,
    QEffDiffusionGemmaForBlockDiffusion,
)
from QEfficient.transformers.models.modeling_auto import (
    _QEFFAutoModelForImageTextToTextSingleQPC,
    _is_diffusion_gemma_arch,
    diffusion_gemma_generate,
)
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform

try:
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaDecoderTextLayer,
        DiffusionGemmaEncoderTextLayer,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaRMSNorm,
    )

    from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        QEffDiffusionGemmaCustomRMSNormAIC,
        QEffDiffusionGemmaDecoderTextLayer,
        QEffDiffusionGemmaEncoderTextLayer,
    )

    _DIFFUSION_GEMMA_AVAILABLE = True
except Exception:
    _DIFFUSION_GEMMA_AVAILABLE = False


def _build_tiny_diffusion_config():
    text_config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["full_attention", "sliding_attention"],
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        vocab_size=64,
        num_global_key_value_heads=None,
        global_head_dim=None,
    )
    return SimpleNamespace(text_config=text_config, canvas_length=6)


def _build_fake_diffusion_model(config):
    class _FakeDiffusionModel:
        def __init__(self, cfg):
            self.config = cfg
            self.model = SimpleNamespace(encoder=MagicMock(), decoder=MagicMock())
            self.lm_head = torch.nn.Identity()
            self.final_logit_softcapping = None

        def get_dummy_pkv_cache(self, config, batch_size, seq_len):
            return QEffDiffusionGemmaForBlockDiffusion.get_dummy_pkv_cache(self, config, batch_size, seq_len)

    return _FakeDiffusionModel(config)


def test_is_diffusion_gemma_arch_detection():
    yes_cfg = SimpleNamespace(architectures=["DiffusionGemmaForBlockDiffusion"])
    no_cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
    assert _is_diffusion_gemma_arch(yes_cfg) is True
    assert _is_diffusion_gemma_arch(no_cfg) is False


def test_diffusion_gemma_generate_dispatch_paths_are_forwarded(monkeypatch):
    class _Dispatch:
        runtime_result = "runtime_result"
        hf_output = "hf_output"

    captured = {}

    def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return _Dispatch()

    monkeypatch.setattr(
        "QEfficient.transformers.models.modeling_auto.diffusion_gemma_generate_dispatch",
        _fake_dispatch,
    )

    out = diffusion_gemma_generate(
        model=SimpleNamespace(),
        inputs={"input_ids": torch.zeros((1, 1), dtype=torch.long)},
        runtime_ai100=True,
        qpc_path="/tmp/single.qpc",
        encoder_qpc_path="/tmp/enc.qpc",
        decoder_qpc_path="/tmp/dec.qpc",
    )
    assert out == "runtime_result"
    assert captured["qpc_path"].name == "single.qpc"
    assert captured["encoder_qpc_path"].name == "enc.qpc"
    assert captured["decoder_qpc_path"].name == "dec.qpc"

    out = diffusion_gemma_generate(model=SimpleNamespace(), runtime_ai100=False)
    assert out == "hf_output"


def test_single_qpc_generate_passes_correct_diffusion_qpc_kwargs(monkeypatch):
    fake_self = object.__new__(_QEFFAutoModelForImageTextToTextSingleQPC)
    fake_self.model = SimpleNamespace(config=SimpleNamespace(architectures=["DiffusionGemmaForBlockDiffusion"]))

    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("QEfficient.transformers.models.modeling_auto.diffusion_gemma_generate", _fake_generate)

    out = fake_self.generate(
        inputs={"x": 1},
        device_ids=[0],
        runtime_ai100=True,
        qpc_path="/tmp/single.qpc",
        encoder_qpc_path="/tmp/enc.qpc",
        decoder_qpc_path="/tmp/dec.qpc",
    )
    assert out == "ok"
    assert captured["qpc_path"] == "/tmp/single.qpc"
    assert captured["encoder_qpc_path"] == "/tmp/enc.qpc"
    assert captured["decoder_qpc_path"] == "/tmp/dec.qpc"


def test_diffusion_gemma_export_interfaces_and_dynamic_axes():
    model = object.__new__(QEffDiffusionGemmaForBlockDiffusion)
    model.config = _build_tiny_diffusion_config()

    output_names = model.get_output_names()
    assert output_names[0] == "logits"
    assert len(output_names) == 1 + (2 * model.config.text_config.num_hidden_layers)

    dummy_inputs = model.get_dummy_inputs()
    assert "input_ids" in dummy_inputs
    assert "decoder_input_ids" in dummy_inputs
    assert "past_key_values" in dummy_inputs

    dynamic_axes = model.get_onnx_dynamic_axes(comp_ctx_lengths=[16], continuous_batching=True)
    assert dynamic_axes["input_ids"] == {0: "batch_size", 1: "seq_len"}
    assert dynamic_axes["decoder_input_ids"] == {0: "batch_size", 1: "canvas_len"}
    assert dynamic_axes["batch_index"] == {0: "batch_size"}
    assert "comp_ctx_lengths" in dynamic_axes
    assert "past_key.0" in dynamic_axes
    assert "past_value.1" in dynamic_axes


def test_diffusion_gemma_specializations_match_single_qpc_contract():
    model = object.__new__(QEffDiffusionGemmaForBlockDiffusion)
    model.config = _build_tiny_diffusion_config()

    specializations, compiler_options = model.get_specializations(
        batch_size=2,
        prefill_seq_len=16,
        ctx_len=128,
        comp_ctx_lengths_prefill=[32],
        comp_ctx_lengths_decode=[64],
        kv_offload=False,
        continuous_batching=False,
        kv_cache_batch_size=2,
        custom_flag=True,
    )
    assert isinstance(specializations, list)
    assert len(specializations) == 2
    assert specializations[0]["seq_len"] == 16
    assert specializations[0]["canvas_len"] == 6
    assert specializations[1]["canvas_len"] == 6
    assert specializations[1]["seq_len"] == "1"
    assert compiler_options["custom_flag"] is True


def test_diffusion_gemma_split_wrappers_have_required_export_helpers():
    cfg = _build_tiny_diffusion_config()
    fake_model = _build_fake_diffusion_model(cfg)

    encoder = QEffDiffusionGemmaEncoderWrapper(fake_model)
    decoder = QEffDiffusionGemmaDecoderWrapper(fake_model)

    enc_dummy = encoder.get_dummy_inputs()
    dec_dummy = decoder.get_dummy_inputs()
    enc_axes = encoder.get_onnx_dynamic_axes()
    dec_axes = decoder.get_onnx_dynamic_axes()

    assert "input_ids" in enc_dummy
    assert "decoder_input_ids" in dec_dummy
    assert "past_key_values" in dec_dummy
    assert encoder.get_output_names() == [
        "past_key.0_RetainedState",
        "past_value.0_RetainedState",
        "past_key.1_RetainedState",
        "past_value.1_RetainedState",
    ]
    assert decoder.get_output_names() == ["logits"]
    assert "input_ids" in enc_axes
    assert "decoder_input_ids" in dec_axes
    assert "past_key.0" in dec_axes


@pytest.mark.skipif(not _DIFFUSION_GEMMA_AVAILABLE, reason="Transformers diffusion_gemma classes are unavailable")
def test_diffusion_gemma_transform_mappings_are_registered():
    assert CustomOpsTransform._module_mapping[DiffusionGemmaRMSNorm] is QEffDiffusionGemmaCustomRMSNormAIC
    assert KVCacheTransform._module_mapping[DiffusionGemmaForBlockDiffusion] is QEffDiffusionGemmaForBlockDiffusion
    assert KVCacheTransform._module_mapping[DiffusionGemmaEncoderTextLayer] is QEffDiffusionGemmaEncoderTextLayer
    assert KVCacheTransform._module_mapping[DiffusionGemmaDecoderTextLayer] is QEffDiffusionGemmaDecoderTextLayer


def test_image_text_from_pretrained_forces_kv_offload_false_for_diffusion():
    class _DummyHF:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(config=kwargs["config"])

    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

    # Build a tiny subclass wrapper around the real classmethod to avoid heavyweight init paths.
    class _DummyQEFFAutoModelForImageTextToText(QEFFAutoModelForImageTextToText):
        _hf_auto_class = _DummyHF

        def __new__(cls, model: torch.nn.Module, kv_offload=True, continuous_batching=False, qaic_config=None, **kwargs):
            instance = object.__new__(cls)
            instance.model = model
            instance.received = {
                "kv_offload": kv_offload,
                "continuous_batching": continuous_batching,
                "qaic_config": qaic_config,
                "kwargs": kwargs,
            }
            return instance

        def __init__(self, *args, **kwargs):
            pass

    diffusion_cfg = SimpleNamespace(architectures=["DiffusionGemmaForBlockDiffusion"])
    non_diffusion_cfg = SimpleNamespace(architectures=["LlavaForConditionalGeneration"])

    diff_instance = _DummyQEFFAutoModelForImageTextToText.from_pretrained(
        "dummy/model",
        kv_offload=True,
        config=diffusion_cfg,
    )
    assert diff_instance.received["kv_offload"] is False

    non_diff_instance = _DummyQEFFAutoModelForImageTextToText.from_pretrained(
        "dummy/model",
        kv_offload=True,
        config=non_diffusion_cfg,
    )
    assert non_diff_instance.received["kv_offload"] is True
