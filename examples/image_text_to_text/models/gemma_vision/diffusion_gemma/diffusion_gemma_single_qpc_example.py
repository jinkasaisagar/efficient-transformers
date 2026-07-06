# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Explain how diffusion language models denoise a token canvas."
TEXT_PROMPT = "Why is the sky blue?"

BS = 1
PREFILL_SEQ_LEN = 32+256
CTX_LEN = 1024
GENERATION_LEN = 256
# NUM_LANG_HIDDEN_LAYER = 2
NUM_LANG_HIDDEN_LAYER = None

EXPORT_ROOT = Path("/home/jsaisaga/qeff_llama/d_g_npi_full/onnx")
COMPILE_ROOT = Path("/home/jsaisaga/qeff_llama/d_g_npi_full/qpc")

# NODE_PRECISION_INFO: Optional argument.
# - True: generate NPI automatically.
# - str path: use provided NPI file.
# - False/None: skip NPI.
NODE_PRECISION_INFO = True

compiler_kwargs = {
    "num_cores": 16,
    "num_devices": 4,
    "mxfp6_matmul": True,
    # "mxint8_kv_cache": True,
    "aic_enable_depth_first": True,
    # "mos": 1,
    "use_onnx_subfunctions": False,
    # "split_model_io": True,
    "batch_size": BS,
    "node_precision_info": NODE_PRECISION_INFO,
}


def build_compile_kwargs(*, effective_prefill_seq_len: int, effective_ctx_len: int, **kwargs):
    print('NUmber of devices ', kwargs["num_devices"])
    return {
        "prefill_seq_len": effective_prefill_seq_len,
        "ctx_len": effective_ctx_len,
        "num_cores": kwargs["num_cores"],
        "num_devices": kwargs["num_devices"],
        "mxfp6_matmul": kwargs.get("mxfp6_matmul", True),
        "mxint8_kv_cache": kwargs.get("mxint8_kv_cache", False),
        "aic_enable_depth_first": kwargs.get("aic_enable_depth_first", False),
        "mos": kwargs.get("mos", 1),
        "use_onnx_subfunctions": kwargs.get("use_onnx_subfunctions", False),
        "split_model_io": kwargs.get("split_model_io", True),
        "batch_size": kwargs.get("batch_size", 1),
        "node_precision_info": kwargs.get("node_precision_info", False),
    }


def normalize_generated_ids(generated_ids):
    array = np.asarray(generated_ids)
    if array.dtype == object:
        array = np.asarray([np.asarray(row).reshape(-1) for row in generated_ids], dtype=np.int64)
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.int64, copy=False)


def effective_lens(model, prefill_seq_len: int, ctx_len: int, prompt_len: int, generation_len: int, skip_vision: bool):
    del model, skip_vision
    effective_ctx_len = max(ctx_len, prompt_len + generation_len)
    effective_prefill_seq_len = max(prefill_seq_len, prompt_len)
    return effective_prefill_seq_len, effective_ctx_len


def _apply_reduced_layer_config(config, num_lang_layers: int):
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        config.text_config.num_hidden_layers = num_lang_layers
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_lang_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]
    return config


def main():
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPILE_ROOT.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # For testing only (reduced layers).
    if NUM_LANG_HIDDEN_LAYER is not None:
        config = _apply_reduced_layer_config(config, num_lang_layers=NUM_LANG_HIDDEN_LAYER)

    # Single QPC path for DiffusionGemma.
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=False,
        ignore_mismatched_sizes=True,
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": TEXT_PROMPT}],
        },
    ]
    messages = [
    {"role": "user", "content": TEXT_PROMPT}
    ]

    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    print(f'Prompt length is {prompt_len}')
    effective_prefill_seq_len, effective_ctx_len = effective_lens(
        qeff_model,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        prompt_len,
        GENERATION_LEN,
        skip_vision=True,
    )
    compile_kwargs = build_compile_kwargs(
        effective_prefill_seq_len=effective_prefill_seq_len,
        effective_ctx_len=effective_ctx_len,
        **compiler_kwargs,
    )
    print(f'Effective Prompt length which includes prompt length + generation length is {effective_prefill_seq_len}')

    onnx_path = qeff_model.export(
        export_dir=str(EXPORT_ROOT),
        prefill_seq_len=effective_prefill_seq_len,
        use_onnx_subfunctions=compile_kwargs["use_onnx_subfunctions"],
    )
    print('Model Exported')

    qpc_path = qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=str(COMPILE_ROOT),
        **compile_kwargs,
    )

    print('Model Compiled and running is started')
    output = qeff_model.generate(
        inputs=inputs,
        generation_len=GENERATION_LEN,
        qpc_path=qpc_path,
    )

    qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
    print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
    print(output)
    print(f"ONNX_PATH={onnx_path}")
    print(f"QPC_PATH={qpc_path}")


if __name__ == "__main__":
    main()
