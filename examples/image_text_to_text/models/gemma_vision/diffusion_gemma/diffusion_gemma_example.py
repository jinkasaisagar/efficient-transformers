# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
NUM_LANG_HIDDEN_LAYER = 2
PREFILL_SEQ_LEN = 32
CTX_LEN = 1024
BATCH_SIZE = 1
GENERATION_LEN = 128
PROMPT = "Please explain how diffusion language models iteratively denoise a random canvas, denoise a random canvas,  "
EXPORT_ROOT = Path(
    "/home/jsaisaga/qeff_llama/test_diffusion_gemma_2layers/onnx"
)
COMPILE_ROOT = Path(
    "/home/jsaisaga/qeff_llama/test_diffusion_gemma_2layers/qpc"
)


def _apply_reduced_layer_config(config, num_lang_layers: int):
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        config.text_config.num_hidden_layers = num_lang_layers
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_lang_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]
    return config


def main():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config = _apply_reduced_layer_config(config, num_lang_layers=NUM_LANG_HIDDEN_LAYER)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        ignore_mismatched_sizes=True,
    )

    base_model = qeff_model.model

    # 1) Export encoder ONNX
    enc_wrapper = base_model.get_qeff_diffusion_encoder()
    qeff_model.model = enc_wrapper
    encoder_onnx = qeff_model._export(
        enc_wrapper.get_dummy_inputs(),
        output_names=enc_wrapper.get_output_names(),
        dynamic_axes=enc_wrapper.get_onnx_dynamic_axes(),
        export_dir=str(EXPORT_ROOT),
        use_onnx_subfunctions=False,
        offload_pt_weights=False,
    )

    # 2) Export decoder ONNX
    dec_wrapper = base_model.get_qeff_diffusion_decoder()
    qeff_model.model = dec_wrapper
    decoder_onnx = qeff_model._export(
        dec_wrapper.get_dummy_inputs(),
        output_names=dec_wrapper.get_output_names(),
        dynamic_axes=dec_wrapper.get_onnx_dynamic_axes(),
        export_dir=str(EXPORT_ROOT),
        use_onnx_subfunctions=False,
        offload_pt_weights=True,
    )

    qeff_model.model = base_model

    canvas_len = int(getattr(config, "canvas_length", 256))

    # 3) Compile encoder QPC
    encoder_specializations = [{"batch_size": BATCH_SIZE, "seq_len": PREFILL_SEQ_LEN}]
    encoder_qpc = qeff_model._compile(
        onnx_path=str(encoder_onnx),
        compile_dir=str(COMPILE_ROOT / "encoder"),
        specializations=encoder_specializations,
        retained_state=True,
        convert_to_fp16=True,
        mxfp6_matmul=True,
        mdp_ts_num_devices=1,
        aic_num_cores=16,
    )

    # 4) Compile decoder QPC
    decoder_specializations = [{"batch_size": BATCH_SIZE, "seq_len": canvas_len, "ctx_len": CTX_LEN}]
    decoder_qpc = qeff_model._compile(
        onnx_path=str(decoder_onnx),
        compile_dir=str(COMPILE_ROOT / "decoder"),
        specializations=decoder_specializations,
        retained_state=True,
        convert_to_fp16=True,
        mxfp6_matmul=True,
        mdp_ts_num_devices=1,
        aic_num_cores=16,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    output = qeff_model.generate(
        inputs=inputs,
        generation_len=GENERATION_LEN,
        encoder_qpc_path=encoder_qpc,
        decoder_qpc_path=decoder_qpc,
    )

    print(f"GENERATED_IDS_SHAPE={output.generated_ids.shape}")
    print(tokenizer.batch_decode(output.generated_ids, skip_special_tokens=True))

    print(f"ENCODER_ONNX={encoder_onnx}")
    print(f"DECODER_ONNX={decoder_onnx}")
    print(f"ENCODER_QPC={encoder_qpc}")
    print(f"DECODER_QPC={decoder_qpc}")


if __name__ == "__main__":
    main()
