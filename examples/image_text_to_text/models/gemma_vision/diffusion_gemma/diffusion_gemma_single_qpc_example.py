# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path
import onnxruntime as ort

import numpy as np
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion
from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
from QEfficient import QEFFAutoModelForImageTextToText
import torch
from sensivity_analysis import analyze_onnx_fp16_overflow, expose_all_intermediate_outputs_robust

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Explain how diffusion language models denoise a token canvas."
TEXT_PROMPT = "Why is the sky blue?"
TEXT_PROMPT = "How are you?"

BS = 1
PREFILL_SEQ_LEN = 256
CTX_LEN = 256
GENERATION_LEN = 256
# NUM_LANG_HIDDEN_LAYER = 2
NUM_LANG_HIDDEN_LAYER = None

EXPORT_ROOT = Path("/home/jsaisaga/qeff_llama/d_g_full_model/onnx")
COMPILE_ROOT = Path("/home/jsaisaga/qeff_llama/d_g_full_model/qpc")
torch.manual_seed(42)
# NODE_PRECISION_INFO: Optional argument.
# - True: generate NPI automatically.
# - str path: use provided NPI file.
# - False/None: skip NPI.
NODE_PRECISION_INFO = False

compiler_kwargs = {
    "num_cores": 16,
    "num_devices": 4,
    "mxfp6_matmul": True,
    # "mxint8_kv_cache": True,
    "aic_enable_depth_first": True,
    # "mos": 1,
    "use_onnx_subfunctions": False,
    "convert_to_fp16":True,
    # "split_model_io": True,
    "batch_size": BS,
    # "node_precision_info": NODE_PRECISION_INFO,
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
    effective_ctx_len = max(prefill_seq_len, prompt_len) #max(ctx_len, prompt_len + generation_len)
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
        padding="max_length",
        max_length=256,
        truncation=True
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
    # breakpoint()
    compile_kwargs = build_compile_kwargs(
        effective_prefill_seq_len=effective_prefill_seq_len,
        effective_ctx_len=effective_ctx_len,
        node_precision_info = '/home/jsaisaga/LongLLaDA/efficient-transformers/examples/image_text_to_text/models/gemma_vision/diffusion_gemma/fp32_nodes_customrmsnorm.yaml',
        **compiler_kwargs,
    )
    
    # breakpoint()
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
    input_ids = inputs["input_ids"]  # torch.LongTensor [B, S]
    B, S = input_ids.shape
    text_cfg = qeff_model.model.config.text_config

    # Match generate() path: first encoder call with fresh/empty cache.
    pkv = qeff_model.model.get_dummy_pkv_cache(config.text_config, 1, 1024)

    # breakpoint()
    # enc_outputs = qeff_model.model(input_ids=inputs["input_ids"], decoder_input_ids=inputs["input_ids"],is_encode=np.ones((1,), dtype=np.int64), past_key_values=pkv)
    from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
    empty_pkv = QEffGemma4DynamicCache(config=qeff_model.model.config.text_config)

    # breakpoint()
    position_ids_ten = torch.tensor(np.arange(0,256).reshape(1,-1))
    # enc_outputs = qeff_model.model.model.encoder.language_model(input_ids=inputs["input_ids"])
    # breakpoint()
    # enc_outputs = qeff_model.model.model.encoder.language_model(input_ids=inputs["input_ids"],past_key_values=pkv,use_cache=True,
    # position_ids=position_ids_ten)
    # enc_outputs = qeff_model.model.model.encoder.language_model(input_ids=inputs["input_ids"],past_key_values=pkv,use_cache=True,)
    # enc_outputs = qeff_model.model.model.encoder.language_model(input_ids=inputs["input_ids"],past_key_values=empty_pkv, use_cache=True,)
    # breakpoint()
    # enc_outputs = qeff_model.model.model.encoder.language_model(input_ids=inputs["input_ids"],past_key_values=pkv,use_cache=True,)
    # breakpoint()

    # enc_outputs = qeff_model.model.model.encoder.language_model(input_ids=inputs["input_ids"],past_key_values=pkv,use_cache=True, position_ids=position_ids_ten)

    self_conditioning_logits = torch.zeros((1, 256, text_cfg.vocab_size), dtype=torch.float32)
    is_encode = torch.ones((1,))
    self_condition_selector = torch.ones((2,))
    # import ipdb; ipdb.set_trace()
    # output_model = qeff_model.model(input_ids=inputs["input_ids"],past_key_values=pkv,use_cache=True, position_ids=position_ids_ten, 
    #                 is_encode=is_encode, self_condition_selector = self_condition_selector, decoder_input_ids = inputs["input_ids"],
    #                 self_conditioning_logits=self_conditioning_logits, decoder_position_ids=position_ids_ten)
    output = qeff_model.generate(inputs=inputs,generation_len=GENERATION_LEN,qpc_path=qpc_path,)
    breakpoint()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL_ID,dtype="float32",device_map="auto",config=config)
    output_original_model = model.generate(inputs['input_ids'], max_new_tokens=256)

    mad_aic_hf = np.abs((output - output_original_model.detach().float().cpu().numpy())).max()
    mad_aic_qeff = np.abs((output - output_model[0].detach().cpu().numpy())).max()
    mad_qeff_hf = np.abs((output_model[0].detach().cpu().numpy() - output_original_model.detach().float().cpu().numpy())).max()
    print(f'mad_aic_hf:{mad_aic_hf}')
    print(f'mad_aic_qeff:{mad_aic_qeff}')
    print(f'mad_qeff_hf:{mad_qeff_hf}')
    breakpoint()

    session = ort.InferenceSession(str(onnx_path))
    m = qeff_model.model  # QEffDiffusionGemmaForBlockDiffusion

    # Get full input structure expected by ONNX
    dummy = m.get_dummy_inputs(kv_offload=False)

    # Replace with your real prompt
    real_ids = inputs["input_ids"]                      # [B, S]
    B, S = real_ids.shape
    dummy["input_ids"] = real_ids
    dummy["position_ids"] = torch.arange(S).unsqueeze(0).repeat(B, 1).to(real_ids.device)

    # Optional: keep/override mm_token_type_ids if needed
    if "mm_token_type_ids" in inputs:
        dummy["mm_token_type_ids"] = inputs["mm_token_type_ids"]

    # Feed ORT
    ort_inputs = {}
    for k, v in dummy.items():
        if k == "past_key_values":
            for i, (pk, pv) in enumerate(v):
                ort_inputs[f"past_key.{i}"] = pk.numpy()
                ort_inputs[f"past_value.{i}"] = pv.numpy()
        else:
            ort_inputs[k] = v.numpy()
    breakpoint()
    
    output_names = [o.name for o in session.get_outputs()]
    # ort_inputs = {k: v.numpy() for k, v in inputs.items()}
    ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
    expose_all_intermediate_outputs_robust('/home/jsaisaga/qeff_llama/d_g_npi_agent_2layers/onnx-998a1cb621b818cb/DiffusionGemmaForBlockDiffusion.onnx')
    analyze_onnx_fp16_overflow('/home/jsaisaga/qeff_llama/d_g_npi_agent_2layers/onnx-998a1cb621b818cb/DiffusionGemmaForBlockDiffusion.onnx', ort_inputs)
 
    # enc_inputs_embeds = qeff_model.model._inject_vision_embeds(inputs["input_ids"],None,None)
    # enc_outputs = qeff_model.model.model.encoder.language_model(inputs_embeds=enc_inputs_embeds)
    print('Model Compiled and running original model is started')
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL_ID,dtype="float32",device_map="auto",config=config)
    output_original_model = model.generate(inputs['input_ids'], max_new_tokens=256)
    breakpoint()
    mad = np.abs((output - output_original_model.detach().float().cpu().numpy())).max()
    print('Mad score between original and model running on device is ', mad)
    breakpoint()

    qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
    print(output.generated_ids)
    print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
    print(output)
    print(f"ONNX_PATH={onnx_path}")
    print(f"QPC_PATH={qpc_path}")


if __name__ == "__main__":
    main()
