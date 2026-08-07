# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import re
import time
from io import BytesIO

import numpy as np
import onnx
import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.transformers.models.modeling_auto import QEffCausalLMForTextImageToTextModel


FP32_ACCUM_OPS = {"CustomRMSNorm", "Clip", "Softmax", "Add", "Sub", "Mul", "Div", "Tanh", "Pow", "ReduceMean"}


class UnifiedQPC(QEffCausalLMForTextImageToTextModel):
    def __init__(self, model):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_unified_wrapper()
        self.model.qaic_config = None
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False

    @property
    def get_model_config(self):
        return self.model.model.config.__dict__

    def export(self, inputs, output_names, dynamic_axes, **kwargs):
        return self._export(inputs, output_names=output_names, dynamic_axes=dynamic_axes)


def load_model_and_processor(model_id: str, canvas_length: int):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=False,
    )
    qeff_model.model.config.canvas_length = canvas_length
    return processor, qeff_model


def _write_unified_accum_npi(onnx_path):
    graph = onnx.load(onnx_path, load_external_data=False).graph
    producers = {output_name: node for node in graph.node for output_name in node.output}
    keep_nodes = []

    for node in graph.node:
        if node.op_type in FP32_ACCUM_OPS:
            keep_nodes.append(node)
        if "/decoder/self_conditioning/" in node.name or node.name.endswith("/decoder/norm/CustomRMSNorm"):
            keep_nodes.append(node)

    seen_names = set()

    def backtrace(tensor_name, depth=0):
        if tensor_name in seen_names or depth > 8:
            return
        seen_names.add(tensor_name)
        node = producers.get(tensor_name)
        if node is None:
            return
        keep_nodes.append(node)
        for input_name in node.input:
            if input_name in producers:
                backtrace(input_name, depth + 1)

    if graph.output:
        backtrace(graph.output[0].name)

    initializer_names = {initializer.name for initializer in graph.initializer}

    def depends_on_initializer(tensor_name, depth=0):
        if tensor_name in initializer_names:
            return True
        if depth > 4:
            return False
        producer = producers.get(tensor_name)
        if producer is None:
            return False
        return any(depends_on_initializer(input_name, depth + 1) for input_name in producer.input)

    excluded_outputs = {"/decoder/MatMul_output_0", "/lm_head/MatMul_output_0"}
    tensors = []
    seen_tensors = set()
    for node in keep_nodes:
        for output_name in node.output:
            if not output_name or output_name in seen_tensors or output_name in excluded_outputs:
                continue
            if node.op_type == "MatMul" and any(depends_on_initializer(name) for name in node.input):
                continue
            if node.op_type in {
                "Cast",
                "Transpose",
                "Reshape",
                "DequantizeLinear",
                "QuantizeLinear",
            } and depends_on_initializer(output_name):
                continue
            seen_tensors.add(output_name)
            tensors.append(output_name)

    npi_path = os.path.join(os.path.dirname(onnx_path), "npi_fp32_unified_accum.yaml")
    with open(npi_path, "w", encoding="utf-8") as handle:
        handle.write("FP32NodeInstanceNames: [")
        handle.write(", ".join(f"'{name}'" for name in sorted(tensors)))
        handle.write("]\n")
    print(f"  unified fp32 accumulation island: {len(tensors)} tensors -> {npi_path}")
    return npi_path


def compile_unified_qpc(
    qeff_model,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    canvas_length: int,
    num_devices: int,
    num_cores: int,
):
    print(f"Compiling unified single-QPC ({num_devices} devices, {num_cores} cores)...")
    start = time.time()
    unified = UnifiedQPC(qeff_model)
    unified.export(
        unified.model.get_dummy_inputs(),
        unified.model.get_output_names(),
        unified.model.get_onnx_dynamic_axes(),
    )
    specializations, _ = unified.model.get_specializations(
        batch_size=1,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        canvas_length=canvas_length,
    )

    custom_io = {"vision_embeds": "float16"}
    for layer_index in range(qeff_model.config.text_config.num_hidden_layers):
        for kv_name in ("key", "value"):
            custom_io[f"past_{kv_name}.{layer_index}"] = "float16"
            custom_io[f"past_{kv_name}.{layer_index}_RetainedState"] = "float16"

    qpc_path = unified._compile(
        onnx_path=unified.onnx_path,
        compile_dir=None,
        specializations=specializations,
        convert_to_fp16=True,
        mxfp6_matmul=True,
        mdp_ts_num_devices=num_devices,
        aic_num_cores=num_cores,
        custom_io=custom_io,
        retained_state=True,
        aic_enable_depth_first=True,
        node_precision_info=_write_unified_accum_npi(unified.onnx_path),
    )
    print(f"  unified QPC: {qpc_path} ({time.time() - start:.0f}s)")
    return qpc_path


def _vision_embeds_cpu(model_id: str, text_model, vision_inputs):
    hf_vision = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    with torch.no_grad():
        encoder = hf_vision.model.encoder
        pixel_values = vision_inputs["pixel_values"]
        image_position_ids = vision_inputs["image_position_ids"]
        padding_positions = (image_position_ids == -1).all(dim=-1)
        hidden_states = encoder.vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)
        attention_mask = padding_positions.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype) * torch.finfo(
            hidden_states.dtype
        ).min
        attention_mask = attention_mask.expand(-1, 1, hidden_states.shape[1], -1)
        position_embeddings = encoder.vision_tower.encoder.rotary_emb(hidden_states, image_position_ids)
        for layer in encoder.vision_tower.encoder.layers[: encoder.vision_tower.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=image_position_ids,
            )
        hidden_states, _ = encoder.vision_tower.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=encoder.vision_tower.config.default_output_length,
        )
        if encoder.vision_tower.config.standardize:
            hidden_states = (hidden_states - encoder.vision_tower.std_bias) * encoder.vision_tower.std_scale
        vision_embeds = encoder.embed_vision(inputs_embeds=hidden_states).clamp(-60000.0, 60000.0)
        vision_embeds = vision_embeds[:, : text_model._get_mm_tokens_per_image(), :].float()
    del hf_vision
    return vision_embeds


def prepare_prompt_inputs(
    *,
    processor,
    qeff_model,
    model_id: str,
    prompt: str,
    text_only: bool,
    image_url: str,
):
    if text_only:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if not text_only:
        inputs["vision_embeds"] = _vision_embeds_cpu(model_id, qeff_model, inputs)
    return inputs


def clean_diffusion_text(text: str, truncate_first_sentence: bool = True):
    text = text.replace("\ufffd", " ").strip()
    text = re.sub(r"^\s*(thought\s*)+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).replace("。", ".")
    text = re.sub(r"\bfulling shot\b", "full shot", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(light|dark)\s+(blue|green|teal)ing\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(?:of|Of)\b.*$", ".", text)
    match = re.search(r"(.{12,}?[.!?])", text) if truncate_first_sentence else None
    if match:
        text = match.group(1)
    return text.strip(" \n\t\r\"'")


def build_step_callback(tokenizer, verbose_steps: bool):
    def callback(event):
        prefix = (
            f"  block {event['block_index'] + 1:2d} step {event['step'] + 1:2d} "
            f"t={event['temperature']:.2f} "
            f"acc={event['accepted_count']}/{event['canvas_length']}"
        )
        if verbose_steps:
            preview = tokenizer.decode(event["tokens"][0].tolist(), skip_special_tokens=True)
            prefix += f" :: {preview[:60]!r}"
        print(prefix)

    return callback
