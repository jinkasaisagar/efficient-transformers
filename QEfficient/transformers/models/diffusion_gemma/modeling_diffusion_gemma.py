# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Type, Union

import onnx
import torch
import torch.nn as nn
import yaml
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaBlockDiffusionOutputWithPast,
    DiffusionGemmaDecoderModel,
    DiffusionGemmaDecoderTextAttention,
    DiffusionGemmaDecoderTextLayer,
    DiffusionGemmaEncoderModel,
    DiffusionGemmaEncoderTextAttention,
    DiffusionGemmaEncoderTextLayer,
    DiffusionGemmaEncoderTextModel,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaModel,
    DiffusionGemmaModelOutputWithPast,
    DiffusionGemmaRMSNorm,
    DiffusionGemmaTextExperts,
    DiffusionGemmaTextRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants

_FP16_CLAMP_MIN = -65504.0
_FP16_CLAMP_MAX = 65504.0
_DISABLE_EXPORT_FP16_CLAMP = False


def _is_onnx_export() -> bool:
    return torch.onnx.is_in_onnx_export()


def _clamp_to_fp16_range(hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export() or _DISABLE_EXPORT_FP16_CLAMP:
        return hidden_states
    return hidden_states.clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX)


def _saturating_residual_add(residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export() or _DISABLE_EXPORT_FP16_CLAMP:
        return residual + hidden_states
    return (residual.float() + hidden_states.float()).clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX).to(hidden_states.dtype)


def _build_additive_attention_mask(
    position_ids: torch.Tensor,
    target_length: int,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    causal_mask = _create_causal_mask(
        position_ids=position_ids,
        target_length=target_length,
        sliding_window=sliding_window,
    )
    return causal_mask.to(dtype=dtype) * torch.finfo(dtype).min


def _build_diffusion_decoder_additive_attention_mask(
    decoder_attention_mask: Optional[torch.Tensor],
    canvas_length: int,
    dtype: torch.dtype,
    layer_cache_length: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if decoder_attention_mask is None:
        return None

    if decoder_attention_mask.dim() != 2:
        raise ValueError("decoder_attention_mask must be 2D [batch, full_kv_len].")

    full_kv_len = decoder_attention_mask.shape[-1]
    cache_length = full_kv_len - canvas_length
    if cache_length < 0:
        raise ValueError("decoder_attention_mask length must be >= canvas_length.")

    layer_mask = decoder_attention_mask
    if layer_cache_length is not None and layer_cache_length != cache_length:
        layer_cache_length = max(min(int(layer_cache_length), cache_length), 0)
        encoder_mask = decoder_attention_mask[:, cache_length - layer_cache_length : cache_length]
        canvas_mask = decoder_attention_mask[:, cache_length:]
        layer_mask = torch.cat([encoder_mask, canvas_mask], dim=-1)

    invalid_positions = ~layer_mask.bool()
    additive_mask = invalid_positions[:, None, None, :].expand(-1, 1, canvas_length, -1)
    return additive_mask.to(dtype=dtype) * torch.finfo(dtype).min


class QEffDiffusionGemmaTextRouter(DiffusionGemmaTextRouter):
    def __qeff_init__(self):
        if (
            hasattr(self, "norm")
            and not getattr(self.norm, "with_scale", True)
            and not hasattr(self.norm, "_qeff_unit_weight")
        ):
            self.norm.register_buffer("_qeff_unit_weight", torch.ones(self.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        router_probabilities = nn.functional.softmax(self.proj(hidden_states), dim=-1)
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )

        top_k_weights = top_k_weights / torch.einsum("bk->b", top_k_weights).unsqueeze(-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class QEffDiffusionGemmaCustomRMSNormAIC(nn.Module):
    """
    DiffusionGemma RMSNorm replacement that preserves `with_scale=False` behavior
    while still exporting through the compiler-known custom RMSNorm op.
    """

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _is_onnx_export():
            normed_output = self._norm(hidden_states.float())
            if getattr(self, "with_scale", True):
                normed_output = normed_output * self.weight.float()
            return normed_output.type_as(hidden_states)

        if getattr(self, "with_scale", True):
            weight = self.weight
        else:
            weight = getattr(self, "_qeff_unit_weight", None)
            if weight is None:
                weight = hidden_states.new_ones(hidden_states.shape[-1])
        return CustomRMSNormFunc.apply(hidden_states, weight, self.eps)


class QEffDiffusionGemmaTextExperts(DiffusionGemmaTextExperts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        gate_up_proj_t = self.gate_up_proj.transpose(1, 2)
        gate_up_out = torch.matmul(hidden_states, gate_up_proj_t).permute(1, 0, 2)
        gate, up = gate_up_out.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up

        down_proj_t = self.down_proj.transpose(1, 2)
        experts_out = torch.matmul(activated.permute(1, 0, 2), down_proj_t).permute(1, 0, 2)
        expert_weights = torch.zeros(
            hidden_states.shape[0],
            self.num_experts,
            dtype=top_k_weights.dtype,
            device=top_k_weights.device,
        )
        expert_weights.scatter_add_(1, top_k_index, top_k_weights)
        weighted_experts = experts_out.transpose(1, 2)
        combine_weights = expert_weights.to(experts_out.dtype).unsqueeze(-1)
        return torch.bmm(weighted_experts, combine_weights).squeeze(-1)


class QEffDiffusionGemmaEncoderTextAttention(DiffusionGemmaEncoderTextAttention):
    def __qeff_init__(self):
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True) and not hasattr(norm, "_qeff_unit_weight"):
                norm.register_buffer("_qeff_unit_weight", torch.ones(self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            if hasattr(past_key_values, "append_new_layers"):
                past_key_values.append_new_layers(self.layer_idx)
            cache_kwargs = {"position_ids": position_ids} if position_ids is not None else {}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffDiffusionGemmaDecoderTextAttention(DiffusionGemmaDecoderTextAttention):
    def __qeff_init__(self):
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True) and not hasattr(norm, "_qeff_unit_weight"):
                norm.register_buffer("_qeff_unit_weight", torch.ones(self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            if hasattr(past_key_values, "read_only"):
                cache_kwargs = {"position_ids": position_ids} if position_ids is not None else {}
                encoder_key_states, encoder_value_states = past_key_values.read_only(
                    self.layer_idx,
                    cache_kwargs=cache_kwargs,
                )
            else:
                encoder_key_states = past_key_values.layers[self.layer_idx].keys
                encoder_value_states = past_key_values.layers[self.layer_idx].values
            key_states = torch.cat([encoder_key_states, key_states], dim=2)
            value_states = torch.cat([encoder_value_states, value_states], dim=2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffDiffusionGemmaEncoderTextLayer(DiffusionGemmaEncoderTextLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = _clamp_to_fp16_range(hidden_states)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        hidden_states *= self.layer_scalar
        return hidden_states


class QEffDiffusionGemmaDecoderTextLayer(DiffusionGemmaDecoderTextLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = _clamp_to_fp16_range(hidden_states)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        hidden_states *= self.layer_scalar
        return hidden_states


class QEffDiffusionGemmaEncoderTextModel(DiffusionGemmaEncoderTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if isinstance(past_key_values, Cache) and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_cache(self.config, past_key_values)
        elif past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.config, past_key_values)
        elif past_key_values is None:
            past_key_values = QEffGemma4DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            sliding_window = self.config.sliding_window if layer_type == "sliding_attention" else None
            target_length = inputs_embeds.shape[1]
            if past_key_values is not None and len(past_key_values.layers) > i:
                layer_keys = past_key_values.layers[i].keys
                if layer_keys is not None and layer_keys.numel() > 0:
                    target_length = layer_keys.shape[-2]
            layer_attention_mask = _build_additive_attention_mask(
                position_ids=position_ids,
                target_length=target_length,
                dtype=hidden_states.dtype,
                sliding_window=sliding_window,
            )

            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class QEffDiffusionGemmaEncoderModel(DiffusionGemmaEncoderModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask = self.get_placeholder_mask(input_ids, inputs_embeds)

        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features.to(inputs_embeds.device))

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask_mapping = attention_mask
        if not isinstance(causal_mask_mapping, dict):
            causal_mask_mapping = None
            del mm_token_type_ids

        kwargs.pop("return_dict", None)
        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffDiffusionGemmaDecoderModel(DiffusionGemmaDecoderModel):
    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        self_conditioning_mask: Optional[torch.BoolTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        if "use_cache" in kwargs:
            raise ValueError(
                "The decoder of DiffusionGemma always uses a cache, so it doesn't accept the `use_cache` argument"
            )

        if isinstance(past_key_values, Cache) and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_cache(self.text_config, past_key_values)
        elif past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.text_config, past_key_values)

        inputs_embeds = self.embed_tokens(decoder_input_ids)

        if self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
            if self_conditioning_mask is not None:
                soft_embeddings = soft_embeddings * self_conditioning_mask.to(soft_embeddings.dtype)[:, None, None]
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        if decoder_position_ids is None:
            canvas_length = inputs_embeds.shape[1]
            cache_seq_length = past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0
            decoder_position_ids = torch.arange(
                cache_seq_length,
                cache_seq_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            decoder_position_ids = decoder_position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, decoder_position_ids, layer_type)

        canvas_length = decoder_input_ids.shape[1]
        for i, decoder_layer in enumerate(self.layers[: self.text_config.num_hidden_layers]):
            layer_cache_length = None
            if past_key_values is not None and len(past_key_values.layers) > i:
                layer_keys = past_key_values.layers[i].keys
                if layer_keys is not None:
                    layer_cache_length = int(layer_keys.shape[-2])

            layer_attention_mask = _build_diffusion_decoder_additive_attention_mask(
                decoder_attention_mask=decoder_attention_mask,
                canvas_length=canvas_length,
                layer_cache_length=layer_cache_length,
                dtype=hidden_states.dtype,
            )

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.text_config.layer_types[i]],
                attention_mask=layer_attention_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class QEffDiffusionGemmaModel(DiffusionGemmaModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        self_conditioning_mask: Optional[torch.BoolTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DiffusionGemmaModelOutputWithPast:
        encoder_last_hidden_state = None
        if input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            encoder_last_hidden_state = encoder_outputs.last_hidden_state
        elif past_key_values is None:
            raise ValueError("Either `input_ids` or `past_key_values` must be provided.")

        if decoder_input_ids is None:
            batch_size = input_ids.shape[0] if input_ids is not None else past_key_values.layers[0].keys.shape[0]
            decoder_input_ids = torch.randint(
                low=0,
                high=self.config.text_config.vocab_size,
                size=(batch_size, self.config.canvas_length),
                device=self.decoder.device,
            )

        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )

        return DiffusionGemmaModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )


class QEffDiffusionGemmaEncoderWrapper(nn.Module):
    """
    Encoder-only wrapper for split-runtime DiffusionGemma execution.
    Exports prompt encoding into KV-cache outputs so decode can iterate on canvas
    without re-running the encoder each denoising step.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = model.model.encoder
        self.config = model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaEncoderTextLayer}

    def get_output_names(self):
        output_names = []
        for i in range(self.config.text_config.num_hidden_layers):
            output_names.extend([f"past_key.{i}_RetainedState", f"past_value.{i}_RetainedState"])
        return output_names

    def get_dummy_inputs(self):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        return {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
        }

    def get_onnx_dynamic_axes(self):
        return {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        kwargs.pop("return_dict", None)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs,
        )
        next_cache = outputs.past_key_values
        if isinstance(next_cache, Cache):
            next_cache = next_cache.to_legacy_cache()
        return next_cache


class QEffDiffusionGemmaDecoderWrapper(nn.Module):
    """
    Decoder-only wrapper for split-runtime DiffusionGemma execution.
    Consumes encoder KV-cache and iteratively denoises canvas blocks.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.config = model.config
        self.final_logit_softcapping = model.final_logit_softcapping

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaDecoderTextLayer}

    def get_output_names(self):
        return ["logits"]

    def get_dummy_inputs(self):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        canvas_len = getattr(self.config, "canvas_length", 256)
        inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
            "decoder_input_ids": torch.zeros((bs, canvas_len), dtype=torch.int64),
            "self_conditioning_logits": torch.zeros((bs, canvas_len, self.config.text_config.vocab_size), dtype=torch.float32),
            "self_conditioning_mask": torch.ones((bs,), dtype=torch.bool),
            "past_key_values": self.model.get_dummy_pkv_cache(config=self.config, batch_size=bs, seq_len=seq_len),
        }
        return inputs

    def get_onnx_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
            "self_conditioning_logits": {0: "batch_size", 1: "seq_len"},
            "self_conditioning_mask": {0: "batch_size"},
        }
        for i in range(self.config.text_config.num_hidden_layers):
            dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}
        return dynamic_axes

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        self_conditioning_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        del input_ids, attention_mask
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.config.text_config, past_key_values)
        elif isinstance(past_key_values, Cache) and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_cache(self.config.text_config, past_key_values)

        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state).to(torch.float32)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits


class QEffDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    _NPI_EXCLUDED_OPS = {
        "Constant",
        "ConstantOfShape",
        "Concat",
        "CustomRMSNorm",
        "Equal",
        "Gather",
        "MatMul",
        "Range",
        "Reshape",
        "Shape",
        "Slice",
        "Transpose",
        "Unsqueeze",
    }

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaEncoderTextLayer, QEffDiffusionGemmaDecoderTextLayer}

    def get_qeff_diffusion_encoder(self):
        return QEffDiffusionGemmaEncoderWrapper(self)

    def get_qeff_diffusion_decoder(self):
        return QEffDiffusionGemmaDecoderWrapper(self)

    def generate_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        del model_name
        onnx_path = onnx_path or self.onnx_path
        if onnx_path is None:
            raise ValueError("ONNX path is required to generate DiffusionGemma NPI file.")
        onnx_path = Path(onnx_path)
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_diffusion_gemma_npi.yaml")

        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_names = []
        for node in model.graph.node:
            if node.op_type in self._NPI_EXCLUDED_OPS:
                continue
            fp32_names.extend(
                output_name for output_name in node.output if output_name and not output_name.endswith("_RetainedState")
            )
        for function in model.functions:
            if "EncoderTextLayer" not in function.name and "DecoderTextLayer" not in function.name:
                continue
            for node in function.node:
                if node.op_type in self._NPI_EXCLUDED_OPS:
                    continue
                fp32_names.extend(output_name for output_name in node.output if output_name)

        fp32_names = [name for name in list(dict.fromkeys(fp32_names)) if "MatMul" not in name]
        npi_data = {"FP32NodeInstanceNames": fp32_names}
        with open(npi_path, "w") as fp:
            yaml.safe_dump(npi_data, fp, sort_keys=False)
        return str(npi_path)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **kwargs,
    ):
        del kwargs
        batch_size = batch_size if batch_size else 1
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        kv_cache_batch_size = kv_cache_batch_size or full_batch_size or batch_size
        sliding_window = self.config.text_config.sliding_window

        def build_prefill_spec(comp_ctx_lengths: Optional[int] = None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": sliding_window,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            return spec

        def build_decode_spec(comp_ctx_lengths: Optional[int] = None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "sliding_window": sliding_window,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            specializations = [build_prefill_spec(length) for length in comp_ctx_lengths_prefill]
            specializations.extend(build_decode_spec(length) for length in comp_ctx_lengths_decode)
            return specializations

        return [build_prefill_spec(), build_decode_spec()]

    def get_output_names(self, kv_offload: bool = False):
        del kv_offload
        output_names = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ("key", "value"):
                output_names.append(f"past_{kv}.{i}_RetainedState")
        return output_names

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        del kv_offload
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
        canvas_len = getattr(self.config, "canvas_length", 256)
        batch_for_kv = fbs if continuous_batching else bs

        inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "decoder_input_ids": torch.zeros((bs, canvas_len), dtype=torch.int64),
            "self_conditioning_logits": torch.zeros(
                (bs, canvas_len, self.config.text_config.vocab_size), dtype=torch.float32
            ),
            "self_conditioning_mask": torch.ones((bs,), dtype=torch.bool),
            "past_key_values": self.get_dummy_pkv_cache(
                config=self.config, batch_size=batch_for_kv, seq_len=seq_len
            ),
        }
        if continuous_batching:
            inputs["batch_index"] = torch.arange(bs, dtype=torch.int64)
        if comp_ctx_lengths is not None:
            inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
        return inputs

    def get_pkv_dynamic_axes(
        self,
        retain_full_kv: Optional[bool] = False,
        continuous_batching: Optional[bool] = False,
    ):
        del retain_full_kv
        layer_types = self.config.text_config.layer_types
        return [
            (
                {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
                if layer_type == "sliding_attention"
                else {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
            )
            for layer_type in layer_types
        ]

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        continuous_batching: bool = False,
    ):
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "self_conditioning_logits": {0: "batch_size", 1: "seq_len"},
            "self_conditioning_mask": {0: "batch_size"},
        }
        if continuous_batching:
            dynamic_axes["batch_index"] = {0: "batch_size"}

        for i, ctx_axis in enumerate(self.get_pkv_dynamic_axes(continuous_batching=continuous_batching)):
            for kv in ("key", "value"):
                dynamic_axes[f"past_{kv}.{i}"] = ctx_axis

        if comp_ctx_lengths is not None:
            dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        return dynamic_axes

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        text_config = config.text_config
        past_key_values = []
        for layer_type in text_config.layer_types:
            if layer_type == "sliding_attention":
                n_heads = text_config.num_key_value_heads
                d_head = text_config.head_dim
                layer_seq_len = min(text_config.sliding_window, seq_len)
            else:
                n_heads = (
                    text_config.num_global_key_value_heads
                    if getattr(text_config, "num_global_key_value_heads", None) is not None
                    else text_config.num_key_value_heads
                )
                d_head = text_config.global_head_dim if getattr(text_config, "global_head_dim", None) else text_config.head_dim
                layer_seq_len = seq_len
            cache_shape = [batch_size, n_heads, layer_seq_len, d_head]
            past_key_values.append(
                (
                    torch.zeros(cache_shape, dtype=torch.float32),
                    torch.zeros(cache_shape, dtype=torch.float32),
                )
            )
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        self_conditioning_mask: Optional[torch.BoolTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DiffusionGemmaBlockDiffusionOutputWithPast:
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )

        logits = self.lm_head(model_outputs.last_hidden_state)
        logits = logits.to(torch.float32)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        next_cache = model_outputs.past_key_values
        if isinstance(next_cache, Cache):
            next_cache = next_cache.to_legacy_cache()

        return DiffusionGemmaBlockDiffusionOutputWithPast(
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            past_key_values=next_cache,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state,
        )


__all__ = [
    "QEffDiffusionGemmaCustomRMSNormAIC",
    "QEffDiffusionGemmaDecoderModel",
    "QEffDiffusionGemmaDecoderWrapper",
    "QEffDiffusionGemmaDecoderTextAttention",
    "QEffDiffusionGemmaDecoderTextLayer",
    "QEffDiffusionGemmaEncoderWrapper",
    "QEffDiffusionGemmaEncoderModel",
    "QEffDiffusionGemmaEncoderTextAttention",
    "QEffDiffusionGemmaEncoderTextLayer",
    "QEffDiffusionGemmaEncoderTextModel",
    "QEffDiffusionGemmaForBlockDiffusion",
    "QEffDiffusionGemmaModel",
    "QEffDiffusionGemmaTextExperts",
    "QEffDiffusionGemmaTextRouter",
]
