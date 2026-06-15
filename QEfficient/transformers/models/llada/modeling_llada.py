import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import PreTrainedModel
from typing import List, Optional, Tuple, Type, Union, NamedTuple, Sequence
from transformers.cache_utils import Cache
import math
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention, get_attention_blocking_config

def qeff_apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # Cast back to original dtype
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

class QEffLladaRotaryEmbedding(nn.Module):
    """
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config, device=None):
        super().__init__()
        dim = config.d_model // config.n_kv_heads
        self.inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
        self.original_max_seq_len = 131072 or config.max_sequence_length
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    

def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def sample_tokens(logits, temperature):
    if temperature > 0:
        logits = logits / temperature
    # logits = top_k_logits(logits, top_k)
    vals = torch.topk(logits, k=2, dim=-1, largest=True, sorted=True)
    x0 = vals.indices[0][:,0]
    vals2 = vals.values
    confidence = vals2[0][:,0] - vals2[0][:,1]
    return confidence, x0

class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """

class LladaSampler(nn.Module):
    def __init__(self, mask_token_id):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.temperature = 0.2

    def forward(self, x, logits, confidence_mask):
        mask_index = (x == self.mask_token_id)
        if self.temperature is not None:
            logits = logits/self.temperature

        mask_logits = torch.where(mask_index.unsqueeze(-1), logits, -torch.inf)
        confidence, x0 = sample_tokens(mask_logits, temperature=self.temperature)
        confidence = torch.where(confidence_mask>0, confidence, -torch.inf)
        confidence = torch.where(mask_index, confidence, -torch.inf)
        # confidence[:, prompt_shape + (num_block + 1) * block_length:] = -torch.inf
        confidence = confidence.unsqueeze(0)
        number_transfer_tokens = 2
        x_ = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        _, transfer_index = torch.topk(confidence, k=number_transfer_tokens)
        x_ = torch.where(mask_index, x0, x_)
        row_indices = torch.arange(x.size(0), device=x.device, dtype=torch.int64).unsqueeze(1).expand_as(transfer_index)
        
        transfer_index = transfer_index.to(torch.int64)
        x[row_indices,transfer_index] = x_[row_indices,transfer_index]
        return x

class QEffLLaDABlock(nn.Module):
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cos_cached: Optional[torch.Tensor] = None,  # ADD THIS
        sin_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            cos_cached = None
            if cos_cached is not None and sin_cached is not None:
                # Use provided cached embeddings
                print('In QEffLLadaBlock cos_cached and sin_cached applied ')
                q, k = qeff_apply_rotary_pos_emb(q, k, cos_cached.squeeze(), sin_cached.squeeze())
            else:
                # Fallback to original method
                q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        # att = self._scaled_dot_product_attention(
        #     q,
        #     k,
        #     v,
        #     attn_mask=attention_bias,
        #     dropout_p=0.0 if not self.training else self.config.attention_dropout,
        #     is_causal=False,
        # )
        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        print('Going Into blocking attention')
        att = compute_blocked_attention(
            q,
            k,
            v,
            blocking_mode=blocking_mode,
            head_block_size=head_block_size,
            num_kv_blocks=num_kv_blocks,
            num_q_blocks=num_q_blocks,
            attention_mask=attention_bias,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

class QEffLLaDASequentialBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cos_cached: Optional[torch.Tensor] = None, 
        sin_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        print('Inside QEffLladaSequentialBlock')
        if self._activation_checkpoint_fn is not None:
            q, k, v = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos_cached, sin_cached=sin_cached
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos_cached, sin_cached=sin_cached)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class QEffLLaDALlamaBlock(nn.Module):
    def __qeff_init__(self):
        self.attention = QEffLLaDABlock.attention

        print(self)
    
    def forward(
            self,
            x: torch.Tensor,
            attention_bias: Optional[torch.Tensor] = None,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            cos_cached: Optional[torch.Tensor] = None,  # ADD THIS
            sin_cached: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        print('Inside QEffLladaBlockModel')
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        print('Inside QEffLladaLlamaBlock')

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos_cached, sin_cached=sin_cached
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos_cached, sin_cached=sin_cached)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache

class QEffLLaDABlockGroup(nn.ModuleList):
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        cos_cached: Optional[torch.Tensor] = None, 
        sin_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        print('Inside QEffLladaBlockGroupModel')
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos_cached, sin_cached=sin_cached
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos_cached, sin_cached=sin_cached)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values
    
class QEffLLaDAModel(nn.Module):
    def __qeff_init__(self):
        self.rotary_emb = QEffLladaRotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached)

    def get_bidirectional_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (bidirectional_bias := self.__cache.get("bidirectional_attention_bias")) is not None and bidirectional_bias.shape[
            -1
        ] >= seq_len:
            if bidirectional_bias.device != device:
                bidirectional_bias = bidirectional_bias.to(device)
                self.__cache["bidirectional_attention_bias"] = bidirectional_bias
            return bidirectional_bias
        with torch.autocast(device.type, enabled=False):
            bidirectional_bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.float)
        self.__cache["bidirectional_attention_bias"] = bidirectional_bias
        return bidirectional_bias
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        
        seq_len = input_ids.shape[1]
        if self.config.rope:
            sin = self.sin_cached[:seq_len].unsqueeze(1)
            cos = self.cos_cached[:seq_len].unsqueeze(1)
        print('Inside QEffLladaModel ', input_ids.shape, seq_len)
        # Add Basic MDM Model config check
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        # assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None and 0.0 in attention_mask:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                1
            elif attention_bias is None:
                attention_bias = self.get_bidirectional_attention_bias(past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos, sin_cached=sin 
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, cos_cached=cos, sin_cached=sin )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache, cos_cached=cos, sin_cached=sin
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)
    

class QEffLLaDAModelLM(PreTrainedModel):
    def __qeff_init__(self):
        self.mask_token_id = 126336
        print('In qeff init of model, sampler is initialized')
        self.sampler = LladaSampler(mask_token_id = self.mask_token_id)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None,  # This is a hack mitigation of an issue in transformers `4.39.x`
        confidence_mask: Optional[torch.Tensor] = None,
    )-> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache
        print('Inside QEff modified model')
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits
        x_new = self.sampler(input_ids, logits, confidence_mask)
        return x_new