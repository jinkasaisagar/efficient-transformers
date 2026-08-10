# DiffusionGemma Session Handoff

Date: August 7, 2026

## Scope

Investigated text-only output and denoising-step differences between:

- Current example: `examples/image_text_to_text/models/gemma_vision/diffusion_gemma/diffusion_gemma_single_qpc_example_correct.py`
- Comparison example: `/home/jsaisaga/onboarding_agent/efficient-transformers-fork/examples/image_text_to_text/models/diffusion_gemma/diffusion_gemma_single_qpc_example.py`
- Current model: `QEfficient/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`
- Comparison model: `/home/jsaisaga/onboarding_agent/efficient-transformers-fork/QEfficient/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`

No implementation files were modified during the investigation.

## Current Result

The current `diffusion_gemma_single_qpc_example_correct.py` now produces good text-only output and generally completes denoising in fewer steps than the comparison implementation.

## Successful Change To Preserve

The working configuration combines two specific behaviors in the decoder:

1. Read the retained encoder KV tensors directly, without calling `read_only()`:

```python
layer = past_key_values.layers[self.layer_idx]
encoder_key_states = layer.keys
encoder_value_states = layer.values
```

2. Use encoder `position_ids` to construct a validity mask for the fixed-size retained cache:

```python
packed_valid_mask = _build_encoder_kv_valid_mask(
    position_ids=position_ids,
    cache_shape_anchor=kv_proj,
)
per_type_mask = packed_valid_mask.to(torch.int64)
```

This successful combination is important:

- Direct KV access preserves the retained-cache physical layout expected by the compiled QPC.
- The position-derived mask prevents unused cache positions from participating in decoder attention.
- The `past_key_values.read_only(...)` gather/reordering path remains commented out and should not be re-enabled without a controlled hardware comparison.
- The unified graph no longer needs a caller-provided `encoder_attention_mask`; cache validity is derived from the same `position_ids` used by encoder prefill and token commits.

The corresponding current example therefore does not feed `encoder_attention_mask`. For multi-canvas generation, it updates `position_ids` after each committed block so the decoder mask includes the complete committed prefix.

## Sampler Comparison

The two examples use the same core denoising algorithm:

- Default seed: `1234`
- Default sampler: cumulative local freezing
- Entropy budget: `0.1`
- Temperature range: `0.8` to `0.4`
- Self-conditioning disabled on step 1 and enabled afterward
- Generation stops when all canvas positions have been accepted

The acceptance rule is effectively:

```python
sel = (np.cumsum(ent[order]) - ent[order]) <= entropy_bound
accepted_mask = accepted_mask | new_acc
```

Lower-entropy logits allow more positions to be accepted per iteration, reducing the total number of denoising steps.

## Primary Model Difference

Both current decoder implementations directly read retained encoder KV:

```python
layer = past_key_values.layers[self.layer_idx]
encoder_key_states = layer.keys
encoder_value_states = layer.values
```

The important difference is the decoder attention mask applied to that cache.

### Comparison Model

The comparison model constructs an all-ones encoder-KV mask from the retained cache shape. For a 20-token prompt and `ctx_len=1024`, decoder attention includes:

```text
20 real encoder KV slots + 1004 zero-padded KV slots
```

The comparison script supplies `encoder_attention_mask`, but the comparison decoder ignores that caller-provided mask and internally uses an all-ones mask.

### Current Model

The current model passes encoder `position_ids` into the decoder and constructs a valid-cache mask with `_build_encoder_kv_valid_mask(...)`.

For a 20-token prompt, only the first 20 encoder KV positions are visible. Remaining cache positions receive the additive attention-mask value, which is `-1e4` during ONNX/QPC execution.

This prevents empty zero-KV slots from participating in decoder softmax. The resulting attention can be sharper, producing lower-entropy token distributions and therefore accepting more canvas tokens per denoising step.

## Encoder Comparison

No functional encoder-model difference was found between the two `modeling_diffusion_gemma.py` files:

- `QEffDiffusionGemmaEncoderTextAttention` is functionally identical.
- `QEffDiffusionGemmaEncoderTextLayer` is functionally identical; its location in the file changed.
- `QEffDiffusionGemmaEncoderTextModel` is byte-for-byte identical in the compared class extraction.
- `QEffDiffusionGemmaEncoderPrefillWrapper` is byte-for-byte identical in the compared class extraction.
- Active DiffusionGemma transform mappings for encoder attention, layer, and text model are equivalent.

The observed output and step-count differences therefore originate primarily from how the decoder consumes and masks retained encoder KV, not from prompt encoding.

## Other Example Differences

### Default Prompt

The current example defaults to:

```text
What is diffusion based generative learning? Answer in one sentence.
```

The comparison example ultimately defaults to:

```text
What are the seven continents? Answer in one sentence.
```

Prompt content can independently change entropy and denoising-step count.

### Attention-Mask Input

- The comparison example creates and feeds `encoder_attention_mask`.
- The current example does not expose that unified-QPC input.
- The current model instead derives cache validity from encoder `position_ids`.

### Multi-Canvas Position Tracking

- The comparison example maintains a separate integer `cursor`.
- The current example derives the next encoder position from the latest valid `position_ids`.
- This difference does not affect a default single-canvas run, but it affects committed-token handling for multi-canvas generation.

## Controlled Comparison Command

Use the same prompt and runtime options for both scripts:

```bash
--text-only \
--prompt "What is diffusion based generative learning? Answer in one sentence." \
--seed 1234 \
--sampler local \
--max-new-tokens 256 \
--diffusion-steps 48
```

Record the accepted-token count printed at every step. With identical prompt, seed, sampler, canvas length, and compiled shapes, the remaining difference should mainly reflect decoder retained-KV masking.

## Validation Notes

- Python syntax compilation passed for the inspected current model, cache utility, and example.
- Focused pytest execution was unavailable because `/usr/bin/python3` does not have `pytest` installed.
- The working tree already contained multiple staged and unstaged DiffusionGemma changes before this investigation; avoid broad resets or overwrites.

## Suggested Next Diagnostic

If exact attribution is needed, perform a two-variant ablation using the same prompt and seed:

1. Directly read `layer.keys/values` and use the current position-derived valid-KV mask.
2. Directly read `layer.keys/values` and use the comparison all-ones mask.

Compare per-step mean entropy, accepted-token count, generated text, and final step count. This isolates cache masking from cache reading and sampler behavior.
