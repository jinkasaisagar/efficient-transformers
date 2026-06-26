# Gemma4 and DiffusionGemma Onboarding Checklists (QEfficient)

This document summarizes how onboarding is done in QEfficient by comparing:

- Hugging Face baseline:
  - `transformers/models/gemma4/modeling_gemma4.py`
  - `transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`
- QEfficient implementations:
  - `QEfficient/transformers/models/gemma4/modeling_gemma4.py`
  - `QEfficient/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`

---

## 1) Gemma4 Onboarding Checklist (`modeling_gemma4.py`)

### A. Baseline Alignment (start here)
1. Keep class structure close to HF:
   - Attention, decoder layer, text model, causal LM, conditional generation.
2. Preserve model semantics:
   - RoPE, KV sharing, MoE routing, per-layer input path, softcapping.
3. Restrict QEff changes to export/runtime integration and compiler compatibility.

### B. Core QEff Class Mapping
1. `Gemma4TextRouter` -> `QEffGemma4TextRouter`
2. `Gemma4TextExperts` -> `QEffGemma4TextExperts` (+ `QEffPrefillChunckedGemma4TextExperts`)
3. `Gemma4TextAttention` -> `QEffGemma4TextAttention`
4. `Gemma4TextDecoderLayer` -> `QEffGemma4TextDecoderLayer`
5. `Gemma4TextModel` -> `QEffGemma4TextModel`
6. `Gemma4ForCausalLM` -> `QEffGemma4ForCausalLM`
7. `Gemma4ForConditionalGeneration` -> `QEffGemma4ForConditionalGeneration`
8. Add split wrappers:
   - `QEffGemma4EncoderWrapper`
   - `QEffGemma4DecoderWrapper`

### C. Numerical/Export Safety Changes
1. Add ONNX-aware fp16 clamps:
   - `_clamp_to_fp16_range`
   - `_saturating_residual_add`
2. Add explicit additive mask builders:
   - `_build_additive_attention_mask`
   - `_build_bidirectional_vision_attention_mask`
3. Replace RMSNorm export path with compiler op:
   - `QEffGemma4CustomRMSNormAIC` using `CustomRMSNormFunc`.
4. Ensure ONNX-friendly behavior during export in wrapper forward paths.

### D. Cache and Attention Adaptation
1. Convert incoming caches to `QEffGemma4DynamicCache`.
2. Return legacy cache tuples for ONNX/runtime friendliness.
3. In attention:
   - keep KV-sharing logic,
   - support compile-safe attention masks,
   - support multimodal bidirectional vision masking when required.

### E. MoE and Router Adaptation
1. Keep router top-k semantics and normalization.
2. Implement expert combine paths with tensor ops favorable for compile/export.
3. Include chunked prefill expert path for efficient prefill execution.

### F. Causal LM Export/Compile Hooks Added
1. `generate_npi_file(...)` for node precision YAML.
2. `get_specializations(...)` for prefill/decode/CB specialization objects.
3. `get_pkv_dynamic_axes(...)`
4. `get_onnx_dynamic_axes(...)`
5. `get_submodules_for_export(...)`
6. `get_dummy_pkv_cache(...)`
7. ONNX/export-friendly `forward(...)` returning logits + legacy cache form.

### G. Conditional Generation (Vision + Language) Integration
1. Add split runtime accessors:
   - `get_qeff_vision_encoder()`
   - `get_qeff_language_decoder()`
2. Add vision NPI generation:
   - `generate_vision_npi_file(...)`
3. Add multimodal specializations and dynamic axes for dual/single QPC.
4. Add dummy input/output definitions for both vision and language paths.
5. Provide helper utilities:
   - transform cleanup (`remove_fp16clip_transform_if_disabled`)
   - generated-id normalization and effective-length helpers.

### H. Wrapper Responsibilities
1. `QEffGemma4EncoderWrapper`
   - Runs vision tower path and emits fixed-shape `vision_embeds`.
2. `QEffGemma4DecoderWrapper`
   - Injects `vision_embeds` into text placeholders.
   - Handles cache conversion and `position_ids`.
   - Returns `(logits, vision_embeds, image_idx_output, past_key_values)`.

### I. What changed vs HF (high-level)
1. Added QEff-prefixed subclasses for key Gemma4 components.
2. Added compile/export scaffolding (specializations, dummy inputs, dynamic axes, NPI).
3. Added explicit additive mask creation and ONNX-safe numerical guards.
4. Added split encoder/decoder wrappers for dual runtime export/compile.
5. Added QEff cache conversion/legacy cache return strategy for runtime compatibility.

### J. Per-Class Change Summary (Gemma4)
1. `QEffGemma4TextRouter`
   - Preserves HF routing semantics, adds QEff init support for `with_scale=False` norm export behavior.
2. `QEffGemma4CustomRMSNormAIC`
   - Replaces standard RMSNorm export with `CustomRMSNormFunc` while keeping eager runtime numerics aligned.
3. `QEffGemma4TextExperts`
   - Uses dense matmul/bmm combine path for export/compiler friendliness.
4. `QEffPrefillChunckedGemma4TextExperts`
   - Adds chunk-friendly experts forward path for prefill performance/runtime blocking use cases.
5. `QEffGemma4TextAttention`
   - Keeps HF attention semantics; adds QEff cache conversion/handling and optional bidirectional vision mask construction.
6. `QEffGemma4TextDecoderLayer`
   - Keeps HF layer flow but adds ONNX fp16-safe clamp and saturating residual add in sensitive paths.
7. `QEffGemma4TextModel`
   - Converts incoming cache to `QEffGemma4DynamicCache`, builds additive masks per layer type, returns legacy cache outputs.
8. `QEffGemma4ForCausalLM`
   - Adds export/runtime hooks (`NPI`, specializations, dynamic axes, dummy PKV), keeps forward logits path ONNX-friendly.
9. `QEffGemma4DecoderWrapper`
   - New split decoder wrapper: injects vision embeds into image placeholders, handles cache conversion, returns retained-state outputs.
10. `QEffGemma4EncoderWrapper`
   - New split encoder wrapper: runs vision tower and emits fixed-shape `vision_embeds` for dual-QPC pipeline.
11. `QEffGemma4ForConditionalGeneration`
   - Adds split encoder/decoder accessors, multimodal export helpers, dual/single QPC specializations, vision NPI support.

---

## 2) DiffusionGemma Onboarding Checklist (`modeling_diffusion_gemma.py`)

### A. Baseline Alignment (start here)
1. Keep HF DiffusionGemma class hierarchy and API shape intact:
   - Encoder text attention/layer/model
   - Decoder text attention/layer/model
   - Top-level model and block diffusion head
2. Reuse Gemma4 QEff patterns where possible for consistency.

### B. Core QEff Class Mapping
1. `DiffusionGemmaTextRouter` -> `QEffDiffusionGemmaTextRouter`
2. `DiffusionGemmaTextExperts` -> `QEffDiffusionGemmaTextExperts`
3. `DiffusionGemmaEncoderTextAttention` -> `QEffDiffusionGemmaEncoderTextAttention`
4. `DiffusionGemmaDecoderTextAttention` -> `QEffDiffusionGemmaDecoderTextAttention`
5. `DiffusionGemmaEncoderTextLayer` -> `QEffDiffusionGemmaEncoderTextLayer`
6. `DiffusionGemmaDecoderTextLayer` -> `QEffDiffusionGemmaDecoderTextLayer`
7. `DiffusionGemmaEncoderTextModel` -> `QEffDiffusionGemmaEncoderTextModel`
8. `DiffusionGemmaEncoderModel` -> `QEffDiffusionGemmaEncoderModel`
9. `DiffusionGemmaDecoderModel` -> `QEffDiffusionGemmaDecoderModel`
10. `DiffusionGemmaModel` -> `QEffDiffusionGemmaModel`
11. `DiffusionGemmaForBlockDiffusion` -> `QEffDiffusionGemmaForBlockDiffusion`
12. Add split wrappers:
    - `QEffDiffusionGemmaEncoderWrapper`
    - `QEffDiffusionGemmaDecoderWrapper`

### C. Numerical and Masking Adaptation
1. Add ONNX-aware clamp/residual helper functions (same style as Gemma4).
2. Use additive mask builders for export:
   - `_build_additive_attention_mask` (encoder side)
   - `_build_diffusion_decoder_additive_attention_mask` (decoder canvas + cache side)
3. Replace RMSNorm path with export-safe custom op:
   - `QEffDiffusionGemmaCustomRMSNormAIC`.

### D. Cache Strategy Changes
1. Convert to `QEffGemma4DynamicCache` in encoder/decoder paths.
2. Decoder attention supports reading encoder KV from cache and appending canvas KV.
3. Return legacy cache tuples in exported forward outputs.

### E. Decoder Diffusion-Specific Handling
1. Preserve self-conditioning behavior:
   - consume `self_conditioning_logits`
   - optional `self_conditioning_mask`
   - combine with token embeddings before decoding.
2. Maintain bidirectional canvas attention semantics with explicit decoder additive mask.

### F. BlockDiffusion Export/Compile Hooks Added
1. `get_submodules_for_export(...)`
2. `get_qeff_diffusion_encoder()`
3. `get_qeff_diffusion_decoder()`
4. `generate_npi_file(...)`
5. `get_specializations(...)`
6. `get_output_names(...)`
7. `get_dummy_inputs(...)`
8. `get_pkv_dynamic_axes(...)`
9. `get_onnx_dynamic_axes(...)`
10. `get_dummy_pkv_cache(...)`

### G. Split Runtime Wrapper Responsibilities
1. `QEffDiffusionGemmaEncoderWrapper`
   - input: prompt IDs/attention
   - output: retained KV tensors for all layers.
2. `QEffDiffusionGemmaDecoderWrapper`
   - input: `decoder_input_ids` + retained KV + optional self-conditioning.
   - output: `logits`.

### H. What changed vs HF (high-level)
1. Added QEff subclasses with compile/export-safe attention/mask/cache behavior.
2. Added explicit export helpers (dummy inputs/axes/output names/specializations).
3. Added split encoder/decoder wrappers for separate ONNX/QPC flow.
4. Ensured ONNX-friendly outputs (legacy cache tuples, no custom cache object outputs).
5. Reused Gemma4 onboarding patterns for consistency and minimal integration friction.

### I. Per-Class Change Summary (DiffusionGemma)
1. `QEffDiffusionGemmaTextRouter`
   - Keeps HF router logic, with QEff init handling for no-scale norm compatibility.
2. `QEffDiffusionGemmaCustomRMSNormAIC`
   - Replaces RMSNorm export path with `CustomRMSNormFunc` while preserving eager-mode behavior.
3. `QEffDiffusionGemmaTextExperts`
   - Rewrites experts combine path with export-friendly tensor ops (`matmul`/`bmm`).
4. `QEffDiffusionGemmaEncoderTextAttention`
   - Preserves HF encoder attention; adds QEff cache update integration and additive mask compatibility.
5. `QEffDiffusionGemmaDecoderTextAttention`
   - Preserves decoder attention semantics; explicitly concatenates encoder KV with canvas KV in compile-friendly form.
6. `QEffDiffusionGemmaEncoderTextLayer`
   - Keeps HF block logic, adds ONNX fp16-safe clamp and saturating residual add.
7. `QEffDiffusionGemmaDecoderTextLayer`
   - Same as encoder layer adaptation, applied to diffusion decoder block.
8. `QEffDiffusionGemmaEncoderTextModel`
   - Converts/creates QEff dynamic cache, computes additive masks per layer type, returns cache in ONNX-friendly form.
9. `QEffDiffusionGemmaEncoderModel`
   - Preserves multimodal/text embedding path but routes through QEff language model/caching behavior.
10. `QEffDiffusionGemmaDecoderModel`
   - Keeps self-conditioning semantics, applies explicit diffusion decoder additive masking, and QEff cache conversion.
11. `QEffDiffusionGemmaModel`
   - Keeps encoder-then-decoder composition; enforces QEff-compatible cache/data flow between both stages.
12. `QEffDiffusionGemmaEncoderWrapper`
   - New split encoder wrapper that exports retained KV tensors only.
13. `QEffDiffusionGemmaDecoderWrapper`
   - New split decoder wrapper consuming retained KV + optional self-conditioning and returning logits.
14. `QEffDiffusionGemmaForBlockDiffusion`
   - Adds full export surface (`NPI`, specializations, dynamic axes, dummy inputs/PKV, output names) and split-wrapper accessors.

---

## 3) Practical Onboarding Flow You Can Reuse

1. Start from HF source-of-truth class layout.
2. Port class-by-class into QEff subclasses.
3. Add custom RMSNorm export path and mask conversion to additive masks.
4. Add cache conversion to QEff dynamic cache + legacy cache outputs.
5. Add export hooks:
   - specializations, dynamic axes, dummy inputs/PKV, output names, NPI file.
6. Add split runtime wrappers if model needs encoder/decoder separation.
7. Register transforms in:
   - `CustomOpsTransform` (RMSNorm mapping)
   - `KVCacheTransform` (model/layer/attention/router/experts mappings).
8. Validate with compile-safe checks (`py_compile`, symbol presence, export smoke tests).

---

## 4) Onboarded Gemma4 vs Onboarded DiffusionGemma (Key Differences)

### A. Model Objective and Runtime Shape
1. Gemma4 onboarding targets autoregressive text/multimodal generation (next-token decode).
2. DiffusionGemma onboarding targets block diffusion denoising with canvas tokens and self-conditioning.
3. Gemma4 decoder runtime is token-incremental; DiffusionGemma decoder runtime is canvas/block-oriented.

### B. Attention and Masking
1. Gemma4 onboarding adds:
   - causal/sliding additive masks,
   - optional bidirectional vision-block mask for image tokens.
2. DiffusionGemma onboarding adds:
   - encoder causal/sliding additive masks,
   - decoder-specific additive mask that merges encoder-cache region + bidirectional canvas region.
3. DiffusionGemma mask path is more specialized because decoder attends over concatenated encoder KV + canvas KV.

### C. Cache Semantics
1. Both use `QEffGemma4DynamicCache` and return legacy cache tuples for export/runtime compatibility.
2. Gemma4 cache path supports KV-sharing behavior in text layers (following Gemma4 architecture).
3. DiffusionGemma cache path is explicitly encoder-to-decoder handoff oriented:
   - encoder builds retained KV once,
   - decoder consumes encoder KV and appends canvas states.

### D. Decoder Inputs/Outputs
1. Gemma4 decoder wrapper inputs:
   - text `input_ids`, `vision_embeds`, `position_ids`, `image_idx`, PKV, optional token types.
2. DiffusionGemma decoder wrapper inputs:
   - `decoder_input_ids`, PKV, optional `self_conditioning_logits` and `self_conditioning_mask`.
3. Gemma4 decoder wrapper outputs multiple tensors (`logits`, retained vision/image index, PKV), while DiffusionGemma decoder wrapper outputs logits for denoising loop.

### E. Wrapper Pair Purpose
1. Gemma4 split wrappers are vision-encoder + language-decoder for multimodal dual-QPC.
2. DiffusionGemma split wrappers are prompt-encoder + diffusion-decoder for block denoising loops.
3. Gemma4 encoder wrapper emits vision embeddings; DiffusionGemma encoder wrapper emits retained KV states.

### F. Export Hook Emphasis
1. Gemma4 onboarding includes multimodal export helpers:
   - dual/single QPC specializations (`vision` + `lang`),
   - multimodal dynamic axes (`vision_embeds`, image token bookkeeping).
2. DiffusionGemma onboarding includes diffusion-specific export helpers:
   - canvas/self-conditioning inputs,
   - split encoder/decoder exports for denoising runtime.
3. Both include `generate_npi_file`, `get_specializations`, dynamic axes, dummy inputs, and dummy PKV helpers.

### G. Conditional/Top-Level Class Adaptation
1. Gemma4 onboarding extends both:
   - text-only causal LM class,
   - multimodal conditional generation class.
2. DiffusionGemma onboarding centers on:
   - `ForBlockDiffusion` top-level class,
   - encoder/decoder submodels used by block diffusion generation logic.

### H. Practical Integration Impact
1. Gemma4 onboarding mainly enables robust multimodal AR compile/export/runtime parity.
2. DiffusionGemma onboarding mainly enables split encoder/decoder diffusion generation orchestration.
3. Code paths overlap on QEff patterns (RMSNorm, masks, cache conversion), but runtime orchestration goals are different.

---

## 5) `diffusion_gemma_utils.py` Details (Algorithm + Function Map)

File: `QEfficient/transformers/diffusion_gemma_utils.py`

### A. End-to-End Algorithm (how generation is coded)
1. Build runtime config from kwargs/model defaults:
   - parse `max_new_tokens`, `max_denoising_steps`, temperature range, entropy bound, stopping thresholds, PAD/EOS IDs.
2. Choose execution mode in `diffusion_gemma_generate_dispatch(...)`:
   - HF fallback (`runtime_ai100=False`) -> calls `model.generate(...)`.
   - AI100 single-QPC (`qpc_path`) -> `diffusion_gemma_generate_ai100(...)`.
   - AI100 split-QPC (`encoder_qpc_path` + `decoder_qpc_path`) -> `diffusion_gemma_generate_ai100_split(...)`.
3. For split-QPC mode:
   - run encoder once (`_run_encoder_block`) to produce KV cache retained states.
   - initialize random or provided diffusion canvas.
   - run denoising loop (`_run_decoder_denoising_loop`) for up to `max_denoising_steps`.
   - append denoised canvas to sequence.
   - optional EOS/PAD post-processing.
   - if more canvas blocks needed, re-run encoder on updated sequence.
4. For single-QPC mode:
   - one session receives prefix + canvas + decoder mask and denoises via `_run_denoising_step(...)`.
   - repeats per canvas block up to `max_new_tokens`.
5. Track metrics:
   - `decode_forward_passes`, `tokens_per_forward`, `total_time`, and return generated IDs.

### B. Data/Config Classes
1. `DiffusionGemmaRuntimeResult`
   - runtime output container: `generated_ids`, `tokens_per_forward`, `decode_forward_passes`, `total_time`.
2. `DiffusionGemmaGenerateDispatch`
   - union-style output: either `runtime_result` (AI100) or `hf_output`.
3. `EntropyBoundSamplerConfig`
   - validates `entropy_bound > 0`.
4. `DiffusionGemmaGenerationConfig`
   - central generation config with validation for lengths, temperatures, thresholds, sampler config.
5. `DiffusionGemmaGenerationOutput`
   - tensor-style generation output container (`sequences`, optional `tokens_per_forward`).

### C. Sampling/Temperature/Stopping Components
1. `LinearTemperatureScheduleLogitsProcessor`
   - scales logits by linearly scheduled temperature between `t_min` and `t_max`.
2. `EntropyBoundSampler`
   - `initialize_canvas(...)`: random init.
   - `accept_canvas(...)`: accept token updates where entropy criterion passes.
   - `renoise_canvas(...)`: resample rejected token positions.
3. `DiffusionGemmaAdaptiveStopping` (abstract)
   - interface for adaptive denoising stop criteria.
4. `StableAndConfidentStoppingCriteria`
   - stops denoising when argmax canvas is stable for `stability_threshold` history and entropy is below confidence threshold.
5. `StableAndConfidentStopper` alias/class
   - file currently contains both:
     - alias `StableAndConfidentStopper = StableAndConfidentStoppingCriteria`
     - later duplicate class definition with same behavior.
   - effective behavior is stable+confidence stopping in both code paths.

### D. Utility Functions (KV and config plumbing)
1. `_normalize_eos_token_ids(eos_token_id)`
   - normalizes scalar/list EOS IDs to `list[int]`.
2. `_prepare_runtime_generation_config(model, kwargs)`
   - resolves runtime generation args from kwargs, model generation config, and model config defaults.
3. `_retained_output_to_state_input(name)`
   - maps retained-state output names to corresponding state-input names.
4. `_collect_kv_cache_from_outputs(outputs)`
   - extracts KV tensors from encoder outputs into a cache dict.
5. `_build_decoder_kv_inputs(decoder_session, kv_cache)`
   - builds decoder input dict by matching session input names with KV cache entries.
6. `_run_encoder_block(encoder_session, input_ids)`
   - executes encoder and returns KV cache map.

### E. Entropy Mask Function
1. `diffusion_gemma_entropy_accept_mask(logits, entropy_bound)`
   - computes token entropy from logits.
   - per position, selects tokens whose cumulative sorted entropy is within `entropy_bound`.
   - returns boolean acceptance mask used by sampler.

### F. Denoising Step/Loop Functions
1. `_run_denoising_step(...)`
   - single-QPC helper:
     - runs model with prefix+canvas inputs,
     - applies temperature scaling,
     - samples denoiser canvas,
     - applies entropy acceptance and renoising.
2. `_run_decoder_denoising_loop(...)`
   - split-QPC helper:
     - runs decoder repeatedly with retained KV + canvas,
     - applies logits processor + entropy sampler,
     - applies adaptive stopping (`stopper`) per batch item.

### G. Runtime Entry Points
1. `diffusion_gemma_generate_ai100_split(...)`
   - split encoder/decoder runtime:
     - encoder precompute -> decoder denoise -> append canvas -> optional re-encode -> repeat.
2. `diffusion_gemma_generate_ai100(...)`
   - legacy/single-QPC runtime:
     - denoises with combined graph that accepts prefix/canvas together.
3. `diffusion_gemma_generate_dispatch(...)`
   - top-level dispatcher for HF vs AI100 and split vs single QPC runtime selection.

### H. Coding Notes (implementation characteristics)
1. Canvas-block generation:
   - `max_new_tokens` is generated in chunks of `canvas_length`.
2. EOS handling:
   - EOS detection is per canvas block, with optional PAD fill after first EOS per sequence.
3. Metric definition:
   - `tokens_per_forward = valid_generated_tokens / decode_forward_passes`.
4. Input contract:
   - requires `inputs["input_ids"]`; uses `attention_mask` if provided, otherwise all-ones mask.
5. Optional warm starts:
   - supports `decoder_input_ids` and `self_conditioning_logits` as initial values for first block.
