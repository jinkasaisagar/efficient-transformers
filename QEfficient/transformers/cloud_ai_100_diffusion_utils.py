import math
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name
from QEfficient.transformers.diffusion_gemma_utils import (
    DiffusionGemmaGenerateDispatch,
    DiffusionGemmaRuntimeResult,
    EntropyBoundSampler,
    EntropyBoundSamplerConfig,
    LinearTemperatureScheduleLogitsProcessor,
    StableAndConfidentStoppingCriteria,
    _get_allowed_seq_lens_for_input,
    _pad_or_truncate_prefix,
    _pick_target_seq_len,
    _build_decoder_kv_inputs,
    _normalize_eos_token_ids,
    _prepare_runtime_generation_config,
    _run_encoder_block,
)


@torch.no_grad()
def _prepare_denoiser_inputs(
    *,
    model_config,
    batch_size: int,
    canvas_length: int,
    block_idx: int,
    initial_decoder_input_ids: Optional[np.ndarray],
    initial_self_conditioning_logits: Optional[np.ndarray],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    vocab_size = model_config.text_config.vocab_size
    if initial_decoder_input_ids is not None and block_idx == 0:
        current_canvas = torch.from_numpy(initial_decoder_input_ids).to(torch.int64)
    else:
        current_canvas = torch.randint(0, vocab_size, size=(batch_size, canvas_length), dtype=torch.int64)

    if initial_self_conditioning_logits is not None and block_idx == 0:
        self_conditioning_logits = torch.from_numpy(initial_self_conditioning_logits)
    else:
        self_conditioning_logits = None

    return current_canvas, self_conditioning_logits


@torch.no_grad()
def _denoising_step(
    *,
    qpc_session: QAICInferenceSession,
    model_config,
    kv_inputs: Dict[str, np.ndarray],
    current_canvas: torch.Tensor,
    argmax_canvas: torch.Tensor,
    self_conditioning_logits: Optional[torch.Tensor],
    cur_step: int,
    sampler: EntropyBoundSampler,
    logits_processor: LinearTemperatureScheduleLogitsProcessor,
    stopper: Optional[StableAndConfidentStoppingCriteria],
    finished_denoising: torch.Tensor,
    decoder_attention_mask: Optional[np.ndarray] = None,
):
    vocab_size = model_config.text_config.vocab_size
    input_names = set(getattr(qpc_session, "input_names", []))

    model_inputs = {
        "decoder_input_ids": current_canvas.cpu().numpy().astype(np.int64),
        **kv_inputs,
    }
    if "is_encode" in input_names:
        model_inputs["is_encode"] = np.ones((2,), dtype=np.int64)
    if decoder_attention_mask is not None and "decoder_attention_mask" in input_names:
        model_inputs["decoder_attention_mask"] = decoder_attention_mask.astype(np.int64)
    if self_conditioning_logits is not None and "self_conditioning_logits" in input_names:
        model_inputs["self_conditioning_logits"] = self_conditioning_logits.cpu().numpy().astype(np.float32)

    model_outputs = qpc_session.run(model_inputs)
    raw_logits = torch.from_numpy(model_outputs["logits"])
    processed_logits = logits_processor(scores=raw_logits, cur_step=cur_step)

    probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
    batch_size, canvas_length = current_canvas.shape
    denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
    denoiser_canvas = denoiser_canvas.squeeze(-1).view(batch_size, canvas_length)
    new_argmax_canvas = torch.argmax(processed_logits, dim=-1).to(torch.int64)

    accepted_canvas = sampler.accept_canvas(
        current_canvas=current_canvas,
        denoiser_canvas=denoiser_canvas,
        logits=processed_logits,
    )
    new_current_canvas = sampler.renoise_canvas(accepted_canvas).to(torch.int64)

    if stopper is not None:
        if finished_denoising.any():
            new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
            new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
            if self_conditioning_logits is not None:
                processed_logits = torch.where(
                    finished_denoising[:, None, None],
                    self_conditioning_logits,
                    processed_logits,
                )
        finished_denoising |= stopper(new_argmax_canvas, processed_logits)

    return new_current_canvas, new_argmax_canvas, processed_logits, finished_denoising


@torch.no_grad()
def _run_decoder_denoising_loop(
    *,
    qpc_session: QAICInferenceSession,
    model_config,
    kv_cache: Dict[str, np.ndarray],
    current_canvas: torch.Tensor,
    self_conditioning_logits: Optional[torch.Tensor],
    max_denoising_steps: int,
    sampler: EntropyBoundSampler,
    logits_processor: LinearTemperatureScheduleLogitsProcessor,
    stopper: Optional[StableAndConfidentStoppingCriteria],
    finished_sequences: torch.Tensor,
    decoder_forward_passes: torch.Tensor,
    decoder_attention_mask: Optional[np.ndarray] = None,
) -> torch.Tensor:
    kv_inputs = _build_decoder_kv_inputs(decoder_session=qpc_session, kv_cache=kv_cache)
    argmax_canvas = current_canvas.clone()
    finished_denoising = torch.zeros_like(finished_sequences, dtype=torch.bool)

    if stopper is not None:
        stopper.reset()

    for cur_step in reversed(range(1, max_denoising_steps + 1)):
        decoder_forward_passes += (~(finished_denoising | finished_sequences)).to(torch.int64)
        current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising = _denoising_step(
            qpc_session=qpc_session,
            model_config=model_config,
            kv_inputs=kv_inputs,
            current_canvas=current_canvas,
            argmax_canvas=argmax_canvas,
            self_conditioning_logits=self_conditioning_logits,
            cur_step=cur_step,
            sampler=sampler,
            logits_processor=logits_processor,
            stopper=stopper,
            finished_denoising=finished_denoising,
            decoder_attention_mask=decoder_attention_mask,
        )
        if torch.all(finished_denoising):
            break

    return argmax_canvas


@torch.no_grad()
def _finalize_canvas(
    *,
    input_ids_t: torch.Tensor,
    finished_sequences: torch.Tensor,
    canvas_length: int,
    eos_ids: Optional[list],
    pad_token_id: Optional[int],
):
    if finished_sequences.any() and pad_token_id is not None:
        input_ids_t[finished_sequences, -canvas_length:] = int(pad_token_id)

    if eos_ids is None:
        return input_ids_t, finished_sequences

    new_tokens = input_ids_t[:, -canvas_length:]
    eos_tensor = torch.tensor(eos_ids, dtype=new_tokens.dtype, device=new_tokens.device)
    is_eos = torch.isin(new_tokens, eos_tensor)
    finished_this_canvas = is_eos.any(dim=-1)
    just_finished = (~finished_sequences) & finished_this_canvas
    if torch.any(just_finished) and pad_token_id is not None:
        eos_cumsum = torch.cumsum(is_eos.to(torch.int64), dim=-1)
        pad_mask = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)
        new_tokens[just_finished] = torch.where(
            pad_mask[just_finished],
            torch.full_like(new_tokens[just_finished], int(pad_token_id)),
            new_tokens[just_finished],
        )
        input_ids_t[:, -canvas_length:] = new_tokens
    finished_sequences |= finished_this_canvas
    return input_ids_t, finished_sequences


@torch.no_grad()
def _cloud_ai_100_diffusion_generate_single_qpc(
    qpc_session: QAICInferenceSession,
    model_config,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int,
    max_denoising_steps: int,
    entropy_bound: float,
    t_min: float,
    t_max: float,
    stability_threshold: int,
    confidence_threshold: float,
    pad_token_id: Optional[int],
    eos_token_id,
    initial_decoder_input_ids: Optional[np.ndarray] = None,
    initial_self_conditioning_logits: Optional[np.ndarray] = None,
) -> DiffusionGemmaRuntimeResult:
    start_time = perf_counter()
    batch_size = input_ids.shape[0]
    canvas_length = int(getattr(model_config, "canvas_length", 256))
    max_new_canvases = int(math.ceil(max_new_tokens / canvas_length))
    eos_ids = _normalize_eos_token_ids(eos_token_id)

    input_ids_t = torch.from_numpy(input_ids).to(torch.int64)
    attention_mask_t = torch.from_numpy(attention_mask).to(torch.int64)
    finished_sequences = torch.zeros((batch_size,), dtype=torch.bool)
    decoder_forward_passes = torch.zeros((batch_size,), dtype=torch.int64)

    stopper = None
    if stability_threshold is not None and confidence_threshold is not None:
        stopper = StableAndConfidentStoppingCriteria(
            stability_threshold=stability_threshold,
            confidence_threshold=confidence_threshold,
        )

    is_prefill = True
    for block_idx in range(max_new_canvases):
        print(f'Number of canvases is {block_idx}')
        if torch.all(finished_sequences):
            break

        # First block prefill uses full prefix; subsequent blocks encode only new canvas tokens.
        raw_encoder_input_ids = input_ids_t if is_prefill else input_ids_t[:, -canvas_length:]
        raw_encoder_attention_mask = attention_mask_t if is_prefill else attention_mask_t[:, -canvas_length:]
        allowed_seq_lens = _get_allowed_seq_lens_for_input(qpc_session, "input_ids")
        target_seq_len = 256#_pick_target_seq_len(allowed_seq_lens, int(raw_encoder_input_ids.shape[1]))
        encoder_ids_np, encoder_mask_np = _pad_or_truncate_prefix(
            raw_encoder_input_ids.cpu().numpy().astype(np.int64),
            raw_encoder_attention_mask.cpu().numpy().astype(np.int64),
            target_seq_len=target_seq_len,
        )
        encoder_input_ids = torch.from_numpy(encoder_ids_np).to(torch.int64)
        encoder_attention_mask = torch.from_numpy(encoder_mask_np).to(torch.int64)
        encoder_input_ids = input_ids
        hidden_states = _run_encoder_block(
            encoder_session=qpc_session,
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )
        return hidden_states
        is_prefill = False

        current_canvas, self_conditioning_logits = _prepare_denoiser_inputs(
            model_config=model_config,
            batch_size=batch_size,
            canvas_length=canvas_length,
            block_idx=block_idx,
            initial_decoder_input_ids=encoder_input_ids,
            # initial_decoder_input_ids=initial_decoder_input_ids,
            initial_self_conditioning_logits=initial_self_conditioning_logits,
        )
        vocab_size = model_config.text_config.vocab_size

        sampler = EntropyBoundSampler(
            config=EntropyBoundSamplerConfig(entropy_bound=entropy_bound),
            canvas_length=canvas_length,
            vocab_size=vocab_size,
        )
        logits_processor = LinearTemperatureScheduleLogitsProcessor(
            t_min=t_min,
            t_max=t_max,
            max_denoising_steps=max_denoising_steps,
        )
        # decoder_attention_mask = encoder_mask_np

        denoised_canvas = _run_decoder_denoising_loop(
            qpc_session=qpc_session,
            model_config=model_config,
            kv_cache=kv_cache,
            current_canvas=current_canvas,
            self_conditioning_logits=self_conditioning_logits,
            max_denoising_steps=max_denoising_steps,
            sampler=sampler,
            logits_processor=logits_processor,
            stopper=stopper,
            finished_sequences=finished_sequences,
            decoder_forward_passes=decoder_forward_passes,
            # decoder_attention_mask=decoder_attention_mask,
        )

        input_ids_t = torch.cat([input_ids_t, denoised_canvas], dim=-1)
        input_ids_t, finished_sequences = _finalize_canvas(
            input_ids_t=input_ids_t,
            finished_sequences=finished_sequences,
            canvas_length=canvas_length,
            eos_ids=eos_ids,
            pad_token_id=pad_token_id,
        )

        attention_mask_t = torch.cat(
            [attention_mask_t, torch.ones((batch_size, canvas_length), dtype=attention_mask_t.dtype)],
            dim=-1,
        )

    new_tokens = input_ids_t[:, input_ids.shape[1] : input_ids.shape[1] + max_new_tokens]
    if pad_token_id is not None:
        num_valid_tokens = (new_tokens != int(pad_token_id)).sum(dim=-1).to(torch.float32)
    else:
        num_valid_tokens = torch.full((batch_size,), float(new_tokens.shape[1]), dtype=torch.float32)

    denom = torch.clamp(decoder_forward_passes.to(torch.float32), min=1.0)
    tokens_per_forward = (num_valid_tokens / denom).cpu().numpy()
    total_time = max(float(perf_counter() - start_time), 1e-6)

    return DiffusionGemmaRuntimeResult(
        generated_ids=new_tokens.cpu().numpy().astype(np.int64),
        tokens_per_forward=tokens_per_forward,
        decode_forward_passes=decoder_forward_passes.cpu().numpy().astype(np.int64),
        total_time=total_time,
    )


def cloud_ai_100_diffusion_generate_dispatch(
    model,
    inputs,
    runtime_ai100: bool,
    device_id,
    qpc_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> DiffusionGemmaGenerateDispatch:
    generation_config = _prepare_runtime_generation_config(model=model, kwargs=kwargs)
    decoder_input_ids = kwargs.pop("decoder_input_ids", None)
    self_conditioning_logits = kwargs.pop("self_conditioning_logits", None)
    input_ids_tensor = inputs.get("input_ids", None)
    
    if isinstance(input_ids_tensor, np.ndarray):
        input_ids = input_ids_tensor.astype(np.int64)
    else:
        input_ids = input_ids_tensor.cpu().numpy().astype(np.int64)

    attention_mask_tensor = inputs.get("attention_mask", None)
    if attention_mask_tensor is None:
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
    elif isinstance(attention_mask_tensor, np.ndarray):
        attention_mask = attention_mask_tensor.astype(np.int64)
    else:
        attention_mask = attention_mask_tensor.cpu().numpy().astype(np.int64)

    if isinstance(decoder_input_ids, torch.Tensor):
        decoder_input_ids = decoder_input_ids.cpu().numpy().astype(np.int64)
    if isinstance(self_conditioning_logits, torch.Tensor):
        self_conditioning_logits = self_conditioning_logits.cpu().numpy().astype(np.float32)

    if runtime_ai100:
        if qpc_path is None:
            raise ValueError("Pass `qpc_path` for single-QPC diffusion generation.")

        qpc_session = QAICInferenceSession(str(qpc_path), device_id or None)
        qpc_session.skip_buffers(
            [
                x
                for x in qpc_session.input_names + qpc_session.output_names
                if is_retained_state_name(x) or x.endswith("_RetainedState")
            ]
        )
        runtime_result = _cloud_ai_100_diffusion_generate_single_qpc(
            qpc_session=qpc_session,
            model_config=model.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_config.max_new_tokens,
            max_denoising_steps=generation_config.max_denoising_steps,
            entropy_bound=generation_config.sampler_config.entropy_bound,
            t_min=generation_config.t_min,
            t_max=generation_config.t_max,
            stability_threshold=generation_config.stability_threshold,
            confidence_threshold=generation_config.confidence_threshold,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            initial_decoder_input_ids=decoder_input_ids,
            initial_self_conditioning_logits=self_conditioning_logits,
        )
        return DiffusionGemmaGenerateDispatch(runtime_result=runtime_result, hf_output=None)

    hf_output = model.generate(**inputs, **kwargs)
    return DiffusionGemmaGenerateDispatch(runtime_result=None, hf_output=hf_output)
