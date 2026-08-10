# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Optional

import numpy as np
import torch

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.diffusion_gemma_utils import DiffusionGemmaRuntimeResult


@dataclass
class DiffusionGemmaSingleQPCRuntimeResult(DiffusionGemmaRuntimeResult):
    ttft: float
    retained_kv_buffers: int
    total_steps: int
    executed_blocks: int
    total_canvas_time: float
    canvas_length: int


def _to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _binding_dims(session: QAICInferenceSession, name: str):
    for binding in session.bindings:
        if binding.name == name:
            return tuple(binding.dims)
    return None


def _session_feed(session: QAICInferenceSession, feed: Dict[str, np.ndarray]):
    input_names = set(session.input_names)
    return {name: value for name, value in feed.items() if name in input_names}


def _normalize_eos_token_ids(eos_token_id):
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(token_id) for token_id in eos_token_id]


def _infer_context_length(session: QAICInferenceSession) -> Optional[int]:
    context_lengths = []
    for binding in session.bindings:
        if binding.name.startswith("past_") and len(binding.dims) > 2:
            context_lengths.append(int(binding.dims[2]))
    return max(context_lengths) if context_lengths else None


class DiffusionGemmaSingleQPCGenerator:
    def __init__(
        self,
        *,
        model_config,
        session: QAICInferenceSession,
        seed: Optional[int] = 1234,
    ):
        self.model_config = model_config
        self.session = session
        self.rng = np.random.RandomState(seed) if seed is not None and seed >= 0 else np.random.RandomState()

        input_dims = _binding_dims(self.session, "input_ids")
        canvas_dims = _binding_dims(self.session, "decoder_input_ids")
        if input_dims is None or canvas_dims is None:
            raise ValueError("The QPC must expose `input_ids` and `decoder_input_ids` bindings.")
        if int(input_dims[0]) != 1:
            raise ValueError("DiffusionGemma single-QPC generation currently supports batch size 1.")

        self.prefill_seq_len = int(input_dims[1])
        self.canvas_length = int(canvas_dims[1])
        self.vocab_size = int(model_config.text_config.vocab_size)
        self.position_ids = None
        self.input_ids = None
        self.mm_token_type_ids = None
        self.vision_embeds = None
        self.image_idx = None
        self.pad_token_id = 0

    def _next_encoder_position(self) -> int:
        valid_positions = self.position_ids[self.position_ids >= 0]
        return int(valid_positions.max()) + 1 if valid_positions.size else 0

    def _prepare_prompt(self, inputs, pad_token_id: int):
        input_ids = _to_numpy(inputs["input_ids"], np.int64)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("`input_ids` must have shape [1, sequence_length].")

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = _to_numpy(attention_mask, np.int64)
            sequence_length = int(attention_mask[0].sum())
        else:
            sequence_length = int(input_ids.shape[1])

        if sequence_length > self.prefill_seq_len:
            raise ValueError(
                f"Prompt has {sequence_length} tokens, exceeding compiled prefill length {self.prefill_seq_len}."
            )

        input_ids = input_ids[:, :sequence_length]
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            position_ids = np.arange(sequence_length, dtype=np.int64).reshape(1, -1)
        else:
            position_ids = _to_numpy(position_ids, np.int64)[:, :sequence_length]

        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = np.zeros_like(input_ids)
        else:
            mm_token_type_ids = _to_numpy(mm_token_type_ids, np.int64)[:, :sequence_length]

        padding = self.prefill_seq_len - sequence_length
        self.pad_token_id = int(pad_token_id)
        self.input_ids = np.pad(input_ids, ((0, 0), (0, padding)), constant_values=self.pad_token_id)
        self.position_ids = np.pad(position_ids, ((0, 0), (0, padding)), constant_values=-1)
        self.mm_token_type_ids = np.pad(mm_token_type_ids, ((0, 0), (0, padding)))
        vision_dims = _binding_dims(self.session, "vision_embeds")
        if vision_dims is not None:
            self.vision_embeds = np.zeros(vision_dims, dtype=np.float16)
            if inputs.get("vision_embeds") is not None:
                vision_embeds = _to_numpy(inputs["vision_embeds"], np.float16)
                self.vision_embeds[:, : vision_embeds.shape[1], :] = vision_embeds

        image_idx = inputs.get("image_idx")
        self.image_idx = (
            _to_numpy(image_idx, np.int64) if image_idx is not None else np.zeros((1, 1), dtype=np.int64)
        )
        return sequence_length

    def _base_feed(self):
        return {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "vision_embeds": self.vision_embeds,
            "image_idx": self.image_idx,
            "mm_token_type_ids": self.mm_token_type_ids,
        }

    def prefill(self):
        canvas_start = self._next_encoder_position()
        feed = {
            **self._base_feed(),
            "decoder_input_ids": np.zeros((1, self.canvas_length), dtype=np.int64),
            "decoder_position_ids": np.arange(
                canvas_start, canvas_start + self.canvas_length, dtype=np.int64
            ).reshape(1, -1),
            "self_conditioning_logits": np.zeros(
                (1, self.canvas_length, self.vocab_size), dtype=np.float32
            ),
            "is_encode": np.ones((1,), dtype=np.int64),
            "use_self_conditioning": np.zeros((1,), dtype=np.int64),
        }
        start = perf_counter()
        outputs = self.session.run(_session_feed(self.session, feed))
        retained_buffers = [
            name
            for name in self.session.input_names + self.session.output_names
            if name.startswith("past_")
        ]
        self.session.skip_buffers(retained_buffers)
        return perf_counter() - start, len([name for name in outputs if name.startswith("past_")])

    def _denoise_canvas(
        self,
        *,
        block_index: int,
        max_denoising_steps: int,
        sampler: str,
        entropy_bound: float,
        t_min: float,
        t_max: float,
        step_callback: Optional[Callable[[dict], None]],
    ):
        canvas_start = self._next_encoder_position()
        canvas = self.rng.randint(0, self.vocab_size, size=(1, self.canvas_length)).astype(np.int64)
        new_canvas = canvas.copy()
        canvas_positions = np.arange(
            canvas_start, canvas_start + self.canvas_length, dtype=np.int64
        ).reshape(1, -1)
        accepted_mask = np.zeros((1, self.canvas_length), dtype=bool)
        self_conditioning_logits = np.zeros(
            (1, self.canvas_length, self.vocab_size), dtype=np.float32
        )

        start = perf_counter()
        for step in range(max_denoising_steps):
            current_step = max_denoising_steps - step
            temperature = t_min + (t_max - t_min) * current_step / max_denoising_steps
            outputs = self.session.run(
                _session_feed(
                    self.session,
                    {
                        **self._base_feed(),
                        "decoder_input_ids": canvas,
                        "decoder_position_ids": canvas_positions,
                        "self_conditioning_logits": self_conditioning_logits,
                        "is_encode": np.zeros((1,), dtype=np.int64),
                        "use_self_conditioning": np.array([step > 0], dtype=np.int64),
                    },
                )
            )
            canvas_logits = outputs["canvas_logits"].astype(np.float32)
            self_conditioning_logits = canvas_logits
            temperature_logits = canvas_logits / max(temperature, 1e-6)
            uniform = self.rng.uniform(size=temperature_logits.shape).astype(np.float32)
            gumbel = -np.log(-np.log(uniform + 1e-20) + 1e-20)
            denoiser_canvas = (temperature_logits + gumbel).argmax(-1).astype(np.int64)

            shifted_logits = temperature_logits - temperature_logits.max(-1, keepdims=True)
            log_softmax = shifted_logits - np.log(np.exp(shifted_logits).sum(-1, keepdims=True))
            entropy = -(np.exp(log_softmax) * log_softmax).sum(-1)[0]
            entropy_order = np.argsort(entropy)
            selected = (np.cumsum(entropy[entropy_order]) - entropy[entropy_order]) <= entropy_bound
            newly_accepted = np.zeros(self.canvas_length, dtype=bool)
            newly_accepted[entropy_order[selected]] = True
            new_canvas = np.where(newly_accepted[None, :], denoiser_canvas, canvas)
            accepted_mask = accepted_mask | newly_accepted[None, :] if sampler == "local" else newly_accepted[None, :]
            canvas = np.where(
                ~accepted_mask,
                self.rng.randint(0, self.vocab_size, size=(1, self.canvas_length)).astype(np.int64),
                new_canvas,
            )

            accepted_count = int(accepted_mask.sum())
            if step_callback is not None:
                step_callback(
                    {
                        "block_index": block_index,
                        "step": step,
                        "temperature": temperature,
                        "accepted_count": accepted_count,
                        "canvas_length": self.canvas_length,
                        "tokens": new_canvas,
                    }
                )
            if accepted_count >= self.canvas_length:
                break

        return new_canvas, step + 1, perf_counter() - start, int(accepted_mask.sum())

    def _commit_canvas(self, tokens: np.ndarray):
        commit_length = int(tokens.shape[1])
        if commit_length > self.prefill_seq_len:
            raise ValueError(
                f"Commit length {commit_length} exceeds compiled prefill length {self.prefill_seq_len}."
            )

        commit_start = self._next_encoder_position()
        commit_input_ids = np.full((1, self.prefill_seq_len), self.pad_token_id, dtype=np.int64)
        commit_input_ids[:, :commit_length] = tokens
        commit_position_ids = np.full((1, self.prefill_seq_len), -1, dtype=np.int64)
        commit_position_ids[:, :commit_length] = np.arange(
            commit_start, commit_start + commit_length, dtype=np.int64
        )
        commit_mm_token_type_ids = np.zeros((1, self.prefill_seq_len), dtype=np.int64)
        self.session.run(
            _session_feed(
                self.session,
                {
                    "input_ids": commit_input_ids,
                    "position_ids": commit_position_ids,
                    "vision_embeds": self.vision_embeds,
                    "image_idx": np.zeros((1, 1), dtype=np.int64),
                    "mm_token_type_ids": commit_mm_token_type_ids,
                    "decoder_input_ids": np.zeros((1, self.canvas_length), dtype=np.int64),
                    "decoder_position_ids": np.arange(
                        commit_start, commit_start + self.canvas_length, dtype=np.int64
                    ).reshape(1, -1),
                    "self_conditioning_logits": np.zeros(
                        (1, self.canvas_length, self.vocab_size), dtype=np.float32
                    ),
                    "is_encode": np.ones((1,), dtype=np.int64),
                    "use_self_conditioning": np.zeros((1,), dtype=np.int64),
                },
            )
        )
        self.input_ids = commit_input_ids
        self.position_ids = commit_position_ids
        self.mm_token_type_ids = commit_mm_token_type_ids
        self.image_idx = np.zeros((1, 1), dtype=np.int64)

    def generate(
        self,
        *,
        inputs,
        generation_len: int,
        max_denoising_steps: int = 48,
        sampler: str = "local",
        entropy_bound: float = 0.1,
        t_min: float = 0.4,
        t_max: float = 0.8,
        ctx_len: Optional[int] = None,
        pad_token_id: int = 0,
        eos_token_id=None,
        stop_on_eos: bool = True,
        step_callback: Optional[Callable[[dict], None]] = None,
    ) -> DiffusionGemmaSingleQPCRuntimeResult:
        if generation_len <= 0:
            raise ValueError("`generation_len` must be positive.")
        if max_denoising_steps <= 0:
            raise ValueError("`max_denoising_steps` must be positive.")
        if sampler not in {"local", "hf"}:
            raise ValueError("`sampler` must be either 'local' or 'hf'.")
        if not 0 <= t_min <= t_max:
            raise ValueError("Temperature bounds must satisfy 0 <= t_min <= t_max.")

        total_start = perf_counter()
        try:
            sequence_length = self._prepare_prompt(inputs, pad_token_id=pad_token_id)
            compiled_ctx_len = ctx_len or _infer_context_length(self.session)
            target_new_tokens = int(generation_len)
            if compiled_ctx_len is not None:
                target_new_tokens = min(target_new_tokens, max(0, int(compiled_ctx_len) - sequence_length))
            if target_new_tokens <= 0:
                raise ValueError("The compiled context length has no room for generated tokens.")

            ttft, retained_kv_buffers = self.prefill()
            eos_token_ids = _normalize_eos_token_ids(eos_token_id)
            generated = []
            total_steps = 0
            total_canvas_time = 0.0
            num_blocks = int(math.ceil(target_new_tokens / self.canvas_length))

            for block_index in range(num_blocks):
                emitted_tokens = sum(tokens.shape[1] for tokens in generated)
                remaining_tokens = target_new_tokens - emitted_tokens
                canvas_tokens, steps_run, canvas_time, _ = self._denoise_canvas(
                    block_index=block_index,
                    max_denoising_steps=max_denoising_steps,
                    sampler=sampler,
                    entropy_bound=entropy_bound,
                    t_min=t_min,
                    t_max=t_max,
                    step_callback=step_callback,
                )
                total_steps += steps_run
                total_canvas_time += canvas_time
                canvas_tokens = canvas_tokens[:, :remaining_tokens]

                hit_eos = False
                if stop_on_eos and eos_token_ids:
                    eos_positions = np.where(np.isin(canvas_tokens[0], eos_token_ids))[0]
                    if eos_positions.size:
                        canvas_tokens = canvas_tokens[:, : int(eos_positions[0]) + 1]
                        hit_eos = True

                generated.append(canvas_tokens)
                if hit_eos:
                    break
                if block_index + 1 < num_blocks and canvas_tokens.shape[1] > 0:
                    self._commit_canvas(canvas_tokens)

            generated_ids = (
                np.concatenate(generated, axis=1) if generated else np.zeros((1, 0), dtype=np.int64)
            )
            valid_tokens = np.array([generated_ids.shape[1]], dtype=np.float32)
            decode_forward_passes = np.array([total_steps], dtype=np.int64)
            tokens_per_forward = valid_tokens / np.maximum(decode_forward_passes.astype(np.float32), 1.0)
            return DiffusionGemmaSingleQPCRuntimeResult(
                generated_ids=generated_ids,
                tokens_per_forward=tokens_per_forward,
                decode_forward_passes=decode_forward_passes,
                total_time=max(perf_counter() - total_start, 1e-6),
                ttft=ttft,
                retained_kv_buffers=retained_kv_buffers,
                total_steps=total_steps,
                executed_blocks=len(generated),
                total_canvas_time=total_canvas_time,
                canvas_length=self.canvas_length,
            )
        finally:
            self.session.deactivate()


def diffusion_gemma_generate_single_qpc(
    *,
    model_config,
    inputs,
    session: QAICInferenceSession,
    generation_len: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    seed: Optional[int] = 1234,
    **kwargs,
) -> DiffusionGemmaSingleQPCRuntimeResult:
    requested_generation_len = generation_len if generation_len is not None else max_new_tokens
    if requested_generation_len is None:
        requested_generation_len = 256
    generator = DiffusionGemmaSingleQPCGenerator(
        model_config=model_config,
        session=session,
        seed=seed,
    )
    return generator.generate(inputs=inputs, generation_len=int(requested_generation_len), **kwargs)
