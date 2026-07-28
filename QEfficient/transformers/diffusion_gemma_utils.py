import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import torch

from QEfficient.generation.cloud_infer import QAICInferenceSession


@dataclass
class DiffusionGemmaRuntimeResult:
    generated_ids: np.ndarray
    tokens_per_forward: np.ndarray
    decode_forward_passes: np.ndarray
    total_time: float


@dataclass
class DiffusionGemmaGenerateDispatch:
    runtime_result: Optional[DiffusionGemmaRuntimeResult]
    hf_output: Optional[object]


@dataclass
class EntropyBoundSamplerConfig:
    entropy_bound: float = 0.1

    def __post_init__(self):
        if not isinstance(self.entropy_bound, (float, int)) or float(self.entropy_bound) <= 0:
            raise ValueError(f"`entropy_bound` must be a positive float, got {self.entropy_bound}")
        self.entropy_bound = float(self.entropy_bound)


@dataclass
class DiffusionGemmaGenerationConfig:
    max_new_tokens: int = 256
    max_denoising_steps: int = 48
    sampler_config: EntropyBoundSamplerConfig = field(
        default_factory=lambda: EntropyBoundSamplerConfig(entropy_bound=0.1)
    )
    t_min: float = 0.4
    t_max: float = 0.8
    stability_threshold: int = 1
    confidence_threshold: float = 0.005
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[object] = None

    def validate(self):
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if not isinstance(self.max_denoising_steps, int) or self.max_denoising_steps <= 0:
            raise ValueError("`max_denoising_steps` must be a positive integer.")
        if self.t_min < 0 or self.t_max < 0 or self.t_max < self.t_min:
            raise ValueError("Temperature schedule must satisfy: 0 <= t_min <= t_max.")
        if not isinstance(self.stability_threshold, int) or self.stability_threshold < 0:
            raise ValueError("`stability_threshold` must be an integer >= 0.")
        if float(self.confidence_threshold) <= 0:
            raise ValueError("`confidence_threshold` must be > 0.")
        if not isinstance(self.sampler_config, EntropyBoundSamplerConfig):
            raise ValueError("`sampler_config` must be EntropyBoundSamplerConfig.")


@dataclass
class DiffusionGemmaGenerationOutput:
    sequences: torch.LongTensor
    tokens_per_forward: Optional[torch.Tensor] = None


class LinearTemperatureScheduleLogitsProcessor:
    def __init__(self, t_min: float, t_max: float, max_denoising_steps: int):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.max_denoising_steps = int(max_denoising_steps)

    def __call__(self, scores: torch.FloatTensor, cur_step: int) -> torch.FloatTensor:
        temperature = self.t_min + ((self.t_max - self.t_min) * (float(cur_step) / float(self.max_denoising_steps)))
        return scores / max(temperature, 1e-6)


class EntropyBoundSampler:
    def __init__(self, config: EntropyBoundSamplerConfig, canvas_length: int, vocab_size: int):
        self.entropy_bound = float(config.entropy_bound)
        self.canvas_length = int(canvas_length)
        self.vocab_size = int(vocab_size)
        self.accepted_token_mask = None

    def initialize_canvas(self, batch_size: int, device: torch.device) -> torch.LongTensor:
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, self.canvas_length),
            dtype=torch.int64,
            device=device,
        )

    def accept_canvas(
        self,
        current_canvas: torch.LongTensor,
        denoiser_canvas: torch.LongTensor,
        logits: torch.FloatTensor,
    ) -> torch.LongTensor:
        self.accepted_token_mask = diffusion_gemma_entropy_accept_mask(logits, entropy_bound=self.entropy_bound)
        return torch.where(self.accepted_token_mask, denoiser_canvas, current_canvas)

    def renoise_canvas(self, accepted_canvas: torch.LongTensor) -> torch.LongTensor:
        if self.accepted_token_mask is None:
            return accepted_canvas
        random_canvas = self.initialize_canvas(accepted_canvas.shape[0], accepted_canvas.device)
        return torch.where(~self.accepted_token_mask, random_canvas, accepted_canvas)


class DiffusionGemmaAdaptiveStopping(ABC):
    @abstractmethod
    def __call__(self, argmax_canvas: torch.Tensor, logits: torch.Tensor) -> torch.BoolTensor:
        pass

    def reset(self):
        pass


class StableAndConfidentStoppingCriteria(DiffusionGemmaAdaptiveStopping):
    def __init__(self, stability_threshold: int, confidence_threshold: float):
        self.stability_threshold = max(int(stability_threshold), 0)
        self.confidence_threshold = float(confidence_threshold)
        self.argmax_canvas_history: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset(self):
        self.argmax_canvas_history = None

    @torch.no_grad()
    def __call__(self, argmax_canvas: torch.Tensor, logits: torch.Tensor) -> torch.BoolTensor:
        batch_size = logits.shape[0]
        if self.stability_threshold == 0:
            stable = torch.ones((batch_size,), dtype=torch.bool, device=logits.device)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas

        entropy = torch.distributions.Categorical(logits=logits).entropy().mean(dim=-1)
        confident = entropy <= self.confidence_threshold
        return stable & confident


# Backward-compatible alias for existing code paths.
StableAndConfidentStopper = StableAndConfidentStoppingCriteria


def _normalize_eos_token_ids(eos_token_id) -> Optional[list[int]]:
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple)):
        return [int(x) for x in eos_token_id]
    return [int(eos_token_id)]


def _prepare_runtime_generation_config(
    model,
    kwargs: dict,
) -> DiffusionGemmaGenerationConfig:
    generation_len = kwargs.pop("generation_len", kwargs.pop("max_new_tokens", 256))
    max_denoising_steps = kwargs.pop("max_denoising_steps", 48)
    entropy_bound = kwargs.pop("entropy_bound", 0.1)
    t_min = kwargs.pop("t_min", 0.4)
    t_max = kwargs.pop("t_max", 0.8)
    stability_threshold = kwargs.pop("stability_threshold", 1)
    confidence_threshold = kwargs.pop("confidence_threshold", 0.005)

    pad_token_id = kwargs.pop("pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)

    eos_token_id = kwargs.pop("eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)

    generation_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=int(generation_len),
        max_denoising_steps=int(max_denoising_steps),
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=float(entropy_bound)),
        t_min=float(t_min),
        t_max=float(t_max),
        stability_threshold=int(stability_threshold),
        confidence_threshold=float(confidence_threshold),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    generation_config.validate()
    return generation_config


def _retained_output_to_state_input(name: str) -> str:
    if name.endswith("_InternalRetainedState"):
        return name[: -len("_InternalRetainedState")]
    if name.endswith("_RetainedState"):
        return name[: -len("_RetainedState")]
    return name


def _collect_kv_cache_from_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cache = {}
    for name, value in outputs.items():
        if "past_key." in name or "past_value." in name:
            cache[_retained_output_to_state_input(name)] = value
    return cache


def _find_matching_kv_tensor(input_name: str, kv_cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if input_name in kv_cache:
        return kv_cache[input_name]

    basename = input_name.rsplit("/", 1)[-1]
    if basename in kv_cache:
        return kv_cache[basename]

    candidates = [input_name, basename]
    for name in list(candidates):
        if name.endswith("_InternalRetainedState"):
            base = name[: -len("_InternalRetainedState")]
            candidates.extend([base, f"{base}_RetainedState"])
        elif name.endswith("_RetainedState"):
            base = name[: -len("_RetainedState")]
            candidates.extend([base, f"{base}_InternalRetainedState"])

    for candidate in candidates:
        if candidate in kv_cache:
            return kv_cache[candidate]
    return None


def _align_to_binding_shape(
    array: np.ndarray,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != expected_dtype:
        arr = arr.astype(expected_dtype, copy=False)

    if tuple(arr.shape) == tuple(expected_shape):
        return arr

    if arr.ndim != len(expected_shape):
        raise ValueError(
            f"KV rank mismatch: got {arr.shape}, expected rank {len(expected_shape)} for shape {expected_shape}."
        )

    aligned = np.zeros(expected_shape, dtype=expected_dtype)
    slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(arr.shape, expected_shape))
    aligned[slices] = arr[slices]
    return aligned


def _get_allowed_seq_lens_for_input(qpc_session: QAICInferenceSession, input_name: str) -> list[int]:
    if not hasattr(qpc_session, "binding_index_map") or input_name not in qpc_session.binding_index_map:
        return []
    if not hasattr(qpc_session, "allowed_shapes"):
        return []

    binding_index = qpc_session.binding_index_map[input_name]
    seq_lens = []
    for allowed in getattr(qpc_session, "allowed_shapes", []):
        try:
            dims = allowed[binding_index][1]
            if len(dims) >= 2:
                seq_lens.append(int(dims[1]))
        except Exception:
            continue
    return sorted(set(seq_lens))


def _pick_target_seq_len(seq_lens: list[int], current_len: int) -> int:
    if not seq_lens:
        return int(current_len)
    for seq_len in seq_lens:
        if current_len <= seq_len:
            return int(seq_len)
    return int(max(seq_lens))


def _pad_or_truncate_prefix(
    ids: np.ndarray, mask: np.ndarray, target_seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.int64)
    if ids.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"Expected 2D prefix ids/mask, got ids={ids.shape}, mask={mask.shape}")
    if ids.shape != mask.shape:
        raise ValueError(f"Prefix ids/mask shape mismatch: ids={ids.shape}, mask={mask.shape}")

    cur_len = ids.shape[1]
    if cur_len == target_seq_len:
        return ids, mask
    if cur_len > target_seq_len:
        return ids[:, -target_seq_len:], mask[:, -target_seq_len:]

    pad_len = target_seq_len - cur_len
    ids_pad = np.zeros((ids.shape[0], pad_len), dtype=np.int64)
    mask_pad = np.zeros((mask.shape[0], pad_len), dtype=np.int64)
    return np.concatenate([ids, ids_pad], axis=1), np.concatenate([mask, mask_pad], axis=1)


def _build_decoder_kv_inputs(
    decoder_session: QAICInferenceSession, kv_cache: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    decoder_inputs: Dict[str, np.ndarray] = {}
    for input_name in getattr(decoder_session, "input_names", []):
        if "past_key." in input_name or "past_value." in input_name:
            kv_tensor = _find_matching_kv_tensor(input_name, kv_cache)
            if kv_tensor is None:
                continue
            if input_name not in decoder_session.binding_index_map:
                continue

            binding_idx = decoder_session.binding_index_map[input_name]
            binding = decoder_session.bindings[binding_idx]
            expected_shape = tuple(int(x) for x in binding.dims)
            expected_dtype = decoder_session.aic_to_np_dtype_mapping[binding.type]
            decoder_inputs[input_name] = _align_to_binding_shape(
                kv_tensor,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
            )
    return decoder_inputs


@torch.no_grad()
def _run_encoder_block(
    encoder_session: QAICInferenceSession,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    encoder_inputs = {"input_ids": input_ids}
    # encoder_inputs = {"input_ids": input_ids.cpu().numpy().astype(np.int64)}
    input_names = set(getattr(encoder_session, "input_names", []))
    print(input_names)
    
    if "is_encode" in input_names:
        encoder_inputs["is_encode"] = np.ones((1,), dtype=np.int64)
    # if attention_mask is not None and "attention_mask" in input_names:
    #     encoder_inputs["attention_mask"] = attention_mask.cpu().numpy().astype(np.int64)
    encoder_inputs["position_ids"] = np.arange(0,256).reshape(1,-1)
    encoder_outputs = encoder_session.run(encoder_inputs)
    breakpoint()
    return encoder_outputs['hidden_states']#, _collect_kv_cache_from_outputs(encoder_outputs)


@torch.no_grad()
def diffusion_gemma_entropy_accept_mask(logits: torch.Tensor, entropy_bound: float) -> torch.BoolTensor:
    dist = torch.distributions.Categorical(logits=logits)
    token_entropy = dist.entropy()
    sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
    cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
    sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= entropy_bound
    return torch.scatter(
        input=torch.zeros_like(sorted_selection_mask),
        dim=-1,
        index=sorted_indices,
        src=sorted_selection_mask,
    )


class StableAndConfidentStopper:
    def __init__(self, stability_threshold: int, confidence_threshold: float):
        self.stability_threshold = max(int(stability_threshold), 0)
        self.confidence_threshold = float(confidence_threshold)
        self.argmax_canvas_history: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset(self):
        self.argmax_canvas_history = None

    @torch.no_grad()
    def __call__(self, argmax_canvas: torch.Tensor, logits: torch.Tensor) -> torch.BoolTensor:
        batch_size = logits.shape[0]
        if self.stability_threshold == 0:
            stable = torch.ones((batch_size,), dtype=torch.bool, device=logits.device)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas

        entropy = torch.distributions.Categorical(logits=logits).entropy().mean(dim=-1)
        confident = entropy <= self.confidence_threshold
        return stable & confident

'''
@torch.no_grad()
def _run_denoising_step(
    qpc_session,
    model_config,
    prefix_ids: np.ndarray,
    prefix_mask: np.ndarray,
    current_canvas: torch.Tensor,
    decoder_attention_mask: np.ndarray,
    self_conditioning_logits: Optional[torch.Tensor],
    cur_step: int,
    max_denoising_steps: int,
    entropy_bound: float,
    t_min: float,
    t_max: float,
):
    step_frac = float(cur_step) / float(max_denoising_steps)
    temperature = float(t_min + ((t_max - t_min) * step_frac))
    input_names = set(getattr(qpc_session, "input_names", []))
    allowed_seq_lens = _get_allowed_seq_lens_for_input(qpc_session, "input_ids")
    target_prefix_seq_len = _pick_target_seq_len(allowed_seq_lens, int(prefix_ids.shape[1]))
    prefix_ids_aligned, prefix_mask_aligned = _pad_or_truncate_prefix(
        prefix_ids,
        prefix_mask,
        target_seq_len=target_prefix_seq_len,
    )

    model_inputs = {"decoder_input_ids": current_canvas.cpu().numpy().astype(np.int64)}
    if "input_ids" in input_names:
        model_inputs["input_ids"] = prefix_ids_aligned
    if "is_encode" in input_names:
        model_inputs["is_encode"] = np.ones((2,), dtype=np.int64)
    if "attention_mask" in input_names:
        model_inputs["attention_mask"] = prefix_mask_aligned
    if "position_ids" in input_names:
        batch_size = prefix_ids_aligned.shape[0]
        model_inputs["position_ids"] = np.broadcast_to(
            np.arange(target_prefix_seq_len, dtype=np.int64),
            (batch_size, target_prefix_seq_len),
        ).copy()
    if "decoder_attention_mask" in input_names:
        model_inputs["decoder_attention_mask"] = decoder_attention_mask
    if "self_conditioning_mask" in input_names:
        model_inputs["self_conditioning_mask"] = np.ones((current_canvas.shape[0],), dtype=np.bool_)
    if self_conditioning_logits is not None and "self_conditioning_logits" in input_names:
        model_inputs["self_conditioning_logits"] = self_conditioning_logits.cpu().numpy()

    print(f'In diffusion utils step, {prefix_ids_aligned.shape}, mask: {prefix_mask_aligned.shape}, decoder_input_ids: {current_canvas.shape}')
    model_outputs = qpc_session.run(model_inputs)
    logits = torch.from_numpy(model_outputs["logits"])
    processed_logits = logits / max(temperature, 1e-6)

    vocab_size = model_config.text_config.vocab_size
    probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
    batch_size, canvas_length = current_canvas.shape
    denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
    denoiser_canvas = denoiser_canvas.squeeze(-1).view(batch_size, canvas_length)
    argmax_canvas = torch.argmax(processed_logits, dim=-1).to(torch.int64)

    accepted_mask = diffusion_gemma_entropy_accept_mask(processed_logits, entropy_bound=entropy_bound)
    accepted_canvas = torch.where(accepted_mask, denoiser_canvas, current_canvas)
    random_canvas = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, canvas_length),
        dtype=torch.int64,
        device=accepted_canvas.device,
    )
    next_canvas = torch.where(~accepted_mask, random_canvas, accepted_canvas).to(torch.int64)
    return next_canvas, argmax_canvas, processed_logits
'''



@torch.no_grad()
def diffusion_gemma_generate_ai100(
    qpc_session,
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
        stopper = StableAndConfidentStopper(stability_threshold=stability_threshold, confidence_threshold=confidence_threshold)

    for _ in range(max_new_canvases):
        if torch.all(finished_sequences):
            break
        
        # print(model_config)
        vocab_size = model_config.text_config.vocab_size #int(getattr(model_config.text_config, "vocab_size", model_config.vocab_size))
        if initial_decoder_input_ids is not None:
            current_canvas = torch.from_numpy(initial_decoder_input_ids).to(torch.int64)
            initial_decoder_input_ids = None
        else:
            current_canvas = torch.randint(0, vocab_size, size=(batch_size, canvas_length), dtype=torch.int64)
        argmax_canvas = current_canvas.clone()
        if initial_self_conditioning_logits is not None:
            self_conditioning_logits = torch.from_numpy(initial_self_conditioning_logits)
            initial_self_conditioning_logits = None
        else:
            self_conditioning_logits = None
        finished_denoising = torch.zeros((batch_size,), dtype=torch.bool)
        if stopper is not None:
            stopper.reset()

        decoder_attention_mask = torch.ones((batch_size, input_ids_t.shape[1] + canvas_length), dtype=torch.int64).numpy()
        decoder_attention_mask[:, : attention_mask_t.shape[1]] = attention_mask_t.numpy()
        prefix_ids_np = input_ids_t.numpy().astype(np.int64)
        prefix_mask_np = attention_mask_t.numpy().astype(np.int64)

        for cur_step in reversed(range(1, max_denoising_steps + 1)):
            decoder_forward_passes += (~(finished_denoising | finished_sequences)).to(torch.int64)
            next_canvas, new_argmax_canvas, processed_logits = _run_denoising_step(
                qpc_session=qpc_session,
                model_config=model_config,
                prefix_ids=prefix_ids_np,
                prefix_mask=prefix_mask_np,
                current_canvas=current_canvas,
                decoder_attention_mask=decoder_attention_mask,
                self_conditioning_logits=self_conditioning_logits,
                cur_step=cur_step,
                max_denoising_steps=max_denoising_steps,
                entropy_bound=entropy_bound,
                t_min=t_min,
                t_max=t_max,
            )

            if stopper is not None:
                if finished_denoising.any():
                    new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
                    next_canvas = torch.where(finished_denoising[:, None], current_canvas, next_canvas)
                    if self_conditioning_logits is not None:
                        processed_logits = torch.where(
                            finished_denoising[:, None, None],
                            self_conditioning_logits,
                            processed_logits,
                        )
                finished_denoising |= stopper(new_argmax_canvas, processed_logits)

            current_canvas = next_canvas
            argmax_canvas = new_argmax_canvas
            self_conditioning_logits = processed_logits

            if torch.all(finished_denoising):
                break

        input_ids_t = torch.cat([input_ids_t, argmax_canvas], dim=-1)

        if finished_sequences.any() and pad_token_id is not None:
            input_ids_t[finished_sequences, -canvas_length:] = int(pad_token_id)

        if eos_ids is not None:
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

def diffusion_gemma_generate_dispatch(
    model,
    qpc_path,
    inputs,
    runtime_ai100: bool,
    device_id,
    **kwargs,
) -> DiffusionGemmaGenerateDispatch:
    encoder_qpc_path = kwargs.pop("encoder_qpc_path", None)
    decoder_qpc_path = kwargs.pop("decoder_qpc_path", None)
    generation_config = _prepare_runtime_generation_config(model=model, kwargs=kwargs)
    decoder_input_ids = kwargs.pop("decoder_input_ids", None)
    self_conditioning_logits = kwargs.pop("self_conditioning_logits", None)

    input_ids_tensor = inputs.get("input_ids", None)
    if input_ids_tensor is None:
        raise ValueError("`inputs` must contain `input_ids` for DiffusionGemma generate.")
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

    print('In utils encoder',encoder_qpc_path)
    print('In utils decoder',decoder_qpc_path)
    if encoder_qpc_path is not None and decoder_qpc_path is not None:
        encoder_session = QAICInferenceSession(str(encoder_qpc_path), device_id or None)
        print('Encoder has finished')
        decoder_session = QAICInferenceSession(str(decoder_qpc_path), device_id or None)
        runtime_result = diffusion_gemma_generate_ai100_split(
            encoder_session=encoder_session,
            decoder_session=decoder_session,
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
    else:
        print()
        qpc_session = QAICInferenceSession(str(qpc_path), None)
        runtime_result = diffusion_gemma_generate_ai100(
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