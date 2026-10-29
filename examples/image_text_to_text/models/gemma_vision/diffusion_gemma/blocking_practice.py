from QEfficient.generation.cloud_infer import QAICInferenceSession
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from QEfficient import QEFFAutoModel
import numpy as np
from QEfficient.diffusers.models.blocking_configurator import build_transformer_blocking_config

# model_path = "Dream-org/Dream-v0-Instruct-7B"
model_path = "GSAI-ML/LLaDA-8B-Instruct"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# config.rope_scaling = 4
# config.num_hidden_layers = 1

model = QEFFAutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, config = config)
print(model.config)

config = build_transformer_blocking_config(
    model_config=model.config,
    specializations={
        "batch_size": 1,
        "cl": 12264,
        "ctx_len": 12264
    },
    compile_config={
        "mdp_ts_num_devices": 16,
        "aic_num_cores": 16,
        "convert_to_fp16": True
    },
    blocking_mode="hqkv"
)
print(config)
