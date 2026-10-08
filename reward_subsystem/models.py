"""Seeding and loading the frozen language model whose hidden states are probed."""

import os
import random

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_lm(model_name: str, dtype: str = "bf16"):
    """Load a causal LM (eval mode, no gradients) and its tokenizer.

    The tokenizer pads and truncates on the right; ``pad_token`` falls back to ``eos_token``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.output_hidden_states = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPES[dtype],
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        config=config,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, tokenizer


def hidden_size(model) -> int:
    """Residual-stream width (handles multimodal configs with a ``text_config``)."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        return model.config.text_config.hidden_size
    raise ValueError("Cannot find hidden_size in model config")
