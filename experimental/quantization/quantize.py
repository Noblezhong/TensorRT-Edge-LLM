# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantize a HuggingFace LLM and export a unified checkpoint.

Loads a model via ``AutoModelForCausalLM`` (with ``AutoModelForImageTextToText``
fallback for VLMs), runs ModelOpt quantization, and writes a unified safetensors
checkpoint consumable by ``llm_loader``.  No ``tensorrt_edgellm`` dependency.
"""

import os
import json
import time
from contextlib import contextmanager
from typing import Optional
from pathlib import Path
import sys

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoProcessor, AutoTokenizer)

from experimental.llm_loader.checkpoint.loader import load_weights

from .quantization_configs import build_quant_config


def _openfly_project_root() -> Path:
    env_root = os.environ.get("OPENFLY_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    candidates = [
        Path("/HDD1/code/OpenFly-Platform"),
        Path.home() / "code" / "BS_UAV-VLN",
        Path.home() / "OpenFly-Platform",
    ]
    for candidate in candidates:
        if (candidate / "eval" / "extern" / "hf" / "configuration_prismatic.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Unable to locate OpenFly project root. Set OPENFLY_PROJECT_ROOT to the "
        "OpenFly-Platform / BS_UAV-VLN checkout that contains eval/extern/hf/."
    )


def _is_openvla_checkpoint(model_dir: str) -> bool:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            root = json.load(f)
    except Exception:
        return False
    return root.get("model_type") == "openvla"


def _register_openvla_hf_classes() -> None:
    openfly_root = _openfly_project_root()
    openfly_eval_dir = openfly_root / "eval"
    openfly_eval_dir_str = str(openfly_eval_dir)
    if openfly_eval_dir_str not in sys.path:
        sys.path.insert(0, openfly_eval_dir_str)

    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf import modeling_prismatic as _modeling_prismatic
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    from transformers import AutoConfig, AutoModelForImageTextToText

    _modeling_prismatic.PrismaticPreTrainedModel._supports_sdpa = property(lambda self: False)
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoModelForImageTextToText.register(OpenFlyConfig, OpenVLAForActionPrediction)


@contextmanager
def _force_timm_pretrained_false():
    """Construct OpenVLA without downloading or initializing timm pretrained weights.

    OpenVLA's vision backbone is checkpoint-initialized after construction, so
    the timm pretrained weights are unnecessary here.  More importantly, the
    HF quantization path may instantiate the model under a meta-device context,
    and timm's pretrained initialization touches meta tensors.  For quantization
    we want a real module tree, so force timm's backbone constructors to build
    from structure only.
    """
    import timm

    orig_create_model = timm.create_model

    def _wrapped_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return orig_create_model(*args, **kwargs)

    timm.create_model = _wrapped_create_model
    try:
        yield
    finally:
        timm.create_model = orig_create_model


def _load_openvla_model(model_dir: str, dtype: str, device: str):
    """Load OpenVLA by instantiating the local class and manually loading weights."""
    openfly_root = _openfly_project_root()
    openfly_eval_dir = openfly_root / "eval"
    openfly_eval_dir_str = str(openfly_eval_dir)
    if openfly_eval_dir_str not in sys.path:
        sys.path.insert(0, openfly_eval_dir_str)

    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf import modeling_prismatic as _modeling_prismatic
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    def _patched_tie_weights(self, *args, **kwargs):
        # Transformers >= 4.50 passes recompute_mapping=False here. OpenVLA's
        # original signature is legacy and only expects a bare call, so accept
        # arbitrary args for compatibility while preserving the original logic.
        return self.language_model.tie_weights()

    _modeling_prismatic.PrismaticForConditionalGeneration.tie_weights = _patched_tie_weights

    cfg_path = Path(model_dir) / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg = OpenFlyConfig(**cfg_dict)
    with _force_timm_pretrained_false():
        model = OpenVLAForActionPrediction(cfg)

    model = model.to(device)
    load_weights(model, model_dir, device=device)

    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]

    if dtype == "fp16":
        model = model.to(torch.float16)
    else:
        model = model.to(torch.bfloat16)
    return model


def _text_calib_dataloader(tokenizer,
                           dataset_name="cnn_dailymail",
                           batch_size=1,
                           num_samples=512,
                           max_length=512):
    """Return a DataLoader of tokenised ``input_ids`` for calibration."""
    if "cnn_dailymail" in dataset_name:
        ds = load_dataset(dataset_name, name="3.0.0", split="train")
        texts = ds["article"][:num_samples]
    elif os.path.isdir(dataset_name):
        ds = load_dataset(dataset_name, split="train")
        texts = ds["text"][:num_samples]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    enc = tokenizer(texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length)
    return DataLoader(enc["input_ids"], batch_size=batch_size, shuffle=False)


def _openfly_text_calib_dataloader(tokenizer,
                                   batch_size=1,
                                   num_samples=512,
                                   max_length=512):
    """Return a local OpenFly-style text calibration loader.

    We intentionally avoid any Hugging Face dataset download here. The goal is
    to quantize the language backbone with representative embodied-navigation
    prompts while keeping the vision tower untouched.
    """
    prompts = [
        "You are an embodied UAV policy. Output only the action.",
        "What action should the UAV take to move toward the destination?",
        "Move straight ahead and then turn slightly left.",
        "Proceed forward, then adjust right toward the target building.",
        "Keep moving in the current direction and stop when close to the goal.",
        "You are controlling a drone in an outdoor navigation task.",
        "Interpret the scene and output the next discrete flight action.",
        "Go forward carefully and avoid deviating too much from the path.",
    ]
    texts = [prompts[i % len(prompts)] for i in range(num_samples)]
    enc = tokenizer(texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length)
    return DataLoader(enc["input_ids"], batch_size=batch_size, shuffle=False)


def _load_model(model_dir, dtype="fp16", device="cuda"):
    """Load model + tokenizer + optional processor via Auto* classes."""
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    if _is_openvla_checkpoint(model_dir):
        _register_openvla_hf_classes()
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True,
                                                  min_pixels=128 * 28 * 28,
                                                  max_pixels=2048 * 32 * 32)
    except Exception:
        processor = None

    if _is_openvla_checkpoint(model_dir):
        model = _load_openvla_model(model_dir, dtype=dtype, device=device)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device)
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device)

    model.to(torch_dtype)

    # modelopt export_hf_checkpoint crashes when architectures is None
    # (e.g. Qwen3.5 resolves to text_config with architectures=None).
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor


def _calibrate(model, dataloader):
    """Forward-loop calibration pass."""
    for data in tqdm(dataloader, desc="Calibrating"):
        model(data.to(model.device))


def _is_hybrid_model(model):
    """Return True if the model has hybrid Mamba+Attention layers.

    Checks multiple signals: ``layers_block_type`` in config (NemotronH),
    ``mamba_ssm_dtype`` in config (Qwen3.5), or ``linear_attn`` submodules.
    """
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    if getattr(config, "layers_block_type", None) is not None:
        return True
    if getattr(config, "mamba_ssm_dtype", None) is not None:
        return True
    if any("linear_attn" in n for n, _ in model.named_modules()):
        return True
    return False


@contextmanager
def _skip_resmooth_for_hybrid(model):
    """WAR for ModelOpt resmoothing bug on hybrid Mamba+Attention models.

    ``export_hf_checkpoint`` calls ``requantize_resmooth_fused_llm_layers``
    which averages AWQ pre_quant_scales across all linear modules that share
    the same input and re-quantizes their weights.  For hybrid models the
    dummy forward used to detect shared inputs does not propagate through
    Mamba layers correctly, and the Mamba projections (qkv, z, a, b) get
    incorrectly fused — corrupting the int4 weights.

    This context manager patches the resmoothing function to a no-op when the
    model is a hybrid architecture.  Standard transformer models are
    unaffected.

    TODO: Remove once ModelOpt fixes hybrid model support upstream.
    """
    if not _is_hybrid_model(model):
        yield
        return

    import modelopt.torch.export.unified_export_hf as _ueh
    _orig = _ueh.requantize_resmooth_fused_llm_layers

    def _noop(m):
        print("[WAR] Skipping requantize_resmooth_fused_llm_layers "
              "for hybrid model (ModelOpt bug workaround)")

    _ueh.requantize_resmooth_fused_llm_layers = _noop
    try:
        yield
    finally:
        _ueh.requantize_resmooth_fused_llm_layers = _orig


def quantize_and_export(
    model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    dataset: str = "cnn_dailymail",
    num_samples: int = 512,
) -> str:
    """Load a HuggingFace model, quantize it, and export a unified checkpoint."""
    t0 = time.time()
    model, tokenizer, processor = _load_model(model_dir, dtype, device)

    if is_quantized(model):
        print("Model already quantized — skipping.")
    else:
        quant_cfg = build_quant_config(quantization, lm_head_quantization,
                                       kv_cache_quantization)
        batch_size = 16 if quantization in (None, "int4_awq") else 1
        if _is_openvla_checkpoint(model_dir):
            loader = _openfly_text_calib_dataloader(tokenizer,
                                                    batch_size=batch_size,
                                                    num_samples=num_samples)
        else:
            loader = _text_calib_dataloader(tokenizer,
                                            dataset,
                                            batch_size=batch_size,
                                            num_samples=num_samples)
        mtq.quantize(model,
                     quant_cfg,
                     forward_loop=lambda m: _calibrate(m, loader))
        mtq.print_quant_summary(model)

    print(f"Quantization: {time.time() - t0:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode(), _skip_resmooth_for_hybrid(model):
        export_hf_checkpoint(model, export_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)

    print(f"Saved to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir
