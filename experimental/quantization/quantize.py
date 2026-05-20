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
    from extern.hf.processing_prismatic import PrismaticProcessor

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    _modeling_prismatic.PrismaticPreTrainedModel._supports_sdpa = property(lambda self: False)
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoModelForImageTextToText.register(OpenFlyConfig, OpenVLAForActionPrediction)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)


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


def _copy_processor_files(src_dir, dst_dir):
    """Copy processor sidecar files that AutoProcessor needs to initialise."""
    import shutil
    for name in ("preprocessor_config.json", "tokenizer.model",
                 "special_tokens_map.json", "added_tokens.json",
                 "generation_config.json", "dataset_statistics.json"):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)


def _openfly_calib_collate(batch):
    """Collate dict batches, stacking pixel_values as [B*3, 6, 224, 224]."""
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    result = {"input_ids": input_ids, "pixel_values": pixel_values}
    if "attention_mask" in batch[0]:
        result["attention_mask"] = torch.cat(
            [item["attention_mask"] for item in batch], dim=0)
    return result


def _append_openvla_boundary_token(input_ids: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None):
    """Match OpenVLA predict_action() by appending the trailing 29871 token."""
    if input_ids.numel() == 0:
        return input_ids, attention_mask
    if int(input_ids.reshape(-1)[-1].item()) == 29871:
        return input_ids, attention_mask

    boundary = torch.tensor([[29871]],
                            dtype=input_ids.dtype,
                            device=input_ids.device)
    input_ids = torch.cat([input_ids, boundary], dim=1)
    if attention_mask is not None:
        boundary_mask = torch.ones((attention_mask.shape[0], 1),
                                   dtype=attention_mask.dtype,
                                   device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, boundary_mask], dim=1)
    return input_ids, attention_mask


def _openfly_real_calib_dataloader(calib_dir: str,
                                   batch_size=1,
                                   num_samples=512):
    """Return a DataLoader built from previously collected eval calibration data.

    Reads ``.pt`` files written by the OpenFly eval hook (base_station.py).
    Each file contains the already-processed ``input_ids``, ``attention_mask``
    and ``pixel_values`` tensors, so this loader mirrors exact inference inputs.
    """
    files = sorted(Path(calib_dir).glob("calib_*.pt"))[:num_samples]
    if not files:
        return None
    dataset = []
    for fp in files:
        item = torch.load(fp, weights_only=True)
        pv = item["pixel_values"]
        if pv.dtype != torch.float16:
            pv = pv.to(torch.float16)
        input_ids, attention_mask = _append_openvla_boundary_token(
            item["input_ids"], item["attention_mask"])
        dataset.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pv,
        })
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=_openfly_calib_collate)


def _make_openvla_dummy_pixel_values(batch_size=1, seed=0):
    """Return dummy pixel_values with OpenVLA-normalised channel layout.

    Shape: ``[batch_size * 3, 6, 224, 224]`` float16.
    The first 3 channels are ImageNet-normalised (DINOv2 backbone),
    the last 3 are [-1, 1]-normalised (SigLIP backbone).

    When *seed* is non-zero a per-sample random noise pattern is used so the
    vision backbone produces **diverse** embeddings across calibration samples.
    """
    import numpy as _np
    rng = _np.random.RandomState(seed)
    # Mid-gray base with small per-pixel jitter to create diverse textures.
    # Range stays within [0, 255] uint8 so the subsequent normalisation
    # produces values in the same ballpark as real outdoor scenes.
    gray = _np.full((3, 224, 224), 128, dtype=_np.float32)
    if seed != 0:
        jitter = rng.randint(-64, 64, (3, 224, 224)).astype(_np.float32)
        gray = _np.clip(gray + jitter, 0, 255)
    gray01 = gray / 255.0
    dino = _np.stack([
        (gray01[0] - 0.485) / 0.229,
        (gray01[1] - 0.456) / 0.224,
        (gray01[2] - 0.406) / 0.225,
    ], axis=0)
    siglip = _np.stack([
        (gray01[0] - 0.5) / 0.5,
        (gray01[1] - 0.5) / 0.5,
        (gray01[2] - 0.5) / 0.5,
    ], axis=0)
    chw = _np.concatenate([dino, siglip], axis=0)  # [6, 224, 224]
    frames = _np.stack([chw] * 3 * batch_size, axis=0).astype(_np.float16)
    return torch.tensor(frames, dtype=torch.float16)


def _openfly_text_calib_dataloader(tokenizer,
                                   eval_dataset_path=None,
                                   batch_size=1,
                                   num_samples=512,
                                   max_length=512):
    """Return a calibration DataLoader built from the OpenFly eval dataset.

    Each batch is a dict with ``input_ids`` and ``pixel_values``.
    Pixel values are dummy mid-gray frames normalised identically to the
    OpenVLA PrismaticImageProcessor (dual-backbone ImageNet + [-1,1] format).

    When *eval_dataset_path* points to a JSON file (e.g. ``configs/eval_test.json``),
    the ``gpt_instruction`` field from each entry is wrapped with the standard
    OpenFly prompt template (``"What action should the robot take to ..."``).
    If the file cannot be read, the function falls back to a small set of
    built-in embodied-navigation prompts.
    """
    # --- collect prompt texts ---
    prompts = []
    if eval_dataset_path is not None:
        try:
            with open(eval_dataset_path, "r", encoding="utf-8") as fh:
                eval_entries = json.load(fh)
            for item in eval_entries[:num_samples]:
                instruction = item.get("gpt_instruction", "")
                if instruction:
                    txt = f"What action should the robot take to {instruction.lower().strip()}?"
                    prompts.append(txt)
        except Exception:
            pass

    if not prompts:
        prompts = [
            "What action should the robot take to move toward the destination?",
            "What action should the robot take to go to the red building?",
            "What action should the robot take to fly forward and then turn left slightly?",
            "What action should the robot take to navigate around the obstacle?",
            "What action should the robot take to approach the landing site?",
            "What action should the robot take to follow the road and then turn right?",
            "What action should the robot take to avoid the tall building?",
            "What action should the robot take to descend slightly and move forward?",
            "What action should the robot take to stay on the current path?",
            "What action should the robot take to turn toward the bridge?",
        ]

    texts = [prompts[i % len(prompts)] for i in range(num_samples)]

    dataset = []
    for idx, text in enumerate(texts):
        enc = tokenizer(text,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=max_length)
        input_ids, attention_mask = _append_openvla_boundary_token(
            enc["input_ids"], enc["attention_mask"])
        # Each calibration sample gets unique randomised pixel_values so
        # AWQ sees diverse visual features — otherwise every sample shares
        # identical dummy frames and the activation statistics collapse to
        # a single visual-text combination.
        seed = idx
        pixel_varied = _make_openvla_dummy_pixel_values(batch_size=1, seed=seed)
        dataset.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_varied,
        })
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=_openfly_calib_collate)


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
        if isinstance(data, dict):
            data = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            model(**data)
        else:
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
        effective_lm_head_quantization = lm_head_quantization
        if _is_openvla_checkpoint(model_dir) and lm_head_quantization is not None:
            print(
                "[WARN] OpenVLA exports keep lm_head in FP16 for TRT engine "
                "compatibility; ignoring --lm_head_quantization.")
            effective_lm_head_quantization = None

        quant_cfg = build_quant_config(quantization,
                                       effective_lm_head_quantization,
                                       kv_cache_quantization)
        batch_size = 16 if quantization in (None, "int4_awq") else 1
        if _is_openvla_checkpoint(model_dir):
            batch_size = 1  # OpenVLA uses 3-frame pixel_values; batch dims must match
            # Prefer real calibration data collected from AirSim eval when available.
            calib_dir = os.environ.get("OPENFLY_CALIB_OUTPUT_DIR", "")
            loader = None
            if calib_dir:
                loader = _openfly_real_calib_dataloader(calib_dir,
                                                        batch_size=batch_size,
                                                        num_samples=num_samples)
                if loader is not None:
                    print(f"[OpenVLA calib] Using real eval data from {calib_dir}")
            if loader is None:
                eval_json = os.path.join(_openfly_project_root(), "configs",
                                         "eval_test.json")
                loader = _openfly_text_calib_dataloader(tokenizer,
                                                        eval_dataset_path=eval_json,
                                                        batch_size=batch_size,
                                                        num_samples=num_samples)
                print("[OpenVLA calib] Using dummy pixel_values with eval prompts "
                      "(set OPENFLY_CALIB_OUTPUT_DIR for real data)")
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
    _copy_processor_files(model_dir, output_dir)

    print(f"Saved to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir
