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
"""
CLI: export ALL components of a multimodal checkpoint to ONNX in one command.

Detects model type from ``config.json`` and exports:
    - LLM backbone        → ``<output_dir>/llm/model.onnx``
    - Visual encoder      → ``<output_dir>/visual/model.onnx``    (VLMs)
    - Audio encoder       → ``<output_dir>/audio/model.onnx``     (speech models)
    - Code2Wav vocoder    → ``<output_dir>/code2wav/model.onnx``  (Qwen3-Omni)

Usage::

    # From experimental/ directory inside the repo:
    python -m llm_loader.export_all_cli /path/to/checkpoint /tmp/onnx_out

    # With explicit dtype
    python -m llm_loader.export_all_cli /path/to/Qwen3-VL-7B /tmp/out --dtype float16

Supported model types
----------------------
VLMs (LLM + visual encoder):
    openvla                       (OpenFly / OpenVLA-style fused dual-vision VLM)
    qwen3_vl, qwen3_omni          (Qwen3-VL / Qwen3-Omni)
    qwen3_5                       (Qwen3.5)
    qwen2_5_vl                    (Qwen2.5-VL)
    internvl_chat                 (InternVL3)
    internvl                      (InternVL3.5)
    phi4mm, phi4_multimodal       (Phi-4 Multimodal)
    NemotronH_Nano_VL_V2          (Nemotron-Omni)

Audio models (LLM + audio encoder):
    qwen3_asr, qwen3_omni, qwen3_omni_thinker
    NemotronH_Nano_VL_V2          (Nemotron-Omni)

LLM + Talker decoder (no audio encoder):
    qwen3_tts    (Talker/CodePredictor are LLM decoders — use --skip-audio)

LLM-only:
    All other model types supported by :mod:`llm_loader.model.AutoModel`.
"""

import argparse
import json
import logging
import os
import sys

from .checkpoint.checkpoint_utils import normalize_rope_scaling_for_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_loader.export_all_cli")

# ---------------------------------------------------------------------------
# Model type classification
# ---------------------------------------------------------------------------

_VLM_MODEL_TYPES = frozenset([
    "openvla",
    "qwen3_vl",
    "qwen3_omni",
    "qwen3_5",
    "qwen2_5_vl",
    "internvl",
    "internvl_chat",
    "phi4mm",
    "phi4_multimodal",
    "NemotronH_Nano_VL_V2",
])

_AUDIO_MODEL_TYPES = frozenset([
    "qwen3_asr",
    "qwen3_omni",
    "qwen3_omni_thinker",
    "NemotronH_Nano_VL_V2",
    # qwen3_tts intentionally excluded: Qwen3-TTS has NO audio encoder.
    # Its Talker and CodePredictor are LLM decoders exported via the LLM pipeline.
])

_TTS_MODEL_TYPES = frozenset([
    "qwen3_tts",
])

_CODE2WAV_MODEL_TYPES = frozenset([
    "qwen3_omni",
])


def _has_visual(model_type: str) -> bool:
    return model_type in _VLM_MODEL_TYPES


def _has_audio(model_type: str) -> bool:
    return model_type in _AUDIO_MODEL_TYPES


def _is_tts(model_type: str) -> bool:
    return model_type in _TTS_MODEL_TYPES


def _has_code2wav(model_type: str) -> bool:
    return model_type in _CODE2WAV_MODEL_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. "
                     "Install it or provide a local path.")
        sys.exit(1)
    logger.info("Downloading %s from Hugging Face Hub ...", model)
    return snapshot_download(model)


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        logger.error("config.json not found in %s", model_dir)
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


def _find_token_id(model_dir: str, token_str: str) -> "Optional[int]":
    """Return the token ID for *token_str* by scanning tokenizer files."""
    # Try tokenizer.json added_tokens list first
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tok = json.load(f)
        for entry in tok.get("added_tokens", []):
            if entry.get("content") == token_str:
                return int(entry["id"])
    # Fall back to added_tokens_decoder in tokenizer.json
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tok = json.load(f)
        for id_str, entry in tok.get("added_tokens_decoder", {}).items():
            if entry.get("content") == token_str:
                return int(id_str)
    # Try added_tokens.json
    added_path = os.path.join(model_dir, "added_tokens.json")
    if os.path.exists(added_path):
        with open(added_path) as f:
            added = json.load(f)
        if token_str in added:
            return int(added[token_str])
    logger.warning("Could not find token ID for %r in %s", token_str,
                   model_dir)
    return None


def _load_all_weights(model_dir: str) -> dict:
    """Load all safetensors shards in *model_dir* into a flat dict."""
    import glob

    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        logger.error("No safetensors files found in %s", model_dir)
        sys.exit(1)
    weights: dict = {}
    for shard in shards:
        logger.info("  Loading shard: %s", os.path.basename(shard))
        weights.update(load_file(shard, device="cpu"))
    return weights


def _to_fp16(tensors: dict) -> dict:
    """Cast bfloat16 tensors to float16 for C++ runtime compatibility.

    The C++ runtime requires FP16 (or FP8) for sidecar weight files.
    Checkpoints often store weights in bfloat16.
    """
    import torch
    return {
        k: v.to(torch.float16) if v.dtype == torch.bfloat16 else v
        for k, v in tensors.items()
    }


def _dtype_from_str(s: str) -> "torch.dtype":
    import torch
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if s not in mapping:
        logger.error("Unknown dtype %r. Choose from: %s", s,
                     ", ".join(mapping))
        sys.exit(1)
    return mapping[s]


def _openvla_llm_key_remap(key: str) -> "str | None":
    """Map OpenFly wrapper weights onto the generic CausalLM key layout."""
    prefix = "language_model."
    if not key.startswith(prefix):
        return key
    return key.removeprefix(prefix)


# ---------------------------------------------------------------------------
# Export stages
# ---------------------------------------------------------------------------


def _export_llm(model_dir: str,
                llm_out_dir: str,
                model_type: str = "",
                eagle_base: bool = False,
                fp8_embedding: bool = False) -> None:
    """Export LLM backbone via the standard llm_loader pipeline."""
    os.makedirs(llm_out_dir, exist_ok=True)
    output_path = os.path.join(llm_out_dir, "model.onnx")

    logger.info("[LLM] Loading checkpoint from %s", model_dir)
    try:
        from .model import AutoModel
        key_remap = _openvla_llm_key_remap if model_type == "openvla" else None
        model = AutoModel.from_pretrained(
            model_dir,
            device="cpu",
            key_remap=key_remap,
            eagle_base=eagle_base,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[LLM] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[LLM] Exporting to %s", output_path)
    try:
        from .onnx.export import export_onnx
        export_onnx(model,
                    output_path,
                    model_dir=model_dir,
                    fp8_embedding=fp8_embedding)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[LLM] ONNX export failed")
        raise SystemExit(1) from exc

    # For VLM models, patch image_token_id into the LLM config so the C++
    # runtime can identify image tokens in the token stream.
    if model_type in _VLM_MODEL_TYPES:
        cfg_path = os.path.join(llm_out_dir, "config.json")
        image_token_id = None
        audio_token_id = None
        if model_type == "NemotronH_Nano_VL_V2":
            # Nemotron-Omni keeps the real image/audio token ids in-stream,
            # so runtime dispatch needs both of them in the engine config.
            src_cfg = _load_config(model_dir)
            image_token_id = src_cfg.get("img_context_token_id")
            audio_token_id = src_cfg.get("sound_context_token_id")
        else:
            image_token_id = _find_token_id(model_dir, "<|image_pad|>")
        if os.path.exists(cfg_path) and (image_token_id is not None
                                         or audio_token_id is not None):
            with open(cfg_path) as _f:
                cfg = json.load(_f)
            if image_token_id is not None:
                cfg["image_token_id"] = image_token_id
                logger.info("[LLM] Added image_token_id=%d to config.json",
                            image_token_id)
            if audio_token_id is not None:
                cfg["audio_token_id"] = audio_token_id
                logger.info("[LLM] Added audio_token_id=%d to config.json",
                            audio_token_id)
            with open(cfg_path, "w") as _f:
                json.dump(cfg, _f, indent=2)

    logger.info("[LLM] Done: %s", output_path)


def _export_visual(model_dir: str, visual_out_dir: str, weights: dict,
                   config: dict, model_type: str,
                   dtype: "torch.dtype") -> None:
    """Export visual encoder via from-scratch llm_loader pipeline."""
    os.makedirs(visual_out_dir, exist_ok=True)
    output_path = os.path.join(visual_out_dir, "model.onnx")

    logger.info("[Visual] Exporting %s visual encoder to %s", model_type,
                output_path)
    try:
        from .onnx.export_encoder import export_visual_onnx
        export_visual_onnx(
            model_dir=model_dir,
            output_path=output_path,
            weights=weights,
            config=config,
            model_type=model_type,
            dtype=dtype,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Visual] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[Visual] Done: %s", output_path)

    # Write a config.json for the C++ runtime.
    # The visual builder will merge builder_config into this file when it
    # builds the engine. Fields needed vary by model family:
    #   InternVL*: image_token_id, text_config (for vocab_size)
    #   all:       model_type, vision_config
    # Qwen3-Omni nests vision_config / text_config / token IDs under
    # thinker_config; other Qwen VL variants keep them at the root.
    _thinker_cfg = config.get("thinker_config", {}) or {}
    vis_cfg = (config.get("vision_config") or _thinker_cfg.get("vision_config")
               or config)
    # C++ stringToModelType maps "internvl" and "internvl_vision" to INTERNVL;
    # "internvl_chat" is NOT registered.  Normalize both variants to "internvl".
    top_level_model_type = "internvl" if model_type in (
        "internvl", "internvl_chat") else model_type
    vis_cfg_out: dict = {
        "model_type": top_level_model_type,
        "vision_config": vis_cfg,
    }
    if model_type == "openvla":
        text_cfg = dict(config.get("text_config", {}) or {})
        if "hidden_size" not in text_cfg and config.get("hf_llm_id"):
            try:
                from transformers import AutoConfig
                base_cfg = AutoConfig.from_pretrained(
                    config["hf_llm_id"], trust_remote_code=True).to_dict()
                base_text_cfg = dict(base_cfg.get("text_config")
                                     or base_cfg or {})
                text_cfg = {**base_text_cfg, **text_cfg}
            except (OSError, ValueError, ImportError) as exc:
                logger.warning(
                    "[Visual] Failed to resolve base LLM config for openvla from %s (%s)",
                    config.get("hf_llm_id"),
                    exc,
                )
        if config.get("llm_max_length") is not None:
            text_cfg["max_position_embeddings"] = config["llm_max_length"]
        if config.get("pad_token_id") is not None:
            text_cfg["pad_token_id"] = config["pad_token_id"]
        if text_cfg:
            vis_cfg_out["text_config"] = text_cfg
        if config.get("hf_llm_id") is not None:
            vis_cfg_out["hf_llm_id"] = config["hf_llm_id"]
        if config.get("llm_backbone_id") is not None:
            vis_cfg_out["llm_backbone_id"] = config["llm_backbone_id"]
    if model_type in ("qwen2_5_vl", "qwen3_vl", "qwen3_omni", "qwen3_5"):
        # C++ QwenViTRunner reads these token IDs and rope_theta from config.json.
        # For Qwen3-VL the token IDs are at the root level, but vocab_size and
        # rope_theta live inside text_config.  Fall back to text_config for any
        # key that is absent from the root.  For Qwen3-Omni all of these live
        # under thinker_config (token IDs) and thinker_config.text_config.
        _text_cfg = (config.get("text_config")
                     or _thinker_cfg.get("text_config") or {})
        # rope_theta may live in text_config.rope_parameters (newer transformers)
        _rope_params = _text_cfg.get("rope_parameters") or _text_cfg.get(
            "rope_scaling") or {}
        for key in ("vision_start_token_id", "vision_end_token_id",
                    "image_token_id", "video_token_id", "vocab_size",
                    "rope_theta"):
            if key in config:
                vis_cfg_out[key] = config[key]
            elif key in _thinker_cfg:
                vis_cfg_out[key] = _thinker_cfg[key]
            elif key in _text_cfg:
                vis_cfg_out[key] = _text_cfg[key]
            elif key in _rope_params:
                vis_cfg_out[key] = _rope_params[key]
        # Include rope_scaling (contains mrope_section) for Qwen VL models.
        # The C++ QwenViTRunner reads mrope_section from rope_scaling.
        # Quantized checkpoints may use rope_parameters instead of
        # rope_scaling — normalize to rope_scaling for the C++ runtime.
        _rope_scaling = (_text_cfg.get("rope_scaling")
                         or _text_cfg.get("rope_parameters")
                         or config.get("rope_scaling")
                         or config.get("rope_parameters"))
        if _rope_scaling:
            vis_cfg_out["rope_scaling"] = normalize_rope_scaling_for_runtime(
                _rope_scaling)
        # Also copy preprocessor_config.json to the output dir so the runner
        # can find patch_size, temporal_patch_size, merge_size, image_mean, image_std.
        import shutil
        pp_src = os.path.join(model_dir, "preprocessor_config.json")
        if os.path.exists(pp_src):
            shutil.copy2(pp_src, visual_out_dir)
            logger.info("[Visual] Copied preprocessor_config.json to %s",
                        visual_out_dir)
        else:
            # Newer quantized checkpoints store image processor config inside
            # processor_config.json under the "image_processor" key.  Extract
            # it and write a standalone preprocessor_config.json.
            proc_src = os.path.join(model_dir, "processor_config.json")
            if os.path.exists(proc_src):
                with open(proc_src) as _pf:
                    proc_cfg = json.load(_pf)
                img_proc = proc_cfg.get("image_processor", {})
                if img_proc:
                    pp_dst = os.path.join(visual_out_dir,
                                          "preprocessor_config.json")
                    with open(pp_dst, "w") as _pf:
                        json.dump(img_proc, _pf, indent=2)
                    logger.info(
                        "[Visual] Extracted preprocessor_config.json from "
                        "processor_config.json to %s", visual_out_dir)
                else:
                    logger.warning(
                        "[Visual] processor_config.json has no "
                        "image_processor key at %s", proc_src)
            else:
                logger.warning(
                    "[Visual] Neither preprocessor_config.json nor "
                    "processor_config.json found at %s", model_dir)
    if model_type in ("phi4mm", "phi4_multimodal"):
        # C++ Phi4MMViTRunner reads vocab_size and embd_layer from the top level
        # of config.json.  For phi4mm the raw config.json is flat (no vision_config
        # sub-key), so flatten the required fields from vision_config → top level.
        vc = vis_cfg_out.get("vision_config", {})
        for key in ("vocab_size", "embd_layer"):
            if key in vc:
                vis_cfg_out[key] = vc[key]
    if model_type in ("internvl", "internvl_chat"):
        # C++ internViTRunner reads image_token_id and text_config.vocab_size.
        # For internvl_chat the image token is <IMG_CONTEXT>; find it from the
        # tokenizer added_tokens list if it's not in config.json directly.
        image_token_id = config.get("image_token_id")
        if image_token_id is None:
            image_token_id = _find_token_id(model_dir, "<IMG_CONTEXT>")
        if image_token_id is not None:
            vis_cfg_out["image_token_id"] = image_token_id
        # text_config (vocab_size) — internvl_chat may use llm_config instead
        text_cfg = config.get("text_config") or config.get("llm_config")
        if text_cfg:
            vis_cfg_out["text_config"] = text_cfg
        # The C++ visual builder reads vision_config.model_type first.
        # intern_vit_6b (old arch) is not registered; override to "internvl".
        if "vision_config" in vis_cfg_out and "model_type" in vis_cfg_out[
                "vision_config"]:
            vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
            vis_cfg_out["vision_config"]["model_type"] = "internvl"
        # C++ builder reads patch_size[0]/[1] and image_size[0]/[1] as arrays.
        # Convert scalar ints to [H, W] pairs if needed.
        vc_out = vis_cfg_out["vision_config"]
        if isinstance(vc_out.get("patch_size"), int):
            vis_cfg_out["vision_config"] = dict(vc_out)
            vis_cfg_out["vision_config"]["patch_size"] = [
                vc_out["patch_size"], vc_out["patch_size"]
            ]
        if isinstance(vc_out.get("image_size"), int):
            vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
            vis_cfg_out["vision_config"]["image_size"] = [
                vc_out["image_size"], vc_out["image_size"]
            ]
    if model_type == "NemotronH_Nano_VL_V2":
        # The ckpt's "NemotronH_Nano_VL_V2" is not registered in C++
        # stringToModelType().  Override to the registered tag both at top
        # level (read by MultimodalRunner::create) and under vision_config
        # (preferred by visualBuilder).
        vis_cfg_out["model_type"] = "nemotron_omni_vision_encoder"
        vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
        vis_cfg_out["vision_config"][
            "model_type"] = "nemotron_omni_vision_encoder"
        # NemotronOmniViTRunner reads these top-level fields; visualBuilder
        # additionally reads patch_size and downsample_ratio.
        for key in ("llm_config", "img_context_token_id", "img_start_token_id",
                    "img_end_token_id", "force_image_size", "norm_mean",
                    "norm_std", "patch_size", "downsample_ratio"):
            if key in config:
                vis_cfg_out[key] = config[key]
    cfg_out_path = os.path.join(visual_out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(vis_cfg_out, f, indent=2)
    logger.info("[Visual] Wrote config.json: %s", cfg_out_path)


def _export_audio(model_dir: str, audio_out_dir: str, weights: dict,
                  config: dict, model_type: str, dtype: "torch.dtype") -> None:
    """Export audio encoder via from-scratch llm_loader pipeline."""
    os.makedirs(audio_out_dir, exist_ok=True)
    output_path = os.path.join(audio_out_dir, "model.onnx")

    logger.info("[Audio] Exporting %s audio encoder to %s", model_type,
                output_path)
    try:
        from .onnx.export_encoder import export_audio_onnx
        export_audio_onnx(
            model_dir=model_dir,
            output_path=output_path,
            weights=weights,
            config=config,
            model_type=model_type,
            dtype=dtype,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Audio] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[Audio] Done: %s", output_path)

    # Write config.json for the C++ runtime
    if model_type == "NemotronH_Nano_VL_V2":
        audio_cfg_out = dict(config)
        sound_model_type = config.get("sound_config", {}).get("model_type")
        if sound_model_type is None:
            raise ValueError(
                "sound_config.model_type not found in config.json")
        audio_cfg_out["model_type"] = sound_model_type
    elif model_type in ("qwen3_asr", "qwen3_omni", "qwen3_omni_thinker"):
        audio_cfg = config.get("thinker_config",
                               {}).get("audio_config",
                                       config.get("audio_config", {}))
        _AUDIO_MODEL_TYPE_MAP = {
            "qwen3_asr": "qwen3_asr_thinker",
        }
        audio_model_type = _AUDIO_MODEL_TYPE_MAP.get(model_type, model_type)
        audio_cfg_out = {
            "model_type": audio_model_type,
            "audio_config": audio_cfg,
        }
    else:
        raise ValueError(f"Unsupported audio model_type: {model_type}")
    cfg_out_path = os.path.join(audio_out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(audio_cfg_out, f, indent=2)
    logger.info("[Audio] Wrote config.json: %s", cfg_out_path)


# ---------------------------------------------------------------------------
# Code2Wav export (Qwen3-Omni vocoder)
# ---------------------------------------------------------------------------


def _export_code2wav(model_dir: str, c2w_out_dir: str, weights: dict,
                     config: dict, dtype: "torch.dtype") -> None:
    """Export Qwen3-Omni Code2Wav vocoder via the standalone llm_loader
    implementation.

    The vocoder converts discrete RVQ codec tokens
    ``[batch, num_quantizers, code_length]`` into continuous audio
    waveforms ``[batch, 1, code_length * total_upsample]``.

    ``code2wav_config`` is expected at the root of ``config.json``; weights
    are extracted from the shared checkpoint using the ``code2wav.`` prefix.
    """
    os.makedirs(c2w_out_dir, exist_ok=True)
    output_path = os.path.join(c2w_out_dir, "model.onnx")

    c2w_cfg = config.get("code2wav_config")
    if not c2w_cfg:
        logger.error(
            "code2wav_config not found in config.json — cannot export Code2Wav"
        )
        sys.exit(1)

    logger.info("[Code2Wav] Building model and loading weights")
    try:
        from .models.qwen3_omni import build_code2wav, export_code2wav_onnx
        model = build_code2wav(c2w_cfg, weights, dtype)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[Code2Wav] Failed to build model")
        raise SystemExit(1) from exc

    logger.info("[Code2Wav] Exporting ONNX to %s", output_path)
    try:
        export_code2wav_onnx(model, output_path, c2w_cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Code2Wav] ONNX export failed")
        raise SystemExit(1) from exc

    # Write a config.json that the C++ runtime / engine builder can consume.
    # Match the layout produced by tensorrt_edgellm.export_code2wav_config:
    # top-level model_type is "qwen3_omni_code2wav" and the sub-config
    # carries the same model_type for parser compatibility.
    c2w_cfg_out = dict(c2w_cfg)
    c2w_cfg_out["model_type"] = "qwen3_omni_code2wav"
    cfg_out_path = os.path.join(c2w_out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(
            {
                "model_type": "qwen3_omni_code2wav",
                "code2wav_config": c2w_cfg_out,
            },
            f,
            indent=2)
    logger.info("[Code2Wav] Wrote config.json: %s", cfg_out_path)
    logger.info("[Code2Wav] Done: %s", output_path)


# ---------------------------------------------------------------------------
# TTS Talker export
# ---------------------------------------------------------------------------


def _extract_tts_weights(model_dir: str, out_dir: str) -> None:
    """Extract TTS-specific weight files from the full checkpoint.

    Saves:
    - ``text_embedding.safetensors``  — thinker text embedding [text_vocab_size, hidden]
    - ``text_projection.safetensors`` — MLP weights (fc1/fc2 weight+bias)
    """
    from safetensors.torch import save_file

    weights = _load_all_weights(model_dir)

    # text_embedding: talker.model.text_embedding.weight [151936, 2048]
    text_emb_key = "talker.model.text_embedding.weight"
    if text_emb_key not in weights:
        logger.error("Key %r not found in checkpoint", text_emb_key)
        sys.exit(1)
    text_emb = weights[text_emb_key].cpu()
    save_file(_to_fp16({"text_embedding": text_emb}),
              os.path.join(out_dir, "text_embedding.safetensors"))
    logger.info("[TTS] Wrote text_embedding.safetensors %s",
                list(text_emb.shape))

    # text_projection: talker.text_projection.linear_fc1/fc2 weight/bias
    proj_keys = {
        "linear_fc1.weight": "talker.text_projection.linear_fc1.weight",
        "linear_fc1.bias": "talker.text_projection.linear_fc1.bias",
        "linear_fc2.weight": "talker.text_projection.linear_fc2.weight",
        "linear_fc2.bias": "talker.text_projection.linear_fc2.bias",
    }
    proj_tensors = {}
    for save_name, ckpt_key in proj_keys.items():
        if ckpt_key not in weights:
            logger.error("Key %r not found in checkpoint", ckpt_key)
            sys.exit(1)
        proj_tensors[save_name] = weights[ckpt_key].cpu()
    save_file(_to_fp16(proj_tensors),
              os.path.join(out_dir, "text_projection.safetensors"))
    logger.info("[TTS] Wrote text_projection.safetensors (4 tensors)")


def _patch_tts_config(model_dir: str, out_dir: str) -> None:
    """Patch the exported config.json with TTS-specific fields.

    Reads the root HF config and injects codec token IDs, TTS token IDs,
    thinker_hidden_size, and speaker_id mapping into the already-written
    config.json in *out_dir*.
    """
    root_config = _load_config(model_dir)
    talker_cfg = root_config.get("talker_config", {})

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # TTS token IDs (from root config)
    for key in ("tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id"):
        if key in root_config:
            cfg[key] = root_config[key]

    # Codec token IDs (from talker_config)
    for key in ("codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id",
                "codec_pad_id", "codec_bos_id", "codec_eos_token_id",
                "codec_think_id"):
        if key in talker_cfg:
            cfg[key] = talker_cfg[key]

    # thinker_hidden_size and text_vocab_size
    if "text_hidden_size" in talker_cfg:
        cfg["thinker_hidden_size"] = talker_cfg["text_hidden_size"]
    if "text_vocab_size" in talker_cfg:
        cfg["text_vocab_size"] = talker_cfg["text_vocab_size"]

    # Speaker ID mapping
    if "spk_id" in talker_cfg:
        cfg["speaker_id"] = talker_cfg["spk_id"]
    if "default_speaker_id" not in cfg and talker_cfg.get("spk_id"):
        # Use the first speaker as default
        cfg["default_speaker_id"] = next(iter(talker_cfg["spk_id"].values()))

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("[TTS] Patched config.json with TTS/codec fields")


def _talker_key_remap(key: str) -> "Optional[str]":
    """Rename ``codec_embedding`` → ``embed_tokens`` in talker checkpoint keys."""
    return key.replace("codec_embedding",
                       "embed_tokens") if "codec_embedding" in key else key


def _export_talker(model_dir: str, llm_out_dir: str) -> None:
    """Export TTS Talker LLM backbone + extract TTS-specific weight files.

    The Talker is architecturally a standard Qwen3 CausalLM with
    ``vocab_size=3072`` (codec tokens).  This function:
    1. Exports the talker backbone ONNX via the standard LLM pipeline.
       ``key_prefix="talker."`` strips the checkpoint prefix;
       ``codec_embedding`` is renamed to ``embed_tokens`` via a minimal remap.
    2. Extracts ``text_embedding.safetensors`` and ``text_projection.safetensors``
       from the checkpoint.
    3. Patches ``config.json`` with TTS/codec token IDs and speaker mapping.
    """
    os.makedirs(llm_out_dir, exist_ok=True)
    output_path = os.path.join(llm_out_dir, "model.onnx")

    logger.info("[Talker] Loading checkpoint from %s", model_dir)
    try:
        from .checkpoint.loader import load_weights
        from .config import ModelConfig
        from .models.qwen3_tts import TalkerCausalLM

        config = ModelConfig.from_pretrained(model_dir)
        model = TalkerCausalLM(config)
        model.to("cpu")
        load_weights(model,
                     model_dir,
                     device="cpu",
                     key_prefix="talker.",
                     key_remap=_talker_key_remap)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[Talker] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[Talker] Exporting ONNX to %s", output_path)
    try:
        from .onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Talker] ONNX export failed")
        raise SystemExit(1) from exc

    logger.info("[Talker] Extracting TTS weight files ...")
    _extract_tts_weights(model_dir, llm_out_dir)

    logger.info("[Talker] Patching config.json with TTS fields ...")
    _patch_tts_config(model_dir, llm_out_dir)

    logger.info("[Talker] Done: %s", output_path)


# ---------------------------------------------------------------------------
# TTS CodePredictor export
# ---------------------------------------------------------------------------


def _export_code_predictor(model_dir: str, cp_out_dir: str) -> None:
    """Export TTS CodePredictor ONNX + extract codec embeddings, lm_heads, projection.

    The CodePredictor is a small 5-layer Qwen3 decoder with:
    - ``lm_head_weight`` as an ONNX input (dynamic, 15 different heads)
    - ``hidden_states`` as an additional output (for residual connection)
    - MLP FP16 overflow WAR applied to all layers

    Outputs:
    - ``model.onnx`` — CodePredictor ONNX graph
    - ``codec_embeddings.safetensors`` — 15 embedding tables
    - ``lm_heads.safetensors`` — 15 lm_head weights
    - ``small_to_mtp_projection.safetensors`` — talker→CP projection
    - ``config.json`` — LLM config with ``use_embeddings_input: true``
    """
    os.makedirs(cp_out_dir, exist_ok=True)
    output_path = os.path.join(cp_out_dir, "model.onnx")

    # Build CodePredictor config from the checkpoint's code_predictor sub-config
    root_config = _load_config(model_dir)
    talker_cfg = root_config.get("talker_config", {})
    cp_cfg = talker_cfg.get("code_predictor_config", {})
    if not cp_cfg.get("hidden_size"):
        logger.error("code_predictor_config not found in talker_config")
        sys.exit(1)

    # Write a temporary config.json for the CodePredictor so ModelConfig can
    # parse it.  The CP sub-config is a valid standalone Qwen3 config.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy safetensors index/files for has_qk_norm detection
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors") or fname.endswith(
                    ".safetensors.index.json"):
                src = os.path.join(model_dir, fname)
                dst = os.path.join(tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        # Write the CP config
        tmp_cfg_path = os.path.join(tmp_dir, "config.json")
        with open(tmp_cfg_path, "w") as f:
            json.dump(cp_cfg, f)

        from .config import ModelConfig
        config = ModelConfig.from_pretrained(tmp_dir)

    # Override model_type for runtime identification
    config.model_type = "qwen3_tts_code_predictor"

    # Create CodePredictorCausalLM and load weights
    from .models.qwen3_tts import (CodePredictorCausalLM,
                                   apply_code_predictor_mlp_war)

    model = CodePredictorCausalLM(config)
    model.to("cpu")

    from .checkpoint.loader import load_weights
    load_weights(model,
                 model_dir,
                 device="cpu",
                 key_prefix="talker.code_predictor.")

    # Apply MLP FP16 overflow WAR
    apply_code_predictor_mlp_war(model)

    logger.info("[CodePredictor] Exporting ONNX to %s", output_path)
    try:
        from .onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[CodePredictor] ONNX export failed")
        raise SystemExit(1) from exc

    # Extract CodePredictor-specific weight files
    logger.info("[CodePredictor] Extracting weight files ...")
    _extract_code_predictor_weights(model_dir, cp_out_dir, talker_cfg)

    # Patch config.json with use_embeddings_input and num_code_groups
    cfg_path = os.path.join(cp_out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["use_embeddings_input"] = True
        cfg["num_code_groups"] = talker_cfg.get("num_code_groups", 16)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info("[CodePredictor] Patched config.json with "
                    "use_embeddings_input and num_code_groups")

    logger.info("[CodePredictor] Done: %s", output_path)


def _extract_code_predictor_weights(model_dir: str, out_dir: str,
                                    talker_cfg: dict) -> None:
    """Extract codec_embeddings, lm_heads, and small_to_mtp_projection."""
    from safetensors.torch import save_file

    weights = _load_all_weights(model_dir)

    # codec_embeddings: talker.code_predictor.model.codec_embedding.{i}.weight
    num_code_groups = talker_cfg.get("num_code_groups", 16)
    num_embeddings = num_code_groups - 1  # 15 for TTS (16-1=15)
    embedding_dict = {}
    for i in range(num_embeddings):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key not in weights:
            logger.error("Key %r not found in checkpoint", key)
            sys.exit(1)
        embedding_dict[f"embedding_{i}"] = weights[key].cpu()
    save_file(_to_fp16(embedding_dict),
              os.path.join(out_dir, "codec_embeddings.safetensors"))
    logger.info(
        "[CodePredictor] Wrote codec_embeddings.safetensors "
        "(%d embeddings, shape %s)", num_embeddings,
        list(embedding_dict["embedding_0"].shape))

    # lm_heads: talker.code_predictor.lm_head.{i}.weight
    lm_head_dict = {}
    for i in range(num_embeddings):
        key = f"talker.code_predictor.lm_head.{i}.weight"
        if key not in weights:
            logger.error("Key %r not found in checkpoint", key)
            sys.exit(1)
        lm_head_dict[f"lm_head_{i}.weight"] = weights[key].cpu()
    save_file(_to_fp16(lm_head_dict),
              os.path.join(out_dir, "lm_heads.safetensors"))
    logger.info(
        "[CodePredictor] Wrote lm_heads.safetensors "
        "(%d heads, shape %s)", num_embeddings,
        list(lm_head_dict["lm_head_0.weight"].shape))

    # small_to_mtp_projection: talker.code_predictor.small_to_mtp_projection
    proj_w_key = "talker.code_predictor.small_to_mtp_projection.weight"
    proj_b_key = "talker.code_predictor.small_to_mtp_projection.bias"
    proj_dict = {}
    if proj_w_key in weights:
        proj_dict["weight"] = weights[proj_w_key].cpu()
        if proj_b_key in weights:
            proj_dict["bias"] = weights[proj_b_key].cpu()
        save_file(_to_fp16(proj_dict),
                  os.path.join(out_dir, "small_to_mtp_projection.safetensors"))
        logger.info(
            "[CodePredictor] Wrote small_to_mtp_projection.safetensors "
            "(weight shape %s)", list(proj_dict["weight"].shape))
    else:
        logger.warning("[CodePredictor] small_to_mtp_projection not found "
                       "(may be Omni-style without projection)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m llm_loader.export_all_cli",
        description=(
            "Export ALL components of a multimodal checkpoint to ONNX "
            "(LLM + optional visual/audio encoder) in one command."),
    )
    p.add_argument(
        "model",
        help="Local checkpoint directory or Hugging Face model ID.",
    )
    p.add_argument(
        "output_dir",
        help=
        "Root output directory. Sub-dirs llm/, visual/, audio/ are created as needed.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        help="Weight dtype for visual/audio models (default: float16).",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM backbone export (export only visual/audio encoders).",
    )
    p.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual encoder export.",
    )
    p.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio encoder export.",
    )
    p.add_argument(
        "--skip-code2wav",
        action="store_true",
        help="Skip Code2Wav vocoder export (Qwen3-Omni only).",
    )
    p.add_argument(
        "--eagle-base",
        action="store_true",
        help=
        "Export as EAGLE3 base model (adds tree-attention I/O and hidden_states output).",
    )
    p.add_argument(
        "--fp8-embedding",
        "--fp8_embedding",
        dest="fp8_embedding",
        action="store_true",
        help=
        "Write embedding.safetensors in FP8 E4M3 format with per-row block scales.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device for export tracing (default: cuda).",
    )
    args = p.parse_args()

    model_dir = _resolve_model_dir(args.model)
    config = _load_config(model_dir)
    model_type: str = config.get("model_type", "unknown")
    dtype = _dtype_from_str(args.dtype)

    has_vis = _has_visual(model_type) and not args.skip_visual
    has_aud = _has_audio(model_type) and not args.skip_audio
    has_c2w = _has_code2wav(model_type) and not args.skip_code2wav
    is_tts = _is_tts(model_type)

    logger.info("=" * 60)
    logger.info("Model type     : %s", model_type)
    logger.info("Checkpoint     : %s", model_dir)
    logger.info("Output dir     : %s", args.output_dir)
    logger.info("Visual export  : %s", "yes" if has_vis else "no")
    logger.info("Audio export   : %s", "yes" if has_aud else "no")
    logger.info("Code2Wav export: %s", "yes" if has_c2w else "no")
    logger.info("TTS talker     : %s", "yes" if is_tts else "no")
    logger.info("FP8 embedding  : %s", "yes" if args.fp8_embedding else "no")
    logger.info("=" * 60)

    # Load weights once (shared by visual, audio, and code2wav exporters)
    weights: dict = {}
    if has_vis or has_aud or has_c2w:
        logger.info("Loading safetensors weights ...")
        weights = _load_all_weights(model_dir)

    # --- LLM (or TTS Talker + CodePredictor) ---
    if not args.skip_llm:
        llm_out = os.path.join(args.output_dir, "llm")
        if is_tts:
            if args.fp8_embedding:
                logger.warning(
                    "--fp8-embedding is not supported for TTS talker/code_predictor; using FP16 embeddings."
                )
            _export_talker(model_dir, llm_out)
            # CodePredictor: small decoder for residual codec prediction
            cp_out = os.path.join(args.output_dir, "code_predictor")
            _export_code_predictor(model_dir, cp_out)
        else:
            _export_llm(model_dir,
                        llm_out,
                        model_type=model_type,
                        eagle_base=args.eagle_base,
                        fp8_embedding=args.fp8_embedding)

    # --- Visual encoder ---
    if has_vis:
        vis_out = os.path.join(args.output_dir, "visual")
        _export_visual(model_dir, vis_out, weights, config, model_type, dtype)

    # --- Audio encoder ---
    if has_aud:
        aud_out = os.path.join(args.output_dir, "audio")
        _export_audio(model_dir, aud_out, weights, config, model_type, dtype)

    # --- Code2Wav vocoder (Qwen3-Omni) ---
    if has_c2w:
        c2w_out = os.path.join(args.output_dir, "code2wav")
        _export_code2wav(model_dir, c2w_out, weights, config, dtype)

    # Summary
    print()
    print("=" * 60)
    print("Export complete")
    print(f"  output dir: {args.output_dir}")
    for sub in ["llm", "code_predictor", "visual", "audio", "code2wav"]:
        p_sub = os.path.join(args.output_dir, sub)
        if os.path.isdir(p_sub):
            onnx = os.path.join(p_sub, "model.onnx")
            mb = os.path.getsize(onnx) / 1e6 if os.path.exists(onnx) else 0
            print(f"  {sub:8s}: {onnx}  ({mb:.1f} MB)")
            # Show TTS sidecar files if present
            for sidecar in ("text_embedding.safetensors",
                            "text_projection.safetensors",
                            "codec_embeddings.safetensors",
                            "lm_heads.safetensors",
                            "small_to_mtp_projection.safetensors"):
                sc_path = os.path.join(p_sub, sidecar)
                if os.path.exists(sc_path):
                    sc_mb = os.path.getsize(sc_path) / 1e6
                    print(f"           + {sidecar}  ({sc_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
