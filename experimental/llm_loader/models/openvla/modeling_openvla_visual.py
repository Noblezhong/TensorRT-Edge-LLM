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
"""OpenVLA / OpenFly visual encoder exporter.

This module mirrors the OpenFly ``PrismaticVisionBackbone`` + ``PrismaticProjector``
path from ``eval/extern/hf/modeling_prismatic.py`` and exports it as a single
visual ONNX graph that outputs LLM-ready embeddings.

The OpenFly checkpoint keeps its multimodal inputs at the root config and uses a
dual-vision fused backbone:
    - ``vision_backbone``: DINOv2 + SigLIP
    - ``projector``: 2- or 3-layer MLP depending on fused mode

The exported visual graph expects the processor output for one policy step:
    ``pixel_values`` with shape ``[3, 2 * 3, H, W]`` when fused.

The graph output is flattened projected embeddings with shape
``[total_image_tokens, llm_hidden_size]``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import LayerScale

__all__ = ["OpenVLAVisualModel", "build_openvla_visual"]


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class PrismaticVisionBackbone(nn.Module):

    def __init__(self, use_fused_vision_backbone: bool,
                 image_sizes: List[int], timm_model_ids: List[str],
                 timm_override_act_layers: List[Optional[str]]) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        if len(timm_model_ids) not in (1, 2):
            raise ValueError(
                f"OpenVLA expects 1 or 2 vision backbones, got {len(timm_model_ids)}"
            )
        if len(image_sizes) != len(timm_model_ids):
            raise ValueError(
                "image_sizes and timm_model_ids must have the same length")

        # Build without pretrained weights. The checkpoint loader fills them in
        # from the OpenFly safetensors shard.
        self.featurizer = timm.create_model(timm_model_ids[0],
                                            pretrained=False,
                                            num_classes=0,
                                            img_size=image_sizes[0])
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers,
                    n={len(self.featurizer.blocks) - 2},
                    return_prefix_tokens=True))
        self.embed_dim = self.featurizer.embed_dim

        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers,
                        n={len(self.fused_featurizer.blocks) - 2},
                        return_prefix_tokens=True))
            self.embed_dim += self.fused_featurizer.embed_dim

            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

        # Keep the modules in eval mode by default; export only traces forward.
        self.featurizer.eval()
        if self.use_fused_vision_backbone:
            self.fused_featurizer.eval()

    def forward(self, pixel_values: torch.Tensor, grid_size: int = 16) -> List[torch.Tensor]:
        """Run the three per-step frames through the fused vision backbone.

        The current OpenFly processor stacks three frames:
        - frame 0: current observation
        - frame 1: history frame 1
        - frame 2: history frame 2
        """
        if pixel_values.shape[0] != 3:
            raise ValueError(
                f"OpenVLA expects 3 stacked frames, got batch={pixel_values.shape[0]}"
            )

        if self.use_fused_vision_backbone and pixel_values.shape[1] != 6:
            raise ValueError(
                f"OpenVLA fused vision expects 6 input channels, got {pixel_values.shape[1]}"
            )
        if (not self.use_fused_vision_backbone) and pixel_values.shape[1] != 3:
            raise ValueError(
                f"OpenVLA single vision expects 3 input channels, got {pixel_values.shape[1]}"
            )

        if self.use_fused_vision_backbone:
            img3, img_fused3 = torch.split(pixel_values[0:1], [3, 3], dim=1)
            patches3 = self.featurizer(img3)[1][:, :1, :]
            patches_fused3 = self.post_process(self.fused_featurizer(img_fused3)[0], grid_size)

            img4, img_fused4 = torch.split(pixel_values[1:2], [3, 3], dim=1)
            patches4 = self.featurizer(img4)[1][:, :1, :]
            patches_fused4 = self.post_process(self.fused_featurizer(img_fused4)[0], grid_size)

            img5, img_fused5 = torch.split(pixel_values[2:3], [3, 3], dim=1)
            patches5 = self.featurizer(img5)[0]
            patches_fused5 = self.fused_featurizer(img_fused5)[0]

            return [
                torch.cat([patches3, patches_fused3], dim=2),
                torch.cat([patches4, patches_fused4], dim=2),
                torch.cat([patches5, patches_fused5], dim=2),
            ]

        patches3 = self.featurizer(pixel_values[0:1])[1][:, :1, :]
        patches4 = self.featurizer(pixel_values[1:2])[1][:, :1, :]
        patches5 = self.featurizer(pixel_values[2:3])[0]
        return [patches3, patches4, patches5]

    @staticmethod
    def post_process(tensor_a: torch.Tensor, grid_size: int) -> torch.Tensor:
        batch, _, channels = tensor_a.size()
        tensor_a_like = tensor_a.view(batch, 16, 16, channels)
        pooled_tensor_a = F.avg_pool2d(tensor_a_like.permute(0, 3, 1, 2),
                                       kernel_size=grid_size)
        return pooled_tensor_a.permute(0, 2, 3, 1).flatten(1, 2)


class PrismaticProjector(nn.Module):

    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int,
                 llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim,
                                 initial_projection_dim,
                                 bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)
        return projected_features


class OpenVLAVisualModel(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.use_fused_vision_backbone = bool(
            config.get("use_fused_vision_backbone", True))
        image_sizes = list(config.get("image_sizes", []))
        timm_model_ids = list(config.get("timm_model_ids", []))
        timm_override_act_layers = list(
            config.get("timm_override_act_layers", [None] * len(timm_model_ids)))

        if not image_sizes or not timm_model_ids:
            raise ValueError(
                "OpenVLA visual export requires image_sizes and timm_model_ids in config.json"
            )

        text_cfg = dict(config.get("text_config", {}) or {})
        llm_hidden_size = text_cfg.get("hidden_size")
        if llm_hidden_size is None:
            hf_llm_id = config.get("hf_llm_id")
            if not hf_llm_id:
                raise ValueError(
                    "OpenVLA visual export requires text_config.hidden_size or hf_llm_id"
                )
            from transformers import AutoConfig

            base_llm_cfg = AutoConfig.from_pretrained(
                hf_llm_id, trust_remote_code=True).to_dict()
            base_text_cfg = dict(base_llm_cfg.get("text_config")
                                 or base_llm_cfg or {})
            text_cfg = {**base_text_cfg, **text_cfg}
            llm_hidden_size = text_cfg.get("hidden_size")
        if llm_hidden_size is None:
            raise ValueError(
                "Failed to determine OpenVLA LLM hidden_size from text_config or hf_llm_id"
            )
        llm_hidden_size = int(llm_hidden_size)

        self.vision_backbone = PrismaticVisionBackbone(
            self.use_fused_vision_backbone,
            image_sizes,
            timm_model_ids,
            timm_override_act_layers,
        )
        self.projector = PrismaticProjector(
            self.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=llm_hidden_size,
        )
        self._llm_hidden_size = llm_hidden_size
        self._image_channels = 3 * (2 if self.use_fused_vision_backbone else 1)
        self._resolved_text_config = text_cfg

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_features = self.vision_backbone(pixel_values, grid_size=16)
        projected_patch_embedding = [self.projector(patch_feature)
                                     for patch_feature in patch_features]
        projected_patch_embeddings = torch.cat(projected_patch_embedding,
                                               dim=1)
        return projected_patch_embeddings.reshape(-1, self._llm_hidden_size)

    def get_onnx_export_args(self, config: dict,
                             device: str) -> Tuple[tuple, list[str], list[str], dict]:
        image_sizes = list(config.get("image_sizes", []))
        image_size = int(image_sizes[0])
        pixel_values = torch.zeros(3,
                                   self._image_channels,
                                   image_size,
                                   image_size,
                                   dtype=torch.float16,
                                   device=device)
        args = (pixel_values, )
        input_names = ["input"]
        output_names = ["output"]
        dynamic_shapes: dict = {}
        return args, input_names, output_names, dynamic_shapes


def _load_weights(model: nn.Module, weights: dict) -> None:
    vision_state: dict = {}
    projector_state: dict = {}
    for key, value in weights.items():
        if key.startswith("vision_backbone."):
            vision_state[key.removeprefix("vision_backbone.")] = value
        elif key.startswith("projector."):
            projector_state[key.removeprefix("projector.")] = value

    missing_v, unexpected_v = model.vision_backbone.load_state_dict(
        vision_state, strict=False)
    missing_p, unexpected_p = model.projector.load_state_dict(projector_state,
                                                              strict=False)

    logger = logging.getLogger(__name__)
    if missing_v or unexpected_v:
        logger.warning("OpenVLA visual backbone load mismatch: missing=%s unexpected=%s",
                       missing_v[:10], unexpected_v[:10])
    if missing_p or unexpected_p:
        logger.warning("OpenVLA projector load mismatch: missing=%s unexpected=%s",
                       missing_p[:10], unexpected_p[:10])


def build_openvla_visual(config: dict,
                         weights: dict,
                         dtype: torch.dtype = torch.float16) -> OpenVLAVisualModel:
    model = OpenVLAVisualModel(config)
    model.to(dtype)
    _load_weights(model, weights)
    model.eval()
    return model
