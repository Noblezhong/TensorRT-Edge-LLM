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

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import \
        Qwen3MoeSparseMoeBlock
except ImportError:
    Qwen3MoeSparseMoeBlock = None

from ...common import ONNX_OPSET_VERSION
from .int4_gemm_plugin import unpack_int4_weights_gptq

# Pre-computed Marlin tensor core layout indices (128 threads, 8 values each)
# fmt: off
_MARLIN_PACK_IDX = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)

_MARLIN_OUT_IDX = np.array([
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
    2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
    66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127
], dtype=np.int32)

# ROW_IDX: 4 unique patterns repeated 32 times each
_ROW_PATTERN = np.array([
    [0, 1, 8, 9, 0, 1, 8, 9],
    [2, 3, 10, 11, 2, 3, 10, 11],
    [4, 5, 12, 13, 4, 5, 12, 13],
    [6, 7, 14, 15, 6, 7, 14, 15]
], dtype=np.int32)
_MARLIN_ROW_IDX = np.tile(_ROW_PATTERN, (32, 1))

# COL_IDX: columns follow pattern based on warp_id and thread_id
_MARLIN_COL_IDX = np.array([
    [0, 0, 0, 0, 8, 8, 8, 8], [0, 0, 0, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 8, 8, 8], [0, 0, 0, 0, 8, 8, 8, 8],
    [1, 1, 1, 1, 9, 9, 9, 9], [1, 1, 1, 1, 9, 9, 9, 9],
    [1, 1, 1, 1, 9, 9, 9, 9], [1, 1, 1, 1, 9, 9, 9, 9],
    [2, 2, 2, 2, 10, 10, 10, 10], [2, 2, 2, 2, 10, 10, 10, 10],
    [2, 2, 2, 2, 10, 10, 10, 10], [2, 2, 2, 2, 10, 10, 10, 10],
    [3, 3, 3, 3, 11, 11, 11, 11], [3, 3, 3, 3, 11, 11, 11, 11],
    [3, 3, 3, 3, 11, 11, 11, 11], [3, 3, 3, 3, 11, 11, 11, 11],
    [4, 4, 4, 4, 12, 12, 12, 12], [4, 4, 4, 4, 12, 12, 12, 12],
    [4, 4, 4, 4, 12, 12, 12, 12], [4, 4, 4, 4, 12, 12, 12, 12],
    [5, 5, 5, 5, 13, 13, 13, 13], [5, 5, 5, 5, 13, 13, 13, 13],
    [5, 5, 5, 5, 13, 13, 13, 13], [5, 5, 5, 5, 13, 13, 13, 13],
    [6, 6, 6, 6, 14, 14, 14, 14], [6, 6, 6, 6, 14, 14, 14, 14],
    [6, 6, 6, 6, 14, 14, 14, 14], [6, 6, 6, 6, 14, 14, 14, 14],
    [7, 7, 7, 7, 15, 15, 15, 15], [7, 7, 7, 7, 15, 15, 15, 15],
    [7, 7, 7, 7, 15, 15, 15, 15], [7, 7, 7, 7, 15, 15, 15, 15],
    [16, 16, 16, 16, 24, 24, 24, 24], [16, 16, 16, 16, 24, 24, 24, 24],
    [16, 16, 16, 16, 24, 24, 24, 24], [16, 16, 16, 16, 24, 24, 24, 24],
    [17, 17, 17, 17, 25, 25, 25, 25], [17, 17, 17, 17, 25, 25, 25, 25],
    [17, 17, 17, 17, 25, 25, 25, 25], [17, 17, 17, 17, 25, 25, 25, 25],
    [18, 18, 18, 18, 26, 26, 26, 26], [18, 18, 18, 18, 26, 26, 26, 26],
    [18, 18, 18, 18, 26, 26, 26, 26], [18, 18, 18, 18, 26, 26, 26, 26],
    [19, 19, 19, 19, 27, 27, 27, 27], [19, 19, 19, 19, 27, 27, 27, 27],
    [19, 19, 19, 19, 27, 27, 27, 27], [19, 19, 19, 19, 27, 27, 27, 27],
    [20, 20, 20, 20, 28, 28, 28, 28], [20, 20, 20, 20, 28, 28, 28, 28],
    [20, 20, 20, 20, 28, 28, 28, 28], [20, 20, 20, 20, 28, 28, 28, 28],
    [21, 21, 21, 21, 29, 29, 29, 29], [21, 21, 21, 21, 29, 29, 29, 29],
    [21, 21, 21, 21, 29, 29, 29, 29], [21, 21, 21, 21, 29, 29, 29, 29],
    [22, 22, 22, 22, 30, 30, 30, 30], [22, 22, 22, 22, 30, 30, 30, 30],
    [22, 22, 22, 22, 30, 30, 30, 30], [22, 22, 22, 22, 30, 30, 30, 30],
    [23, 23, 23, 23, 31, 31, 31, 31], [23, 23, 23, 23, 31, 31, 31, 31],
    [23, 23, 23, 23, 31, 31, 31, 31], [23, 23, 23, 23, 31, 31, 31, 31],
    [32, 32, 32, 32, 40, 40, 40, 40], [32, 32, 32, 32, 40, 40, 40, 40],
    [32, 32, 32, 32, 40, 40, 40, 40], [32, 32, 32, 32, 40, 40, 40, 40],
    [33, 33, 33, 33, 41, 41, 41, 41], [33, 33, 33, 33, 41, 41, 41, 41],
    [33, 33, 33, 33, 41, 41, 41, 41], [33, 33, 33, 33, 41, 41, 41, 41],
    [34, 34, 34, 34, 42, 42, 42, 42], [34, 34, 34, 34, 42, 42, 42, 42],
    [34, 34, 34, 34, 42, 42, 42, 42], [34, 34, 34, 34, 42, 42, 42, 42],
    [35, 35, 35, 35, 43, 43, 43, 43], [35, 35, 35, 35, 43, 43, 43, 43],
    [35, 35, 35, 35, 43, 43, 43, 43], [35, 35, 35, 35, 43, 43, 43, 43],
    [36, 36, 36, 36, 44, 44, 44, 44], [36, 36, 36, 36, 44, 44, 44, 44],
    [36, 36, 36, 36, 44, 44, 44, 44], [36, 36, 36, 36, 44, 44, 44, 44],
    [37, 37, 37, 37, 45, 45, 45, 45], [37, 37, 37, 37, 45, 45, 45, 45],
    [37, 37, 37, 37, 45, 45, 45, 45], [37, 37, 37, 37, 45, 45, 45, 45],
    [38, 38, 38, 38, 46, 46, 46, 46], [38, 38, 38, 38, 46, 46, 46, 46],
    [38, 38, 38, 38, 46, 46, 46, 46], [38, 38, 38, 38, 46, 46, 46, 46],
    [39, 39, 39, 39, 47, 47, 47, 47], [39, 39, 39, 39, 47, 47, 47, 47],
    [39, 39, 39, 39, 47, 47, 47, 47], [39, 39, 39, 39, 47, 47, 47, 47],
    [48, 48, 48, 48, 56, 56, 56, 56], [48, 48, 48, 48, 56, 56, 56, 56],
    [48, 48, 48, 48, 56, 56, 56, 56], [48, 48, 48, 48, 56, 56, 56, 56],
    [49, 49, 49, 49, 57, 57, 57, 57], [49, 49, 49, 49, 57, 57, 57, 57],
    [49, 49, 49, 49, 57, 57, 57, 57], [49, 49, 49, 49, 57, 57, 57, 57],
    [50, 50, 50, 50, 58, 58, 58, 58], [50, 50, 50, 50, 58, 58, 58, 58],
    [50, 50, 50, 50, 58, 58, 58, 58], [50, 50, 50, 50, 58, 58, 58, 58],
    [51, 51, 51, 51, 59, 59, 59, 59], [51, 51, 51, 51, 59, 59, 59, 59],
    [51, 51, 51, 51, 59, 59, 59, 59], [51, 51, 51, 51, 59, 59, 59, 59],
    [52, 52, 52, 52, 60, 60, 60, 60], [52, 52, 52, 52, 60, 60, 60, 60],
    [52, 52, 52, 52, 60, 60, 60, 60], [52, 52, 52, 52, 60, 60, 60, 60],
    [53, 53, 53, 53, 61, 61, 61, 61], [53, 53, 53, 53, 61, 61, 61, 61],
    [53, 53, 53, 53, 61, 61, 61, 61], [53, 53, 53, 53, 61, 61, 61, 61],
    [54, 54, 54, 54, 62, 62, 62, 62], [54, 54, 54, 54, 62, 62, 62, 62],
    [54, 54, 54, 54, 62, 62, 62, 62], [54, 54, 54, 54, 62, 62, 62, 62],
    [55, 55, 55, 55, 63, 63, 63, 63], [55, 55, 55, 55, 63, 63, 63, 63],
    [55, 55, 55, 55, 63, 63, 63, 63], [55, 55, 55, 55, 63, 63, 63, 63],
], dtype=np.int32)
# fmt: on

int4_moe_plugin_schema = OpSchema(
    name="Int4MoePlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    "Custom TensorRT Int4 MoE plugin. Inputs: router_logits (FP32, from traced GEMM + cast in Python), hidden_states, expert weights. Plugin does softmax+topk then expert GEMMs.",
    inputs=[
        OpSchema.FormalParameter(
            name="router_logits",
            description=
            "Router logits (B*S, E) FP32, from gate GEMM + cast, before softmax",
            type_str="tensor(float)"),
        OpSchema.FormalParameter(name="hidden_states",
                                 description="Input hidden states (B, S, D)",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="fc_gate_up_qweights",
            description="Fused gate+up proj quantized weights (E, I, D)",
            type_str="tensor(int8)"),
        OpSchema.FormalParameter(
            name="fc_gate_up_scales",
            description="Fused gate+up proj scales (E, D/G, I)",
            type_str="T"),
        OpSchema.FormalParameter(
            name="fc_down_qweights",
            description="Down proj quantized weights (E, D/2, I)",
            type_str="tensor(int8)"),
        OpSchema.FormalParameter(name="fc_down_scales",
                                 description="Down proj scales (E, I/G, D)",
                                 type_str="T"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="Output tensor (B, S, D)",
                                 type_str="T"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], "FP16 data type."),
    ],
    attributes=[
        OpSchema.Attribute(name="num_experts",
                           type=OpSchema.AttrType.INT,
                           description="Number of experts",
                           required=True),
        OpSchema.Attribute(name="top_k",
                           type=OpSchema.AttrType.INT,
                           description="Top K experts per token",
                           required=True),
        OpSchema.Attribute(name="hidden_size",
                           type=OpSchema.AttrType.INT,
                           description="Hidden size D",
                           required=True),
        OpSchema.Attribute(name="moe_inter_size",
                           type=OpSchema.AttrType.INT,
                           description="MoE intermediate size I",
                           required=True),
        OpSchema.Attribute(name="activation_type",
                           type=OpSchema.AttrType.INT,
                           description="Activation function type",
                           required=True),
        OpSchema.Attribute(
            name="quantization_group_size",
            type=OpSchema.AttrType.INT,
            description="Quantization group size G",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(int4_moe_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "i", "i", "i", "i",
                            "i", "i")
def symbolic_int4_moe_plugin(
    g: Any,
    router_logits: torch._C.Value,
    hidden_states: torch._C.Value,
    fc_gate_up_qweights: torch._C.Value,
    fc_gate_up_scales: torch._C.Value,
    fc_down_qweights: torch._C.Value,
    fc_down_scales: torch._C.Value,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
):
    output = g.op(
        "trt::Int4MoePlugin",
        router_logits,
        hidden_states,
        fc_gate_up_qweights,
        fc_gate_up_scales,
        fc_down_qweights,
        fc_down_scales,
        num_experts_i=num_experts,
        top_k_i=top_k,
        hidden_size_i=hidden_size,
        moe_inter_size_i=moe_inter_size,
        activation_type_i=activation_type,
        quantization_group_size_i=quantization_group_size,
    )

    input_type = hidden_states.type()
    output_sizes = _get_tensor_sizes(hidden_states)
    output.setType(input_type.with_sizes(output_sizes))

    return output


@torch.library.custom_op("trt::int4_moe_plugin", mutates_args=())
def int4_moe_plugin(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    fc_gate_up_qweights: torch.Tensor,
    fc_gate_up_scales: torch.Tensor,
    fc_down_qweights: torch.Tensor,
    fc_down_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    output = torch.zeros(batch_size,
                         seq_len,
                         hidden_size,
                         dtype=hidden_states.dtype,
                         device=hidden_states.device)
    return output


def _is_gptq_quant_linear(module: nn.Module) -> bool:
    """Detect if a module uses GPTQ format (qweight packed along K dimension)."""
    if not hasattr(module, "qweight") or not hasattr(module, "in_features"):
        return False
    qw = module.qweight
    in_feat = getattr(module, "in_features", None)
    if in_feat is None or qw is None:
        return False
    # GPTQ: qweight (K/8, N), so qweight.shape[0] * 8 == in_features
    return qw.shape[0] * 8 == in_feat


def _unpack_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ qzeros (num_groups, N/8) to (num_groups, N) int32 [0,15]."""
    device = qzeros.device
    pack_factor = 8
    wf = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                      dtype=torch.int64,
                      device=device).view(1, 1, -1)
    z = qzeros.unsqueeze(2).expand(-1, -1, pack_factor).to(torch.int64)
    zeros = torch.bitwise_and(torch.bitwise_right_shift(z, wf),
                              15).reshape(qzeros.shape[0], -1)
    return zeros


def _assert_identity_g_idx(proj: nn.Module, group_size: int) -> None:
    """Assert that g_idx is an identity mapping (g_idx[i] == i // group_size)."""
    if not hasattr(proj, "g_idx") or proj.g_idx is None:
        return
    g_idx = proj.g_idx
    K = g_idx.shape[0]
    expected = torch.arange(K, device=g_idx.device) // group_size
    if not torch.equal(g_idx, expected):
        raise ValueError(
            "Non-identity g_idx detected. Only identity g_idx (g_idx[i] == i // group_size) "
            "is supported. Non-trivial g_idx (desc_act) support is not yet implemented."
        )


def _extract_gptq_proj_for_marlin(
    proj: nn.Module, ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (weights [N,K], scales [N, num_groups]) from a GPTQ quant linear for Marlin.
    Marlin uses zero_point=8; GPTQ uses qzeros. We remap: q_marlin = clamp(q_gptq - zero + 8, 0, 15)
    so that scale * (q_marlin - 8) = scale * (q_gptq - zero) exactly.
    """
    unpacked = unpack_int4_weights_gptq(proj.qweight)  # [K, N]
    group_size = getattr(proj, "group_size", 128)
    if group_size == -1:
        group_size = proj.in_features
    _assert_identity_g_idx(proj, group_size)

    if hasattr(proj, "qzeros") and proj.qzeros is not None:
        zeros = _unpack_qzeros(proj.qzeros)  # [num_groups, N]
        K, N = unpacked.shape
        group_ids = torch.arange(K, device=unpacked.device) // group_size
        zeros_expanded = zeros[group_ids.clamp(max=zeros.shape[0] -
                                               1)]  # [K, N]
        unpacked = torch.clamp(
            unpacked.to(torch.int32) - zeros_expanded.to(torch.int32) + 8, 0,
            15).to(torch.int16)

    weights = unpacked.transpose(0, 1).contiguous()  # [N, K]
    scales = proj.scales.data.to(torch.float16)  # [K/group_size, N]
    scales = scales.transpose(0, 1).contiguous()  # [N, num_groups]
    return weights, scales


def _marlin_permute_scales(s, size_k, size_n, group_size):
    """Permute scale columns to match the Marlin kernel's shared-memory read pattern.
    Adapted from https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L327.
    Input s: [num_groups, N] for one expert. Returns permuted [num_groups, N]."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def pack_int4_awq_marlin(
        weights_q: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack INT4 weights (stored as int16) and scales into AWQ Marlin format.
    
    Args:
        weights_q: INT4 weights as int16 tensor, shape [E, N, K] with values in [0, 15].
            - E: number of experts
            - N: output features
            - K: input features (must be divisible by 16)
            - N must be divisible by 64
        scales: Scale tensor, shape [E, N, num_groups] where num_groups = K // group_size.
        group_size: Quantization group size (default: 128).
    
    Returns:
        Tuple of:
        - weights_marlin: Marlin-formatted weights [E, K//16, 2*N] as int32.
        - scales_marlin: Permuted scales [E, num_groups, N] for Marlin kernel.
    """
    num_experts, N, K = weights_q.shape
    assert K % 16 == 0, f"K={K} must be divisible by 16"
    assert N % 64 == 0, f"N={N} must be divisible by 64"
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    num_groups = K // group_size
    assert scales.shape == (num_experts, N, num_groups), \
        f"scales shape {scales.shape} must be [{num_experts}, {N}, {num_groups}]"

    device = weights_q.device
    weights_marlin_list = []

    for expert_id in range(num_experts):
        w_np = weights_q[expert_id].transpose(
            0, 1).contiguous().cpu().numpy().astype(np.uint32)  # [K, N]

        k_tiles, n_tiles = K // 16, N // 64
        tiles = w_np.reshape(k_tiles, 16, n_tiles, 64).transpose(0, 2, 1, 3)
        gathered = tiles[:, :, _MARLIN_ROW_IDX,
                         _MARLIN_COL_IDX][:, :, :,
                                          _MARLIN_PACK_IDX].astype(np.uint32)

        packed_out = (gathered[:, :, :, 0] | (gathered[:, :, :, 1] << 4) |
                      (gathered[:, :, :, 2] << 8) |
                      (gathered[:, :, :, 3] << 12) |
                      (gathered[:, :, :, 4] << 16) |
                      (gathered[:, :, :, 5] << 20) |
                      (gathered[:, :, :, 6] << 24) |
                      (gathered[:, :, :, 7] << 28))

        out = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
        for n_tile_id in range(n_tiles):
            out[:,
                n_tile_id * 128 + _MARLIN_OUT_IDX] = packed_out[:,
                                                                n_tile_id, :]

        weights_marlin_list.append(
            torch.from_numpy(out.view(np.int32)).to(device))

    weights_marlin = torch.stack(weights_marlin_list, dim=0)

    # Permute scales for Marlin kernel: [E, N, num_groups] -> [E, num_groups, N]
    # Then apply column permutation to match Marlin kernel's shared-memory access pattern
    scales_marlin = scales.transpose(1, 2).contiguous()
    for e in range(num_experts):
        scales_marlin[e] = _marlin_permute_scales(scales_marlin[e], K, N,
                                                  group_size)

    return weights_marlin, scales_marlin


class Int4MoePluginModule(nn.Module):
    """
    MoE block for ONNX export: gate (FP16 GEMM) is traced as standard MatMul;
    plugin receives router_logits + hidden_states and does softmax+topk + expert INT4 GEMMs.
    Supports GPTQ expert weights only (TorchFusedQuantLinear).
    """

    def __init__(self,
                 moe_block: Qwen3MoeSparseMoeBlock,
                 group_size: int = 128):
        super().__init__()

        self.num_experts = moe_block.num_experts
        self.top_k = moe_block.top_k

        first_expert = moe_block.experts[0]
        self.hidden_size = first_expert.hidden_size
        self.moe_inter_size = first_expert.intermediate_size
        self.group_size = group_size
        self.activation_type = 0

        # Gate as nn.Linear so it traces to FP16 GEMM (MatMul+Add) outside the plugin
        gate_layer = moe_block.gate
        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=gate_layer.bias is not None,
            dtype=torch.float16,
        )
        self.gate.weight.data = gate_layer.weight.data.clone().to(
            torch.float16)
        if gate_layer.bias is not None:
            self.gate.bias.data = gate_layer.bias.data.clone().to(
                torch.float16)

        # Collect unpacked weights and scales for all experts
        # Gate+Up projection: input D, output I (concatenated gate I/2 + up I/2)
        gate_up_weights_list: List[torch.Tensor] = []
        gate_up_scales_list: List[torch.Tensor] = []

        # Down projection: input I, output D
        down_weights_list: List[torch.Tensor] = []
        down_scales_list: List[torch.Tensor] = []

        assert _is_gptq_quant_linear(first_expert.gate_proj), \
            "Int4MoePluginModule supports GPTQ only; expert gate_proj must be GPTQ format"

        for expert in moe_block.experts:
            gate_weights, gate_scales = _extract_gptq_proj_for_marlin(
                expert.gate_proj)
            up_weights, up_scales = _extract_gptq_proj_for_marlin(
                expert.up_proj)
            gate_up_weights_list.append(
                torch.cat([gate_weights, up_weights], dim=0))
            gate_up_scales_list.append(
                torch.cat([gate_scales, up_scales], dim=0))

            down_weights, down_scales = _extract_gptq_proj_for_marlin(
                expert.down_proj)
            down_weights_list.append(down_weights)
            down_scales_list.append(down_scales)

        # Stack all experts: [E, I, D] and [E, D, I]
        gate_up_weights_stacked = torch.stack(gate_up_weights_list,
                                              dim=0)  # [E, I, D] as int16
        gate_up_scales_stacked = torch.stack(gate_up_scales_list,
                                             dim=0)  # [E, I, D/G]
        down_weights_stacked = torch.stack(down_weights_list,
                                           dim=0)  # [E, D, I] as int16
        down_scales_stacked = torch.stack(down_scales_list,
                                          dim=0)  # [E, D, I/G]

        gate_up_weights_marlin, gate_up_scales_marlin = pack_int4_awq_marlin(
            gate_up_weights_stacked, gate_up_scales_stacked, group_size)

        down_weights_marlin, down_scales_marlin = pack_int4_awq_marlin(
            down_weights_stacked, down_scales_stacked, group_size)
        # Output: weights_marlin [E, I//16, 2*D] as int32, scales_marlin [E, I/G, D]

        # Convert int32 Marlin weights to int8 for plugin
        gate_up_weights_int8 = gate_up_weights_marlin.view(
            torch.int8).contiguous()
        down_weights_int8 = down_weights_marlin.view(torch.int8).contiguous()

        self.register_buffer("fc_gate_up_qweights", gate_up_weights_int8)
        self.register_buffer("fc_gate_up_scales",
                             gate_up_scales_marlin.contiguous())
        self.register_buffer("fc_down_qweights", down_weights_int8)
        self.register_buffer("fc_down_scales", down_scales_marlin.contiguous())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_flat).float()
        return int4_moe_plugin(
            router_logits,
            hidden_states,
            self.fc_gate_up_qweights,
            self.fc_gate_up_scales,
            self.fc_down_qweights,
            self.fc_down_scales,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_inter_size,
            self.activation_type,
            self.group_size,
        )


def register_int4_moe_plugin_onnx_symbolic_functions() -> None:
    register_custom_op_symbolic("trt::int4_moe_plugin",
                                symbolic_int4_moe_plugin, ONNX_OPSET_VERSION)
    print("Registered ONNX symbolic functions for custom Int4MoePlugin")


def replace_moe_blocks_with_plugin(model: nn.Module,
                                   group_size: int = 128) -> nn.Module:
    """
    Replace Qwen3MoeSparseMoeBlock with Int4MoePluginModule for ONNX export.

    Supports GPTQ expert weights only. Call before the first forward pass so
    qweight is still in GPTQ format. Use _fix_gptq_moe_gate_weights() first
    to replace quantized gate with nn.Linear.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            new_module = Int4MoePluginModule(module, group_size)
            parent = model
            if "." in name:
                parent_name, module_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name
            setattr(parent, module_name, new_module)
    return model


def is_moe_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", "")
    return "moe" in model_type.lower()
