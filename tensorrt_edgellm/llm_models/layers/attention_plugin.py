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
Dummy Attention Plugin for TensorRT Integration

This module provides a custom TensorRT operation for attention computation that can be
exported to ONNX format. It includes RoPE (Rotary Position Embedding) application,
KV cache management, and attention computation in a single fused operation.

The module contains:
- attention_plugin: Dummy TensorRT operation for attention computation, this is not used in the actual inference.
- ONNX export utilities for the custom operation
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import onnx
import torch
from onnx.defs import OpSchema
from torch._C import Value
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION

# Define ONNX OpSchema for AttentionPlugin
attention_plugin_schema = OpSchema(
    name="AttentionPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    "Custom TensorRT attention plugin with RoPE, KV cache, and attention computation.",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description="KV cache tensor",
            type_str="T_KV",
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Context length tensor",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="RoPE rotary embeddings (FP32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="kvcache_start_index",
            description=
            "KV cache start index tensor of shape [kv_cache_start_batch_size]",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="attention_mask",
            description="Attention mask tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="attention_pos_id",
            description="Position IDs tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor (always FP16)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="present_key_value",
            description=
            "Updated KV cache tensor with dynamic shape [batch_size, 2, num_kv_heads, present_kv_cache_len, head_size]",
            type_str="T_KV",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Input Q/K/V and attention output data type.",
        ),
        (
            "T_KV",
            ["tensor(float16)", "tensor(float8e4m3fn)"],
            "KV cache data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_q_heads",
            type=OpSchema.AttrType.INT,
            description="Number of query heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_kv_heads",
            type=OpSchema.AttrType.INT,
            description="Number of key-value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_tree_attention",
            type=OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0(false), 1(true))",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_fp8_kv_cache",
            type=OpSchema.AttrType.INT,
            description=
            "Whether to use FP8 KV cache (0(false), 1(true)). Optional.",
            required=False,
        ),
        OpSchema.Attribute(
            name="sliding_window_size",
            type=OpSchema.AttrType.INT,
            description=
            "Sliding window size for attention (-1 = no sliding window, >0 = window size).",
            required=False,
        ),
        OpSchema.Attribute(
            name="qkv_scales",
            type=OpSchema.AttrType.FLOATS,
            description=
            "Optional QKV dequant scales [q, k, v] for FP8 attention.",
            required=False,
        ),
    ],
)
onnx.defs.register_schema(attention_plugin_schema)

# Define ONNX OpSchema for ViTAttentionPlugin
vit_attention_plugin_schema = OpSchema(
    name="ViTAttentionPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    "Custom TensorRT ViT attention plugin (separate Q/K/V, no KV cache, no RoPE).",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor in head-major layout [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor in head-major layout [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor in head-major layout [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cu_seqlens",
            description="Prefix sum of sequence lengths (int32, shape [B+1])",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="max_seqlen_carrier",
            description=
            "Shape-only input used to carry runtime max sequence length hint; tensor values are ignored.",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor [total_S, H, D]",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Input Q/K/V data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_heads",
            type=OpSchema.AttrType.INT,
            description="Number of attention heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(vit_attention_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "i", "i", "b",
                            "i", "b", "i", "v", "v", "none")
def symbolic_attention_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    q: torch._C.Value,
    k: torch._C.Value,
    v: torch._C.Value,
    past_key_value: torch._C.Value,
    context_lengths: torch._C.Value,
    rope_rotary_cos_sin: torch._C.Value,
    kvcache_start_index: torch._C.Value,
    num_q_heads: torch._C.Value,
    num_kv_heads: torch._C.Value,
    enable_tree_attention: torch._C.Value,
    head_size: torch._C.Value,
    enable_fp8_kv_cache: torch._C.Value,
    sliding_window_size: torch._C.Value,
    attention_mask: Optional[torch._C.Value] = None,
    position_ids: Optional[torch._C.Value] = None,
    qkv_scales=None,
):
    """Custom attention plugin operation for ONNX export."""

    inputs = [
        q, k, v, past_key_value, context_lengths, rope_rotary_cos_sin,
        kvcache_start_index
    ]
    if enable_tree_attention:
        assert attention_mask is not None and attention_mask.type().kind(
        ) != 'NoneType', "attention_mask should be provided for tree attention"
        assert position_ids is not None and position_ids.type().kind(
        ) != 'NoneType', "position_ids should be provided for tree attention"
        inputs.append(attention_mask)
        inputs.append(position_ids)

    q_type = q.type()
    past_key_value_type = past_key_value.type()
    attrs = dict[str, Value | int](
        num_q_heads_i=num_q_heads,
        num_kv_heads_i=num_kv_heads,
        head_size_i=head_size,
        enable_tree_attention_i=1 if enable_tree_attention else 0,
        enable_fp8_kv_cache_i=1 if enable_fp8_kv_cache else 0,
        sliding_window_size_i=sliding_window_size,
    )
    if qkv_scales is not None:
        if isinstance(qkv_scales, torch._C.Value):
            if not qkv_scales.node().mustBeNone():
                node = qkv_scales.node()
                if node.kind() == 'prim::ListConstruct':
                    attrs["qkv_scales_f"] = [
                        float(v.node().f('value')) for v in node.inputs()
                    ]
                elif node.kind() == 'onnx::Constant':
                    attrs["qkv_scales_f"] = node.t('value').tolist()
                else:
                    raise RuntimeError(
                        f"Unsupported qkv_scales JIT node kind: "
                        f"{node.kind()}. Expected prim::ListConstruct "
                        f"or onnx::Constant.")
        else:
            attrs["qkv_scales_f"] = list(qkv_scales)

    attn_output, present_key_value = g.op("trt::AttentionPlugin",
                                          *inputs,
                                          **attrs,
                                          outputs=2)

    q_sizes = _get_tensor_sizes(q)
    attn_output_sizes = q_sizes[:-1] + [num_q_heads, head_size]
    attn_output.setType(q_type.with_sizes(attn_output_sizes))

    # KV Cache output has the same shape as input past_key_value except for dimension 3 (sequence length)
    # Shape: [batch_size, 2, num_kv_heads, present_kv_cache_len (dynamic), head_size]
    past_kv_sizes = _get_tensor_sizes(past_key_value)
    present_key_value.setType(past_key_value_type.with_sizes(past_kv_sizes))

    return attn_output, present_key_value


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "i")
def symbolic_vit_attention_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    q: torch._C.Value,
    k: torch._C.Value,
    v: torch._C.Value,
    cu_seqlens: torch._C.Value,
    max_seqlen_carrier: torch._C.Value,
    num_heads: torch._C.Value,
    head_size: torch._C.Value,
):
    """Custom ViT attention plugin operation for ONNX export."""
    attn_output = g.op(
        "trt::ViTAttentionPlugin",
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen_carrier,
        num_heads_i=num_heads,
        head_size_i=head_size,
        outputs=1,
    )
    # Attention output has the same shape as q: [total_S, H, D]
    attn_output.setType(q.type())
    return attn_output


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    enable_tree_attention: bool,
    head_size: int,
    enable_fp8_kv_cache: bool,
    sliding_window_size: int = -1,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    qkv_scales: Optional[Sequence[float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dummy TensorRT operation for attention computation, this is not used in the actual inference.

    This operation wraps the logic after v_proj and before o_proj into a single
    AttentionPlugin operation during ONNX export. It handles RoPE application,
    KV cache management, and attention computation in a fused manner.

    Args:
        q: Query tensor of shape (batch_size, seq_len, num_q_heads * head_size)
        k: Key tensor of shape (batch_size, seq_len, num_kv_heads * head_size)
        v: Value tensor of shape (batch_size, seq_len, num_kv_heads * head_size)
        past_key_value: KV cache tensor of shape (batch_size, 2, num_kv_heads, past_len, head_size)
        context_lengths: Context length tensor of shape (batch_size,) indicating current position in cache
        rope_rotary_cos_sin: RoPE tensor of shape (batch_size, seq_len, rotary_dim) containing cos and sin values
        kvcache_start_index: Start index of KV cache of shape (kv_cache_start_batch_size,), required
        num_q_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        enable_tree_attention: Whether to enable tree attention
        head_size: Size of each attention head
        enable_fp8_kv_cache: Whether to use FP8 KV cache
        sliding_window_size: Sliding window size for attention, optional
        attention_mask: Attention mask of shape (batch_size, seq_len, seq_len + past_len), optional
        position_ids: Position IDs tensor of shape (batch_size, seq_len), optional
        qkv_scales: QKV dequant scales [q, k, v] as host floats, or None.
            Required when enable_fp8_kv_cache is True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Attention output (FP16) and updated KV cache

    Raises:
        AssertionError: If enable_tree_attention is True but required tensors are missing
    """
    if enable_tree_attention:
        assert attention_mask is not None, "attention_mask should be provided for tree attention"
        assert position_ids is not None, "position_ids should be provided for tree attention"
    if enable_fp8_kv_cache:
        assert qkv_scales is not None and len(qkv_scales) == 3, \
            "qkv_scales must have 3 elements [q, k, v] when FP8 KV cache is enabled"

    batch_size_q, seq_len_q, q_size = q.shape
    batch_size_k, seq_len_k, k_size = k.shape
    batch_size_v, seq_len_v, v_size = v.shape
    assert (
        batch_size_q == batch_size_k == batch_size_v
    ), f"batch_size of q/k/v should be equal. Got {batch_size_q}, {batch_size_k}, {batch_size_v}"
    assert (
        seq_len_q == seq_len_k == seq_len_v
    ), f"seq_len of q/k/v should be equal. Got {seq_len_q}, {seq_len_k}, {seq_len_v}"

    batch_size, seq_len = batch_size_q, seq_len_q

    assert (
        q_size == head_size * num_q_heads
    ), f"q_size {q_size} should be equal to head_size * num_q_heads {head_size * num_q_heads}"
    assert (
        k_size == head_size * num_kv_heads
    ), f"k_size {k_size} should be equal to head_size * num_kv_heads {head_size * num_kv_heads}"
    assert (
        v_size == head_size * num_kv_heads
    ), f"v_size {v_size} should be equal to head_size * num_kv_heads {head_size * num_kv_heads}"

    assert (
        past_key_value.shape[0] == batch_size
    ), f"batch_size of kv_cache {past_key_value.shape[0]} should be equal to batch_size of q/k/v {batch_size}"
    assert past_key_value.shape[
        1] == 2, f"kv_cache {past_key_value.shape[1]} should have 2 tensors"
    assert (
        past_key_value.shape[2] == num_kv_heads
    ), f"num_kv_heads of kv_cache {past_key_value.shape[2]} should be equal to num_kv_heads of k/v {num_kv_heads}"
    assert (
        past_key_value.shape[4] == head_size
    ), f"head_size of kv_cache {past_key_value.shape[4]} should be equal to head_size of q/k/v {head_size}"

    assert q.dtype == torch.float16, f"q {q.dtype} should be in float16"
    assert k.dtype == torch.float16, f"k {k.dtype} should be in float16"
    assert v.dtype == torch.float16, f"v {v.dtype} should be in float16"
    assert past_key_value.dtype == torch.float16 or past_key_value.dtype == torch.float8_e4m3fn, f"past_key_value {past_key_value.dtype} should be in float16, float8_e4m3fn"

    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=q.dtype,
                              device=q.device)

    return attn_output, past_key_value.clone()


@torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
def vit_attention_plugin(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_carrier: torch.Tensor,
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """
    Dummy TensorRT operation for ViT attention during ONNX export.

    Args:
        q: Query tensor [total_S, H, D] in head-major layout.
        k: Key tensor [total_S, H, D] in head-major layout.
        v: Value tensor [total_S, H, D] in head-major layout.
        cu_seqlens: Prefix sum of sequence lengths [B+1].
        max_seqlen_carrier: Shape-only input carrying max sequence length hint.
        num_heads: Number of heads.
        head_size: Head size.
    """
    # Output has the same shape as q: [total_S, H, D]
    return torch.zeros_like(q)


def register_attention_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::attention_plugin",
                                symbolic_attention_plugin, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::vit_attention_plugin",
                                symbolic_vit_attention_plugin,
                                ONNX_OPSET_VERSION)

    print("Registered ONNX symbolic functions for custom attention plugin")
