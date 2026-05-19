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
ONNX / PyTorch integration for TensorRT ``Nvfp4MoePlugin`` (FP16 hidden + NVFP4 weights).

Mirrors ``int4_moe_plugin.py``: ONNX ``OpSchema``, symbolic export, ``torch.library.custom_op`` stub,
and ``nn.Module`` wrapper. TensorRT engine build and inference live in tests (see
``tests/python-unittests/test_nvfp4_moe_plugin_accuracy.py``).

**Plugin version 2 contract (breaking vs v1):** the plugin accepts FP16 hidden states only.
``hidden_block_scale`` is removed from the input list; activation FP4 quantization needed by
the prefill path is computed inside the plugin via ``fp4Quantize``. The user supplies only
``hidden_global_scale`` — a ``[2]`` FP32 tensor with ``[0]`` = FC1 activation global scale and
``[1]`` = FC2 activation global scale. Engines emitted by this file are not compatible with
plugin version 1 — re-export + rebuild is required.

The C++ plugin uses router logits ``[batch * seq_len, E]`` and FP16 hidden states
``[batch, seq_len, hidden_size]`` (``seq_len`` may be ``> 1``). Routing inside the plugin is
``kernel::moeTopkSoftmax`` (softmax over experts, top-k, renormalize); see
:meth:`NemotronHMoEW4A4Plugin.moe_topk_softmax_renormalize_torch`. ``hidden_size`` and ``moe_inter_size`` must be
multiples of 64 (decode GEMV / Marlin tile chunks). **Plugin v1 weight byte layout (both FCs N-major):**
Up qweights are INT8 ``[E, hidden_size, moe_inter_size/2]`` (N = moe_inter_size innermost; two NVFP4 nibbles
per byte along N) with prefill block scales ``[E, moe_inter_size, hidden_size/16]``. Down weights are INT8
``[E, moe_inter_size, hidden_size/2]`` (N = hidden_size innermost) with block scales
``[E, hidden_size, moe_inter_size/16]``. Activations are dense FP16. Plugin output tensors are FP16.

HuggingFace ``NemotronHMoE`` (``transformers.models.nemotron_h.modeling_nemotron_h``) uses
``NemotronHTopkRouter`` / ``DeepseekV3TopkRouter`` for **pre-routing logits** (FP32 matmul), then
``route_tokens_to_experts`` (sigmoid, grouped top-k, correction bias). That post-processing is **not** in
the TRT plugin; :class:`NemotronHMoEW4A4Plugin` matches the router **linear** only via
:meth:`NemotronHMoEW4A4Plugin.nemotron_h_plugin_router_logits`.

**Expert weight layout:** HuggingFace ``NemotronHExperts`` stores ``up_proj`` as ``[E, I, H_in]`` and
``down_proj`` as ``[E, H_in, I]`` (``I`` = ``moe_intermediate_size``, ``H_in`` = ``moe_latent_size`` or
``hidden_size``). The TensorRT ``Nvfp4MoePlugin`` / Marlin pack path uses ``w_up_ehi`` ``[E, H_in, I]`` and
``w_down_eih`` ``[E, I, H_in]``; use ``experts.up_proj.data.transpose(1, 2)`` /
``experts.down_proj.data.transpose(1, 2)`` into Marlin layout (see :meth:`NemotronHMoEW4A4Plugin.pack_experts_weights_to_marlin`).
"""

from __future__ import annotations

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION
from ..marlin_converter import MarlinConverter

try:
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE
except Exception:  # pragma: no cover - optional dependency surface
    NemotronHMoE = None  # type: ignore[misc, assignment]

nvfp4_moe_plugin_schema = OpSchema(
    name="Nvfp4MoePlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    ("Custom TensorRT Nvfp4MoePlugin (v1): Top-K routing + FP16 hidden + NVFP4 weights; the plugin runtime "
     "dispatches between decode (per-token GEMV) and prefill (CuteDSL grouped GEMM) based on B*S. "
     "Hidden states are FP16 only; activation FP4 quantization for prefill is computed inside the plugin. "
     "FP32 router logits; up qweights INT8 [E, H, I/2] (N-major: N=I innermost) + block scales [E, I, H/16] "
     "(prefill SF atom M=I, K=H/16); down qweights INT8 [E, I, H/2] (N-major: N=H innermost) + "
     "block scales [E, H, I/16] (prefill SF atom M=H, K=I/16); FP32 per-expert scales for up and down; "
     "FP32 length-2 hidden_global_scale (FC1, FC2)."),
    inputs=[
        OpSchema.FormalParameter(
            name="router_logits",
            description=
            ("Router logits (batch * seq_len, E) FP32 before plugin routing; Nvfp4MoePlugin applies "
             "softmax + top-k + renormalize (moeTopkSoftmax) internally."),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="hidden_states",
            description=
            "FP16 activations (batch, seq_len, hidden_size). Activation NVFP4 quantization "
            "(if needed by the prefill path) is computed inside the plugin.",
            type_str="tensor(float16)",
        ),
        OpSchema.FormalParameter(
            name="hidden_global_scale",
            description=
            ("FP32 length-2 tensor of calibrated activation global scales. Slot [0] is the FC1 "
             "(up-proj) activation global scale; slot [1] is the FC2 (down-proj) activation global "
             "scale. Consumed by the prefill path's internal fp4Quantize / α compute; both slots are "
             "populated unconditionally. Forward-direction convention — the plugin computes 1/GS "
             "inside the quantize kernel."),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_qweights",
            description=
            ("NVFP4 up-proj quantized payload (E, H, I/2) INT8 with H=hidden_size, "
             "I=moe_inter_size (N-major byte layout: N axis innermost, 2 FP4 nibbles "
             "per byte along I)."),
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_blocks_scale",
            description=
            ("Up-proj NVFP4 block scales (E, I, H/16) INT8 — prefill-friendly SF "
             "atom swizzle with M=I, K=H/16; four raw FP8 E4M3 bytes per 4-group "
             "K tile."),
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_global_scale",
            description=
            ("Per-expert FP32 scale for up weights (E): S_max/448 with S_max the max FP32 block scale "
             "(max|w|/6 per 16-lane group) for that expert; FP8 block scales store (s/S_max)*448."
             ),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_qweights",
            description=
            ("NVFP4 down-proj quantized payload (E, I, H/2) INT8 with H=hidden_size, "
             "I=moe_inter_size (N-major byte layout: N axis innermost, 2 FP4 nibbles "
             "per byte along H)."),
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_blocks_scale",
            description=
            ("Down-proj NVFP4 block scales (E, H, I/16) INT8 — prefill-friendly SF "
             "atom swizzle with M=H, K=I/16; four raw FP8 E4M3 bytes per 4-group "
             "K tile."),
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_global_scale",
            description=
            ("Per-expert FP32 scale for down weights (E); same convention as fc_up_global_scale "
             "(S_max/448)."),
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="fc_up_blocks_scale_decode",
            description=
            ("Up-proj NVFP4 block scales (E, H/16, I) INT8 — **decode-friendly** "
             "row-major transposed layout. Same scheme-B scale values as "
             "fc_up_blocks_scale, transposed so the Marlin decode GEMV reads 64 "
             "contiguous per-element scale bytes per tile. Bytes are Marlin-"
             "projected (`decodeSfByteToFloat`-compatible), not raw IEEE FP8 E4M3."
             ),
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_blocks_scale_decode",
            description=
            ("Down-proj NVFP4 block scales (E, I/16, H) INT8 — **decode-friendly** "
             "row-major transposed layout. Same scheme-B scale values as "
             "fc_down_blocks_scale, transposed for decode GEMV tile reads; "
             "bytes are Marlin-projected."),
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="e_score_correction_bias",
            description=
            ("Per-expert bias FP32 (num_experts,). Used as a load-balancing bias by "
             "``moeSigmoidGroupTopk`` (``routing_mode == 1``) and as an optional pre-softmax "
             "bias by ``moeTopkSoftmax`` (``routing_mode == 0``). Pass zeros when no bias is "
             "desired."),
            type_str="tensor(float)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description=
            "MoE output (batch, seq_len, hidden_size) FP16, same shape as hidden_states.",
            type_str="tensor(float16)",
        ),
    ],
    type_constraints=[],
    attributes=[
        OpSchema.Attribute(
            name="num_experts",
            type=OpSchema.AttrType.INT,
            description="Number of experts E.",
            required=True,
        ),
        OpSchema.Attribute(
            name="top_k",
            type=OpSchema.AttrType.INT,
            description="Top-K experts per token.",
            required=True,
        ),
        OpSchema.Attribute(
            name="hidden_size",
            type=OpSchema.AttrType.INT,
            description="Hidden dimension (multiple of 64).",
            required=True,
        ),
        OpSchema.Attribute(
            name="moe_inter_size",
            type=OpSchema.AttrType.INT,
            description=
            "MoE intermediate size (multiple of 64; decode kernel strip tiling).",
            required=True,
        ),
        OpSchema.Attribute(
            name="activation_type",
            type=OpSchema.AttrType.INT,
            description=
            ("Expert MLP nonlinearity after up-proj: 0 = ReLU², 1 = SiLU "
             "(MoEActivationKind; see nvfp4MoePlugin.cpp nvfp4StoredActivationToKernelKind)."
             ),
            required=True,
        ),
        OpSchema.Attribute(
            name="quantization_group_size",
            type=OpSchema.AttrType.INT,
            description=
            "Marlin NVFP4 block scale group size along hidden_size (must be 16 for Nvfp4MoePlugin).",
            required=True,
        ),
        OpSchema.Attribute(
            name="n_group",
            type=OpSchema.AttrType.INT,
            description=
            ("Number of expert groups (used when ``routing_mode == 1``; must divide "
             "``num_experts``)."),
            required=False,
        ),
        OpSchema.Attribute(
            name="topk_group",
            type=OpSchema.AttrType.INT,
            description=
            ("Number of expert groups to select per token (used when ``routing_mode == 1``; "
             "must be in ``[1, n_group]``)."),
            required=False,
        ),
        OpSchema.Attribute(
            name="norm_topk_prob",
            type=OpSchema.AttrType.INT,
            description=
            ("Renormalize the selected top-k weights so they sum to 1 (0 = no, 1 = yes). "
             "Serialized as INT for ONNX portability."),
            required=False,
        ),
        OpSchema.Attribute(
            name="routed_scaling_factor",
            type=OpSchema.AttrType.FLOAT,
            description=(
                "Scalar multiplier applied to the selected top-k weights (only "
                "``moeSigmoidGroupTopk``; default ``1.0``)."),
            required=False,
        ),
        OpSchema.Attribute(
            name="routing_mode",
            type=OpSchema.AttrType.INT,
            description=
            ("Router kernel selector: ``0`` = ``moeTopkSoftmax`` (softmax + flat top-k + "
             "renormalize, default), ``1`` = ``moeSigmoidGroupTopk`` (sigmoid + grouped top-k "
             "+ renormalize + scale; NemotronH)."),
            required=False,
        ),
    ],
)
onnx.defs.register_schema(nvfp4_moe_plugin_schema)


@symbolic_helper.parse_args(
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "v",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "i",
    "f",
    "i",
)
def symbolic_nvfp4_moe_plugin(
    g: object,
    router_logits: torch._C.Value,
    hidden_states: torch._C.Value,
    hidden_global_scale: torch._C.Value,
    fc_up_qweights: torch._C.Value,
    fc_up_blocks_scale: torch._C.Value,
    fc_up_global_scale: torch._C.Value,
    fc_down_qweights: torch._C.Value,
    fc_down_blocks_scale: torch._C.Value,
    fc_down_global_scale: torch._C.Value,
    fc_up_blocks_scale_decode: torch._C.Value,
    fc_down_blocks_scale_decode: torch._C.Value,
    e_score_correction_bias: torch._C.Value,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
):
    output = g.op(
        "trt::Nvfp4MoePlugin",
        router_logits,
        hidden_states,
        hidden_global_scale,
        fc_up_qweights,
        fc_up_blocks_scale,
        fc_up_global_scale,
        fc_down_qweights,
        fc_down_blocks_scale,
        fc_down_global_scale,
        fc_up_blocks_scale_decode,
        fc_down_blocks_scale_decode,
        e_score_correction_bias,
        num_experts_i=num_experts,
        top_k_i=top_k,
        hidden_size_i=hidden_size,
        moe_inter_size_i=moe_inter_size,
        activation_type_i=activation_type,
        quantization_group_size_i=quantization_group_size,
        n_group_i=n_group,
        topk_group_i=topk_group,
        norm_topk_prob_i=norm_topk_prob,
        routed_scaling_factor_f=routed_scaling_factor,
        routing_mode_i=routing_mode,
    )
    hs_sizes = _get_tensor_sizes(hidden_states)
    output.setType(router_logits.type().with_dtype(torch.float16).with_sizes(
        [hs_sizes[0], hs_sizes[1], hs_sizes[2]]))
    return output


@torch.library.custom_op("trt::nvfp4_moe_plugin", mutates_args=())
def nvfp4_moe_plugin(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_global_scale: torch.Tensor,
    fc_up_qweights: torch.Tensor,
    fc_up_blocks_scale: torch.Tensor,
    fc_up_global_scale: torch.Tensor,
    fc_down_qweights: torch.Tensor,
    fc_down_blocks_scale: torch.Tensor,
    fc_down_global_scale: torch.Tensor,
    fc_up_blocks_scale_decode: torch.Tensor,
    fc_down_blocks_scale_decode: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
) -> torch.Tensor:
    """
    Placeholder for ONNX tracing; TensorRT ``Nvfp4MoePlugin`` executes the real path (router
    kernel dispatch between ``moeTopkSoftmax`` and ``moeSigmoidGroupTopk`` on ``router_logits``).
    """
    del (
        hidden_global_scale,
        fc_up_qweights,
        fc_up_blocks_scale,
        fc_up_global_scale,
        fc_down_qweights,
        fc_down_blocks_scale,
        fc_down_global_scale,
        fc_up_blocks_scale_decode,
        fc_down_blocks_scale_decode,
        e_score_correction_bias,
        num_experts,
        top_k,
        moe_inter_size,
        activation_type,
        quantization_group_size,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
        routing_mode,
    )
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be (batch, seq_len, hidden_size), got shape {tuple(hidden_states.shape)}"
        )
    b, s, hd = hidden_states.shape
    if int(hd) != int(hidden_size):
        raise ValueError(
            f"hidden_states last dim {hd} != hidden_size {hidden_size}")
    num_tokens = int(b) * int(s)
    if int(router_logits.shape[0]) != num_tokens:
        raise ValueError(
            f"router_logits leading dim {int(router_logits.shape[0])} must equal batch*seq_len ({num_tokens})"
        )
    return torch.zeros(
        b,
        s,
        hidden_size,
        dtype=torch.float16,
        device=router_logits.device,
    )


class NemotronHMoEW4A4Plugin(nn.Module):
    """
    ONNX export wrapper for TensorRT ``Nvfp4MoePlugin`` (Nemotron-style routed experts only).

    Experts follow **Nemotron-H** layout: ``down( act( up(x) ) )`` with no fused Qwen3-style
    ``gate_up_proj`` / GLU. Only :class:`NemotronHMoE` is supported (``NemotronHMoE(config, layer_idx)``).

    Router logits are computed with :meth:`nemotron_h_plugin_router_logits` so they match
    ``NemotronHTopkRouter``’s FP32 linear (see ``modeling_nemotron_h.py``). The TRT plugin then runs
    :meth:`moe_topk_softmax_renormalize_torch`’s equivalent in CUDA—not HF ``route_tokens_to_experts``
    (sigmoid, ``e_score_correction_bias``, grouped top-k).

    Router logits are ``(num_tokens, num_experts)`` with ``num_tokens = batch * seq_len``; FP16
    ``hidden_states`` are ``(batch, seq_len, hidden_size)`` (same linear memory as a flattened
    ``(num_tokens, hidden_size)`` view). The custom op returns ``(batch, seq_len, hidden_size)`` in FP16,
    matching HF layout.

    Packed expert weights follow plugin layout ``w_up_ehi`` ``[E,H,I]`` / ``w_down_eih`` ``[E,I,H]``;
    HF ``NemotronHExperts`` stores ``up_proj`` / ``down_proj`` as ``[E,I,H]`` / ``[E,H,I]`` (transpose
    last two axes for Marlin packing).
    """

    @staticmethod
    def moe_topk_softmax_renormalize_torch(
        router_logits: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch equivalent of TensorRT ``kernel::moeTopkSoftmax`` with ``renormalize=True`` (see
        ``cpp/kernels/moe/moeTopkSoftmaxKernels.h`` / ``Nvfp4MoePlugin::enqueue``).

        Applies softmax on the expert dimension, takes the top-k probabilities, then renormalizes those
        weights so each row sums to 1. Returned indices are ``int32`` to match plugin workspace tensors.

        **Tie-breaking:** ``torch.topk`` ordering for equal probabilities is not guaranteed to match CUDA
        ``moeTopkSoftmax`` (lower expert index wins). For strict parity with the TRT plugin, use
        :meth:`moe_topk_softmax_renormalize_numpy` on logits in NumPy.
        """
        logits = router_logits.float()
        probs = F.softmax(logits, dim=-1)
        k = min(int(top_k), probs.shape[-1])
        topw, topi = torch.topk(probs, k, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
        return topw, topi.to(torch.int32)

    @staticmethod
    def moe_activation_numpy(z: np.ndarray,
                             activation_type: int) -> np.ndarray:
        """
        FP32 expert nonlinearity matching plugin / C++ ``referenceMoeActivation``: ``0`` = ReLU², ``1`` = SiLU
        (clipped to [-50, 50] before ``sigmoid`` form).
        """
        z = np.asarray(z, dtype=np.float32)
        if int(activation_type) == 1:
            zc = np.clip(z, -50.0, 50.0)
            return (zc / (1.0 + np.exp(-zc))).astype(np.float32)
        t = np.maximum(z, 0.0)
        return (t * t).astype(np.float32)

    @staticmethod
    def moe_topk_softmax_renormalize_numpy(
        router_logits: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NumPy softmax + top-k + renormalize aligned with CUDA ``moeTopkSoftmax``: iterative argmax with
        masking, **lower expert id wins on equal probability** (same as the fused kernel's warp reduce).
        """
        logits = np.asarray(router_logits, dtype=np.float32)
        m = np.max(logits, axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        probs = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)
        num_tokens, num_experts = probs.shape
        k = min(int(top_k), int(num_experts))
        topw = np.zeros((num_tokens, k), dtype=np.float32)
        topi = np.zeros((num_tokens, k), dtype=np.int32)
        for t in range(int(num_tokens)):
            row = probs[t]
            used = np.zeros(num_experts, dtype=bool)
            for ki in range(k):
                best_e = -1
                best_p = np.float32(-1.0)
                for e in range(int(num_experts)):
                    if used[e]:
                        continue
                    p = row[e]
                    if best_e < 0 or p > best_p or (p == best_p
                                                    and e < best_e):
                        best_p = p
                        best_e = e
                used[best_e] = True
                topi[t, ki] = np.int32(best_e)
                topw[t, ki] = best_p
            denom = float(np.sum(topw[t])) + 1e-20
            topw[t] = (topw[t] / denom).astype(np.float32)
        return topw, topi

    @staticmethod
    def nemotron_h_plugin_router_logits(hidden_flat: torch.Tensor,
                                        gate: nn.Linear) -> torch.Tensor:
        """
        Compute router logits for the C++ ``moeTopkSoftmax`` kernel, matching HuggingFace
        ``NemotronHTopkRouter`` which uses sigmoid-based routing (not softmax).

        HF forward: ``scores = sigmoid(F.linear(x, w))``, top-k, renormalize, × routed_scaling_factor.
        The C++ plugin applies ``softmax(logits) → top-k → renorm``.

        To make ``softmax(z) / renorm == sigmoid(l) / renorm`` for the same top-k set, we pass
        ``z = log(sigmoid(l)) = F.logsigmoid(l)`` so that ``exp(z) = sigmoid(l)`` and
        ``softmax(log_sigmoid) == sigmoid(l) / Σ sigmoid`` — exactly the HF weight distribution.

        Expert selection is unchanged (log, sigmoid are both monotone → same top-k order).
        ``routed_scaling_factor`` is applied separately in ``forward()`` to match the HF ×scale step.
        """
        x = hidden_flat.float()
        w = gate.weight.float()
        if gate.bias is not None:
            logits = F.linear(x, w, gate.bias.float())
        else:
            logits = F.linear(x, w)
        # Transform raw logits → log(sigmoid(l)) so moeTopkSoftmax produces sigmoid-normalized weights.
        return F.logsigmoid(logits)

    @staticmethod
    def _nemotron_h_moe_gate_as_linear(gate: nn.Module, hidden_size: int,
                                       num_experts: int) -> nn.Linear:
        """
        Nemotron-H ``NemotronHMoE`` may use ``nn.Linear`` or ``NemotronHTopkRouter`` (DeepSeek-style
        ``weight`` matrix + ``F.linear``, no bias). Return an ``nn.Linear`` with the same mapping for
        plugin ONNX tracing.
        """
        if isinstance(gate, nn.Linear):
            return gate
        weight = getattr(gate, "weight", None)
        if isinstance(weight, nn.Parameter) and tuple(
                weight.shape) == (num_experts, hidden_size):
            lin = nn.Linear(
                hidden_size,
                num_experts,
                bias=False,
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                lin.weight.copy_(weight)
            return lin
        raise TypeError(
            "NemotronHMoEW4A4Plugin expects moe_block.gate to be nn.Linear or a top-k router with "
            f"parameter weight of shape ({num_experts}, {hidden_size}); got {type(gate).__name__}"
        )

    def __init__(
        self,
        moe_block: "NemotronHMoE",
        *,
        activation_type: int = 0,
        quantization_group_size: int = 16,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: int = 1,
        routed_scaling_factor: float = 1.0,
        routing_mode: int = 0,
    ):
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers.models.nemotron_h import failed)."
            )
        if not isinstance(moe_block, NemotronHMoE):
            raise TypeError(
                f"NemotronHMoEW4A4Plugin only supports NemotronHMoE, not {type(moe_block).__name__} "
                "(Qwen3 MoE uses fused gate/up and is incompatible with this plugin)."
            )
        super().__init__()
        cfg = moe_block.config
        hidden_size = int(cfg.hidden_size)
        latent = getattr(cfg, "moe_latent_size", None)
        expert_in = int(latent) if latent is not None else hidden_size
        if expert_in != hidden_size:
            raise ValueError(
                "NemotronHMoEW4A4Plugin needs expert input dim == hidden_size "
                f"(set moe_latent_size=None or {hidden_size}); got moe_latent_size={latent}"
            )
        # trust_remote_code NemotronHMoE: n_routed_experts/top_k live on .gate or
        # .experts.num_experts (NemotronHExperts has no __len__, use lazy eval).
        if hasattr(moe_block, "n_routed_experts"):
            num_experts = int(moe_block.n_routed_experts)
        elif hasattr(moe_block.gate, "n_routed_experts"):
            num_experts = int(moe_block.gate.n_routed_experts)
        else:
            num_experts = int(moe_block.experts.num_experts)
        gate_linear = NemotronHMoEW4A4Plugin._nemotron_h_moe_gate_as_linear(
            moe_block.gate, hidden_size, num_experts)
        self._init_from_gate_and_dims(
            num_experts=num_experts,
            top_k=int(moe_block.top_k),
            hidden_size=hidden_size,
            moe_inter_size=int(cfg.moe_intermediate_size),
            gate_layer=gate_linear,
            activation_type=int(activation_type),
            quantization_group_size=int(quantization_group_size),
            n_group=int(n_group),
            topk_group=int(topk_group),
            norm_topk_prob=int(norm_topk_prob),
            routed_scaling_factor=float(routed_scaling_factor),
            routing_mode=int(routing_mode),
        )

    @classmethod
    def from_nemotron_h_moe(
        cls,
        moe_block: "NemotronHMoE",
        *,
        activation_type: int = 0,
        quantization_group_size: int = 16,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: int = 1,
        routed_scaling_factor: float = 1.0,
        routing_mode: int = 0,
    ) -> "NemotronHMoEW4A4Plugin":
        """Alias for ``NemotronHMoEW4A4Plugin(moe_block, ...)``."""
        return cls(
            moe_block,
            activation_type=activation_type,
            quantization_group_size=quantization_group_size,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            routing_mode=routing_mode,
        )

    def _init_from_gate_and_dims(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_inter_size: int,
        gate_layer: nn.Linear,
        activation_type: int = 0,
        quantization_group_size: int = 16,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: int = 1,
        routed_scaling_factor: float = 1.0,
        routing_mode: int = 0,
    ) -> None:
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.moe_inter_size = int(moe_inter_size)
        self.activation_type = int(activation_type)
        self.quantization_group_size = int(quantization_group_size)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.norm_topk_prob = int(norm_topk_prob)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.routing_mode = int(routing_mode)

        if self.quantization_group_size != 16:
            raise ValueError(
                "Nvfp4MoePlugin only supports quantization_group_size == 16 "
                f"(Marlin NVFP4 tile scales); got {self.quantization_group_size}"
            )
        if self.hidden_size % self.quantization_group_size != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be a multiple of "
                f"quantization_group_size ({self.quantization_group_size})")

        if self.hidden_size % 64 != 0:
            raise ValueError(
                f"hidden_size must be a multiple of 64, got {self.hidden_size}"
            )
        if self.moe_inter_size % 64 != 0:
            raise ValueError(
                f"moe_inter_size must be a multiple of 64 for Nvfp4MoePlugin (decode GEMV strips), "
                f"got {self.moe_inter_size}")

        self.num_chunks = self.hidden_size // 64

        if gate_layer.out_features != self.num_experts:
            raise ValueError(
                f"gate.out_features ({gate_layer.out_features}) must equal num_experts ({self.num_experts})"
            )
        if gate_layer.in_features != self.hidden_size:
            raise ValueError(
                f"gate.in_features ({gate_layer.in_features}) must equal hidden_size ({self.hidden_size})"
            )

        # FP32 weights match ``NemotronHTopkRouter`` / ``DeepseekV3TopkRouter`` (FP32 matmul). Using FP16 here
        # would quantize router weights and diverge from HF logits; call ``module.half()`` before export if FP16
        # parameters are required for the rest of the graph.
        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=gate_layer.bias is not None,
            dtype=torch.float32,
        )
        self.gate.weight.data = gate_layer.weight.data.clone().to(
            torch.float32)
        if gate_layer.bias is not None:
            self.gate.bias.data = gate_layer.bias.data.clone().to(
                torch.float32)

        # Plugin v3 uses K-major (CuteDSL-compatible) weight byte layout to unify decode and
        # prefill on a single weight copy. FC1: bytes [E, I, H/2] with H as inner (contraction),
        # 2 FP4 nibbles per byte packed along H. FC2: bytes [E, H, I/2] with I as inner.
        # SF atom-swizzle has M=N (FC1 M=I, FC2 M=H) and 16-element blocks run along the K axis.
        e, inter = self.num_experts, self.moe_inter_size
        h_half = self.hidden_size // 2
        i_half = self.moe_inter_size // 2
        h_per_group = self.hidden_size // self.quantization_group_size
        i_per_group = self.moe_inter_size // self.quantization_group_size
        # FC1 up: N-major weight byte layout [E, H, I/2] — N (= I) innermost,
        # 2 FP4 nibbles per byte along I. Matches what the CuteDSL N-major
        # prefill kernel and the Marlin decode GEMV both consume.
        self.register_buffer(
            "fc_up_qweights",
            torch.zeros(e, self.hidden_size, i_half, dtype=torch.int8))
        # FC1 up SF: prefill-friendly atom swizzle (M=I, K=H/16). Scheme-B
        # quantization — scales per ``(n, k/16) = (i, h/16)`` — matching
        # ``tcgen05.mma`` block-scaled MMA contract and ModelOpt's NVFP4
        # convention (quantize along the last/K axis). Atom layout requires
        # M padded to 128 and K padded to 4 — matches the plugin's
        # ``supportsFormatCombination`` case 4 contract.
        up_sf_m_padded = ((inter + 127) // 128) * 128
        up_sf_k_padded = ((h_per_group + 3) // 4) * 4
        self.register_buffer(
            "fc_up_blocks_scale",
            torch.zeros(e, up_sf_m_padded, up_sf_k_padded, dtype=torch.int8))
        self.register_buffer("fc_up_global_scale",
                             torch.ones(e, dtype=torch.float32))
        # FC2 down: N-major weight byte layout [E, I, H/2] — N (= H) innermost,
        # 2 FP4 nibbles per byte along H. Symmetric to FC1 in v5; both FC1 and
        # FC2 now use N-major bytes and a single weight copy feeds the plugin's
        # prefill and (when re-enabled) Marlin decode paths.
        self.register_buffer("fc_down_qweights",
                             torch.zeros(e, inter, h_half, dtype=torch.int8))
        # FC2 down SF: prefill-friendly atom swizzle (M=H, K=I/16). Scheme-B
        # scales per ``(h, i/16)``. Atom-padded to match
        # ``supportsFormatCombination`` case 7.
        dn_sf_m_padded = ((self.hidden_size + 127) // 128) * 128
        dn_sf_k_padded = ((i_per_group + 3) // 4) * 4
        self.register_buffer(
            "fc_down_blocks_scale",
            torch.zeros(e, dn_sf_m_padded, dn_sf_k_padded, dtype=torch.int8))
        self.register_buffer("fc_down_global_scale",
                             torch.ones(e, dtype=torch.float32))
        # Decode-friendly SF: row-major transposed, Marlin-projected FP8 bytes.
        # Shape [E, H/16, I] / [E, I/16, H] — same scheme-B scale values as the
        # prefill SFs above, transposed so the Marlin decode GEMV reads 64
        # contiguous per-element scale bytes per (expert, h_group=j/16, inter
        # chunk) tile. Byte encoding is Marlin-projected (shift-by-7 recovers
        # the FP16 after decodeSfByteToFloat), distinct from the raw IEEE FP8
        # E4M3 bytes CuteDSL consumes from the prefill SF.
        self.register_buffer(
            "fc_up_blocks_scale_decode",
            torch.zeros(e, h_per_group, inter, dtype=torch.int8))
        self.register_buffer(
            "fc_down_blocks_scale_decode",
            torch.zeros(e, i_per_group, self.hidden_size, dtype=torch.int8))
        # FP32 length-2 forward activation global scales: [0] FC1, [1] FC2. Populated by the
        # calibration pipeline (see populate_hidden_global_scales). Plugin v2 does not consume a
        # separate block-scale input — activation block scales are produced inside the plugin
        # via fp4Quantize.
        self.register_buffer("hidden_global_scale",
                             torch.ones(2, dtype=torch.float32))
        # Per-expert load-balancing bias FP32 (num_experts,). Consumed by moeSigmoidGroupTopk
        # (routing_mode == 1) and optionally by moeTopkSoftmax (routing_mode == 0). Default zero
        # so softmax routing is a no-op.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(
                f"hidden_dim {hidden_dim} != hidden_size {self.hidden_size}")

        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = type(self).nemotron_h_plugin_router_logits(
            hidden_flat, self.gate)

        if hidden_states.dtype != torch.float16:
            raise TypeError(
                f"Nvfp4MoePlugin (v2) accepts FP16 hidden_states only; got {hidden_states.dtype}."
                " Cast with ``.half()`` before passing the tensor to this module."
            )
        out = nvfp4_moe_plugin(
            router_logits,
            hidden_states,
            self.hidden_global_scale,
            self.fc_up_qweights,
            self.fc_up_blocks_scale,
            self.fc_up_global_scale,
            self.fc_down_qweights,
            self.fc_down_blocks_scale,
            self.fc_down_global_scale,
            self.fc_up_blocks_scale_decode,
            self.fc_down_blocks_scale_decode,
            self.e_score_correction_bias,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_inter_size,
            self.activation_type,
            self.quantization_group_size,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
            self.routing_mode,
        )
        return out

    def populate_prefill_plugin_buffers(
        self,
        w_up_ehi,
        w_down_eih,
    ) -> None:
        """Pack dense weights into the plugin-v5 byte layout (N-major FC1 AND FC2).

        ``w_up_ehi`` is ``[E, H, I]`` dense FP32 / BF16 (same HF-transposed layout the
        legacy Marlin packer consumes); ``w_down_eih`` is ``[E, I, H]``. The output
        byte layouts:

        - FC1 up: ``fc_up_qweights[E, H, I/2]`` bytes — outer axis H, inner axis I/2,
          2 FP4 nibbles per byte along I (**N-major**, N = I).
        - FC2 down: ``fc_down_qweights[E, I, H/2]`` bytes — outer axis I, inner axis H/2,
          2 FP4 nibbles per byte along H (**N-major**, N = H; new in v5).
        - Both FC1 and FC2 bytes match the layout the CuteDSL N-major prefill kernels
          read AND what the Marlin decode GEMV expects, so a single weight copy serves
          both prefill and decode end-to-end.
        - SF atom swizzle: 128×4 tile with M = N, K = K/16 — prefill-friendly for both
          FC1 (M=I, K=H/16) and FC2 (M=H, K=I/16). SF layouts are unchanged vs v3/v4;
          only weight byte axes move.
        - SF bytes are **raw IEEE FP8 E4M3 bytes** (not Marlin-projected). CuteDSL
          reinterprets them directly as ``__nv_fp8_e4m3`` values.

        Implementation note: both FCs quantize in the K-major iteration order (outer =
        N axis, inner = 64 K values per tile) so the per-tile scales land in the
        prefill SF layout. The packed FP4 bytes are produced in K-major layout in a
        per-expert scratch buffer and then nibble-transposed to the final N-major
        layout. This avoids a second quantization pass while still emitting the
        correct byte layout.

        Per-expert global scale convention is ``s_max / 448`` with
        ``s_max = max|W_e| / 6``.
        """
        e, h, inter = w_up_ehi.shape
        assert w_down_eih.shape == (e, inter, h)
        if int(e) != int(self.num_experts):
            raise ValueError(
                f"w_up_ehi expert count {e} != self.num_experts {self.num_experts}"
            )
        if int(h) != int(self.hidden_size) or int(inter) != int(
                self.moe_inter_size):
            raise ValueError(
                f"w_up_ehi shape ({e},{h},{inter}) mismatches plugin "
                f"(hidden_size={self.hidden_size}, moe_inter_size={self.moe_inter_size})"
            )
        num_hidden_chunks = h // 64
        num_inter_chunks = inter // 64
        assert num_hidden_chunks * 64 == h
        assert num_inter_chunks * 64 == inter

        w_up_f = w_up_ehi.float()
        w_dn_f = w_down_eih.float()
        s_max_up = (w_up_f.abs().amax(dim=(1, 2)) / 6.0).clamp(min=1e-12)
        s_max_dn = (w_dn_f.abs().amax(dim=(1, 2)) / 6.0).clamp(min=1e-12)
        s_max_up_np = s_max_up.detach().cpu().numpy()
        s_max_dn_np = s_max_dn.detach().cpu().numpy()
        k_fp8 = MarlinConverter.FP8_MAX

        up_pl = self.fc_up_qweights
        up_bs = self.fc_up_blocks_scale
        dn_pl = self.fc_down_qweights
        dn_bs = self.fc_down_blocks_scale
        up_bs_decode = self.fc_up_blocks_scale_decode
        dn_bs_decode = self.fc_down_blocks_scale_decode

        # v6 scheme-B: quantization blocks run along the **K (contraction) axis**
        # of the ``[N, K]`` weight — H for FC1, I for FC2 — matching the NVFP4
        # ecosystem convention (ModelOpt emits per-``(n, k/16)`` scales) and the
        # ``tcgen05.mma`` block-scaled MMA contract consumed by the CuteDSL
        # prefill kernel. Prefill SF is stored at atom ``(M=N, K=K/16)`` with
        # raw IEEE FP8 E4M3 bytes (CuteDSL ``__nv_fp8_e4m3`` cast). Decode SF
        # is a second atom copy at ``(M=H, K=I/16)`` / ``(M=I, K=H/16)`` with
        # Marlin-projected bytes — see the decode-SF-orientation analysis doc
        # for why the decode kernel's tile model forces a different axis; with
        # scheme-B scales the decode path is approximate (≈0.90 cos) and is
        # currently gated off at ``kPrefillDispatchThreshold = 0``.
        num_sf_cols_up = h // 16  # FC1 prefill / decode: K = H → SF cols = H/16
        num_sf_cols_dn = inter // 16  # FC2 prefill / decode: K = I → SF cols = I/16

        # Numpy views of the dense weights for fast indexing inside the packing loops.
        w_up_np = np.ascontiguousarray(w_up_f.detach().cpu().numpy())
        w_dn_np = np.ascontiguousarray(w_dn_f.detach().cpu().numpy())

        # Helper: quantize 64 FP32 values → (32 bytes FP4 nibbles, 4 raw FP8 E4M3
        # bytes, 4 FP32 normalized targets).
        # Unlike MarlinConverter's output, which Marlin-projects the FP8 bits so their
        # `dequant_fp8_scales` decoder recovers the right FP16, this writes the IEEE
        # FP8 E4M3 byte directly so CuteDSL's `__nv_fp8_e4m3` cast gets the right value.
        # The raw FP32 targets (scales / s_max * 448) are returned so callers can
        # Marlin-project them for the decode SF copy.
        def _quantize_fp4_tile_with_raw_fp8_sf(
                vec64: np.ndarray,
                s_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            v = np.asarray(vec64, dtype=np.float32).reshape(64)
            scales = np.empty(4, dtype=np.float32)
            for g in range(4):
                blk = v[g * 16:(g + 1) * 16]
                scales[g] = max(float(np.max(np.abs(blk))) / 6.0, 1e-12)
            # FP4 payload (reuse existing tile quantizer — it's layout-agnostic for the nibble side)
            pl, _ = MarlinConverter.quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
                v, expert_block_scale_max_fp32=s_max)
            # FP32 targets for the FP8 step: each is in `[0, 448]` range.
            sf_targets = (scales / s_max * k_fp8).astype(np.float32)
            # Raw FP8 E4M3 bytes from the same targets.
            sf_bytes = torch.from_numpy(sf_targets).to(
                torch.float8_e4m3fn).view(torch.uint8).numpy()
            return pl, sf_bytes, sf_targets

        for ex in range(e):
            s_max_up_ex = float(s_max_up_np[ex])
            s_max_dn_ex = float(s_max_dn_np[ex])

            # ---- FC1 up (scheme-B): iterate outer I, inner 64-H chunks.
            # Each tile quantizes 64 consecutive H values at fixed i → scale
            # per (i, h/16). Accumulate FP4 bytes in a scratch [I, H/2] buffer
            # (K-major), then nibble-transpose to the N-major [H, I/2] layout
            # that fc_up_qweights expects.
            up_kmajor_scratch = torch.zeros(inter,
                                            self.hidden_size // 2,
                                            dtype=torch.int8)
            up_kmajor_flat = up_kmajor_scratch.reshape(-1)
            up_bs_flat = up_bs[ex].view(torch.uint8).reshape(-1)
            for i in range(inter):
                for c in range(num_hidden_chunks):
                    up_seg = w_up_np[ex, c * 64:(c + 1) * 64,
                                     i].astype(np.float32, copy=False)
                    pl_u, fp8_u, _tgt_u = _quantize_fp4_tile_with_raw_fp8_sf(
                        up_seg, s_max_up_ex)
                    tile_u = i * num_hidden_chunks + c
                    up_kmajor_flat[tile_u * 32:(tile_u + 1) * 32].copy_(
                        torch.from_numpy(pl_u.view(np.int8)))
                    # Prefill SF: raw IEEE FP8 E4M3 bytes at atom(i, h/16).
                    for g in range(4):
                        sf_col = c * 4 + g
                        off = MarlinConverter.atom_sf_offset(
                            i, sf_col, num_sf_cols_up)
                        up_bs_flat[off] = int(fp8_u[g])

            # K-major [I, H/2] → N-major [H, I/2] via nibble transpose.
            kmajor_u8 = up_kmajor_scratch.view(torch.uint8)
            lo = kmajor_u8 & 0x0F
            hi = (kmajor_u8 >> 4) & 0x0F
            nib_ih = torch.empty(inter, self.hidden_size, dtype=torch.uint8)
            nib_ih[:, 0::2] = lo
            nib_ih[:, 1::2] = hi
            nib_hi = nib_ih.transpose(0, 1).contiguous()
            i_half_local = inter // 2
            lo_out = nib_hi[:, 0::2]
            hi_out = nib_hi[:, 1::2]
            packed = (lo_out | (hi_out << 4)).view(torch.int8)
            up_pl[ex].copy_(packed.reshape(self.hidden_size, i_half_local))

            # ---- FC2 down (scheme-B): iterate outer H, inner 64-I chunks.
            dn_kmajor_scratch = torch.zeros(h, inter // 2, dtype=torch.int8)
            dn_kmajor_flat = dn_kmajor_scratch.reshape(-1)
            dn_bs_flat = dn_bs[ex].view(torch.uint8).reshape(-1)
            for hh in range(h):
                for c in range(num_inter_chunks):
                    dn_seg = w_dn_np[ex, c * 64:(c + 1) * 64,
                                     hh].astype(np.float32, copy=False)
                    pl_d, fp8_d, _tgt_d = _quantize_fp4_tile_with_raw_fp8_sf(
                        dn_seg, s_max_dn_ex)
                    tile_d = hh * num_inter_chunks + c
                    dn_kmajor_flat[tile_d * 32:(tile_d + 1) * 32].copy_(
                        torch.from_numpy(pl_d.view(np.int8)))
                    for g in range(4):
                        sf_col = c * 4 + g
                        off = MarlinConverter.atom_sf_offset(
                            hh, sf_col, num_sf_cols_dn)
                        dn_bs_flat[off] = int(fp8_d[g])

            # K-major [H, I/2] → N-major [I, H/2].
            dn_kmajor_u8 = dn_kmajor_scratch.view(torch.uint8)
            lo = dn_kmajor_u8 & 0x0F
            hi = (dn_kmajor_u8 >> 4) & 0x0F
            nib_hi_axis = torch.empty(h, inter, dtype=torch.uint8)
            nib_hi_axis[:, 0::2] = lo
            nib_hi_axis[:, 1::2] = hi
            nib_ih_axis = nib_hi_axis.transpose(0, 1).contiguous()
            h_half_local = h // 2
            lo_out = nib_ih_axis[:, 0::2]
            hi_out = nib_ih_axis[:, 1::2]
            packed_dn = (lo_out | (hi_out << 4)).view(torch.int8)
            dn_pl[ex].copy_(packed_dn.reshape(inter, h_half_local))

            # ---- FC1 decode SF (scheme-B grouping, Marlin-projected, row-major [H/16, I]).
            # Same per-block scales as the prefill SF loop above (16 consecutive H values
            # per scale at each I). The decode kernel reads 64 contiguous scale bytes per
            # (e, h_group=j/16, inter_chunk c) tile at linear offset
            # ``up_bs_decode[ex][h_group * inter + c*64 .. c*64+63]``. Bytes are
            # Marlin-projected so ``decodeSfByteToFloat`` recovers the FP16 directly.
            w_up_ex = w_up_np[ex]  # [H, I]
            # Scheme-B scale per (h_group, i): max over 16 consecutive H rows.
            sf_up_scheme_b = np.abs(w_up_ex).reshape(num_sf_cols_up, 16,
                                                     inter).max(axis=1)
            sf_up_scheme_b = np.maximum(sf_up_scheme_b / 6.0, 1e-12)
            sf_up_targets_b = (sf_up_scheme_b / s_max_up_ex * k_fp8).astype(
                np.float32)  # [H/16, I]
            up_bs_decode_flat = up_bs_decode[ex].view(torch.uint8).reshape(-1)
            num_i_tiles_up = (inter + 3) // 4
            for h_g in range(num_sf_cols_up):
                row_base = h_g * inter
                for k_tile in range(num_i_tiles_up):
                    s = [0.0, 0.0, 0.0, 0.0]
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < inter:
                            s[g] = float(sf_up_targets_b[h_g, col])
                    scale_word = MarlinConverter.fp32x4_to_marlin_f8x4_block_scale(
                        s[0], s[1], s[2], s[3])
                    marlin_bytes = MarlinConverter.marlin_scale_word_to_raw_fp8_bytes(
                        scale_word)
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < inter:
                            up_bs_decode_flat[row_base + col] = int(
                                marlin_bytes[g])

            # ---- FC2 decode SF (scheme-B grouping, Marlin-projected, row-major [I/16, H]).
            w_dn_ex = w_dn_np[ex]  # [I, H]
            # Scheme-B scale per (i_group, h): max over 16 consecutive I rows.
            sf_dn_scheme_b = np.abs(w_dn_ex).reshape(num_sf_cols_dn, 16,
                                                     h).max(axis=1)
            sf_dn_scheme_b = np.maximum(sf_dn_scheme_b / 6.0, 1e-12)
            sf_dn_targets_b = (sf_dn_scheme_b / s_max_dn_ex * k_fp8).astype(
                np.float32)  # [I/16, H]
            dn_bs_decode_flat = dn_bs_decode[ex].view(torch.uint8).reshape(-1)
            num_h_tiles_dn = (h + 3) // 4
            for i_g in range(num_sf_cols_dn):
                row_base = i_g * h
                for k_tile in range(num_h_tiles_dn):
                    s = [0.0, 0.0, 0.0, 0.0]
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < h:
                            s[g] = float(sf_dn_targets_b[i_g, col])
                    scale_word = MarlinConverter.fp32x4_to_marlin_f8x4_block_scale(
                        s[0], s[1], s[2], s[3])
                    marlin_bytes = MarlinConverter.marlin_scale_word_to_raw_fp8_bytes(
                        scale_word)
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < h:
                            dn_bs_decode_flat[row_base + col] = int(
                                marlin_bytes[g])

        self.fc_up_global_scale.copy_(
            (s_max_up / k_fp8).to(device=self.fc_up_global_scale.device,
                                  dtype=self.fc_up_global_scale.dtype))
        self.fc_down_global_scale.copy_(
            (s_max_dn / k_fp8).to(device=self.fc_down_global_scale.device,
                                  dtype=self.fc_down_global_scale.dtype))

    def populate_decode_plugin_buffers(self) -> None:
        """Populate decode-friendly row-major transposed SF from the prefill SF.

        Reads ``fc_up_blocks_scale`` (atom-swizzled, raw IEEE FP8 E4M3 bytes at
        ``atom_sf_offset(i, h_group, H/16)``) and writes
        ``fc_up_blocks_scale_decode`` (row-major ``[E, H/16, I]`` with Marlin-
        projected bytes at ``[h_group * I + i]``). Similarly for FC2:
        ``fc_down_blocks_scale`` (atom ``(h, i_group)``) → ``fc_down_blocks_scale_decode``
        (row-major ``[E, I/16, H]`` at ``[i_group * H + h]``).

        The byte encoding is converted from raw IEEE FP8 E4M3 (CuteDSL-friendly)
        to Marlin-projected (the W4A16 decode GEMV's ``decodeSfByteToFloat`` shift-by-7
        trick). ``populate_prefill_plugin_buffers`` already writes both, so calling
        this afterward is idempotent (produces the same bytes).

        This helper is O(E · H · I / 16) — acceptable as a one-time load step; follow
        up with a vectorized variant if it becomes a bottleneck.
        """
        e = self.num_experts
        h = self.hidden_size
        inter = self.moe_inter_size
        h_per_group = h // self.quantization_group_size  # H/16
        i_per_group = inter // self.quantization_group_size  # I/16
        k_fp8 = MarlinConverter.FP8_MAX

        for ex in range(e):
            # --- FC1: read prefill atom (i, h_group) raw FP8 → decode linear [h_group, i] Marlin.
            up_bs_flat = self.fc_up_blocks_scale[ex].view(
                torch.uint8).reshape(-1)
            up_dec_flat = self.fc_up_blocks_scale_decode[ex].view(
                torch.uint8).reshape(-1)
            # Gather raw FP8 bytes at (i, h_g) positions into a [I, H/16] array.
            raw_up = np.empty((inter, h_per_group), dtype=np.uint8)
            for i in range(inter):
                for h_g in range(h_per_group):
                    off = MarlinConverter.atom_sf_offset(i, h_g, h_per_group)
                    raw_up[i, h_g] = int(up_bs_flat[off])
            # FP8 E4M3 byte → FP32 scalar (vectorized via torch FP8 dtype).
            sf_up_fp32 = torch.from_numpy(raw_up).view(
                torch.float8_e4m3fn).float().numpy()  # [I, H/16]
            # Write Marlin-projected bytes at linear [h_g * I + i], groups of 4 along I.
            num_i_tiles = (inter + 3) // 4
            for h_g in range(h_per_group):
                row_base = h_g * inter
                for k_tile in range(num_i_tiles):
                    s = [0.0, 0.0, 0.0, 0.0]
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < inter:
                            s[g] = float(sf_up_fp32[col, h_g])
                    scale_word = MarlinConverter.fp32x4_to_marlin_f8x4_block_scale(
                        s[0], s[1], s[2], s[3])
                    marlin_bytes = MarlinConverter.marlin_scale_word_to_raw_fp8_bytes(
                        scale_word)
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < inter:
                            up_dec_flat[row_base + col] = int(marlin_bytes[g])

            # --- FC2: read prefill atom (h, i_group) raw FP8 → decode linear [i_group, h] Marlin.
            dn_bs_flat = self.fc_down_blocks_scale[ex].view(
                torch.uint8).reshape(-1)
            dn_dec_flat = self.fc_down_blocks_scale_decode[ex].view(
                torch.uint8).reshape(-1)
            raw_dn = np.empty((h, i_per_group), dtype=np.uint8)
            for hh in range(h):
                for i_g in range(i_per_group):
                    off = MarlinConverter.atom_sf_offset(hh, i_g, i_per_group)
                    raw_dn[hh, i_g] = int(dn_bs_flat[off])
            sf_dn_fp32 = torch.from_numpy(raw_dn).view(
                torch.float8_e4m3fn).float().numpy()  # [H, I/16]
            num_h_tiles = (h + 3) // 4
            for i_g in range(i_per_group):
                row_base = i_g * h
                for k_tile in range(num_h_tiles):
                    s = [0.0, 0.0, 0.0, 0.0]
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < h:
                            s[g] = float(sf_dn_fp32[col, i_g])
                    scale_word = MarlinConverter.fp32x4_to_marlin_f8x4_block_scale(
                        s[0], s[1], s[2], s[3])
                    marlin_bytes = MarlinConverter.marlin_scale_word_to_raw_fp8_bytes(
                        scale_word)
                    for g in range(4):
                        col = k_tile * 4 + g
                        if col < h:
                            dn_dec_flat[row_base + col] = int(marlin_bytes[g])
        _ = k_fp8  # retained for symmetry with populate_prefill_plugin_buffers

    def populate_hidden_global_scales(
        self,
        fc1_input_scale: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        *,
        atol: float = 1e-6,
    ) -> None:
        """Populate :attr:`hidden_global_scale` from calibrated per-expert input scales.

        The checkpoint schema stores one ``input_scale`` per expert (shape ``[E]`` or a
        scalar). Across the 5,888 FC1 / FC2 tensors verified so far, those values are
        constant per layer — so the prefill path folds them to a single scalar per FC.
        This method asserts the per-expert invariance ``(max - min) / mean < atol`` and
        writes the max value into the corresponding ``hidden_global_scale`` slot. Failing
        the assertion likely indicates a checkpoint schema change that breaks the design's
        single-scalar-per-FC contract; fail loud rather than silently collapsing.
        """

        def _fold(tensor: torch.Tensor, label: str) -> float:
            flat = tensor.detach().to(torch.float32).flatten()
            if flat.numel() == 0:
                raise ValueError(f"{label}: empty tensor")
            if flat.numel() == 1:
                return float(flat.item())
            mean = float(flat.mean().item())
            if mean <= 0.0:
                raise ValueError(
                    f"{label}: non-positive mean ({mean}); expected forward-direction scale"
                )
            spread = float((flat.max() - flat.min()).item()) / mean
            if spread >= atol:
                raise ValueError(
                    f"{label}: per-expert spread (max-min)/mean={spread:.3e} exceeds atol={atol:.0e}. "
                    "The plugin v2 prefill path passes a single scalar per FC per layer; "
                    "checkpoints with diverging per-expert activation scales are not supported "
                    "without a kernel-level extension.")
            return float(flat.max().item())

        fc1_scalar = _fold(fc1_input_scale, "fc1_input_scale")
        fc2_scalar = _fold(fc2_input_scale, "fc2_input_scale")
        with torch.no_grad():
            self.hidden_global_scale.copy_(
                torch.tensor([fc1_scalar, fc2_scalar],
                             dtype=self.hidden_global_scale.dtype,
                             device=self.hidden_global_scale.device))

    def pack_experts_weights_to_marlin(self, moe: "NemotronHMoE") -> None:
        """
        Pack **all** NemotronH routed experts from HuggingFace ``NemotronHExperts`` into ``self`` buffers.

        Reads ``moe.experts.up_proj`` / ``down_proj`` (HF ``[E,I,H]`` / ``[E,H,I]``) with ``transpose(1,2)``
        into Marlin layout. Only supports ``moe.config.moe_latent_size is None`` (expert input dim ==
        ``hidden_size``); latent MoE raises ``ValueError``.
        """
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers.models.nemotron_h import failed)."
            )
        if not isinstance(moe, NemotronHMoE):
            raise TypeError(
                f"moe must be NemotronHMoE, got {type(moe).__name__!r}")
        if not hasattr(moe, "experts"):
            raise TypeError(
                f"moe must have ``experts``; got {type(moe).__name__!r}")
        # trust_remote_code NemotronHMoE: n_routed_experts lives on .gate or
        # .experts.num_experts (NemotronHExperts has no __len__).
        _gate = getattr(moe, "gate", None)
        if hasattr(moe, "n_routed_experts"):
            _n_experts = int(moe.n_routed_experts)
        elif _gate is not None and hasattr(_gate, "n_routed_experts"):
            _n_experts = int(_gate.n_routed_experts)
        else:
            _n_experts = int(moe.experts.num_experts)
        if _n_experts != int(self.num_experts):
            raise ValueError(
                f"moe n_routed_experts ({_n_experts}) != self.num_experts ({self.num_experts})"
            )

        cfg = getattr(moe, "config", None)
        if cfg is None:
            raise TypeError(
                "packing Marlin weights from HF needs ``moe.config`` (NemotronHConfig) "
                "to resolve expert weight shapes (moe_intermediate_size, moe_latent_size, hidden_size)"
            )
        if int(cfg.moe_intermediate_size) != int(self.moe_inter_size):
            raise ValueError(
                f"moe.config.moe_intermediate_size ({cfg.moe_intermediate_size}) != "
                f"self.moe_inter_size ({self.moe_inter_size})")
        latent = getattr(cfg, "moe_latent_size", None)
        expert_in = int(latent) if latent is not None else int(cfg.hidden_size)
        if expert_in != int(self.hidden_size):
            raise ValueError(
                "Nvfp4MoePlugin / NemotronHMoEW4A4Plugin expect expert input dim == model hidden_size "
                f"(use ``moe_latent_size=None``); got expert_input_dim={expert_in} from moe.config, "
                f"self.hidden_size={self.hidden_size}")

        experts = moe.experts
        up = experts.up_proj
        down = experts.down_proj
        if not isinstance(up, nn.Parameter) or not isinstance(
                down, nn.Parameter):
            raise TypeError(
                "expected experts.up_proj and experts.down_proj to be nn.Parameter stacked tensors "
                f"(got up_proj={type(up).__name__!r}, down_proj={type(down).__name__!r})"
            )
        e_ct = int(_n_experts)
        if int(up.shape[0]) != e_ct or int(down.shape[0]) != e_ct:
            raise ValueError(
                f"expert dim mismatch: n_routed_experts={e_ct}, up.shape={tuple(up.shape)}, "
                f"down.shape={tuple(down.shape)}")
        w_up_ehi = up.data.transpose(1, 2).contiguous()
        w_down_eih = down.data.transpose(1, 2).contiguous()
        # Plugin v1: both FC1 and FC2 weights are N-major (Marlin-compatible
        # bytes); prefill SFs use raw IEEE FP8 E4M3, decode SFs use Marlin
        # projection. populate_prefill fills both; populate_decode is an
        # idempotent refill used on weight-reload paths.
        self.populate_prefill_plugin_buffers(w_up_ehi, w_down_eih)
        self.populate_decode_plugin_buffers()


def register_nvfp4_moe_plugin_onnx_symbolic_functions() -> None:
    register_custom_op_symbolic(
        "trt::nvfp4_moe_plugin",
        symbolic_nvfp4_moe_plugin,
        ONNX_OPSET_VERSION,
    )


def replace_moe_blocks_with_nvfp4_plugin(model: nn.Module) -> nn.Module:
    """Replace ``NemotronHMoE`` blocks with ``NemotronHMoEW4A4Plugin`` (NVFP4 buffers must be filled).

    Routing parameters (``n_group`` / ``topk_group`` / ``norm_topk_prob`` /
    ``routed_scaling_factor`` / ``routing_mode``) are inferred from ``moe_block.config``
    when present; NemotronH configs expose them via the DeepSeek-style sigmoid-group
    router, so the plugin runs ``moeSigmoidGroupTopk`` by default for those models.
    """
    if NemotronHMoE is None:
        return model
    for name, module in list(model.named_modules()):
        new_module = None
        if isinstance(module, NemotronHMoE):
            cfg = getattr(module, "config", None)
            n_group = int(getattr(cfg, "n_group", 1) or 1)
            topk_group = int(getattr(cfg, "topk_group", 1) or 1)
            norm_topk_prob = int(bool(getattr(cfg, "norm_topk_prob", True)))
            routed_scaling_factor = float(
                getattr(cfg, "routed_scaling_factor", 1.0) or 1.0)
            # Default to sigmoid-group-topk when the config advertises a grouped router
            # (n_group > 1), else fall back to the legacy softmax-topk path.
            routing_mode = 1 if n_group > 1 else 0
            try:
                new_module = NemotronHMoEW4A4Plugin(
                    module,
                    n_group=n_group,
                    topk_group=topk_group,
                    norm_topk_prob=norm_topk_prob,
                    routed_scaling_factor=routed_scaling_factor,
                    routing_mode=routing_mode,
                )
                e_score_bias = getattr(module.gate, "e_score_correction_bias",
                                       None)
                if e_score_bias is None:
                    e_score_bias = getattr(module, "e_score_correction_bias",
                                           None)
                if isinstance(e_score_bias, torch.Tensor):
                    with torch.no_grad():
                        new_module.e_score_correction_bias.copy_(
                            e_score_bias.detach().to(
                                dtype=torch.float32,
                                device=new_module.e_score_correction_bias.
                                device))
            except (TypeError, ValueError):
                new_module = None
        if new_module is None:
            continue
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
