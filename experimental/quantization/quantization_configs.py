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
"""Quantization recipe configurations for ModelOpt."""

from typing import Any, Dict, Optional

import modelopt.torch.quantization as mtq

FP8_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "default": {
            "enable": False
        },
    }
}

INT4_AWQ_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {
                -1: 128,
                "type": "static"
            },
            "enable": True,
        },
        "default": {
            "enable": False
        },
    }
}

NVFP4_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True,
        },
        "default": {
            "enable": False
        },
    }
}

MXFP8_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "default": {
            "enable": False
        },
    }
}

FP8_ATTN = {
    "quant_cfg": {
        "*q_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "*k_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "*v_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
    }
}

DISABLE_NON_LLM = {
    "quant_cfg": {
        k: {
            "enable": False
        }
        for k in (
            "*vision_backbone.*",
            "*visual.*",
            "*vision_tower.*",
            "*multi_modal_projector.*",
            "*projector.*",
            "*mlp1.*",
            "*audio_tower.*",
            "*audio_embed.*",
            "*image_embed.*",
            "*code_predictor.*",
            "*code2wav.*",
        )
    }
}

_BACKBONE_CFG_MAP = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
}

_LM_HEAD_CFG_MAP = {
    "fp8": FP8_LM_HEAD,
    "int4_awq": INT4_AWQ_LM_HEAD,
    "nvfp4": NVFP4_LM_HEAD,
    "mxfp8": MXFP8_LM_HEAD,
}


def build_quant_config(
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a composite ModelOpt quantization config from method names."""
    if quantization is None:
        cfg = {"quant_cfg": {}, "algorithm": "max"}
    elif quantization in _BACKBONE_CFG_MAP:
        cfg = _BACKBONE_CFG_MAP[quantization].copy()
    else:
        raise ValueError(f"Unsupported quantization: {quantization}. "
                         f"Choose from: {list(_BACKBONE_CFG_MAP)}")

    if lm_head_quantization is not None:
        if lm_head_quantization not in _LM_HEAD_CFG_MAP:
            raise ValueError(
                f"Unsupported lm_head_quantization: {lm_head_quantization}. "
                f"Choose from: {list(_LM_HEAD_CFG_MAP)}")
        cfg["quant_cfg"] = {
            k: v
            for k, v in cfg["quant_cfg"].items() if "*lm_head" not in k
        }
        cfg["quant_cfg"].update(
            _LM_HEAD_CFG_MAP[lm_head_quantization]["quant_cfg"])

    if kv_cache_quantization == "fp8":
        cfg["quant_cfg"].update(mtq.FP8_KV_CFG["quant_cfg"])
        cfg["quant_cfg"].update(FP8_ATTN["quant_cfg"])

    cfg["quant_cfg"].update(DISABLE_NON_LLM["quant_cfg"])
    return cfg
