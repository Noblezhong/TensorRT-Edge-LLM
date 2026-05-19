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
Quantization utilities for TensorRT Edge-LLM.

This module provides functions for quantizing LLM and visual models using NVIDIA ModelOpt.
"""

# FP8 (E4M3) quantization constants
# Max finite value representable by NVIDIA FP8 E4M3 format; used to derive per-tensor KV cache scale.
# NOTE: This constant must be defined BEFORE any imports to avoid circular import issues.
FP8_E4M3_MAX: float = 448.0

try:
    from .llm_quantization import quantize_and_save_draft, quantize_and_save_llm
except ImportError:
    # ModelOpt is optional for export-only / vocab-reduction workflows.
    quantize_and_save_draft = None
    quantize_and_save_llm = None

__all__ = [
    "quantize_and_save_llm",
    "quantize_and_save_draft",
    "FP8_E4M3_MAX",
]
